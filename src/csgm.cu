#define GRB_USE_APSPIE
#define private public

#include "csgm.cuh"

#include "utils.cu"
#include "cli.h"
#include "auction.cu"
#include "timer.cuh"

void init_P(FloatMatrix* P, int num_seeds) {
    std::vector<graphblas::Index> p_row_indices, p_col_indices;
    std::vector<float> p_values;

    for(graphblas::Index i = 0; i < num_seeds; i++) {
      p_row_indices.push_back(i);
      p_col_indices.push_back(i);
      p_values.push_back(1.0f);
    }
    P->build(&p_row_indices, &p_col_indices, &p_values, num_seeds, GrB_NULL);
}

void init_T(FloatMatrix* T, int* d_ascending, int* d_person2item, float* d_ones, int num_nodes) {
  int* h_ascending = (int*) malloc((num_nodes+1)*sizeof(int));;
  float* h_ones    = (float*) malloc(num_nodes*sizeof(int));
  for (int i = 0; i < num_nodes; ++i) {
    h_ascending[i] = i;
    h_ones[i]      = 1.f;
  }
  h_ascending[num_nodes] = num_nodes;

  cudaMemcpy(d_ones, h_ones, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ascending, h_ascending, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

  T->build(d_ascending, d_person2item, d_ones, num_nodes);

  free(h_ones);
  free(h_ascending);
}

int main(int argc, char** argv)
{
  // ----------------------------------------------------------------------
  // CLI

  po::variables_map vm;

  parseArgsSGM(argc, argv, vm);
  int num_seeds;
  try {
    num_seeds = vm["num-seeds"].as<int>();
  } catch(const std::exception& e) {
    std::cerr << "csgm: must specify `--num-seeds`" << std::endl;
    exit(1);
  }
  int num_iters     = vm["num-iters"].as<int>();
  bool verbose      = vm["sgm-debug"].as<bool>();
  float tolerance   = vm["tolerance"].as<float>();
  float auction_max_eps = vm["auction-max-eps"].as<float>();
  float auction_min_eps = vm["auction-min-eps"].as<float>();
  float auction_factor  = vm["auction-factor"].as<float>();

  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  // ----------------------------------------------------------------------
  // IO

  std::vector<graphblas::Index> a_row_indices, b_row_indices, p_row_indices, t_row_indices;
  std::vector<graphblas::Index> a_col_indices, b_col_indices, p_col_indices, t_col_indices;
  std::vector<float> a_values, b_values, p_values, t_values;
  graphblas::Index num_nodes;
  graphblas::Index a_num_edges, b_num_edges;

  std::string A_path = vm["A"].as<std::string>();
  std::string B_path = vm["B"].as<std::string>();

  readMtx(A_path.c_str(), a_row_indices, a_col_indices, a_values, num_nodes, num_nodes, a_num_edges, 0, false);
  readMtx(B_path.c_str(), b_row_indices, b_col_indices, b_values, num_nodes, num_nodes, b_num_edges, 0, false);

  // ----------------------------------------------------------------------
  // Initialize data structures

  FloatMatrix   _A(num_nodes, num_nodes); FloatMatrix* A   = &_A;
  FloatMatrix   _B(num_nodes, num_nodes); FloatMatrix* B   = &_B;
  FloatMatrix   _P(num_nodes, num_nodes); FloatMatrix* P   = &_P;
  FloatMatrix  _AP(num_nodes, num_nodes); FloatMatrix* AP  = &_AP;
  FloatMatrix _APB(num_nodes, num_nodes); FloatMatrix* APB = &_APB;
  FloatMatrix   _T(num_nodes, num_nodes); FloatMatrix* T   = &_T;
  FloatMatrix  _AT(num_nodes, num_nodes); FloatMatrix* AT  = &_AT;
  FloatMatrix _ATB(num_nodes, num_nodes); FloatMatrix* ATB = &_ATB;
  FloatMatrix  _PB(num_nodes, num_nodes); FloatMatrix* PB  = &_PB;
  FloatMatrix  _TB(num_nodes, num_nodes); FloatMatrix* TB  = &_TB;

  FloatVector AP_rowsum(num_nodes);
  FloatVector AT_rowsum(num_nodes);
  FloatVector B_rowsum(num_nodes);
  FloatVector BP_sum(num_nodes);
  FloatVector BT_sum(num_nodes);
  FloatVector PAP_sum(num_nodes);
  FloatVector PAT_sum(num_nodes);
  FloatVector TAP_sum(num_nodes);
  FloatVector TAT_sum(num_nodes);

  int P_num_values;
  int BP_sum_num_values;
  int BT_sum_num_values;

  A->build(&a_row_indices, &a_col_indices, &a_values, a_num_edges, GrB_NULL);
  B->build(&b_row_indices, &b_col_indices, &b_values, b_num_edges, GrB_NULL);
  init_P(P, num_seeds);

  easy_mxm(AP,   A, P, &desc);
  easy_mxm(APB, AP, B, &desc);

  int* d_ascending;
  int* d_person2item;
  float* d_ones;
  cudaMalloc((void **)&d_ones, num_nodes * sizeof(float));
  cudaMalloc((void **)&d_ascending, (num_nodes+1) * sizeof(int));
  cudaMalloc((void **)&d_person2item, num_nodes * sizeof(int));
  init_T(T, d_ascending, d_person2item, d_ones, num_nodes);

  GpuTimer iter_timer;
  GpuTimer total_timer;
  total_timer.Start();
  float num_diff;
  for(int iter = 0; iter < num_iters; iter++) {
    if(verbose) {
      std::cerr << "===== iter=" << iter << " ================================" << std::endl;
    }
    iter_timer.Start();

    // --------------------------
    // Solve LAP

    int APB_num_entries; APB->nvals(&APB_num_entries);
    std::cerr << "APB_num_entries=" << APB_num_entries << std::endl;
    run_auction(
        num_nodes,
        APB_num_entries,

        APB->matrix_.sparse_.d_csrVal_,
        APB->matrix_.sparse_.d_csrRowPtr_,
        APB->matrix_.sparse_.d_csrColInd_,

        d_person2item,

        auction_max_eps,
        auction_min_eps,
        auction_factor,

        1,           // num_runs
        int(verbose) // verbose
    );


    cudaMemcpy(T->matrix_.sparse_.d_csrColInd_, d_person2item, num_nodes * sizeof(int), cudaMemcpyDeviceToDevice);

    // --------------------------
    // Matmuls

    AT->clear();  easy_mxm(AT,   A,  T, &desc);
    ATB->clear(); easy_mxm(ATB, AT,  B, &desc);
    PB->clear();  easy_mxm(PB,   P,  B, &desc);
    TB->clear();  easy_mxm(TB,   T,  B, &desc);

    // --------------------------
    // Step size + convergence checking

    float APPB_trace;
    float APTB_trace;
    float ATPB_trace;
    float ATTB_trace;

    graphblas::traceMxmTranspose(&APPB_trace, graphblas::PlusMultipliesSemiring<float>(), AP, PB, &desc);
    graphblas::traceMxmTranspose(&APTB_trace, graphblas::PlusMultipliesSemiring<float>(), AP, TB, &desc);
    graphblas::traceMxmTranspose(&ATPB_trace, graphblas::PlusMultipliesSemiring<float>(), AT, PB, &desc);
    graphblas::traceMxmTranspose(&ATTB_trace, graphblas::PlusMultipliesSemiring<float>(), AT, TB, &desc);

    rowsum(&AP_rowsum,  AP, &desc);
    rowsum(&AT_rowsum,  AT, &desc);
    rowsum( &B_rowsum,  B, &desc);

    easy_vxm(&BP_sum, &B_rowsum, P, &desc);
    easy_vxm(&BT_sum, &B_rowsum, T, &desc);
    easy_mxv(&PAP_sum,  P, &AP_rowsum, &desc);
    easy_mxv(&PAT_sum,  P, &AT_rowsum, &desc);
    easy_mxv(&TAP_sum,  T, &AP_rowsum, &desc);
    easy_mxv(&TAT_sum,  T, &AT_rowsum, &desc);

    P->nvals(&P_num_values);
    BP_sum.nvals(&BP_sum_num_values);
    BT_sum.nvals(&BT_sum_num_values);

    float P_sum      = sum_reduce(P->matrix_.sparse_.d_csrVal_, P_num_values);
    float BP_sum_sum = sum_reduce(BP_sum.vector_.sparse_.d_val_, BP_sum_num_values);
    float BT_sum_sum = sum_reduce(BT_sum.vector_.sparse_.d_val_, BT_sum_num_values);

    float PAP_sum_sum = sum_reduce(PAP_sum.vector_.dense_.d_val_, num_nodes);
    float PAT_sum_sum = sum_reduce(PAT_sum.vector_.dense_.d_val_, num_nodes);
    float TAP_sum_sum = sum_reduce(TAP_sum.vector_.dense_.d_val_, num_nodes);
    float TAT_sum_sum = sum_reduce(TAT_sum.vector_.dense_.d_val_, num_nodes);

    float ps_grad_P  = 4 * APPB_trace + (float)num_nodes * P_sum - 2 * (PAP_sum_sum + BP_sum_sum);
    float ps_gradt_P = 4 * ATPB_trace + (float)num_nodes * P_sum - 2 * (PAT_sum_sum + BP_sum_sum);
    float ps_grad_T  = 4 * APTB_trace + (float)num_nodes * float(num_nodes) - 2 * (TAP_sum_sum + BT_sum_sum);
    float ps_gradt_T = 4 * ATTB_trace + (float)num_nodes * float(num_nodes) - 2 * (TAT_sum_sum + BT_sum_sum);

    // --
    // Check convergence

    float cde = (ps_grad_P + ps_gradt_T) - (ps_gradt_P + ps_grad_T);
    float d2e = ps_gradt_P + ps_grad_T - 2 * ps_gradt_T;
    float alpha, falpha;
    if((cde == 0) && (d2e == 0)) {
      alpha  = 0.0;
      falpha = -1;
    } else {
      if(cde == 0) {
        alpha  = -1.0;
        falpha = -1;
      } else {
        alpha = - d2e / (2 * cde);
        falpha = cde * pow(alpha, 2) + d2e * alpha;
      }
    }

    float f1 = ps_grad_P - ps_gradt_T;
    num_diff = a_num_edges + b_num_edges - 2 * ATTB_trace; // Number of disagreements (unweighted graph)

    if(verbose) {
      std::cerr << "APPB_trace = " << std::setprecision(9) << APPB_trace << std::endl;
      std::cerr << "APTB_trace = " << std::setprecision(9) << APTB_trace << std::endl;
      std::cerr << "ATPB_trace = " << std::setprecision(9) << ATPB_trace << std::endl;
      std::cerr << "ATTB_trace = " << std::setprecision(9) << ATTB_trace << std::endl;
      std::cerr << "ps_grad_P  = " << std::setprecision(9) << ps_grad_P  << std::endl;
      std::cerr << "ps_grad_T  = " << std::setprecision(9) << ps_grad_T  << std::endl;
      std::cerr << "ps_gradt_P = " << std::setprecision(9) << ps_gradt_P << std::endl;
      std::cerr << "ps_gradt_T = " << std::setprecision(9) << ps_gradt_T << std::endl;
      std::cerr << "alpha      = " << std::setprecision(9) << alpha << std::endl;
      std::cerr << "falpha     = " << std::setprecision(9) << falpha << std::endl;
      std::cerr << "f1         = " << std::setprecision(9) << f1 << std::endl;
      std::cerr << "num_diff   = " << num_diff << std::endl;
      std::cerr << "------------"  << std::endl;
    }

    if((alpha > 0) && (alpha < tolerance) && (falpha > 0) && (falpha > f1)) {
      std::cerr << "alpha > 0" << std::endl;
      spmm_convex_combination(P, T, alpha, 1 - alpha);
      spmm_convex_combination(APB, ATB, alpha, 1 - alpha);
      spmm_convex_combination(AP, AT, alpha, 1 - alpha);

    } else if(f1 < 0) {
      std::cerr << "f1 < 0" << std::endl;
      P->clear();
      AP->clear();
      APB->clear();

      P->dup(T);
      AP->dup(AT);
      APB->dup(ATB);

    } else {
      break;
    }
    iter_timer.Stop();
    std::cerr << "iter_timer=" << iter_timer.ElapsedMillis() << std::endl;

  }
  iter_timer.Stop();
  std::cerr << "iter_timer=" << iter_timer.ElapsedMillis() << std::endl;

  total_timer.Stop();
  std::cerr << "total_timer=" << total_timer.ElapsedMillis() << " | num_diff=" << num_diff << std::endl;
}
