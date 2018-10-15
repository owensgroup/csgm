#define GRB_USE_APSPIE
#define private public
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

#include "auction.cu"
#include "utils.cu"

#define NUM_SEEDS 100

int main( int argc, char** argv )
{
  bool verbose = true;
  float tolerance = 1.0;

  // ----------------------------------------------------------------------
  // CLI

  po::variables_map vm;
  parseArgs( argc, argv, vm );
  graphblas::Descriptor desc;
  desc.loadArgs(vm);

  // ----------------------------------------------------------------------
  // IO

  std::vector<graphblas::Index> a_row_indices, b_row_indices, p_row_indices, t_row_indices;
  std::vector<graphblas::Index> a_col_indices, b_col_indices, p_col_indices, t_col_indices;
  std::vector<float> a_values, b_values, p_values, t_values;
  graphblas::Index num_nodes;
  graphblas::Index a_num_edges, b_num_edges;

  readMtx("data/A.mtx", a_row_indices, a_col_indices, a_values, num_nodes, num_nodes, a_num_edges, 0, false);
  readMtx("data/B.mtx", b_row_indices, b_col_indices, b_values, num_nodes, num_nodes, b_num_edges, 0, false);

  // ----------------------------------------------------------------------
  // Initialize data structures

  graphblas::Matrix<float>   _A(num_nodes, num_nodes); graphblas::Matrix<float>* A   = &_A;
  graphblas::Matrix<float>   _B(num_nodes, num_nodes); graphblas::Matrix<float>* B   = &_B;
  graphblas::Matrix<float>   _P(num_nodes, num_nodes); graphblas::Matrix<float>* P   = &_P;
  graphblas::Matrix<float>  _AP(num_nodes, num_nodes); graphblas::Matrix<float>* AP  = &_AP;
  graphblas::Matrix<float> _APB(num_nodes, num_nodes); graphblas::Matrix<float>* APB = &_APB;
  graphblas::Matrix<float>   _T(num_nodes, num_nodes); graphblas::Matrix<float>* T   = &_T;
  graphblas::Matrix<float>  _AT(num_nodes, num_nodes); graphblas::Matrix<float>* AT  = &_AT;
  graphblas::Matrix<float> _ATB(num_nodes, num_nodes); graphblas::Matrix<float>* ATB = &_ATB;
  graphblas::Matrix<float>  _PB(num_nodes, num_nodes); graphblas::Matrix<float>* PB  = &_PB;
  graphblas::Matrix<float>  _TB(num_nodes, num_nodes); graphblas::Matrix<float>* TB  = &_TB;
  graphblas::Matrix<float>* tmp;

  A->build(&a_row_indices, &a_col_indices, &a_values, a_num_edges, GrB_NULL);
  B->build(&b_row_indices, &b_col_indices, &b_values, b_num_edges, GrB_NULL);

  for(graphblas::Index i = 0; i < NUM_SEEDS; i++) {
    p_row_indices.push_back(i);
    p_col_indices.push_back(i);
    p_values.push_back(1.0f);
  }
  P->build(&p_row_indices, &p_col_indices, &p_values, NUM_SEEDS, GrB_NULL);
  easy_mxm(AP,   A, P, &desc);
  easy_mxm(APB, AP, B, &desc);

  int* h_ascending = (int*) malloc((num_nodes+1)*sizeof(int));;
  float* h_ones    = (float*) malloc(num_nodes*sizeof(int));
  for (int i = 0; i < num_nodes; ++i) {
    h_ascending[i] = i;
    h_ones[i]      = 1.f;
  }
  h_ascending[num_nodes] = num_nodes;

  CpuTimer timer;
  for(int iter = 0; iter < 20; iter++) {
    if(verbose) {
      std::cerr << "===== iter=" << iter << " ================================" << std::endl;
    }
    timer.Start();

    // --------------------------
    // Solve LAP

    int* d_person2item;
    cudaMalloc((void **)&d_person2item, num_nodes * sizeof(int));

    int APB_num_edges; APB->nvals(&APB_num_edges);
    run_auction(
        num_nodes,
        APB_num_edges,

        APB->matrix_.sparse_.d_csrVal_,
        APB->matrix_.sparse_.d_csrRowPtr_,
        APB->matrix_.sparse_.d_csrColInd_,

        d_person2item,

        0.1,
        0.1,
        0.0,

        1,
        1
    );

    float* d_ones;
    cudaMalloc((void **)&d_ones, num_nodes * sizeof(float));
    cudaMemcpy(d_ones, h_ones, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    int* d_ascending;
    cudaMalloc((void **)&d_ascending, (num_nodes+1) * sizeof(int));
    cudaMemcpy(d_ascending, h_ascending, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

    T->build(d_ascending, d_person2item, d_ones, num_nodes);

    // --------------------------
    // Matmuls

    AT->clear();  easy_mxm(AT,   A,  T, &desc);
    ATB->clear(); easy_mxm(ATB, AT,  B, &desc);
    PB->clear();  easy_mxm(PB,   P,  B, &desc);
    TB->clear();  easy_mxm(TB,   T,  B, &desc);

    // --------------------------
    // Step size + convergence checking

    float APPB_trace = gpu_trace(AP, PB, &desc);
    float APTB_trace = gpu_trace(AP, TB, &desc);
    float ATPB_trace = gpu_trace(AT, PB, &desc);
    float ATTB_trace = gpu_trace(AT, TB, &desc);

    float T_sum = (float)num_nodes;
    int P_num_values; P->nvals(&P_num_values);
    float P_sum = sum_reduce(P->matrix_.sparse_.d_csrVal_, P_num_values);

    graphblas::Vector<float> AP_rowsum(num_nodes); rowsum(&AP_rowsum,  AP, &desc);
    graphblas::Vector<float> AT_rowsum(num_nodes); rowsum(&AT_rowsum,  AT, &desc);
    graphblas::Vector<float> B_rowsum(num_nodes);  rowsum( &B_rowsum,   B, &desc);

    graphblas::Vector<float> PAP_sum(num_nodes); easy_mxv(&PAP_sum,  P, &AP_rowsum, &desc);
    graphblas::Vector<float> PAT_sum(num_nodes); easy_mxv(&PAT_sum,  P, &AT_rowsum, &desc);
    graphblas::Vector<float> TAP_sum(num_nodes); easy_mxv(&TAP_sum,  T, &AP_rowsum, &desc);
    graphblas::Vector<float> TAT_sum(num_nodes); easy_mxv(&TAT_sum,  T, &AT_rowsum, &desc);

    graphblas::Vector<float> BP_sum(num_nodes); easy_vxm(&BP_sum, &B_rowsum, P, &desc);
    graphblas::Vector<float> BT_sum(num_nodes); easy_vxm(&BT_sum, &B_rowsum, T, &desc);

    float PAP_sum_sum = sum_reduce(PAP_sum.vector_.dense_.d_val_, num_nodes);
    float PAT_sum_sum = sum_reduce(PAT_sum.vector_.dense_.d_val_, num_nodes);
    float TAP_sum_sum = sum_reduce(TAP_sum.vector_.dense_.d_val_, num_nodes);
    float TAT_sum_sum = sum_reduce(TAT_sum.vector_.dense_.d_val_, num_nodes);
    float BP_sum_sum  = sum_reduce(BP_sum.vector_.sparse_.d_val_, num_nodes);
    float BT_sum_sum  = sum_reduce(BT_sum.vector_.sparse_.d_val_, num_nodes);

    float ps_grad_P  = 4 * APPB_trace + (float)num_nodes * P_sum - 2 * (PAP_sum_sum + BP_sum_sum);
    float ps_grad_T  = 4 * APTB_trace + (float)num_nodes * T_sum - 2 * (TAP_sum_sum + BT_sum_sum);
    float ps_gradt_P = 4 * ATPB_trace + (float)num_nodes * P_sum - 2 * (PAT_sum_sum + BP_sum_sum);
    float ps_gradt_T = 4 * ATTB_trace + (float)num_nodes * T_sum - 2 * (TAT_sum_sum + BT_sum_sum);

    // --
    // Check convergence

    float c = ps_grad_P;
    float d = ps_gradt_P + ps_grad_T;
    float e = ps_gradt_T;

    float cde = c + e - d;
    float d2e = d - 2 * e;
    float alpha, falpha;
    if((cde == 0) && (d2e == 0)) {
      alpha  = 0.0;
      falpha = -1;
    } else {
      if(cde == 0) {
        alpha  = -1;
        falpha = -1;
      } else {
        alpha = - d2e / (2 * cde);
        falpha = cde * pow(alpha, 2) + d2e * alpha;
      }
    }

    float f1 = c - e;

    float num_diff = pow(num_nodes, 2) - ps_grad_P; // Number of disagreements (unweighted graph)

    if(verbose) {
      std::cerr << "APPB_trace= " << std::setprecision(9) << APPB_trace << std::endl;
      std::cerr << "APTB_trace= " << std::setprecision(9) << APTB_trace << std::endl;
      std::cerr << "ATPB_trace= " << std::setprecision(9) << ATPB_trace << std::endl;
      std::cerr << "ATTB_trace= " << std::setprecision(9) << ATTB_trace << std::endl;
      std::cerr << "ps_grad_P=  " << std::setprecision(9) << ps_grad_P  << std::endl;
      std::cerr << "ps_grad_T=  " << std::setprecision(9) << ps_grad_T  << std::endl;
      std::cerr << "ps_gradt_P= " << std::setprecision(9) << ps_gradt_P << std::endl;
      std::cerr << "ps_gradt_T= " << std::setprecision(9) << ps_gradt_T << std::endl;
      std::cerr << "alpha=      " << alpha << std::endl;
      std::cerr << "falpha=     " << falpha << std::endl;
      std::cerr << "f1=         " << f1 << std::endl;
      std::cerr << "num_diff=   " << num_diff << std::endl;
      std::cerr << "------------" << std::endl;
    }

    if((alpha > 0) && (alpha < tolerance) && (falpha > 0) && (falpha > f1)) {
      graphblas::Matrix<float> new_P(num_nodes, num_nodes);
      add_matrix(P, T, &new_P, alpha, 1 - alpha);
      P->clear();
      P = &new_P;

      graphblas::Matrix<float> new_APB(num_nodes, num_nodes);
      add_matrix(APB, ATB, &new_APB, alpha, 1 - alpha);
      APB->clear();
      APB = &new_APB;

      graphblas::Matrix<float> new_AP(num_nodes, num_nodes);
      add_matrix(AP, AT, &new_AP, alpha, 1 - alpha);
      AP->clear();
      AP = &new_AP;

      timer.Stop();
      std::cerr << "timer=" << timer.ElapsedMillis() << std::endl;
    } else if(f1 < 0) {
      P->clear();
      AP->clear();
      APB->clear();

      std::swap(P, T);
      std::swap(AP, AT);
      std::swap(APB, ATB);

      timer.Stop();
      std::cerr << "timer=" << timer.ElapsedMillis() << std::endl;
    } else {
      break;
    }

  }
  timer.Stop();
  std::cerr << "timer=" << timer.ElapsedMillis() << std::endl;
}
