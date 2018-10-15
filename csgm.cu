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
#include "timer.cuh"

#define NUM_SEEDS 100
#define NUM_ITERS 20
#define TOLERANCE 1.0;

typedef graphblas::Matrix<float> Matrix;
typedef graphblas::Vector<float> Vector;


int main( int argc, char** argv )
{
  bool verbose  = false;
  int num_seeds = NUM_SEEDS;
  int num_iters = NUM_ITERS;
  float tolerance = TOLERANCE;

  // ----------------------------------------------------------------------
  // CLI

  po::variables_map vm;
  parseArgs(argc, argv, vm);
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

  Matrix   _A(num_nodes, num_nodes); Matrix* A   = &_A;
  Matrix   _B(num_nodes, num_nodes); Matrix* B   = &_B;
  Matrix   _P(num_nodes, num_nodes); Matrix* P   = &_P;
  Matrix  _AP(num_nodes, num_nodes); Matrix* AP  = &_AP;
  Matrix _APB(num_nodes, num_nodes); Matrix* APB = &_APB;
  Matrix   _T(num_nodes, num_nodes); Matrix* T   = &_T;
  Matrix  _AT(num_nodes, num_nodes); Matrix* AT  = &_AT;
  Matrix _ATB(num_nodes, num_nodes); Matrix* ATB = &_ATB;
  Matrix  _PB(num_nodes, num_nodes); Matrix* PB  = &_PB;
  Matrix  _TB(num_nodes, num_nodes); Matrix* TB  = &_TB;

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

  // Data structures for auction
  int* h_ascending = (int*) malloc((num_nodes+1)*sizeof(int));;
  float* h_ones    = (float*) malloc(num_nodes*sizeof(int));
  for (int i = 0; i < num_nodes; ++i) {
    h_ascending[i] = i;
    h_ones[i]      = 1.f;
  }
  h_ascending[num_nodes] = num_nodes;

  float* d_ones;
  int* d_ascending;
  int* d_person2item;

  cudaMalloc((void **)&d_ones, num_nodes * sizeof(float));
  cudaMalloc((void **)&d_ascending, (num_nodes+1) * sizeof(int));
  cudaMalloc((void **)&d_person2item, num_nodes * sizeof(int));

  cudaMemcpy(d_ones, h_ones, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ascending, h_ascending, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

  T->build(d_ascending, d_person2item, d_ones, num_nodes);

  GpuTimer timer;
  for(int iter = 0; iter < num_iters; iter++) {
    if(verbose) {
      std::cerr << "===== iter=" << iter << " ================================" << std::endl;
    }
    timer.Start();

    // --------------------------
    // Solve LAP

    int APB_num_entries; APB->nvals(&APB_num_entries);
    run_auction(
        num_nodes,
        APB_num_entries,

        APB->matrix_.sparse_.d_csrVal_,
        APB->matrix_.sparse_.d_csrRowPtr_,
        APB->matrix_.sparse_.d_csrColInd_,

        d_person2item,

        0.1,
        0.1,
        0.0,

        1,
        int(verbose)
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

    float APPB_trace = compute_trace(AP, PB, &desc);
    float APTB_trace = compute_trace(AP, TB, &desc);
    float ATPB_trace = compute_trace(AT, PB, &desc);
    float ATTB_trace = compute_trace(AT, TB, &desc);

    float T_sum = (float)num_nodes;
    int P_num_values; P->nvals(&P_num_values);
    float P_sum = sum_reduce(P->matrix_.sparse_.d_csrVal_, P_num_values);

    Vector AP_rowsum(num_nodes); rowsum(&AP_rowsum,  AP, &desc);
    Vector AT_rowsum(num_nodes); rowsum(&AT_rowsum,  AT, &desc);
    Vector B_rowsum(num_nodes);  rowsum( &B_rowsum,   B, &desc);

    Vector PAP_sum(num_nodes); easy_mxv(&PAP_sum,  P, &AP_rowsum, &desc);
    Vector PAT_sum(num_nodes); easy_mxv(&PAT_sum,  P, &AT_rowsum, &desc);
    Vector TAP_sum(num_nodes); easy_mxv(&TAP_sum,  T, &AP_rowsum, &desc);
    Vector TAT_sum(num_nodes); easy_mxv(&TAT_sum,  T, &AT_rowsum, &desc);

    Vector BP_sum(num_nodes); easy_vxm(&BP_sum, &B_rowsum, P, &desc);
    Vector BT_sum(num_nodes); easy_vxm(&BT_sum, &B_rowsum, T, &desc);

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
      // std::cerr << "APPB_trace= " << std::setprecision(9) << APPB_trace << std::endl;
      // std::cerr << "APTB_trace= " << std::setprecision(9) << APTB_trace << std::endl;
      // std::cerr << "ATPB_trace= " << std::setprecision(9) << ATPB_trace << std::endl;
      // std::cerr << "ATTB_trace= " << std::setprecision(9) << ATTB_trace << std::endl;
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
      spmm_convex_combination(P, T, alpha, 1 - alpha);
      spmm_convex_combination(APB, ATB, alpha, 1 - alpha);
      spmm_convex_combination(AP, AT, alpha, 1 - alpha);

    } else if(f1 < 0) {
      P->clear();
      AP->clear();
      APB->clear();

      // std::swap(P, T);
      // std::swap(AP, AT);
      // std::swap(APB, ATB);
      P->dup(T);
      AP->dup(AT);
      APB->dup(ATB);
    } else {
      break;
    }
    timer.Stop();
    std::cerr << "timer=" << timer.ElapsedMillis() << std::endl;

  }
  timer.Stop();
  std::cerr << "timer=" << timer.ElapsedMillis() << std::endl;
}
