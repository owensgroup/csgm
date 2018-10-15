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
  bool DEBUG = false;
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
  graphblas::Index num_rows, num_cols;
  graphblas::Index a_num_edges, b_num_edges;

  // Load A
  readMtx("data/A.mtx", a_row_indices, a_col_indices, a_values, num_rows, num_cols, a_num_edges, 0, false);
  graphblas::Matrix<float> A(num_rows, num_cols);
  A.build(&a_row_indices, &a_col_indices, &a_values, a_num_edges, GrB_NULL);

  // Load B
  readMtx("data/B.mtx", b_row_indices, b_col_indices, b_values, num_rows, num_cols, b_num_edges, 0, false);
  graphblas::Matrix<float> B(num_rows, num_cols);
  B.build(&b_row_indices, &b_col_indices, &b_values, b_num_edges, GrB_NULL);

  // Create P
  for(graphblas::Index i = 0; i < NUM_SEEDS; i++) {
    p_row_indices.push_back(i);
    p_col_indices.push_back(i);
    p_values.push_back(1.0f);
  }
  graphblas::Matrix<float> _P(num_rows, num_cols);
  _P.build(&p_row_indices, &p_col_indices, &p_values, NUM_SEEDS, GrB_NULL);
  graphblas::Matrix<float>* P = &_P;

  // ----------------------------------------------------------------------
  // Allocation for intermediate/final results

  graphblas::Matrix<float>  _AP(num_rows, num_cols);
  graphblas::Matrix<float> _APB(num_rows, num_cols);
  graphblas::Matrix<float>    T(num_rows, num_cols);
  graphblas::Matrix<float>   AT(num_rows, num_cols);
  graphblas::Matrix<float>  ATB(num_rows, num_cols);
  graphblas::Matrix<float>   PB(num_rows, num_cols);
  graphblas::Matrix<float>   TB(num_rows, num_cols);

  easy_mxm(&_AP,  &A,  P, &desc); graphblas::Matrix<float>* AP = &_AP;
  easy_mxm(&_APB, AP, &B, &desc); graphblas::Matrix<float>* APB = &_APB;

  int* h_ascending = (int*) malloc((num_rows+1)*sizeof(int));;
  float* h_ones    = (float*) malloc(num_rows*sizeof(int));
  for (int i = 0; i < num_rows; ++i) {
    h_ascending[i] = i;
    h_ones[i]      = 1.f;
  }
  h_ascending[num_rows] = num_rows;

  int* d_person2item;
  int* d_ascending;
  float* d_ones;
  cudaMalloc((void **)&d_person2item,   num_rows * sizeof(int));
  cudaMalloc((void **)&d_ascending, (num_rows+1) * sizeof(int));
  cudaMalloc((void **)&d_ones,          num_rows * sizeof(float));

  cudaMemcpy(d_ascending, h_ascending, (num_rows+1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ones, h_ones, num_rows*sizeof(int), cudaMemcpyHostToDevice);

  for(int iter = 0; iter < 20; iter++) {
    std::cerr << "===== iter=" << iter << std::endl;

    // --------------------------
    // Solve LAP

    int APB_num_edges; APB->nvals(&APB_num_edges);
    run_auction(
        num_rows,
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
    cudaDeviceSynchronize(); // ??

// <<
    // ----------------------------------------------------------------------
    // Read results
    int* h_person2item = (int*)malloc(num_rows * sizeof(int));
    cudaMemcpy(h_person2item, d_person2item, num_rows * sizeof(int), cudaMemcpyDeviceToHost);
    APB->matrix_.sparse_.gpuToCpu();
    float score = 0;
    for (int i = 0; i < num_rows; i++) {
      int start = APB->matrix_.sparse_.h_csrRowPtr_[i];
      int end   = APB->matrix_.sparse_.h_csrRowPtr_[i + 1];
      for(int j = start; j < end; j++) {
        if(APB->matrix_.sparse_.h_csrColInd_[j] == h_person2item[i]) {
          score += APB->matrix_.sparse_.h_csrVal_[j];
        }
      }
    }
    std::cout << "score=" << score << std::endl;
// >>

    float trace1 = -1;
    if(iter > 0) {
      trace1 = gpu_trace(P, &T, &desc);
    }
    T.build(d_ascending, d_person2item, d_ones, num_rows);
    float trace2 = gpu_trace(P, &T, &desc);

    std::cerr << "trace1= " << std::setprecision(9) << trace1 << std::endl;
    std::cerr << "trace2= " << std::setprecision(9) << trace2 << std::endl;

    // --------------------------
    // Matmuls

    AT.clear();  easy_mxm(&AT,   &A, &T,  &desc);
    ATB.clear(); easy_mxm(&ATB, &AT, &B,  &desc);
    PB.clear();  easy_mxm(&PB,    P, &B,  &desc);
    TB.clear();  easy_mxm(&TB,   &T, &B,  &desc);

    // --------------------------
    // Step size + convergence checking

    float APPB_trace = gpu_trace(AP, &PB, &desc);
    float APTB_trace = gpu_trace(AP, &TB, &desc);
    float ATPB_trace = gpu_trace(&AT, &PB, &desc);
    float ATTB_trace = gpu_trace(&AT, &TB, &desc);

    std::cerr << "APPB_trace= " << std::setprecision(9) << APPB_trace << std::endl;
    std::cerr << "APTB_trace= " << std::setprecision(9) << APTB_trace << std::endl;
    std::cerr << "ATPB_trace= " << std::setprecision(9) << ATPB_trace << std::endl;
    std::cerr << "ATTB_trace= " << std::setprecision(9) << ATTB_trace << std::endl;

    float T_sum = (float)num_rows;
    int P_num_values; P->nvals(&P_num_values);
    float P_sum = sum_reduce(P->matrix_.sparse_.d_csrVal_, P_num_values);

    graphblas::Vector<float> AP_rowsum(num_rows); rowsum(&AP_rowsum,  AP, &desc);
    graphblas::Vector<float> AT_rowsum(num_rows); rowsum(&AT_rowsum, &AT, &desc);
    graphblas::Vector<float> B_rowsum(num_rows);  rowsum( &B_rowsum,  &B, &desc);

    graphblas::Vector<float> PAP_sum(num_rows); easy_mxv(&PAP_sum,  P, &AP_rowsum, &desc);
    graphblas::Vector<float> PAT_sum(num_rows); easy_mxv(&PAT_sum,  P, &AT_rowsum, &desc);
    graphblas::Vector<float> TAP_sum(num_rows); easy_mxv(&TAP_sum, &T, &AP_rowsum, &desc);
    graphblas::Vector<float> TAT_sum(num_rows); easy_mxv(&TAT_sum, &T, &AT_rowsum, &desc);

    graphblas::Vector<float> BP_sum(num_rows); easy_vxm(&BP_sum, &B_rowsum, P, &desc);
    graphblas::Vector<float> BT_sum(num_rows); easy_vxm(&BT_sum, &B_rowsum, &T, &desc);

    float PAP_sum_sum = sum_reduce(PAP_sum.vector_.dense_.d_val_, num_rows);
    float PAT_sum_sum = sum_reduce(PAT_sum.vector_.dense_.d_val_, num_rows);
    float TAP_sum_sum = sum_reduce(TAP_sum.vector_.dense_.d_val_, num_rows);
    float TAT_sum_sum = sum_reduce(TAT_sum.vector_.dense_.d_val_, num_rows);
    float BP_sum_sum  = sum_reduce(BP_sum.vector_.sparse_.d_val_, num_rows);
    float BT_sum_sum  = sum_reduce(BT_sum.vector_.sparse_.d_val_, num_rows);

    float ps_grad_P  = 4 * APPB_trace + (float)num_rows * P_sum - 2 * (PAP_sum_sum + BP_sum_sum);
    float ps_grad_T  = 4 * APTB_trace + (float)num_rows * T_sum - 2 * (TAP_sum_sum + BT_sum_sum);
    float ps_gradt_P = 4 * ATPB_trace + (float)num_rows * P_sum - 2 * (PAT_sum_sum + BP_sum_sum);
    float ps_gradt_T = 4 * ATTB_trace + (float)num_rows * T_sum - 2 * (TAT_sum_sum + BT_sum_sum);

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

    std::cerr << "ps_grad_P=  " << std::setprecision(9) << ps_grad_P  << std::endl;
    std::cerr << "ps_grad_T=  " << std::setprecision(9) << ps_grad_T  << std::endl;
    std::cerr << "ps_gradt_P= " << std::setprecision(9) << ps_gradt_P << std::endl;
    std::cerr << "ps_gradt_T= " << std::setprecision(9) << ps_grad_T  << std::endl;
    std::cerr << "alpha=      " << alpha << std::endl;
    std::cerr << "falpha=     " << falpha << std::endl;
    std::cerr << "f1=         " << f1 << std::endl;
    std::cerr << "============" << std::endl;

    if((alpha > 0) && (alpha < tolerance) && (falpha > 0) && (falpha > f1)) {
      graphblas::Matrix<float> new_P(num_rows, num_cols);
      add_matrix(P, &T, &new_P, alpha, 1 - alpha);
      P->clear();
      P = &new_P;

      graphblas::Matrix<float> new_APB(num_rows, num_cols);
      add_matrix(APB, &ATB, &new_APB, alpha, 1 - alpha);
      APB->clear();
      APB = &new_APB;

      graphblas::Matrix<float> new_AP(num_rows, num_cols);
      add_matrix(AP, &AT, &new_AP, alpha, 1 - alpha);
      AP->clear();
      AP = &new_AP;

    } else if(f1 < 0) {
      P->clear();
      APB->clear();
      AP->clear();

      P   = &T;
      APB = &ATB;
      AP  = &AT;
    } else {
      break;
    }
  }
}
