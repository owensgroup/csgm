#define GRB_USE_APSPIE
#define private public
#include <iostream>
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
  bool DEBUG = true;
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
  std::cerr << "loading A" << std::endl;
  readMtx("data/A.mtx", a_row_indices, a_col_indices, a_values, num_rows, num_cols, a_num_edges, 0, false);
  graphblas::Matrix<float> A(num_rows, num_cols);
  A.build(&a_row_indices, &a_col_indices, &a_values, a_num_edges, GrB_NULL);

  // Load B
  std::cerr << "loading B" << std::endl;
  readMtx("data/B.mtx", b_row_indices, b_col_indices, b_values, num_rows, num_cols, b_num_edges, 0, false);
  graphblas::Matrix<float> B(num_rows, num_cols);
  B.build(&b_row_indices, &b_col_indices, &b_values, b_num_edges, GrB_NULL);

  // Creating P
  std::cerr << "creating P" << std::endl;
  for(graphblas::Index i = 0; i < NUM_SEEDS; i++) {
    p_row_indices.push_back(i);
    p_col_indices.push_back(i);
    p_values.push_back(1.0f);
  }
  graphblas::Matrix<float> _P(num_rows, num_cols);
  _P.build(&p_row_indices, &p_col_indices, &p_values, NUM_SEEDS, GrB_NULL);
  graphblas::Matrix<float>* P = &_P;

  // ----------------------------------------------------------------------
  // Run SGM

  graphblas::Matrix<float>  _AP(num_rows, num_cols);
  graphblas::Matrix<float> _APB(num_rows, num_cols);
  graphblas::Matrix<float>    T(num_rows, num_cols);
  graphblas::Matrix<float>   AT(num_rows, num_cols);
  graphblas::Matrix<float>  ATB(num_rows, num_cols);
  graphblas::Matrix<float>   PB(num_rows, num_cols);
  graphblas::Matrix<float>   TB(num_rows, num_cols);

  easy_mxm(&_AP, &A, P, &desc);   graphblas::Matrix<float>* AP = &_AP;
  easy_mxm(&_APB, AP, &B, &desc); graphblas::Matrix<float>* APB = &_APB;

  int* d_person2item;
  cudaMalloc((void **)&d_person2item, num_rows * sizeof(int));

  for(int iter = 0; iter < 20; iter++) {
    std::cerr << "******** iter=" << iter << "**********" << std::endl;

    std::cerr << "\t Solve LAP" << std::endl;
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
    int* h_person2item = (int *)malloc(num_rows * sizeof(int));
    cudaMemcpy(h_person2item, d_person2item, num_rows * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Build T matrix from `h_person2item` -- could be faster?
    T.clear(); t_row_indices.clear(); t_col_indices.clear(); t_values.clear();
    for(graphblas::Index i = 0; i < num_rows; i++) {
      t_row_indices.push_back(i);
      t_col_indices.push_back(h_person2item[i]);
      t_values.push_back(1.0f);
    }
    T.build(&t_row_indices, &t_col_indices, &t_values, num_rows, GrB_NULL);

    std::cerr << "\t Matrix multiply" << std::endl;

    std::cerr << "\t AT" << std::endl;
    AT.clear();  easy_mxm(&AT,   &A, &T,  &desc);
    std::cerr << "\t ATB" << std::endl;
    ATB.clear(); easy_mxm(&ATB, &AT, &B,  &desc);
    std::cerr << "\t AP" << std::endl;
    PB.clear();  easy_mxm(&PB,    P, &B,  &desc);
    std::cerr << "\t TB" << std::endl;
    TB.clear();  easy_mxm(&TB,   &T, &B,  &desc);

    std::cerr << "\t Check convergence" << std::endl;
    float APPB_trace = trace(AP, &PB, &desc);
    float APTB_trace = trace(AP, &TB, &desc);
    float ATPB_trace = trace(&AT, &PB, &desc);
    float ATTB_trace = trace(&AT, &TB, &desc);

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

    std::cerr << "============"  << std::endl;
    std::cerr << "ps_grad_P=  " << ps_grad_P  << std::endl;
    std::cerr << "ps_grad_T=  " << ps_grad_T  << std::endl;
    std::cerr << "ps_gradt_P= " << ps_gradt_P << std::endl;
    std::cerr << "ps_gradt_T= " << ps_grad_T  << std::endl;
    std::cerr << "alpha=      " << alpha << std::endl;
    std::cerr << "falpha=     " << falpha << std::endl;
    std::cerr << "f1=         " << f1 << std::endl;
    std::cerr << "============"  << std::endl;

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

  // ----------------------------------------------------------------------
  // Read results

  // APB.matrix_.sparse_.gpuToCpu();
  // float score = 0;
  // for (int i = 0; i < num_rows; i++) {
  //   // std::cout << i << " " << h_person2item[i] << std::endl;
  //   int start = APB.matrix_.sparse_.h_csrRowPtr_[i];
  //   int end   = APB.matrix_.sparse_.h_csrRowPtr_[i + 1];
  //   for(int j = start; j < end; j++) {
  //     if(APB.matrix_.sparse_.h_csrColInd_[j] == h_person2item[i]) {
  //       score += APB.matrix_.sparse_.h_csrVal_[j];
  //     }
  //   }
  // }
  // std::cout << "score=" << score << std::endl;
}
