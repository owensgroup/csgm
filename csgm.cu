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
  graphblas::Matrix<float> P(num_rows, num_cols);
  P.build(&p_row_indices, &p_col_indices, &p_values, NUM_SEEDS, GrB_NULL);

  // ----------------------------------------------------------------------
  // Run SGM

  // AP = A.dot(P)
  graphblas::Matrix<float> AP(num_rows, num_cols);
  dot(&A, &P, &AP, &desc);

  // APB = AP.dot(B)
  graphblas::Matrix<float> APB(num_rows, num_cols);
  dot(&AP, &B, &APB, &desc);

  // solve_lap
  // std::cerr << "run_auction" << std::endl;
  // int APB_num_edges; APB.nvals(&APB_num_edges);
  // APB.matrix_.sparse_.gpuToCpu();
  // int* d_person2item;
  // cudaMalloc((void **)&d_person2item, num_rows * sizeof(int));
  // run_auction(
  //     num_rows,
  //     APB_num_edges,

  //     APB.matrix_.sparse_.d_csrVal_,
  //     APB.matrix_.sparse_.d_csrRowPtr_,
  //     APB.matrix_.sparse_.d_csrColInd_,

  //     d_person2item,

  //     0.1,
  //     0.1,
  //     0.0,

  //     1,
  //     1
  // );
  // int* h_person2item = (int *)malloc(num_rows * sizeof(int));
  // cudaMemcpy(h_person2item, d_person2item, num_rows * sizeof(int), cudaMemcpyDeviceToHost);

  // Build T matrix from `h_person2item` -- could be faster?
  for(graphblas::Index i = 0; i < num_rows; i++) {
    t_row_indices.push_back(i);
    // t_col_indices.push_back(h_person2item[i]);
    t_col_indices.push_back(i);
    t_values.push_back(1.0f);
  }
  graphblas::Matrix<float> T(num_rows, num_cols);
  T.build(&t_row_indices, &t_col_indices, &t_values, num_rows, GrB_NULL);

  std::cerr << "AT = A.dot(T)" << std::endl;
  graphblas::Matrix<float> AT(num_rows, num_cols);
  dot(&A, &T, &AT, &desc);

  std::cerr << "PB = P.dot(B)" << std::endl;
  graphblas::Matrix<float> PB(num_rows, num_cols);
  dot(&P, &B, &PB, &desc);

  std::cerr << "TB = T.dot(B)" << std::endl;
  graphblas::Matrix<float> TB(num_rows, num_cols);
  dot(&T, &B, &TB, &desc);

  std::cerr << "computing traces: start" << std::endl;
  float APPB_trace = trace(&AP, &PB, &desc);
  float APTB_trace = trace(&AP, &TB, &desc);
  float ATPB_trace = trace(&AT, &PB, &desc);
  float ATTB_trace = trace(&AT, &TB, &desc);
  std::cerr << "computing traces: done" << std::endl;

  float T_sum = (float)num_rows;

  int P_num_values; P.nvals(&P_num_values);
  float P_sum = sum_values(P.matrix_.sparse_.d_csrVal_, P_num_values);

  graphblas::Vector<float> AP_rowsum(num_rows);
  graphblas::Vector<float> AT_rowsum(num_rows);
  graphblas::Vector<float> B_rowsum(num_rows);

  graphblas::reduce<float,float,float>(&AP_rowsum, GrB_NULL, GrB_NULL, graphblas::PlusMonoid<float>(), &AP, &desc);
  graphblas::reduce<float,float,float>(&AT_rowsum, GrB_NULL, GrB_NULL, graphblas::PlusMonoid<float>(), &AT, &desc);
  graphblas::reduce<float,float,float>(&B_rowsum,  GrB_NULL, GrB_NULL, graphblas::PlusMonoid<float>(), &B,  &desc);

  float* h_B_rowsum = (float*)malloc(num_rows * sizeof(float));
  cudaMemcpy(h_B_rowsum, B_rowsum.vector_.sparse_.d_val_, 10 * sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 10; i++) {
    std::cerr << "h_B_rowsum[" << i << "]=" << h_B_rowsum[i] << std::endl;
  }


  graphblas::Vector<float> PAP_sum(num_rows);
  graphblas::Vector<float> PAT_sum(num_rows);
  graphblas::Vector<float> TAP_sum(num_rows);
  graphblas::Vector<float> TAT_sum(num_rows);

  graphblas::mxv<float, float, float, float>(&PAP_sum, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), &P, &AP_rowsum, &desc);
  graphblas::mxv<float, float, float, float>(&PAT_sum, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), &P, &AT_rowsum, &desc);
  graphblas::mxv<float, float, float, float>(&TAP_sum, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), &T, &AP_rowsum, &desc);
  graphblas::mxv<float, float, float, float>(&TAT_sum, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), &T, &AT_rowsum, &desc);

  float PAP_sum_sum = sum_values(PAP_sum.vector_.dense_.d_val_, num_rows);
  float PAT_sum_sum = sum_values(PAT_sum.vector_.dense_.d_val_, num_rows);
  float TAP_sum_sum = sum_values(TAP_sum.vector_.dense_.d_val_, num_rows);
  float TAT_sum_sum = sum_values(TAT_sum.vector_.dense_.d_val_, num_rows);

  // !!!!!!!!!!!!!!!!!!!!!
  // !!! This is wrong, but works for the first iteration, because P is symmetric at that point
  // !!! Should be doing `vxm` -- need Carl's help
  std::cerr << "**** BP_sum *****" << std::endl;
  graphblas::Vector<float> BP_sum(num_rows);
  graphblas::mxv<float, float, float, float>(&BP_sum, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), &P, &B_rowsum, &desc);

  float* h_BP_sum = (float*)malloc(num_rows * sizeof(float));
  cudaMemcpy(h_BP_sum, BP_sum.vector_.dense_.d_val_, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 10; i++) {
    std::cerr << "h_BP_sum[" << i << "]=" << h_BP_sum[i] << std::endl;
  }
  std::cerr << "**** done *****" << std::endl;

  std::cerr << "**** BP_sum2 *****" << std::endl;
  graphblas::Vector<float> BP_sum2(num_rows);
  graphblas::vxm<float, float, float, float>(&BP_sum2, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), &B_rowsum, &P, &desc);
  float* h_BP_sum2 = (float*)malloc(num_rows * sizeof(float));
  cudaMemcpy(h_BP_sum2, BP_sum2.vector_.sparse_.d_val_, 101 * sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 101; i++) {
    std::cerr << "h_BP_sum2[" << i << "]=" << h_BP_sum2[i] << std::endl;
  }
  std::cerr << "**** done *****" << std::endl;

  graphblas::Vector<float> BT_sum(num_rows);
  graphblas::mxv<float, float, float, float>(&BT_sum, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), &T, &B_rowsum, &desc);
  // !!!!!!!!!!!!!!!!!!!!!

  float BP_sum_sum = sum_values(BP_sum.vector_.dense_.d_val_, num_rows);
  float BT_sum_sum = sum_values(BT_sum.vector_.dense_.d_val_, num_rows);

  float ps_grad_P  = 4 * APPB_trace + (float)num_rows * P_sum - 2 * (PAP_sum_sum + BP_sum_sum);
  float ps_grad_T  = 4 * APTB_trace + (float)num_rows * T_sum - 2 * (TAP_sum_sum + BT_sum_sum);
  float ps_gradt_P = 4 * ATPB_trace + (float)num_rows * P_sum - 2 * (PAT_sum_sum + BP_sum_sum);
  float ps_gradt_T = 4 * ATTB_trace + (float)num_rows * T_sum - 2 * (TAT_sum_sum + BT_sum_sum);

  std::cerr << "ps_grad_P="  << ps_grad_P << std::endl;
  std::cerr << "ps_grad_T="  << ps_grad_T << std::endl;
  std::cerr << "ps_gradt_P=" << ps_gradt_P << std::endl;
  std::cerr << "ps_gradt_T=" << ps_gradt_T << std::endl;

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

  if((alpha > 0) && (alpha < tolerance) && (falpha > 0) && (falpha > f1)) {
    // P <- (alpha * P) + (1 - alpha) * T
  } else if(f1 < 0) {
    // P   <- T
    // APB <- ATB
    // AP  <- AT
    // Probably more?
  } else {
    // break;
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

  // if(DEBUG) AP.print();
}
