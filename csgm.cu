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

#define NUM_SEEDS 5

int main( int argc, char** argv )
{
  bool DEBUG = true;

  // ----------------------------------------------------------------------
  // CLI

  po::variables_map vm;
  parseArgs( argc, argv, vm );

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

  // A.matrix_.sparse_.gpuToCpu();
  // B.matrix_.sparse_.gpuToCpu();
  // A.matrix_.sparse_.cpuToGpu();
  // B.matrix_.sparse_.cpuToGpu();
  trace(&A, &B);

  // // Creating P
  // std::cerr << "creating P" << std::endl;
  // for(graphblas::Index i = 0; i < NUM_SEEDS; i++) {
  //   p_row_indices.push_back(i);
  //   p_col_indices.push_back(i);
  //   p_values.push_back(1.0f);
  // }
  // graphblas::Matrix<float> P(num_rows, num_cols);
  // P.build(&p_row_indices, &p_col_indices, &p_values, NUM_SEEDS, GrB_NULL);

  // // ----------------------------------------------------------------------
  // // Running SGM

  // // AP = A.dot(P)
  // graphblas::Matrix<float> AP(num_rows, num_cols);
  // graphblas::Descriptor AP_desc;
  // dot(&A, &P, &AP, &AP_desc);

  // // APB = AP.dot(B)
  // graphblas::Matrix<float> APB(num_rows, num_cols);
  // graphblas::Descriptor APB_desc;
  // dot(&AP, &B, &APB, &APB_desc);

  // // solve_lap
  // int APB_num_edges; APB.nvals(&APB_num_edges);
  // int* h_person2item = (int *)malloc(sizeof(int) * num_rows);
  // run_auction( // !! Extracts data, which isn't necessary
  //     num_rows,
  //     APB_num_edges,

  //     APB.matrix_.sparse_.d_csrVal_,
  //     APB.matrix_.sparse_.d_csrRowPtr_,
  //     APB.matrix_.sparse_.d_csrColInd_,

  //     h_person2item,

  //     0.1,
  //     0.1,
  //     0.0,

  //     1,
  //     0
  // );

  // // Build T matrix from `h_person2item` -- could be faster?
  // for(graphblas::Index i = 0; i < num_rows; i++) {
  //   t_row_indices.push_back(i);
  //   t_col_indices.push_back(h_person2item[i]);
  //   t_values.push_back(1.0f);
  // }
  // graphblas::Matrix<float> T(num_rows, num_cols);
  // T.build(&t_row_indices, &t_col_indices, &t_values, num_rows, GrB_NULL);

  // // AT = A.dot(T)
  // graphblas::Matrix<float> AT(num_rows, num_cols);
  // graphblas::Descriptor AT_desc;
  // dot(&A, &T, &AT, &AP_desc);



  // // ----------------------------------------------------------------------
  // // Read results

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
