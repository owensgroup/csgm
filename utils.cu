#define THREADS 1024

// --
// Less verbose wrappers for GraphBLAS functions

void rowsum(
  graphblas::Vector<float>* out,
  graphblas::Matrix<float>* X,
  graphblas::Descriptor* desc
)
{
  graphblas::reduce<float,float,float>(out, GrB_NULL, GrB_NULL, graphblas::PlusMonoid<float>(), X, desc);
}

void easy_mxv(
  graphblas::Vector<float>* out,
  graphblas::Matrix<float>* X,
  graphblas::Vector<float>* y,
  graphblas::Descriptor* desc
)
{
  graphblas::mxv<float, float, float, float>(out, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), X, y, desc);
}

void easy_vxm(
  graphblas::Vector<float>* out,
  graphblas::Vector<float>* y,
  graphblas::Matrix<float>* X,
  graphblas::Descriptor* desc
)
{
  graphblas::vxm<float, float, float, float>(out, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(), y, X, desc);
}

void easy_mxm(
  graphblas::Matrix<float>* out,
  graphblas::Matrix<float>* A,
  graphblas::Matrix<float>* B,
  graphblas::Descriptor* desc
)
{
   graphblas::mxm<float,float,float,float>(
       out,
       GrB_NULL,
       GrB_NULL,
       graphblas::PlusMultipliesSemiring<float>(),
       A,
       B,
       desc
   );
}

float sum_reduce(
  float* d_in,
  int num_items
)
{
  float* d_out;
  cudaMalloc((void**)&d_out, 1 * sizeof(float));

  void   *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

  float* h_sum = (float*)malloc(sizeof(float));
  cudaMemcpy(h_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  return *h_sum;
}


// --
// Custome kernels

// __global__ void __flatten(int* row_ptr, int* cols, int n) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   if(i < n) {
//     int start = row_ptr[i];
//     int end   = row_ptr[i + 1];
//     for(int offset = start; offset < end; offset++) {
//       cols[offset] = cols[offset] + (i * n);
//     }
//   }
// }

// void flatten(graphblas::Matrix<float>* X, graphblas::Matrix<float>* flat, bool transpose) {
//     int num_edges; X->nvals(&num_edges);
//     int num_rows;  X->nrows(&num_rows);

//     // Flatten matrix to vector
//     int* Xv;
//     cudaMalloc((void**)&Xv, num_edges * sizeof(int));
//     cudaMemcpy(Xv, X->matrix_.sparse_.d_csrColInd_, num_edges * sizeof(int), cudaMemcpyDeviceToDevice);

//     int X_blocks = 1 + (num_rows / THREADS);
//     __flatten<<<X_blocks, THREADS>>>(X->matrix_.sparse_.d_csrRowPtr_, Xv, num_rows);

//     // Convert Xv back to GraphBLXS matrix
//     int* h_Xv = (int*)malloc(num_edges * sizeof(int));
//     cudaMemcpy(h_Xv, Xv, num_edges * sizeof(int), cudaMemcpyDeviceToHost);

//     std::vector<int>   flat_row(num_edges, 0);
//     std::vector<int>   flat_col(h_Xv, h_Xv + num_edges);
//     std::vector<float> flat_val(X->matrix_.sparse_.h_csrVal_, X->matrix_.sparse_.h_csrVal_ + num_edges);
//     if(!transpose) {
//       flat->build(&flat_row, &flat_col, &flat_val, num_edges, GrB_NULL);
//     } else {
//       flat->build(&flat_col, &flat_row, &flat_val, num_edges, GrB_NULL);
//     }
// }



// float cpu_trace(
//   graphblas::Matrix<float>* A,
//   graphblas::Matrix<float>* B,
//   graphblas::Descriptor* desc
// )
// {
//     A->matrix_.sparse_.gpuToCpu();
//     B->matrix_.sparse_.gpuToCpu();

//     int nrows; A->nrows(&nrows);

//     graphblas::Matrix<float> flat_A(1, nrows * nrows); flatten(A, &flat_A, false);
//     cudaDeviceSynchronize();

//     graphblas::Matrix<float> flat_B(nrows * nrows, 1); flatten(B, &flat_B, true);
//     cudaDeviceSynchronize();

//     graphblas::Matrix<float> trace_mtx(1, 1);
//     easy_mxm(&trace_mtx, &flat_A, &flat_B, desc);
//     trace_mtx.matrix_.sparse_.gpuToCpu();
//     return trace_mtx.matrix_.sparse_.h_csrVal_[0];
// }


__global__ void __flatten2(int* out, int* row_ptr, int* cols, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    for(int offset = start; offset < end; offset++) {
      out[offset] = cols[offset] + (i * n);
    }
  }
}


float rowvector_dot(
  float* a_val,
  int*   a_rowptr,
  int*   a_colind,
  float* b_val,
  int*   b_rowptr,
  int*   b_colind,

  int nvals_a,
  int nvals_b,

  int dim
)
{
    cusparseHandle_t handle = 0;
    cusparseStatus_t status = cusparseCreate(&handle);

    // --
    // Transpose B

    int* tb_colind;
    int* tb_rowptr;
    float* tb_val;
    cudaMalloc((void**)&tb_colind, nvals_b * sizeof(int));
    cudaMalloc((void**)&tb_rowptr, (dim + 1) * sizeof(int));
    cudaMalloc((void**)&tb_val, nvals_b * sizeof(float));

    cusparseScsr2csc(handle, 1, dim, nvals_b,
                     b_val, b_rowptr, b_colind,
                     tb_val, tb_colind, tb_rowptr,
                     CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);

    // --
    // Compute dot product

    int* out_row;
    int* out_col;
    float* out_val;
    cudaMalloc((void**)&out_row, sizeof(int)*2);
    cudaMalloc((void**)&out_col, sizeof(int)*1);
    cudaMalloc((void**)&out_val, sizeof(float)*1);

    cusparseMatDescr_t desc_a;   cusparseCreateMatDescr(&desc_a);
    cusparseMatDescr_t desc_b;   cusparseCreateMatDescr(&desc_b);
    cusparseMatDescr_t desc_out; cusparseCreateMatDescr(&desc_out);
    cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            1, dim, 1,
            desc_a, nvals_a, a_val, a_rowptr, a_colind,
            desc_b, nvals_b, tb_val, tb_rowptr, tb_colind,
            desc_out, out_val, out_row, out_col);

    float* h_out_val = (float*)malloc(1 * sizeof(float));
    cudaMemcpy(h_out_val, out_val, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    float result = h_out_val[0];

    cudaFree(tb_colind);
    cudaFree(tb_rowptr);
    cudaFree(tb_val);
    cudaFree(out_row);
    cudaFree(out_col);
    cudaFree(out_val);

    return result;
}

void gpu_flatten_matrix(
  int* flat_rowptr,
  int* flat_colind,
  int* rowptr,
  int* colind,
  int nrows,
  int nnz
)
{
    int blocks = 1 + (nrows / THREADS);

    // Flatten columns
    __flatten2<<<blocks, THREADS>>>(flat_colind, rowptr, colind, nrows);

    // Dummy rows
    int * h_flat_rowptr = (int*)malloc(2 * sizeof(int));
    h_flat_rowptr[0] = 0;
    h_flat_rowptr[1] = nnz;
    cudaMemcpy(flat_rowptr, h_flat_rowptr, 2 * sizeof(int), cudaMemcpyHostToDevice);
}


float gpu_trace(
  graphblas::Matrix<float>* A,
  graphblas::Matrix<float>* B,
  graphblas::Descriptor* desc
)
{
    int nrows; A->nrows(&nrows);
    int blocks = 1 + (nrows / THREADS);

    // --
    // Flatten A

    int A_nnz; A->nvals(&A_nnz);
    int* Af_colind;
    int* Af_rowptr;
    cudaMalloc((void**)&Af_colind, A_nnz * sizeof(int));
    cudaMalloc((void**)&Af_rowptr, 2 * sizeof(int));
    gpu_flatten_matrix(
      Af_rowptr,
      Af_colind,
      A->matrix_.sparse_.d_csrRowPtr_,
      A->matrix_.sparse_.d_csrColInd_,
      nrows,
      A_nnz
    );

    // --
    // Flatten B

    int B_nnz; B->nvals(&B_nnz);
    int* Bf_colind;
    int* Bf_rowptr;
    cudaMalloc((void**)&Bf_colind, B_nnz * sizeof(int));
    cudaMalloc((void**)&Bf_rowptr, 2 * sizeof(int));
    gpu_flatten_matrix(
      Bf_rowptr,
      Bf_colind,
      B->matrix_.sparse_.d_csrRowPtr_,
      B->matrix_.sparse_.d_csrColInd_,
      nrows,
      B_nnz
    );


    // --
    // Compute trace

    float trace = rowvector_dot(
      A->matrix_.sparse_.d_csrVal_,
      Af_rowptr,
      Af_colind,

      B->matrix_.sparse_.d_csrVal_, // B will be transposed
      Bf_rowptr,
      Bf_colind,

      A_nnz,
      B_nnz,
      nrows * nrows
    );

    cudaFree(Af_colind);
    cudaFree(Af_rowptr);
    cudaFree(Bf_colind);
    cudaFree(Bf_rowptr);

    return trace;
}



// float trace(
//   graphblas::Matrix<float>* A,
//   graphblas::Matrix<float>* B,
//   graphblas::Descriptor* desc
// )
// {
//     A->matrix_.sparse_.gpuToCpu();
//     B->matrix_.sparse_.gpuToCpu();

//     int nrows; A->nrows(&nrows);

//     graphblas::Matrix<float> flat_A(1, nrows * nrows); flatten(A, &flat_A);
//     cudaDeviceSynchronize();

//     graphblas::Matrix<float> flat_B(1, nrows * nrows); flatten(B, &flat_B);
//     cudaDeviceSynchronize();

//     cusparseHandle_t handle = 0;
//     cusparseStatus_t status = cusparseCreate(&handle);
//     cusparseMatDescr_t desc_a;   cusparseCreateMatDescr(&desc_a);
//     cusparseMatDescr_t desc_b;   cusparseCreateMatDescr(&desc_b);
//     cusparseMatDescr_t desc_out; cusparseCreateMatDescr(&desc_out);

//     int baseC, nnzC;
//     int* csrRowPtrC;
//     int* csrColIndC;
//     float* csrValC;

//     int nvals_a; flat_A.nvals(&nvals_a);

//     int * a_row;
//     int * a_col;
//     float * a_val;
//     cudaMalloc((void**)&a_row, sizeof(int) * 2);
//     cudaMalloc((void**)&a_col, sizeof(int) * nvals_a);
//     cudaMalloc((void**)&a_val, sizeof(float) * nvals_a);
//     cudaMemcpy(a_row, flat_A.matrix_.sparse_.d_csrRowPtr_, sizeof(int)   * 2, cudaMemcpyDeviceToDevice);
//     cudaMemcpy(a_col, flat_A.matrix_.sparse_.d_csrColInd_, sizeof(int)   * nvals_a, cudaMemcpyDeviceToDevice);
//     cudaMemcpy(a_val, flat_A.matrix_.sparse_.d_csrVal_,    sizeof(float) * nvals_a, cudaMemcpyDeviceToDevice);

//     int * h_a_row = (int*)malloc(sizeof(int) * 2);
//     int * h_a_col = (int*)malloc(sizeof(int) * nvals_a);
//     float * h_a_val = (float*)malloc(sizeof(float) * nvals_a);
//     cudaMemcpy(h_a_row, a_row, sizeof(int) * 2, cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_a_col, a_col, sizeof(int) * nvals_a, cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_a_val, a_val, sizeof(float) * nvals_a, cudaMemcpyDeviceToHost);

//     std::cerr << "nvals_a=" << nvals_a << std::endl;
//     for(int i = 0; i < 2; i ++) {
//       std::cerr << i << " " << h_a_row[i] << std::endl;
//     }
//     for(int i = 0; i < 10; i ++) {
//       std::cerr << i << " " << h_a_col[i] << " " << h_a_val[i] << std::endl;
//     }

//     // --

//     int nvals_b; flat_B.nvals(&nvals_b);

//     int * b_row;
//     int * b_col;
//     float * b_val;
//     cudaMalloc((void**)&b_row, sizeof(int) * 2);
//     cudaMalloc((void**)&b_col, sizeof(int) * nvals_b);
//     cudaMalloc((void**)&b_val, sizeof(float) * nvals_b);
//     cudaMemcpy(b_row, flat_B.matrix_.sparse_.d_csrRowPtr_, sizeof(int)   * 2, cudaMemcpyDeviceToDevice);
//     cudaMemcpy(b_col, flat_B.matrix_.sparse_.d_csrColInd_, sizeof(int)   * nvals_b, cudaMemcpyDeviceToDevice);
//     cudaMemcpy(b_val, flat_B.matrix_.sparse_.d_csrVal_,    sizeof(float) * nvals_b, cudaMemcpyDeviceToDevice);

//     int * h_b_row = (int*)malloc(sizeof(int) * 2);
//     int * h_b_col = (int*)malloc(sizeof(int) * nvals_b);
//     float * h_b_val = (float*)malloc(sizeof(float) * nvals_b);
//     cudaMemcpy(h_b_row, b_row, sizeof(int) * 2, cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_b_col, b_col, sizeof(int) * nvals_b, cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_b_val, b_val, sizeof(float) * nvals_b, cudaMemcpyDeviceToHost);

//     std::cerr << "nvals_b=" << nvals_b << std::endl;
//     for(int i = 0; i < 2; i ++) {
//       std::cerr << i << " " << h_b_row[i] << std::endl;
//     }
//     for(int i = 0; i < 10; i ++) {
//       std::cerr << i << " " << h_b_col[i] << " " << h_b_val[i] << std::endl;
//     }

//     // nnzTotalDevHostPtr points to host memory
//     int *nnzTotalDevHostPtr = &nnzC;
//     cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
//     cudaMalloc((void**)&csrRowPtrC, sizeof(int) * 2);
//     cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
//             1, nrows * nrows, 1,
//             desc_a, nvals_a, a_row, a_col,
//             desc_b, nvals_b, b_row, b_col,
//             desc_out, csrRowPtrC, nnzTotalDevHostPtr );

//     // // cudaDeviceSynchronize();
//     // // if (NULL != nnzTotalDevHostPtr){
//     // //     nnzC = *nnzTotalDevHostPtr;
//     // // }else{
//     // //     cudaMemcpy(&nnzC, csrRowPtrC + 1, sizeof(int), cudaMemcpyDeviceToHost);
//     // //     cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
//     // //     nnzC -= baseC;
//     // // }
//     // // cudaDeviceSynchronize();
//     // // cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
//     // // cudaMalloc((void**)&csrValC, sizeof(float)*nnzC);
//     // // cudaDeviceSynchronize();
//     // // cusparseScsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
//     // //         1, nrows * nrows, 1,
//     // //         desc_a, nvals_a, a_val, a_row, a_col,
//     // //         desc_b, nvals_b, b_val, b_row, b_col,
//     // //         desc_out, csrValC, csrRowPtrC, csrColIndC);
//     // // cudaDeviceSynchronize();

//     std::cerr << "returning" << std::endl;

//     return -1;

//     // graphblas::Matrix<float> trace_mtx(1, 1);
//     // easy_mxm(&trace_mtx, &flat_A, &flat_B, desc);
//     // trace_mtx.matrix_.sparse_.gpuToCpu();
//     // return trace_mtx.matrix_.sparse_.h_csrVal_[0];
// }



void add_matrix(
    graphblas::Matrix<float>* A,
    graphblas::Matrix<float>* B,
    graphblas::Matrix<float>* C,
    const float alpha,
    const float beta
)
{
  // This copies data GPU -> CPU -> GPU
  // ATTN CARL

  cusparseHandle_t handle = 0;
  cusparseStatus_t status = cusparseCreate(&handle);

  int nrows; A->nrows(&nrows);
  int ncols; A->ncols(&ncols);
  int nvals_a; A->nvals(&nvals_a);
  int nvals_b; B->nvals(&nvals_b);

  cusparseMatDescr_t desc_a;   cusparseCreateMatDescr(&desc_a);
  cusparseMatDescr_t desc_b;   cusparseCreateMatDescr(&desc_b);
  cusparseMatDescr_t desc_out; cusparseCreateMatDescr(&desc_out);

  int minval_out, nvals_out;
  int* d_indptr_out;
  int* d_indices_out;
  float* d_row_out;

  // nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nvals_out;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  cudaMalloc((void**)&d_indptr_out, sizeof(int) * (nrows + 1));
  cusparseXcsrgeamNnz(
    handle, nrows, ncols,
    desc_a, nvals_a, A->matrix_.sparse_.d_csrRowPtr_, A->matrix_.sparse_.d_csrColInd_,
    desc_b, nvals_b, B->matrix_.sparse_.d_csrRowPtr_, B->matrix_.sparse_.d_csrColInd_,
    desc_out, d_indptr_out, nnzTotalDevHostPtr
  );

  if (NULL != nnzTotalDevHostPtr){
      nvals_out = *nnzTotalDevHostPtr;
  }else{
      cudaMemcpy(&nvals_out, d_indptr_out + nrows, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&minval_out, d_indptr_out, sizeof(int), cudaMemcpyDeviceToHost);
      nvals_out -= minval_out;
  }
  cudaMalloc((void**)&d_indices_out, sizeof(int)   * nvals_out);
  cudaMalloc((void**)&d_row_out,     sizeof(float) * nvals_out);
  cusparseScsrgeam(
    handle, nrows, ncols,

    &alpha,
    desc_a, nvals_a, A->matrix_.sparse_.d_csrVal_, A->matrix_.sparse_.d_csrRowPtr_, A->matrix_.sparse_.d_csrColInd_,

    &beta,
    desc_b, nvals_b, B->matrix_.sparse_.d_csrVal_, B->matrix_.sparse_.d_csrRowPtr_, B->matrix_.sparse_.d_csrColInd_,

    desc_out, d_row_out, d_indptr_out, d_indices_out
  );

  float * h_val     = (float*)malloc(sizeof(float) * nvals_out);
  int   * h_indices = (int*)malloc(sizeof(int)     * nvals_out);
  int   * h_indptr  = (int*)malloc(sizeof(int)     * (nrows + 1));

  cudaMemcpy(h_val,     d_row_out,     sizeof(float) * nvals_out,   cudaMemcpyDeviceToHost);
  cudaMemcpy(h_indices, d_indices_out, sizeof(int)   * nvals_out,   cudaMemcpyDeviceToHost);
  cudaMemcpy(h_indptr,  d_indptr_out,  sizeof(int)   * (nrows + 1), cudaMemcpyDeviceToHost);

  std::cerr << "nvals_out=" << nvals_out << std::endl;
  std::vector<int> vec_row;
  for(int i = 0; i < nrows; i++) {
    int start = h_indptr[i];
    int end   = h_indptr[i + 1];
    for(int offset = start; offset < end; offset++) {
      vec_row.push_back(i);
    }
  }
  std::vector<int>   vec_col(h_indices, h_indices + nvals_out);
  std::vector<float> vec_val(h_val,     h_val + nvals_out);
  C->build(&vec_row, &vec_col, &vec_val, nvals_out, GrB_NULL);
  C->print();
}