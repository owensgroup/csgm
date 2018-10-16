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


float compute_trace(
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


void spmm_convex_combination(
    graphblas::Matrix<float>* A,
    graphblas::Matrix<float>* B,
    const float alpha,
    const float beta
)
{
  cusparseHandle_t handle = 0;
  cusparseStatus_t status = cusparseCreate(&handle);

  int n; A->nrows(&n);
  int nvals_a; A->nvals(&nvals_a);
  int nvals_b; B->nvals(&nvals_b);

  cusparseMatDescr_t desc_a;   cusparseCreateMatDescr(&desc_a);
  cusparseMatDescr_t desc_b;   cusparseCreateMatDescr(&desc_b);
  cusparseMatDescr_t desc_out; cusparseCreateMatDescr(&desc_out);

  int minval_out, nvals_out;
  int* C_rowptr;
  int* C_colind;
  float* C_val;

  // nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nvals_out;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  cudaMalloc((void**)&C_rowptr, sizeof(int) * (n + 1));
  cusparseXcsrgeamNnz(
    handle, n, n,
    desc_a, nvals_a, A->matrix_.sparse_.d_csrRowPtr_, A->matrix_.sparse_.d_csrColInd_,
    desc_b, nvals_b, B->matrix_.sparse_.d_csrRowPtr_, B->matrix_.sparse_.d_csrColInd_,
    desc_out, C_rowptr, nnzTotalDevHostPtr
  );

  if (NULL != nnzTotalDevHostPtr){
      nvals_out = *nnzTotalDevHostPtr;
  }else{
      cudaMemcpy(&nvals_out, C_rowptr + n, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&minval_out, C_rowptr, sizeof(int), cudaMemcpyDeviceToHost);
      nvals_out -= minval_out;
  }
  cudaMalloc((void**)&C_colind, sizeof(int)   * nvals_out);
  cudaMalloc((void**)&C_val,     sizeof(float) * nvals_out);
  cusparseScsrgeam(
    handle, n, n,

    &alpha, desc_a, nvals_a,
    A->matrix_.sparse_.d_csrVal_, A->matrix_.sparse_.d_csrRowPtr_, A->matrix_.sparse_.d_csrColInd_,

    &beta, desc_b, nvals_b,
    B->matrix_.sparse_.d_csrVal_, B->matrix_.sparse_.d_csrRowPtr_, B->matrix_.sparse_.d_csrColInd_,

    desc_out, C_val, C_rowptr, C_colind
  );

  A->clear();
  A->build(C_rowptr, C_colind, C_val, nvals_out);
}