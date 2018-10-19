#include "csgm.cuh"

// --
// Less verbose wrappers for GraphBLAS functions

void rowsum(
  FloatVector* out,
  FloatMatrix* X,
  graphblas::Descriptor* desc
)
{
  graphblas::reduce<float,float,float>(out, GrB_NULL, GrB_NULL, graphblas::PlusMonoid<float>(), X, desc);
}

void easy_mxv(
  FloatVector* out,
  FloatMatrix* X,
  FloatVector* y,
  graphblas::Descriptor* desc
)
{
  graphblas::mxv<float, float, float, float>(out, GrB_NULL, GrB_NULL,
    graphblas::PlusMultipliesSemiring<float>(), X, y, desc);
}

void easy_vxm(
  FloatVector* out,
  FloatVector* y,
  FloatMatrix* X,
  graphblas::Descriptor* desc
)
{
  graphblas::vxm<float, float, float, float>(out, GrB_NULL, GrB_NULL,
    graphblas::PlusMultipliesSemiring<float>(), y, X, desc);
}

void easy_mxm(
  FloatMatrix* out,
  FloatMatrix* A,
  FloatMatrix* B,
  graphblas::Descriptor* desc
)
{
   graphblas::mxm<float,float,float,float>(out, GrB_NULL, GrB_NULL,
       graphblas::PlusMultipliesSemiring<float>(), A, B, desc);
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

void spmm_convex_combination(
    FloatMatrix* A,
    FloatMatrix* B,
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