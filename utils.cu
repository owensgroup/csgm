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

__global__ void __flatten(int* row_ptr, int* cols, int num_rows) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < num_rows) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    for(int offset = start; offset < end; offset++) {
      cols[offset] += i * num_rows;
    }
  }
}

void flatten(graphblas::Matrix<float>* A, graphblas::Matrix<float>* flat, bool transpose) {
    int num_edges; A->nvals(&num_edges);
    int num_rows;  A->nrows(&num_rows);

    // Flatten matrix to vector
    int* Av;
    cudaMalloc((void**)&Av, num_edges * sizeof(int));
    cudaMemcpy(Av, A->matrix_.sparse_.d_csrColInd_, num_edges * sizeof(int), cudaMemcpyDeviceToDevice);

    int A_blocks = 1 + (num_rows / THREADS);
    __flatten<<<A_blocks, THREADS>>>(A->matrix_.sparse_.d_csrRowPtr_, Av, num_rows);

    // Convert Av back to GraphBLAS matrix
    int* h_Av = (int*)malloc(num_edges * sizeof(int));
    cudaMemcpy(h_Av, Av, num_edges * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int>   flat_row(num_edges, 0);
    std::vector<int>   flat_col(h_Av, h_Av + num_edges);
    std::vector<float> flat_val(A->matrix_.sparse_.h_csrVal_, A->matrix_.sparse_.h_csrVal_ + num_edges);
    if(!transpose) {
      flat->build(&flat_row, &flat_col, &flat_val, num_edges, GrB_NULL);
    } else {
      flat->build(&flat_col, &flat_row, &flat_val, num_edges, GrB_NULL);
    }
}

float trace(
  graphblas::Matrix<float>* A,
  graphblas::Matrix<float>* B,
  graphblas::Descriptor* desc
)
{
    A->matrix_.sparse_.gpuToCpu();
    B->matrix_.sparse_.gpuToCpu();

    int num_rows; A->nrows(&num_rows);

    graphblas::Matrix<float> flat_A(1, num_rows * num_rows); flatten(A, &flat_A, false);
    graphblas::Matrix<float> flat_B(num_rows * num_rows, 1); flatten(B, &flat_B, true);

    graphblas::Matrix<float> trace_mtx(1, 1);
    easy_mxm(&trace_mtx, &flat_A, &flat_B, desc);
    trace_mtx.matrix_.sparse_.gpuToCpu();
    return trace_mtx.matrix_.sparse_.h_csrVal_[0];
}



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