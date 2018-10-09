#define THREADS 1024

void dot(
  graphblas::Matrix<float>* A,
  graphblas::Matrix<float>* B,
  graphblas::Matrix<float>* C,
  graphblas::Descriptor* desc
)
{
   graphblas::mxm<float,float,float,float>(
       C,
       GrB_NULL,
       GrB_NULL,
       graphblas::PlusMultipliesSemiring<float>(),
       A,
       B,
       desc
   );
}

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

    int num_rows;
    A->nrows(&num_rows);

    graphblas::Matrix<float> flat_A(1, num_rows * num_rows);
    flatten(A, &flat_A, false);

    graphblas::Matrix<float> flat_B(num_rows * num_rows, 1);
    flatten(B, &flat_B, true);

    graphblas::Matrix<float> trace_mtx(1, 1);
    dot(&flat_A, &flat_B, &trace_mtx, desc);
    trace_mtx.matrix_.sparse_.gpuToCpu();
    return trace_mtx.matrix_.sparse_.h_csrVal_[0];
}


float sum_values(
  float* d_in,
  int num_items
)
{

  // float* h_in = (float*)malloc(num_items * sizeof(float));
  // cudaMemcpy(h_in, d_in, num_items * sizeof(float), cudaMemcpyDeviceToHost);
  // for(int i = 0; i < 10; i ++ ) {
  //   std::cerr << "h_in[" << i << "]=" << h_in[i] << std::endl;
  // }
  // std::cerr << "sum_values: num_items=" << num_items << std::endl;

  float* d_out;
  cudaMalloc((void**)&d_out, 1 * sizeof(float));

  void   *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

  float* h_sum = (float*)malloc(1 * sizeof(float));
  cudaMemcpy(h_sum, d_out, 1 * sizeof(float), cudaMemcpyDeviceToHost);
  // std::cerr << "h_sum[0]=" << h_sum[0] << std::endl;
  return h_sum[0];
}
