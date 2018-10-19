// auction.cu

#ifndef MAIN_AUCTION
#define MAIN_AUCTION

#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include <curand.h>
#include <curand_kernel.h>

#include "auction_kernel_csr.cu"

#define THREADS 8

int run_auction(
    int    num_nodes,
    int    num_edges,

    float* d_data,      // data
    int*   d_offsets,   // offsets for items
    int*   d_columns,

    int*   d_person2item, // results

    float auction_max_eps,
    float auction_min_eps,
    float auction_factor,

    int num_runs,
    int verbose
)
{
    // float* h_data    = (float*)malloc(num_edges * sizeof(float));
    // int*   h_offsets = (int*)malloc((num_nodes + 1) * sizeof(int));
    // int*   h_columns = (int*)malloc(num_edges * sizeof(int));
    // std::cerr << "num_edges=" << num_edges << std::endl;
    // cudaMemcpy(h_columns, d_columns, num_edges * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_data, d_data, num_edges * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < 20; i++) {
    //     std::cerr << "h_columns[" << i << "]=" << h_columns[i] << std::endl;
    //     std::cerr << "   h_data[" << i << "]=" << h_data[i] << std::endl;
    // }

    int block = 1 + num_nodes / THREADS;

    // --
    // Declare variables

    int* d_item2person;

    float* d_bids;
    float* d_prices;
    int*   d_sbids;
    int    h_numAssign;
    int*   d_numAssign;
    float* d_rand;

    // --
    // Allocate device memory

    cudaMalloc((void **)&d_numAssign,   1                     * sizeof(int)) ;
    cudaMalloc((void **)&d_item2person, num_nodes             * sizeof(int));
    cudaMalloc((void **)&d_prices,      num_nodes             * sizeof(float));
    cudaMalloc((void **)&d_sbids,       num_nodes             * sizeof(int));
    cudaMalloc((void **)&d_bids,        num_nodes * num_nodes * sizeof(float));
    cudaMalloc((void **)&d_rand,        num_nodes * num_nodes * sizeof(float));

    // --
    // Copy from host to device

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 123);
    curandGenerateUniform(gen, d_rand, num_nodes * num_nodes);

    for(int run_num = 0; run_num < num_runs; run_num++) {
        cudaMemset(d_prices, 0.0, num_nodes * sizeof(float));

        // Start timer
        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        float auction_eps = auction_max_eps;
        while(auction_eps >= auction_min_eps) {
            h_numAssign = 0;
            cudaMemset(d_person2item,   -1, num_nodes * sizeof(int));
            cudaMemset(d_item2person,   -1, num_nodes * sizeof(int));
            cudaMemset(d_numAssign,      0, 1         * sizeof(int));
            cudaThreadSynchronize();

            int counter = 0;
            while(h_numAssign < num_nodes){
                counter += 1;
                cudaMemset(d_bids,  0, num_nodes * num_nodes * sizeof(float));
                cudaMemset(d_sbids, 0, num_nodes * sizeof(int));

                run_bidding<<<block, THREADS>>>(
                    num_nodes,

                    d_data,
                    d_offsets,
                    d_columns,

                    d_person2item,
                    d_bids,
                    d_sbids,
                    d_prices,
                    auction_eps,
                    d_rand
                );
                run_assignment<<<block, THREADS>>>(
                    num_nodes,
                    d_person2item,
                    d_item2person,
                    d_bids,
                    d_sbids,
                    d_prices,
                    d_numAssign
                );

                cudaMemcpy(&h_numAssign, d_numAssign, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            }
            if(verbose) {
                std::cerr << "counter=" << counter << std::endl;
            }

            auction_eps *= auction_factor;
        }
        cudaThreadSynchronize();

        // Stop timer
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if(verbose) {
            std::cerr <<
                "run_num="         << run_num      <<
                " | h_numAssign="  << h_numAssign  <<
                " | milliseconds=" << milliseconds << std::endl;
        }

        cudaThreadSynchronize();
     }

    cudaFree(d_item2person);
    cudaFree(d_bids);
    cudaFree(d_prices);
    cudaFree(d_sbids);
    cudaFree(d_numAssign);
    cudaFree(d_rand);

    return 0;
} // end run_auction

#endif
