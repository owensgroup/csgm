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

#include "csgm.cuh"
#include "auction_kernel_csr.cu"

#define THREADS 8

// #define DEBUG

int run_auction(
    int    num_nodes,
    int    num_edges,

    float* d_data,      // data
    int*   d_offsets,   // offsets for items
    int*   d_columns,

    float auction_max_eps,
    float auction_min_eps,
    float auction_factor,

    int num_runs,
    int verbose,

    curandGenerator_t &gen,

    AuctionData &ad
)
{
    int block = 1 + num_nodes / THREADS;
    int h_numAssign = 0;

    // curandSetPseudoRandomGeneratorSeed(gen, 123); // To exactly match other implementations
    curandGenerateUniform(gen, ad.d_rand, num_nodes * num_nodes);

    for(int run_num = 0; run_num < num_runs; run_num++) {
        cudaMemset(ad.d_prices, 0.0, num_nodes * sizeof(float));

        float auction_eps = auction_max_eps;
        while(auction_eps >= auction_min_eps) {
            h_numAssign = 0;
            cudaMemset(ad.d_person2item,   -1, num_nodes * sizeof(int));
            cudaMemset(ad.d_item2person,   -1, num_nodes * sizeof(int));
            cudaMemset(ad.d_numAssign,      0, 1         * sizeof(int));

            int counter = 0;
            while(h_numAssign < num_nodes){
                counter += 1;
                cudaMemset(ad.d_bids,  0, num_nodes * num_nodes * sizeof(float));
                cudaMemset(ad.d_sbids, 0, num_nodes * sizeof(int));

                run_bidding<<<block, THREADS>>>(
                    num_nodes,

                    d_data,
                    d_offsets,
                    d_columns,

                    ad.d_person2item,
                    ad.d_bids,
                    ad.d_sbids,
                    ad.d_prices,
                    auction_eps,
                    ad.d_rand
                );
                run_assignment<<<block, THREADS>>>(
                    num_nodes,
                    ad.d_person2item,
                    ad.d_item2person,
                    ad.d_bids,
                    ad.d_sbids,
                    ad.d_prices,
                    ad.d_numAssign
                );

                cudaMemcpy(&h_numAssign, ad.d_numAssign, sizeof(int) * 1, cudaMemcpyDeviceToHost);
#ifdef DEBUG
                std::cerr << "counter=" << counter << " | h_numAssign=" << h_numAssign << std::endl;
                // if(counter > 1000) break;
#endif
            }
            if(verbose) {
                std::cerr << "auction_counter=" << counter << std::endl;
            }
            auction_eps *= auction_factor;
        }
     }

    return h_numAssign;
}

#endif
