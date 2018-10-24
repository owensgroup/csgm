#ifndef __CSGM_HEADER
#define __CSGM_HEADER

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

#define THREADS 1024

typedef graphblas::Matrix<float> FloatMatrix;
typedef graphblas::Vector<float> FloatVector;

struct AuctionData {
    int   *d_numAssign;
    int   *d_person2item;
    int   *d_item2person;
    float *d_prices;
    int   *d_sbids;
    float *d_bids;
    float *d_rand;
} ;

#endif