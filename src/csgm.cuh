#ifndef __CSGM_HEADER
#define __CSGM_HEADER

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

#include "graphblas/graphblas.hpp"
#include "test/test.hpp"

#define THREADS 1024

typedef graphblas::Matrix<float> FloatMatrix;
typedef graphblas::Vector<float> FloatVector;

#endif