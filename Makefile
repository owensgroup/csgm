
GRAPHBLAS_PATH=/home/bjohnson/projects/davis/GraphBLAS

ARCH=\
  -gencode arch=compute_60,code=compute_60 \
  -gencode arch=compute_60,code=sm_60

OPTIONS=-O3 -use_fast_math

all: csgm

csgm: csgm.cu
	nvcc -g $(ARCH) $(OPTIONS) -w -std=c++11 -o csgm \
		csgm.cu \
	    $(GRAPHBLAS_PATH)/ext/moderngpu/src/mgpucontext.cu \
	    $(GRAPHBLAS_PATH)/ext/moderngpu/src/mgpuutil.cpp \
	    -I$(GRAPHBLAS_PATH)/ext/moderngpu/include \
	    -I$(GRAPHBLAS_PATH)/ext/cub/cub \
	    -I$(GRAPHBLAS_PATH)/ \
	    -I/usr/local/cuda/samples/common/inc/ \
	    -lboost_program_options \
	    -lcublas \
	    -lcusparse \
	    -lcurand

clean:
	rm -f csgm
	
