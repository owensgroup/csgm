# Makefile

ARCH=-gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60
OPTIONS=-O3 -use_fast_math

all: csgm

csgm: src/*
	nvcc -g $(ARCH) $(OPTIONS) -w -std=c++11 -o csgm \
		src/csgm.cu \
		GraphBLAS/ext/moderngpu/src/mgpucontext.cu \
		GraphBLAS/ext/moderngpu/src/mgpuutil.cpp \
		-IGraphBLAS/ext/moderngpu/include \
		-IGraphBLAS/ext/cub/cub \
		-IGraphBLAS/ \
		-Isrc/ \
		-I/usr/local/cuda/samples/common/inc/ \
		-lboost_program_options \
		-lcublas \
		-lcusparse \
		-lcurand

clean:
	rm -f csgm