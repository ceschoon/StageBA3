#! /bin/bash

g++ -O3 -fopenmp -std=c++11 -DHAVE_CUBLAS -DADD_\
	-I/usr/local/cuda/include -I/usr/local/magma/include \
	-c operateurs.cpp svd_with_t.cpp main.cpp
	
g++ -fopenmp -o main main.o operateurs.o svd_with_t.o \
	-L/usr/local/magma/lib -lm -lmagma \
	-L/usr/local/cuda/lib64 -lcublas -lcudart \
	-L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential
