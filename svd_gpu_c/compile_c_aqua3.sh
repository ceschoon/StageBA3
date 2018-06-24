#! /bin/bash

gcc -O3 -fopenmp -DHAVE_CUBLAS -DADD_\
	-I/usr/local/cuda/include -I/opt/magma/include \
	-c operateurs.cpp svd_with_t.cpp main.cpp
	
gcc -fopenmp -o main main.o operateurs.o svd_with_t.o \
	-L/opt/magma/lib -lm -lmagma \
	-L/usr/local/cuda/lib64 -lcublas -lcudart \
	-L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential
