#include "magma.h"
#include "operateurs.h"
#include "svd_with_t.h"

int main()
{
	magma_init();
	
	int N = 401;
	double re = 10000;
	double alpha = 0.6;
	double tMax = 100;
	double dt = 0.1;
	int step = 10;
	
	svd_with_t(re,alpha,N,tMax,dt,step);
	
	magma_finalize();
	
	return 0;
}