#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "magma.h"
#include "operateurs.h"
#include "svd_with_t.h"

int main(int argc, char ** argv)
{
	int N = 401;
	double re = 10000;
	double alpha = 0.6;
	double tMax = 100;
	double dt = 0.1;
	int step = 10;
	
	if (argc>1){N = atoi(argv[1]);} 	// attention mauvais inputs
	if (argc>2){re = atof(argv[2]);} 	// attention mauvais inputs
	if (argc>3){alpha = atof(argv[3]);} // attention mauvais inputs
	if (argc>4){tMax = atof(argv[4]);} 	// attention mauvais inputs
	if (argc>5){dt = atof(argv[5]);} 	// attention mauvais inputs
	if (argc>6){step = atoi(argv[6]);} 	// attention mauvais inputs
	
	magma_init();
	
	svd_with_t(re,alpha,N,tMax,dt,step);
	
	magma_finalize();
	
	return 0;
}