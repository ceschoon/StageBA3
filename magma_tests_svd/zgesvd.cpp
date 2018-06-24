# include <stdio.h>
# include <string>
# include <cuda.h>
# include "magma.h"
# include "magma_lapack.h"

int min(int a, int b){return (a<b)?a:b;}

int main(int argc, char ** argv)
{
	int N = 1024;
	if (argc>1)
	{
		std::string str = argv[1];
		N = std::stoi(str);
	}
	
	magma_init ();	// initialize Magma
	real_Double_t gpu_time, cpu_time;
	
	// Matrix size
	magma_int_t m=N, n=N, n2=m*n, min_mn=min(m,n); 
	
	magmaDoubleComplex *a, *r; // a , r - mxn matrices
	magmaDoubleComplex *u, *vt; // u - mxm matrix , vt - nxn matrix on the host
	double *s1, *s2; // vectors of singular values
	double *rwork;
	magma_int_t info;
	magma_int_t ione = 1;
	double work[1], error = 1.; // used in difference computations
	double mone = -1.0;
	magmaDoubleComplex *h_work ; // h_work - workspace
	magma_int_t lwork ; // workspace size
	magma_int_t ISEED [4] = {0 ,0 ,0 ,1}; // seed
	
	// Allocate host memory
	magma_zmalloc_cpu(&a, m*n); // host memory for a
	magma_zmalloc_cpu(&vt, n*n); // host memory for vt
	magma_zmalloc_cpu(&u, m*m); // host memory for u
	magma_dmalloc_cpu(&s1, min_mn); // host memory for s1
	magma_dmalloc_cpu(&s2, min_mn); // host memory for s2
	magma_zmalloc_pinned(&r, m*n); // host memory for r
	magma_int_t nb = magma_get_zgesvd_nb(m, n); // optim. block size
	lwork = min_mn * min_mn +2* min_mn +2* min_mn * nb;
	magma_zmalloc_pinned(& h_work , lwork ); // host mem. for h_work
	magma_dmalloc_cpu(& rwork , 5*min_mn );
	
	// Randomize the matrix a
	lapackf77_zlarnv(&ione, ISEED, &n2, a );
	lapackf77_zlacpy(MagmaFullStr, &m , &n, a, &m, r, &m ); // a -> r (copy)
	
	// Compute the svd of r
	// and optionally the left and right singular vectors :
	// r = u * sigma * vt ; the diagonal elements of sigma ( s1 array )
	// are the singular values of a in descending order
	// the first min (m , n ) columns of u contain the left sing. vec .
	// the first min (m , n ) columns of vt contain the right sing. vec .
	
	// MAGMA
	gpu_time = magma_wtime();
	magma_zgesvd(MagmaNoVec,MagmaNoVec,m,n,r,m,s1,u,m,vt,n,h_work,lwork,rwork,
				 &info);
	gpu_time = magma_wtime() - gpu_time;
	
	// LAPACK
	cpu_time = magma_wtime();
	lapackf77_zgesvd("N","N",&m,&n,a,&m,s2,u,&m,vt,&n,h_work,&lwork,rwork,
					 &info);
	cpu_time = magma_wtime() - cpu_time;
	
	// Difference
	error = 1;//lapackf77_zlange("f",&min_mn,&ione,s1,&min_mn,work);
	//blasf77_daxpy(&min_mn,&mone,s1,&ione,s2,&ione);
	error = 1.;//lapackf77_zlange("f",&min_mn,&ione,s2,&min_mn,work);
	
	// Print results
	printf("%i,%f,%f,%e\n",N,gpu_time,cpu_time,error);
	
	// Free Host memory
	free(a);
	free(vt);
	free(s1);
	free(s2);
	free(u);
	free(rwork);
	magma_free_pinned(h_work);
	magma_free_pinned(r);
	magma_finalize(); // finalize Magma
	
	return EXIT_SUCCESS;	
}