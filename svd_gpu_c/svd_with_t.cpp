#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include "magma.h"
#include "operateurs.h" 
#include "magma_operators.h"
#include "svd_with_t.h"	

// profil de vitesse
void profilU(double *y, double *u, int n)
{
	for (int i=0; i<n; i++)
	{
		u[i] = 1-pow(y[i],2);
	}
}
void profilU2(double *y, double *u2, int n)
{
	for (int i=0; i<n; i++)
	{
		u2[i] = -2;
	}
}

void identity(magmaDoubleComplex *l, int n)
{
	for (int i=0; i<n*n; i++)
	{
		l[i] = MAGMA_Z_MAKE(0,0);
	}
	for (int i=0; i<n; i++)
	{
		l[(n+1)*i] = MAGMA_Z_MAKE(1,0);
	}
}

void fill_zeros(magmaDoubleComplex *l, int n)
{
	for (int i=0; i<n*n; i++)
	{
		l[i] = MAGMA_Z_MAKE(0,0);
	}
}

void linspace(double *y, double min, double max, double n)
{
	for (int i=0; i<n; i++)
	{
		y[i] = min+(max-min)*((double)i)/(n-1);
	}
}

// fonction f dans RK4: dérivée de l(t)
void f(magmaDoubleComplex *k, magmaDoubleComplex *l, 
	   double *d2, double *d4, 
	   double *y, int n, double t, double re, double alpha)
{
	double dy = 2.0/(n+3);
	double *u,*u2;
	magmaDoubleComplex *a_cpu,*b_cpu,*a,*b,*m, *zeros;
	
	magma_dmalloc_cpu(&u , n);
	magma_dmalloc_cpu(&u2 , n);
	magma_zmalloc_cpu(&a_cpu , n*n);
	magma_zmalloc_cpu(&b_cpu , n*n);
	magma_zmalloc_cpu(&zeros , n*n);
	magma_zmalloc(&a , n*n);
	magma_zmalloc(&b , n*n);
	magma_zmalloc(&m , n*n);
	
	// calcule le profil de vitesse
	profilU(y,u,n);
	profilU2(y,u2,n);
	
	// calcul des opérateurs a et b
	for (int i=0; i<n*n; i++)
	{
		b_cpu[i] = MAGMA_Z_MAKE(d2[i],0);
		a_cpu[i] = MAGMA_Z_MAKE(d4[i],0);
		a_cpu[i] = MAGMA_Z_MAKE( u[i%n]*d2[i] ,0)
		           + MAGMA_Z_MAKE(0, 1.0/alpha/re*
		             (d4[i]-2*d2[i]*pow(alpha,2)) );
	}
	for (int i=0; i<n; i++)
	{
		int j = i*(n+1);
		b_cpu[j] = MAGMA_Z_MAKE( d2[j]-pow(alpha,2) ,0);
		a_cpu[j] = MAGMA_Z_MAKE( u[i] ,0)*b_cpu[j] 
			       - MAGMA_Z_MAKE( u2[i] ,0)
			       + MAGMA_Z_MAKE(0,1.0/alpha/re*
		             (d4[j]-2*d2[j]*pow(alpha,2)+pow(alpha,4)) );
	}
	magma_zsetmatrix(n,n,a_cpu,n,a,n);
	magma_zsetmatrix(n,n,b_cpu,n,b,n);
	
	// prépare le calcul de l'inverse de b
	magmaDoubleComplex *dwork;
	magma_int_t ldwork,*piv,info;
	ldwork = n*magma_get_zgetri_nb(n);
	magma_zmalloc(&dwork,ldwork);
	piv=(magma_int_t*)malloc(n*sizeof(magma_int_t));
	// calcul 
	magma_zgetrf_gpu(n,n,b,n,piv,&info); // LU
	magma_zgetri_gpu(n,b,n,piv,dwork,ldwork,&info); // inverse
	// nettoyage
	magma_free(dwork);
	free(piv);
	
	// m = b^-1*a
	fill_zeros(zeros,n);
	magma_zsetmatrix(n,n,zeros,n,m,n);
	magma_zgemm(MagmaNoTrans,MagmaNoTrans,n,n,n,MAGMA_Z_MAKE(0,-alpha),
				b,n,a,n,MAGMA_Z_MAKE(0,0),m,n);
	/*		
	printf("\nm en t=%f:", t);
	magma_zgetmatrix(n,n,m,n,a_cpu,n);
	printf("\nreal: %f", 1000000000*MAGMA_Z_REAL(a_cpu[0]));
	printf("\nimag: %f", 1000000000*MAGMA_Z_IMAG(a_cpu[0]));
	*/
	
	// k = m*l
	magma_zsetmatrix(n,n,zeros,n,k,n);
	magma_zgemm(MagmaNoTrans,MagmaNoTrans,n,n,n,MAGMA_Z_MAKE(1,0),
				m,n,l,n,MAGMA_Z_MAKE(0,0),k,n);
	
	magma_free_cpu(a_cpu);
	magma_free_cpu(b_cpu);
	magma_free_cpu(zeros);
	magma_free_cpu(u);
	magma_free_cpu(u2);
	magma_free(a);
	magma_free(b);
	magma_free(m);
}


void svd_with_t(double re, double alpha, int N, double tMax, double dt, 
				int step)
{
	int n = N-4;
	int nt = int(tMax/dt)+1;
	double t=0;
	double dy = 2.0/(n+3);
	double *tVec,*sMaxVec;
	double *d2,*d4,*y;
	magmaDoubleComplex *l, *l2, *temp_cpu;
	magmaDoubleComplex *k1,*k2,*k3,*k4;
	
	// utiles à la svd
	double *s1, *rwork;
	magma_int_t lwork;
	magmaDoubleComplex *dwork,*u,*vt, *l_cpu;
	
	magma_dmalloc_cpu(&tVec, nt/step);
	magma_dmalloc_cpu(&sMaxVec, nt/step);
	magma_dmalloc_cpu(&d2, n*n);
	magma_dmalloc_cpu(&d4, n*n);
	magma_dmalloc_cpu(&y, n);
	magma_zmalloc_cpu(&temp_cpu, n*n);
	magma_zmalloc_cpu(&l_cpu, n*n);
	magma_zmalloc(&l , n*n);
	magma_zmalloc(&l2, n*n);
	magma_zmalloc(&k1, n*n);
	magma_zmalloc(&k2, n*n);
	magma_zmalloc(&k3, n*n);
	magma_zmalloc(&k4, n*n);
	
	// prépare le calcul de la svd
	magma_dmalloc_cpu(&s1,n);
	magma_dmalloc_cpu(&rwork,5*n);
	magma_int_t nb = magma_get_dgesvd_nb(n,n); // optim. block size
	lwork = n*n+2*n+2*n*nb;
	magma_zmalloc_pinned(&dwork,lwork);
	magma_zmalloc_cpu(&u,n*n);
	magma_zmalloc_cpu(&vt,n*n);
	
	buildD2_forward(d2, n, dy); // opérateurs dérivées
	buildD4_forward(d4, n, dy);
	linspace(y,-1+2*dy,1-2*dy,n);
	
	identity(temp_cpu,n); // init l avec l'identité
	magma_zsetmatrix(n,n,temp_cpu,n,l,n);
	
	// intégration avec RK4
	for (int i=0; i<nt; i++)
	{
		f(k1,l,d2,d4,y,n,t,re,alpha);
		magma_zcopymatrix(n,n,l,n,l2,n);
		magmablas_zgeadd(n,n,MAGMA_Z_MAKE(dt/2,0),k1,n,l2,n);
		f(k2,l2,d2,d4,y,n,t+dt/2,re,alpha);
		magma_zcopymatrix(n,n,l,n,l2,n);
		magmablas_zgeadd(n,n,MAGMA_Z_MAKE(dt/2,0),k2,n,l2,n);
		f(k3,l2,d2,d4,y,n,t+dt/2,re,alpha);
		magma_zcopymatrix(n,n,l,n,l2,n);
		magmablas_zgeadd(n,n,MAGMA_Z_MAKE(dt,0),k3,n,l2,n);
		f(k4,l2,d2,d4,y,n,t+dt,re,alpha);
		
		magmablas_zgeadd(n,n,MAGMA_Z_MAKE(dt/6  ,0),k1,n,l,n);
		magmablas_zgeadd(n,n,MAGMA_Z_MAKE(dt/6*2,0),k2,n,l,n);
		magmablas_zgeadd(n,n,MAGMA_Z_MAKE(dt/6*2,0),k3,n,l,n);
		magmablas_zgeadd(n,n,MAGMA_Z_MAKE(dt/6  ,0),k4,n,l,n);
		t = t+dt;
		
		if ((i+1)%step==0)
		{	
			int j = (i+1)/step;
			magma_int_t info;
			
			magma_zgetmatrix(n,n,l,n,l_cpu,n);
			magma_zgesvd(MagmaNoVec,MagmaNoVec,n,n,l_cpu,n,s1,u,n,vt,n,dwork,
						 lwork,rwork,&info);
						 
			sMaxVec[j-1] = s1[0];
			tVec[j-1] = t;
			/*
			printf("\nl en t=%f:", t);
			magma_zgetmatrix(n,n,l,n,l_cpu,n);
			printf("\nreal: %f", MAGMA_Z_REAL(l_cpu[33+33*n]));
			printf("\nimag: %f", MAGMA_Z_IMAG(l_cpu[33+33*n]));
			*/
		}
	}
	
	// écrit les résultats dans un fichier
	FILE *dataFile;
	
	dataFile = fopen("data/data","w"); // écrase le contenu initial
	fclose(dataFile);
	
	dataFile = fopen("data/data","a");
	fprintf(dataFile, "Temps,ValSingMax\n");
	for (int i=0; i<nt/step; i++)
	{
		fprintf(dataFile, "%f,%f\n", tVec[i], sMaxVec[i]);
	}
	fclose(dataFile);
	
	magma_free_cpu(tVec);
	magma_free_cpu(sMaxVec);
	magma_free_cpu(temp_cpu);
	magma_free_cpu(d2);
	magma_free_cpu(d4);
	magma_free_cpu(y);
	magma_free_cpu(l_cpu);
	magma_free_cpu(rwork);
	magma_free_cpu(s1);
	magma_free_cpu(u);
	magma_free_cpu(vt);
	magma_free_pinned(dwork);
	magma_free(l);
	magma_free(l2);
	magma_free(k1);
	magma_free(k2);
	magma_free(k3);
	magma_free(k4);
}
