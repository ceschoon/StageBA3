#ifndef SVD_WITH_T_H
#define SVD_WITH_T_H

void profilU(double *y, double *u, int n);
void profilU2(double *y, double *u, int n);
void identity(magmaDoubleComplex *l, int n);
void linspace(double *y, double min, double max, double n);
void f(magmaDoubleComplex *k, magmaDoubleComplex *l, 
	   double *d2, double *d4, double *y, int n, double t, double re, 
	   double alpha);
void svd_with_t(double re, double alpha, int N, double tMax, double dt, 
				int step);   


#endif