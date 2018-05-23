#include <math.h>

void buildD2_forward(double *D2, int n, double dy)
{
	for(int i=0; i<n*n; i++)
	{
		D2[i] = 0; // matrice de zéros
	}
	for(int i=0; i<n; i++)
	{
		D2[(n+1)*i] = -2/pow(dy,2); // diagonale
	}
	for(int i=0; i<n-1; i++)
	{
		D2[(n+1)*i+1] = 1/pow(dy,2); // sous-diagonale
		D2[(n+1)*i+n] = 1/pow(dy,2); // sur-diagonale
	}
	// conditions aux bords
	D2[0] = -7.0/4/pow(dy,2);
    D2[n*n-1] = -7.0/4/pow(dy,2);
}

void buildD4_forward(double *D4, int n, double dy)
{
	for(int i=0; i<n*n; i++)
	{
		D4[i] = 0; // matrice de zéros
	}
	for(int i=0; i<n; i++)
	{
		D4[(n+1)*i] = 6/pow(dy,4); // diagonale
	}
	for(int i=0; i<n-1; i++)
	{
		D4[(n+1)*i+1] = -4/pow(dy,4); // sous-diagonale
		D4[(n+1)*i+n] = -4/pow(dy,4); // sur-diagonale
	}
	for(int i=0; i<n-2; i++)
	{
		D4[(n+1)*i+2] = 1/pow(dy,4); // 2e sous-diagonale
		D4[(n+1)*i+2*n] = 1/pow(dy,4); // 2e sur-diagonale
	}
	// conditions aux bords
	D4[0] = 5/pow(dy,4);
	D4[1] = -15.0/4/pow(dy,4);
    D4[n*n-1] = 5/pow(dy,4);
    D4[n*n-2] = -15.0/4/pow(dy,4);
}