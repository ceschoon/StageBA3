#include <Eigen/Eigen/Core>
#include <Eigen/Eigen/SVD>
#include <iostream>
#include <chrono>
#include <string>

using namespace std;
using namespace Eigen;

int main(int argc, char *argv[])
{
	int N = 1000;
	
	if (argc>1)
	{
		string str = argv[1];
		N = stoi(str);
	}
	
	// crée la matrice
	MatrixXf A = MatrixXf::Random(N,N);
	
	chrono::system_clock::time_point start = chrono::system_clock::now();
			
	// effectue la svd
	BDCSVD<MatrixXf> B(A);
	
	chrono::system_clock::time_point finish = chrono::system_clock::now();
	long nanosecsExecution = chrono::duration_cast<chrono::nanoseconds>
		(finish-start).count();
	
	/*
	cout << "Singular values:" << endl;
	cout << B.singularValues().transpose() << endl;
	*/
	
	std::cout << "Execution time for a "<<N<<"x"<<N<<" matrix: "
			  << nanosecsExecution/1000000 << "." 
			  << (nanosecsExecution%1000000)/100000 
			  << (nanosecsExecution%100000)/10000
			  << (nanosecsExecution%10000)/1000
			  << " milliseconds" << endl;
	
	return 0;
}