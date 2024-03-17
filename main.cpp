#include "matrix.hpp"

int main()
{
	Matrix<double> matrixA = { {1.0, 32.1}, {10.7, 2.0} };
	Matrix<double> A = { {0.0, 0.0, 4.7},
						 {0.0, 2.2, 1.5},
						 {1.0, 0.0, 3.2}
						};


	Matrix<double> matrixB = { {0.0, 3.2}, {11.0, 1.2} };
	Matrix<double> matrixAtranspose = matrixA.transpose();

	vector<double> gaussA = {2.2, 4.7, 3.3};

	Matrix<double> gaussianA = A.poorGaussian(gaussA);
	std::cout << "PoorGauss(A)=\n";
	gaussianA.printToStdOut();

	std::cout << "A=\n";
	matrixA.printToStdOut();

	std::cout << "A^T=\n";
	matrixAtranspose.printToStdOut();

	matrixA.swapRows(0, 1);

	std::cout << "\nSwapped\n";
	matrixA.printToStdOut();

	std::cout << "B=\n";
	matrixB.printToStdOut();

	matrixA = matrixA + matrixB;
	
	matrixA.printToStdOut();
	
	Matrix<double> matrixC = matrixA * matrixB;
	std::cout << "C= A * B=\n";
	matrixC.printToStdOut();

	Matrix<double> L;
	Matrix<double> U;

	A.luDecomposition(L, U);

	std::cout << "\nLU decomposition\n";
	std::cout << "L=\n";
	L.printToStdOut();
	std::cout << "U=\n";
	U.printToStdOut();


	std::cout << "Gauss elimination of A=\n";
	A.printToStdOut();
	Matrix<double> matrixAgauss = A.gaussJordanElimination();
	matrixAgauss.printToStdOut();

	std::cout << "\n\n determinant(A)=" << A.determinant() << "\n";

	A.trace();

	return 0;
}