#include "matrix.hpp"
#include <iostream>
#include <vector>

int main() {
    try {
        // Matrix Creation
        Matrix<double> mat1({ {3, -7, 2}, {-3, 5, 1}, {6, -4, 0} });
        std::cout << "Matrix 1:\n";
        mat1.printToStdOut();
        // Expected Output: 
        // 1 2 3
        // 4 5 6
        // 7 8 9

        // Matrix Addition
        Matrix<double> mat2({ {9, 8, 7}, {6, 5, 4}, {3, 2, 1} });
        std::cout << "\nMatrix 1 + Matrix 2:\n";
        (mat1 + mat2).printToStdOut();
        // Expected Output:
        // 10 10 10
        // 10 10 10
        // 10 10 10

        // Scalar Multiplication
        std::cout << "\nMatrix 1 scaled by 2:\n";
        (mat1 * 2).printToStdOut();
        // Expected Output:
        // 2 4 6
        // 8 10 12
        // 14 16 18

        // Matrix Multiplication
        std::cout << "\nMatrix 1 * Matrix 2:\n";
        (mat1 * mat2).printToStdOut();
        // Expected output will depend on the contents of mat2. Assuming standard matrix multiplication.

        // Transpose
        std::cout << "\nTranspose of Matrix 1:\n";
        mat1.transpose().printToStdOut();
        // Expected Output:
        // 1 4 7
        // 2 5 8
        // 3 6 9

        // Determinant (if applicable)
        std::cout << "\nDeterminant of Matrix 1: " << mat1.determinant() << std::endl;
        // Expected output needs calculation.

        // Inverse (if applicable)
        std::cout << "\nInverse of Matrix 1:\n";
        mat1.inverse().printToStdOut();
        // Expected output needs calculation.

        // Eigenvalues and Eigenvectors (if applicable)
        // Expected output needs implementation and calculation.

        // LU Decomposition (if applicable)
        Matrix<double> L, U;
        mat1.luDecomposition(L, U);
        std::cout << "\nL matrix from LU Decomposition of Matrix 1:\n";
        L.printToStdOut();
        // Expected L matrix output
        std::cout << "\nU matrix from LU Decomposition of Matrix 1:\n";
        U.printToStdOut();
        // Expected U matrix output

        std::cout << "L * U=\n";
        auto LU = L * U;
        LU.printToStdOut();

        // Trace
        std::cout << "\nTrace of Matrix 1: " << mat1.trace() << std::endl;
        // Expected Output: Sum of the diagonal elements of mat1.

        // QR Decomposition (if applicable)
        Matrix<double> Q, R;
        mat1.qrDecomposition(Q, R);
        std::cout << "\nQ matrix from QR Decomposition of Matrix 1:\n";
        Q.printToStdOut();
        // Expected Q matrix output
        std::cout << "\nR matrix from QR Decomposition of Matrix 1:\n";
        R.printToStdOut();
        // Expected R matrix output

        // Pseudo Inverse (if applicable)
        std::cout << "\nPseudo Inverse of Matrix 1:\n";
        mat1.pseudoInverse().printToStdOut();
        // Expected output depends on the implementation.

        Matrix<double> CNNM = { {-1, -1, -1, -1, -1, -1, -1, -1, -1 },
                                {-1, 1, -1, -1, -1, -1, -1, 1, -1 },
                                {-1, -1, 1, -1, -1, -1, 1, -1, -1 },
                                {-1, -1, -1, 1, -1, 1, -1, -1, -1 },
                                {-1, -1, -1, -1, 1, -1, -1, -1, -1 },
                                {-1, -1, -1, 1, -1, 1, -1, -1, -1 },
                                {-1, -1, 1, -1, -1, -1, 1, -1, -1 },
                                {-1, 1, -1, -1, -1, -1, -1, 1, -1 },
                                {-1, -1, -1, -1, -1, -1, -1, -1, -1 } };

        Matrix<double> filterM = { {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1} };

        Matrix<double> filteredM = CNNM.filter(filterM);

        filteredM.printToStdOut();

        Matrix<double> permutation = { {0, 0, 1, 0},
                                       {0, 1, 0, 0},
                                       {0, 0, 0, 1},
                                       {1, 0, 0, 0} };

        std::cout << "The matrix P is a ptermuattion matrix: " <<( (permutation.isPermutationMatrix()) ? "True " : "False " )<< "Permuation matrix P = \n";

        permutation.printToStdOut();

        std::cout << "Number of inversions in the matrix " << permutation.numInversions() << ".\n";

        vector<double> inversions = { 3, 2, 4, 1 };
        Matrix<double> permM2(4, 4);

        permM2 = permM2.createPermutationMatrixFromInversion(inversions);

        std::cout << "Creating inversion matrix form {3, 2, 4, 1}!\n";

        permM2.printToStdOut();

        Matrix<double> resh = { {0, 0, 1.4, 0},
                                {0, 1.6, 0, 0},
                                {0, 0, 0, 2.9},
                                {0.7, 0, 0, 0} };

        std::cout << "Reshaping this 4x4 matrix: \n";
        resh.printToStdOut();

        std::cout << "To 2x8:\n";

        Matrix<double> reshaped = resh.reshape(resh, 2, 8);

        reshaped.printToStdOut();

        std::cout << "Scaling matrix:\n";

        Matrix<double> scaled = reshaped.scale(7.0, 3.0);
        scaled.printToStdOut();
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
    }

    return 0;
}
