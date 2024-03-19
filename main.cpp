#include "matrix.hpp"
#include <iostream>


int main() {
    // Testing matrix creation and printing
    std::cout << "Testing matrix creation and printing:" << std::endl;
    Matrix<double> mat1{ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    mat1.printToStdOut();
    // Expected output:
    // 1 2 3
    // 4 5 6
    // 7 8 9

    // Testing addition
    std::cout << "\nTesting matrix addition:" << std::endl;
    Matrix<double> mat2{ {9, 8, 7}, {6, 5, 4}, {3, 2, 1} };
    auto sum = mat1 + mat2;
    sum.printToStdOut();
    // Expected output:
    // 10 10 10
    // 10 10 10
    // 10 10 10

    // Testing scalar multiplication
    std::cout << "\nTesting scalar multiplication:" << std::endl;
    auto scaled = mat1 * 2.0;
    scaled.printToStdOut();
    // Expected output:
    // 2 4 6
    // 8 10 12
    // 14 16 18

    // Testing matrix transpose
    std::cout << "\nTesting matrix transpose:" << std::endl;
    auto transposed = mat1.transpose();
    transposed.printToStdOut();
    // Expected output:
    // 1 4 7
    // 2 5 8
    // 3 6 9

    // Testing matrix multiplication
    std::cout << "\nTesting matrix multiplication:" << std::endl;
    auto product = mat1 * mat2;
    product.printToStdOut();
    // Expected output for mat1 * mat2 (assuming standard matrix multiplication):
    // 30 24 18
    // 84 69 54
    // 138 114 90

    // Testing determinant (if applicable)
    // Please replace with the actual determinant calculation if supported
    std::cout << "\nTesting determinant (for 2x2 matrix example):" << std::endl;
    Matrix<double> matDet{ {4, 7}, {2, 6} };
    std::cout << "Determinant: " << matDet.determinant() << std::endl;
    // Expected output:
    // Determinant: 10 (for the 2x2 example matrix)

    // Initial setup: creating a test matrix
    Matrix<double> A{ {4, 12, -16}, {12, 37, -43}, {-16, -43, 98} };
    std::cout << "Initial matrix A:" << std::endl;
    A.printToStdOut();

    // Testing determinant (assuming 3x3 matrix)
    std::cout << "\nDeterminant of A:" << std::endl;
    std::cout << A.determinant() << std::endl;
    // Expected output: 36

    // Testing inverse (if applicable)
    std::cout << "\nInverse of A:" << std::endl;
    auto A_inv = A.inverse();
    A_inv.printToStdOut();
    // Expected output needs to be calculated based on A.
    // Testing resizing - expanding A
    std::cout << "\nExpanding A to 4x4, filling with default values:" << std::endl;
    A.resize(4, 4, true);
    A.printToStdOut();
    // Expected output: Original matrix with an extra row and column of default values (e.g., zeros).

    // Testing resizing - reducing A back to 3x3
    std::cout << "\nReducing A back to 3x3:" << std::endl;
    A.resize(3, 3, true); // Assuming true retains old data
    A.printToStdOut();
    // Expected output: Should match the initial matrix A.

    // Testing max value and its index
    std::cout << "\nMaximum value in A and its index:" << std::endl;
    auto maxVal = A.max();
    auto maxIdx = A.maxIdx();
    std::cout << "Max value: " << maxVal << " at (" << maxIdx.first << ", " << maxIdx.second << ")" << std::endl;
    // Expected output needs to be based on the current state of A.

    Matrix<double> X = { {1, 2, 3, 4}, 
                         {2, 4, 6, 8}, 
                         {0, 1, 0, 1},
                         {3, 6, 9, 15}};

    std::cout << "X matrix\n";
    X.printToStdOut();
    std::cout << "Poor GaussElmination on {1, 0, 0, 0} on X=\n";
    X.poorGaussian({ 1, 0, 0, 0 }).printToStdOut();
    std::cout << "GaussElmination on X=\n";
    X.gaussJordanElimination().printToStdOut();
    std::cout << "Inverse of X=\n";
    X.inverse().printToStdOut();

    Matrix<double> L, U;

    X.luDecomposition(L, U);

    std::cout << "L = \n";
    L.printToStdOut();
    std::cout << "U = \n";
    U.printToStdOut();

    Matrix<double> CNNM = { {-1, -1, -1, -1, -1, -1, -1, -1, -1 },
                         {-1, 1, -1, -1, -1, -1, -1, 1, -1 },
                         {-1, -1, 1, -1, -1, -1, 1, -1, -1 },
                         {-1, -1, -1, 1, -1, 1, -1, -1, -1 },
                         {-1, -1, -1, -1, 1, -1, -1, -1, -1 },
                         {-1, -1, -1, 1, -1, 1, -1, -1, -1 },
                         {-1, -1, 1, -1, -1, -1, 1, -1, -1 },
                         {-1, 1, -1, -1, -1, -1, -1, 1, -1 },
                         {-1, -1, -1, -1, -1, -1, -1, -1, -1 }};
 
    Matrix<double> filterM = { {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1} };

    Matrix<double> filteredM = CNNM.filter(filterM);

    filteredM.printToStdOut();
    return 0;
}
