// unit_test.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include "matrix.hpp" 

using namespace std;


// Comparing matrices with floating point representation 
template<typename T>
bool matricesEqual(const Matrix<T>& A, const Matrix<T>& B, T epsilon = 1e-5) 
{
    if (A.row() != B.row() || A.col() != B.col()) return false;
    for (size_t i = 0; i < A.row(); ++i)
        for (size_t j = 0; j < A.col(); ++j)
            if (abs(A.getMatrix()[i][j] - B.getMatrix()[i][j]) > epsilon)
                return false;
    return true;
}

// Create matrix from initializer list
template<typename T>
Matrix<T> createMatrix(initializer_list<initializer_list<T>> init) 
{
    size_t rows = init.size();
    size_t cols = init.begin()->size();
    Matrix<T> mat(rows, cols);
    size_t i = 0;
    for (const auto& row : init) {
        size_t j = 0;
        for (const auto& val : row) {
            mat.getMatrix()[i][j++] = val;
        }
        ++i;
    }
    return mat;
}

// Test counters
int testsPassed = 0;
int testsFailed = 0;

// Test reports
void reportTestResult(const string& testName, bool passed) 
{
    if (passed) {
        cout << "[PASS] " << testName << endl;
        testsPassed++;
    }
    else {
        cout << "[FAIL] " << testName << endl;
        testsFailed++;
    }
}

// Test functions

// /////////////////////////////////////////////////////////////////////////////
// tests for operator+ function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: operator+ adds 2 matrix of matching sizes
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Matrix B = [[5, 6],
            [7, 8]]
Expected output
  Matrix C = [[6, 8],
            [10, 12]]
*/
void testOperatorPlus_SameSize_AddsCorrectly() 
{
    string testName = "OperatorPlus_SameSize_AddsCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        Matrix<int> B = createMatrix<int>({ {5, 6}, {7, 8} });

        // Compute
        Matrix<int> C = A + B;

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {6, 8}, {10, 12} });

        // Check
        bool passed = matricesEqual(C, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: operator+ throws exception on size mismatch
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
  Matrix B = [[7, 8],
            [9, 10]]
Expected output
  Runtime error: "Cannot add matrices of not the same size!" 
*/
void testOperatorPlus_DifferentSize_ThrowsException() 
{
    string testName = "OperatorPlus_DifferentSize_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });
        Matrix<int> B = createMatrix<int>({ {7, 8}, {9, 10} });

        // Compute
        Matrix<int> C = A + B;

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot add matrices of not the same size!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests az operator- function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: operator- kivon két azonos méretű Matrixot helyesen.
Inputs
  Matrix A = [[5, 7],
            [9, 11]]
  Matrix B = [[1, 2],
            [3, 4]]
Expected output
  Matrix C = [[4, 5],
            [6, 7]]
*/
void testOperatorMinus_SameSize_SubtractsCorrectly() 
{
    string testName = "OperatorMinus_SameSize_SubtractsCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {5, 7}, {9, 11} });
        Matrix<int> B = createMatrix<int>({ {1, 2}, {3, 4} });

        // Compute
        Matrix<int> C = A - B;

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {4, 5}, {6, 7} });

        // Check
        bool passed = matricesEqual(C, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: operator- throws exception, ha a Matrixok mérete eltérő.
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
  Matrix B = [[7, 8],
            [9, 10]]
Expected output
  Runtime error a "Cannot subtract matrices of not the same size!" 
*/
void testOperatorMinus_DifferentSize_ThrowsException() 
{
    string testName = "OperatorMinus_DifferentSize_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });
        Matrix<int> B = createMatrix<int>({ {7, 8}, {9, 10} });

        // Compute
        Matrix<int> C = A - B;

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot subtract matrices of not the same size!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests on operator* function 
// /////////////////////////////////////////////////////////////////////////////

/*
Test: operator* 2 compatible matrices
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
  Matrix B = [[7, 8],
            [9, 10],
            [11, 12]]
Expected output
  Matrix C = [[58, 64],
            [139, 154]]
*/
void testOperatorMultiply_CompatibleSizes_MultipliesCorrectly() 
{
    string testName = "OperatorMultiply_CompatibleSizes_MultipliesCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });
        Matrix<int> B = createMatrix<int>({ {7, 8}, {9, 10}, {11, 12} });

        // Compute
        Matrix<int> C = A * B;

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {58, 64}, {139, 154} });

        // Check
        bool passed = matricesEqual(C, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: operator* throws exception on size mismatch
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Matrix B = [[5, 6],
            [7, 8],
            [9, 10]]
Expected output
  Runtime error a "Matrix dimensions do not match for multiplication."
*/
void testOperatorMultiply_IncompatibleSizes_ThrowsException() 
{
    string testName = "OperatorMultiply_IncompatibleSizes_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        Matrix<int> B = createMatrix<int>({ {5, 6}, {7, 8}, {9, 10} });

        // Compute
        Matrix<int> C = A * B;

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const invalid_argument& e) {
        // Check if we got the expected exception
        string expectedMsg = "Matrix dimensions do not match for multiplication.";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests on operator* function 
// /////////////////////////////////////////////////////////////////////////////

/*
Test: operator* scalar multiply valid
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Scalar = 3
Expected output
  Matrix C = [[3, 6],
            [9, 12]]
*/
void testOperatorMultiplyScalar_MultipliesCorrectly() 
{
    string testName = "OperatorMultiplyScalar_MultipliesCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        int scalar = 3;

        // Compute
        Matrix<int> C = A * scalar;

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {3, 6}, {9, 12} });

        // Check
        bool passed = matricesEqual(C, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests on operator/= function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: operator/ scalar divide valid
Inputs
  Matrix A = [[2, 4],
            [6, 8]]
  Scalar = 2
Expected output
  Matrix C = [[1, 2],
            [3, 4]]
*/
void testOperatorDivideScalar_DividesCorrectly() 
{
    string testName = "OperatorDivideScalar_DividesCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {2, 4}, {6, 8} });
        int scalar = 2;

        // Compute
        Matrix<int> C = A / scalar;

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {1, 2}, {3, 4} });

        // Check
        bool passed = matricesEqual(C, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: operator/ throws expection on 0 scalar
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Scalar = 0
Expected output
  Runtime error a "Cannot divide by zero, at normalizing row!" 
*/
void testOperatorDivideScalar_DivideByZero_ThrowsException() 
{
    string testName = "OperatorDivideScalar_DivideByZero_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        int scalar = 0;

        // Compute
        Matrix<int> C = A / scalar;

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Invalid scalar, cannot divide by 0!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a swapRows function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: swapRows swap two rows valid
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
  Swicth 0 and 2 rows
Expected output
    [[7, 8, 9],
     [4, 5, 6],
     [1, 2, 3]]
*/
void testSwapRows_ValidIndices_SwapsCorrectly() 
{
    string testName = "SwapRows_ValidIndices_SwapsCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} });

        // Compute
        A.swapRows(0, 2);

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {7, 8, 9}, {4, 5, 6}, {1, 2, 3} });

        // Check
        bool passed = matricesEqual(A, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: swapRowsthrows exception index out of bounds
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Swap 0. és 3. rows (invalid)
Expected output
  Runtime error a "Invalid row indexes for swapping!"
*/
void testSwapRows_InvalidIndices_ThrowsException() 
{
    string testName = "SwapRows_InvalidIndices_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });

        // Compute
        A.swapRows(0, 3);

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Invalid row indexes for swapping!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a swapCols function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: swapCols 2 columns swap valid
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
 Swap  0. és 2. cols
Expected output
    [[3, 2, 1],
     [6, 5, 4],
     [9, 8, 7]]
*/
void testSwapCols_ValidIndices_SwapsCorrectly() 
{
    string testName = "SwapCols_ValidIndices_SwapsCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} });

        // Compute
        A.swapCols(0, 2);

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {3, 2, 1}, {6, 5, 4}, {9, 8, 7} });

        // Check
        bool passed = matricesEqual(A, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: swapCols throws exception
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Swap  0. és 3. cols (invalid)
Expected output
  Runtime error a "Column index out of bounds." 
*/
void testSwapCols_InvalidIndices_ThrowsException() 
{
    string testName = "SwapCols_InvalidIndices_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });

        // Compute
        A.swapCols(0, 3);

        // If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Column index out of bounds.";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a normalizeRowForPosition function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: normalizeRowForPosition valid
Inputs
  Matrix A = [[2, 4, 6],
            [8, 10, 12]]
  Nrom on 0. row with 1. pos  element (4)
Expected output
    [[0.5, 1, 1.5],
     [8, 10, 12]]
  return val: 4
*/
void testNormalizeRowForPosition_ValidIndices_NormalizesCorrectly() 
{
    string testName = "NormalizeRowForPosition_ValidIndices_NormalizesCorrectly";
    try {
        // Prepare
        Matrix<double> A = createMatrix<double>({ {2.0, 4.0, 6.0}, {8.0, 10.0, 12.0} });

        // Compute
        double denominator = A.normalizeRowForPosition(0, 1);

        // Expected result
        Matrix<double> expected = createMatrix<double>({ {0.5, 1.0, 1.5}, {8.0, 10.0, 12.0} });

        // Check
        bool passed = matricesEqual(A, expected, 1e-6) && abs(denominator - 4.0) < 1e-6;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: normalizeRowForPosition throws exception out of bounds
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  norm 2. row with 1. position (invalid)
Expected output
  Runtime error a "Cannot normalize row, indexes out of bounds!" 
*/
void testNormalizeRowForPosition_InvalidIndices_ThrowsException() {
    string testName = "NormalizeRowForPosition_InvalidIndices_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });

        // Compute
        int denominator = A.normalizeRowForPosition(2, 1);

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot normalize row, indexes out of bounds!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

/*
Test: normalizeRowForPosition excpetion, norm val is 0
Inputs
  Matrix A = [[0, 0],
            [1, 2]]
  Nrom 0. row with 1. position element (0)
Expected output
  Runtime error a "Cannot divide by zero, at normalizing row!" 
*/
void testNormalizeRowForPosition_DivideByZero_ThrowsException() {
    string testName = "NormalizeRowForPosition_DivideByZero_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {0, 0}, {1, 2} });

        // Compute
        int denominator = A.normalizeRowForPosition(0, 1);

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot divide by zero";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a scalarMultiplyRow function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: scalarMultiplyRow valid
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
  Scalar = 2
  Row index = 1
Expected output
    [[1, 2, 3],
     [8, 10, 12]]
*/
void testScalarMultiplyRow_ValidRow_MultipliesCorrectly() {
    string testName = "ScalarMultiplyRow_ValidRow_MultipliesCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });
        int scalar = 2;
        size_t rowIdx = 1;

        // Compute
        A.scalarMultiplyRow(scalar, rowIdx);

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {1, 2, 3}, {8, 10, 12} });

        // Check
        bool passed = matricesEqual(A, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: scalarMultiplyRow throws exception out of bounds
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Scalar = 3
  Row index = 2 (invalid)
Expected output
  Runtime error a "Cannot scalar multiply row, row index out of bounds!" 
*/
void testScalarMultiplyRow_InvalidRow_ThrowsException() {
    string testName = "ScalarMultiplyRow_InvalidRow_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        int scalar = 3;
        size_t rowIdx = 2;

        // Compute
        A.scalarMultiplyRow(scalar, rowIdx);

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot scalar multiply row, row index out of bounds!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a scalarMultiplyCol function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: scalarMultiplyCol valid
Inputs
  Matrix A = [[1, 2],
            [3, 4],
            [5, 6]]
  Scalar = 3
  Col index = 0
Expected output
    [[3, 2],
     [9, 4],
     [15, 6]]
*/
void testScalarMultiplyCol_ValidCol_MultipliesCorrectly() {
    string testName = "ScalarMultiplyCol_ValidCol_MultipliesCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4}, {5, 6} });
        int scalar = 3;
        size_t colIdx = 0;

        // Compute
        A.scalarMultiplyCol(scalar, colIdx);

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {3, 2}, {9, 4}, {15, 6} });

        // Check
        bool passed = matricesEqual(A, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: scalarMultiplyCol throws exception, col index out of bounds
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Scalar = 2
  Col index = 3 (invalid)
Expected output
  Runtime error a "Cannot scalar multiply col, col index out of bounds." 
*/
void testScalarMultiplyCol_InvalidCol_ThrowsException() {
    string testName = "ScalarMultiplyCol_InvalidCol_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        int scalar = 2;
        size_t colIdx = 3;

        // Compute
        A.scalarMultiplyCol(scalar, colIdx);

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot scalar multiply col, col index out if bounds!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests az addRow function 
// /////////////////////////////////////////////////////////////////////////////

/*
Test: addRow based on row index
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Row index = 1
  Row to add index = 0
  Mult factor  = 2
Expected output
    [[1, 2],
     [5, 8]]
*/
void testAddRow_RowIndices_AddsCorrectly() {
    string testName = "AddRow_RowIndices_AddsCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        size_t row = 1;
        size_t rowToAdd = 0;
        int sign = 1;
        int howManyTimes = 2;

        // Compute
        A.addRow(row, rowToAdd, sign, howManyTimes);

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {1, 2}, {5, 8} });

        // Check
        bool passed = matricesEqual(A, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: addRow row idx throws exception, out of bounds
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Row index = 0
  Row to add index = 2 (invalid)
  Mult factor = 1
Expected output
  Runtime error a "Cannot add rows, since they are out of bounds!" 
*/
void testAddRow_RowIndices_InvalidRowIndices_ThrowsException() {
    string testName = "AddRow_RowIndices_InvalidRowIndices_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        size_t row = 0;
        size_t rowToAdd = 2;
        int sign = 1;
        int howManyTimes = 1;

        // Compute
        A.addRow(row, rowToAdd, sign, howManyTimes);

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot add rows, since they are out of bounds!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a addRow function vector based
// /////////////////////////////////////////////////////////////////////////////

/*
Test: addRow vektor based valid
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Hozzáadandó vektor = [5, 6]
  Row index = 0
  Sign = -1
  Mult factor  = 1
Expected output
    [[-4, -4],
     [3, 4]]
*/
void testAddRow_Vector_AddsCorrectly() {
    string testName = "AddRow_Vector_AddsCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        vector<int> rowToAdd = { 5, 6 };
        size_t row = 0;
        int sign = -1;
        int howManyTimes = 1;

        // Compute
        A.addRow(row, rowToAdd, sign, howManyTimes);

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {-4, -4}, {3, 4} });

        // Check
        bool passed = matricesEqual(A, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: addRow (vektor) throws exception, vector size mismatch
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  rowToAdd vektor = [5, 6, 7] (méreteltérés)
  Row index = 0
  sign = 1
  Mult factor = 1
Expected output
  Runtime error a "Cannot add row since it is out of bounds, or size doesn't match!" 
*/
void testAddRow_Vector_SizeMismatch_ThrowsException() {
    string testName = "AddRow_Vector_SizeMismatch_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        vector<int> rowToAdd = { 5, 6, 7 };
        size_t row = 0;
        int sign = 1;
        int howManyTimes = 1;

        // Compute
        A.addRow(row, rowToAdd, sign, howManyTimes);

        // If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot add row since it is out of bounds, or size doesnt match!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a dotProduct function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: dotProduct helyesen számolja ki két sor szorzatát.
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]
  row1 = 0
  row2 = 2
Expected output
  Dot Product = 1*7 + 2*8 + 3*9 = 50
*/
void testDotProduct_Rows_CalculatesCorrectly() {
    string testName = "DotProduct_Rows_CalculatesCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} });
        size_t row1 = 0;
        size_t row2 = 2;

        // Compute
        int result = A.dotProduct(row1, row2);

        // Expected result
        int expected = 50;

        // Check
        bool passed = (result == expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: dotProduct (sorok) throws exception, ha a sorindexek kívül esnek a határokon.
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  row1 = 0
  row2 = 3 (invalid)
Expected output
  Runtime error a "Cannot calculate dotproduct because indexes are out of bounds!" 
*/
void testDotProduct_Rows_InvalidIndices_ThrowsException() {
    string testName = "DotProduct_Rows_InvalidIndices_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        size_t row1 = 0;
        size_t row2 = 3;

        // Compute
        int result = A.dotProduct(row1, row2);

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot calulate dotproduct because indexes are out of bounds!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

/*
Test: dotProduct ugyanazon sor szorzata nulla.
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  row1 = 1
  row2 = 1
Expected output
  Dot Product = 0
*/
void testDotProduct_SameRow_ReturnsZero() {
    string testName = "DotProduct_SameRow_ReturnsZero";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        size_t row1 = 1;
        size_t row2 = 1;

        // Compute
        int result = A.dotProduct(row1, row2);

        // Expected result
        int expected = 0;

        // Check
        bool passed = (result == expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a dotProduct (vektor és sor) function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: dotProduct helyesen számolja ki egy sor és egy vektor szorzatát.
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
  Vektor = [7, 8, 9]
  Sor index = 0
Expected output
  Dot Product = 1*7 + 2*8 + 3*9 = 50
*/
void testDotProduct_Vector_Row_CalculatesCorrectly() 
{
    string testName = "DotProduct_Vector_Row_CalculatesCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });
        vector<int> vec = { 7, 8, 9 };
        size_t rowIdx = 0;

        // Compute
        int result = A.dotProduct(vec, rowIdx);

        // Expected result
        int expected = 50;

        // Check
        bool passed = (result == expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: dotProduct (vektor és sor) throws exception, ha a vektor mérete nem egyezik a sor méretével.
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Vektor = [5, 6, 7] (méreteltérés)
  Sor index = 0
Expected output
  Runtime error a "Cannot perform dot product, size doesn't match, vector out of bounds!" 
*/
void testDotProduct_Vector_Row_SizeMismatch_ThrowsException() 
{
    string testName = "DotProduct_Vector_Row_SizeMismatch_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        vector<int> vec = { 5, 6, 7 };
        size_t rowIdx = 0;

        // Compute
        int result = A.dotProduct(vec, rowIdx);

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot perform dot product, size doesnt match, vector out of bounds!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

/*
Test: dotProduct (vektor és sor) throws exception, ha a sorindex kívül esik a határokon.
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Vektor = [5, 6]
  Sor index = 2 (invalid)
Expected output
  Runtime error a "Cannot perform dot product, row index out of bounds!" 
*/
void testDotProduct_Vector_Row_InvalidRowIndex_ThrowsException() 
{
    string testName = "DotProduct_Vector_Row_InvalidRowIndex_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        vector<int> vec = { 5, 6 };
        size_t rowIdx = 2;

        // Compute
        int result = A.dotProduct(vec, rowIdx);

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot perform dot product, row index out of bounds!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        // Other exceptions will result in test failing
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a rowAbs function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: rowAbs helyesen számolja ki egy sor eurklei normáját.
Inputs
  Matrix A = [[3, 4],
            [0, 0]]
  Sor index = 0
Expected output
  Norm = 5.0
*/
void testRowAbs_ValidRow_CalculatesCorrectly() 
{
    string testName = "RowAbs_ValidRow_CalculatesCorrectly";
    try {
        // Prepare
        Matrix<double> A = createMatrix<double>({ {3.0, 4.0}, {0.0, 0.0} });
        size_t rowIdx = 0;

        // Compute
        double norm = A.rowAbs(rowIdx);

        // Expected result
        double expected = 5.0;

        // Check
        bool passed = (abs(norm - expected) < 1e-6);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: rowAbs throws exception, ha a sorindex kívül esik a határokon.
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
  Sor index = 2 (invalid)
Expected output
  Runtime error a "Row index out of bounds, cannot calculate absolute!" 
*/
void testRowAbs_InvalidRowIndex_ThrowsException() 
{
    string testName = "RowAbs_InvalidRowIndex_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });
        size_t rowIdx = 2;

        // Compute
        int norm = A.rowAbs(rowIdx);

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Row index out of bounds, cannot calculate absoulte!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a transpose function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: transpose helyesen visszaadja a Matrix transzponáltját.
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
Expected output
  Matrix C = [[1, 4],
            [2, 5],
            [3, 6]]
*/
void testTranspose_ReturnsCorrectTranspose() 
{
    string testName = "Transpose_ReturnsCorrectTranspose";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });

        // Compute
        Matrix<int> C = A.transpose();

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {1, 4}, {2, 5}, {3, 6} });

        // Check
        bool passed = matricesEqual(C, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: transpose ugyanazt a Matrixot adja vissza, ha az szimmetrikus.
Inputs
  Matrix A = [[1, 2],
            [2, 3]]
Expected output
  Matrix C = [[1, 2],
            [2, 3]]
*/
void testTranspose_SymmetricMatrix_ReturnsSameMatrix() 
{
    string testName = "Transpose_SymmetricMatrix_ReturnsSameMatrix";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {2, 3} });

        // Compute
        Matrix<int> C = A.transpose();

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {1, 2}, {2, 3} });

        // Check
        bool passed = matricesEqual(C, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a trace function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: trace helyesen számolja ki a Matrix nyomát.
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
Expected output
  Trace = 5
*/
void testTrace_SquareMatrix_CalculatesCorrectly() 
{
    string testName = "Trace_SquareMatrix_CalculatesCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });

        // Compute
        int trace = A.trace();

        // Expected result
        int expected = 5;

        // Check
        bool passed = (trace == expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: trace throws exception, ha a Matrix nem négyzetes.
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
Expected output
  Runtime error a "Trace cannot be calculated for non square matrices!" 
*/
void testTrace_NonSquareMatrix_ThrowsException() 
{
    string testName = "Trace_NonSquareMatrix_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });

        // Compute
        int trace = A.trace();

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Trace cannot be calculated for non square matrices!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a setToIdentity és identity function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: setToIdentity helyesen állítja be a Matrixot azonosító Matrixra.
Inputs
  Matrix A = [[5, 6],
            [7, 8]]
Expected output
  Matrix A művelet után:
    [[1, 0],
     [0, 1]]
*/
void testSetToIdentity_SquareMatrix_SetsCorrectly() 
{
    string testName = "SetToIdentity_SquareMatrix_SetsCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {5, 6}, {7, 8} });

        // Compute
        A.setToIdentity();

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {1, 0}, {0, 1} });

        // Check
        bool passed = matricesEqual(A, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: setToIdentity throws exception, ha a Matrix nem négyzetes.
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
Expected output
  Runtime error a "Cannot set non square matrix to identity!" 
*/
void testSetToIdentity_NonSquareMatrix_ThrowsException() 
{
    string testName = "SetToIdentity_NonSquareMatrix_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });

        // Compute
        A.setToIdentity();

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Cannot set non square matrix to identity!";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: identity helyesen létrehoz egy identitási Matrixot adott mérettel.
Inputs
  Méret = 3
Expected output
  Matrix C = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
*/
void testIdentity_CreatesCorrectIdentityMatrix() 
{
    string testName = "Identity_CreatesCorrectIdentityMatrix";
    try {
        // Prepare
        size_t size = 3;
        Matrix<int> A(1, 1); // Csak egy példány; A nem használódik

        // Compute
        Matrix<int> C = A.identity(size);

        // Expected result
        Matrix<int> expected = createMatrix<int>({ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} });

        // Check
        bool passed = matricesEqual(C, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests az isSymmetric function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: isSymmetric igazat ad vissza szimmetrikus Matrixokra.
Inputs
  Matrix A = [[1, 2, 3],
            [2, 4, 5],
            [3, 5, 6]]
Expected output
  true
*/
void testIsSymmetric_SymmetricMatrix_ReturnsTrue() 
{
    string testName = "IsSymmetric_SymmetricMatrix_ReturnsTrue";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {2, 4, 5}, {3, 5, 6} });

        // Compute
        bool result = A.isSymmetric();

        // Expected result
        bool expected = true;

        // Check
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: isSymmetric hamisat ad vissza nem szimmetrikus Matrixokra.
Inputs
  Matrix A = [[1, 0],
            [2, 1]]
Expected output
  false
*/
void testIsSymmetric_NonSymmetricMatrix_ReturnsFalse() 
{
    string testName = "IsSymmetric_NonSymmetricMatrix_ReturnsFalse";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 0}, {2, 1} });

        // Compute
        bool result = A.isSymmetric();

        // Expected result
        bool expected = false;

        // Check
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: isSymmetric hamisat ad vissza nem négyzetes Matrixokra.
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
Expected output
  false
*/
void testIsSymmetric_NonSquareMatrix_ReturnsFalse() 
{
    string testName = "IsSymmetric_NonSquareMatrix_ReturnsFalse";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });

        // Compute
        bool result = A.isSymmetric();

        // Expected result
        bool expected = false;

        // Check
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests az isThisOrthogonal és isThisOrthonormed függvényekhez
// /////////////////////////////////////////////////////////////////////////////

/*
Test: isThisOrthogonal igazat ad vissza ortogonális Matrixokra.
Inputs
  Matrix A = [[1, 0],
            [0, 1]]
Expected output
  true
*/
void testIsThisOrthogonal_OrthogonalMatrix_ReturnsTrue() 
{
    string testName = "IsThisOrthogonal_OrthogonalMatrix_ReturnsTrue";
    try {
        // Prepare
        Matrix<double> A = createMatrix<double>({ {1.0, 0.0}, {0.0, 1.0} });

        // Compute
        bool result = A.isThisOrthogonal();

        // Expected result
        bool expected = true;

        // Check
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: isThisOrthogonal hamisat ad vissza nem ortogonális Matrixokra.
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
Expected output
  false
*/
void testIsThisOrthogonal_NonOrthogonalMatrix_ReturnsFalse() 
{
    string testName = "IsThisOrthogonal_NonOrthogonalMatrix_ReturnsFalse";
    try {
        // Prepare
        Matrix<double> A = createMatrix<double>({ {1.0, 2.0}, {3.0, 4.0} });

        // Compute
        bool result = A.isThisOrthogonal();

        // Expected result
        bool expected = false;

        // Check
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: isThisOrthonormed igazat ad vissza ortonormális Matrixokra.
Inputs
  Matrix A = [[1, 0],
            [0, 1]]
Expected output
  true
*/
void testIsThisOrthonormed_OrthonormalMatrix_ReturnsTrue() 
{
    string testName = "IsThisOrthonormed_OrthonormalMatrix_ReturnsTrue";
    try {
        // Prepare
        Matrix<double> A = createMatrix<double>({ {1.0, 0.0}, {0.0, 1.0} });

        // Compute
        bool result = A.isThisOrthonormed();

        // Expected result
        bool expected = true;

        // Check
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: isThisOrthonormed hamisat ad vissza ortogonális, de nem ortonormális Matrixokra.
Inputs
  Matrix A = [[2, 0],
            [0, 2]]
Expected output
  false
*/
void testIsThisOrthonormed_OrthogonalNotOrthonormal_ReturnsFalse() 
{
    string testName = "IsThisOrthonormed_OrthogonalNotOrthonormal_ReturnsFalse";
    try {
        // Prepare
        Matrix<double> A = createMatrix<double>({ {2.0, 0.0}, {0.0, 2.0} });

        // Compute
        bool result = A.isThisOrthonormed();

        // Expected result
        bool expected = false;

        // Check
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: isThisOrthonormed hamisat ad vissza nem ortogonális Matrixokra.
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
Expected output
  false
*/
void testIsThisOrthonormed_NonOrthogonalMatrix_ReturnsFalse() 
{
    string testName = "IsThisOrthonormed_NonOrthogonalMatrix_ReturnsFalse";
    try {
        // Prepare
        Matrix<double> A = createMatrix<double>({ {1.0, 2.0}, {3.0, 4.0} });

        // Compute
        bool result = A.isThisOrthonormed();

        // Expected result
        bool expected = false;

        // Check
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests a determinant function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: determinant helyesen számolja ki a négyzetes Matrix determinánsát.
Inputs
  Matrix A = [[1, 2],
            [3, 4]]
Expected output
  Determinant = -2
*/
void testDeterminant_SquareMatrix_CalculatesCorrectly() 
{
    string testName = "Determinant_SquareMatrix_CalculatesCorrectly";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {3, 4} });

        // Compute
        int det = A.determinant();

        // Expected result
        int expected = -2;

        // Check
        bool passed = (det == expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: determinant throws exception, ha a Matrix nem négyzetes.
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
Expected output
  Runtime error a "Determinant can only be calculated for square matrices." 
*/
void testDeterminant_NonSquareMatrix_ThrowsException() 
{
    string testName = "Determinant_NonSquareMatrix_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });

        // Compute
        int det = A.determinant();

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Determinant can only be calculated for square matrices.";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// tests az inverse function
// /////////////////////////////////////////////////////////////////////////////

/*
Test: inverse helyesen számolja ki egy invertibilis négyzetes Matrix inverzét.
Inputs
  Matrix A = [[4, 7],
            [2, 6]]
Expected output
  Inverz C = [[0.6, -0.7],
            [-0.2, 0.4]]
*/
void testInverse_InvertibleMatrix_CalculatesCorrectly() 
{
    string testName = "Inverse_InvertibleMatrix_CalculatesCorrectly";
    try {
        // Prepare
        Matrix<double> A = createMatrix<double>({ {4.0, 7.0}, {2.0, 6.0} });

        // Compute
        Matrix<double> C = A.inverse();

        // Expected result
        Matrix<double> expected = createMatrix<double>({ {0.6, -0.7}, {-0.2, 0.4} });

        // Check
        bool passed = matricesEqual(C, expected, 1e-6);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: inverse throws exception, ha a Matrix nem invertibilis (szingular).
Inputs
  Matrix A = [[1, 2],
            [2, 4]]
Expected output
  Runtime error a "Matrix is singular and cannot be inverted." 
*/
void testInverse_SingularMatrix_ThrowsException() 
{
    string testName = "Inverse_SingularMatrix_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2}, {2, 4} });

        // Compute
        Matrix<int> C = A.inverse();

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Matrix is singular and cannot be inverted.";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
Test: inverse throws exception, ha a Matrix nem négyzetes.
Inputs
  Matrix A = [[1, 2, 3],
            [4, 5, 6]]
Expected output
  Runtime error a "Matrix must be square to calculate the inverse." 
*/
void testInverse_NonSquareMatrix_ThrowsException() 
{
    string testName = "Inverse_NonSquareMatrix_ThrowsException";
    try {
        // Prepare
        Matrix<int> A = createMatrix<int>({ {1, 2, 3}, {4, 5, 6} });

        // Compute
        Matrix<int> C = A.inverse();

// If no exception is thrown, the test failed
        reportTestResult(testName, false);
    }
    catch (const runtime_error& e) {
        // Check if we got the expected exception
        string expectedMsg = "Matrix must be square to calculate the inverse.";
        bool passed = string(e.what()).find(expectedMsg) != string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// Tests for span() Function
// /////////////////////////////////////////////////////////////////////////////

/*
 * Test: span of a full-rank matrix correctly returns the basis vectors.
 * Inputs:
 *   Matrix A = [[1, 0],
 *             [0, 1]]
 * Expected Output:
 *   Basis Vectors = [[1, 0],
 *                   [0, 1]]
 */
void testSpan_FullRank_ReturnsCorrectBasis() 
{
    std::string testName = "Span_FullRank_ReturnsCorrectBasis";
    try {
        Matrix<double> A = createMatrix({ {1.0, 0.0}, {0.0, 1.0} });
        Matrix<double> basis = A.span();
        Matrix<double> expected = createMatrix({ {1.0, 0.0}, {0.0, 1.0} });
        bool passed = matricesEqual(basis, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: span of a rank-deficient matrix correctly returns the reduced basis vectors.
 * Inputs:
 *   Matrix A = [[1, 2],
 *             [2, 4]]
 * Expected Output:
 *   Basis Vectors = [[1, 2]]
 */
void testSpan_RankDeficient_ReturnsReducedBasis() 
{
    std::string testName = "Span_RankDeficient_ReturnsReducedBasis";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0}, {2.0, 4.0} });
        Matrix<double> basis = A.span();
        Matrix<double> expected = createMatrix({ {1.0, 2.0} });
        bool passed = matricesEqual(basis, expected);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// Tests for pseudoInverse() Function
// /////////////////////////////////////////////////////////////////////////////

/*
 * Test: pseudoInverse of an invertible matrix equals its inverse.
 * Inputs:
 *   Matrix A = [[4, 7],
 *             [2, 6]]
 * Expected Output:
 *   Pseudo-Inverse = [[0.6, -0.7],
 *                    [-0.2, 0.4]]
 */
void testPseudoInverse_InvertibleMatrix_EqualsInverse() 
{
    std::string testName = "PseudoInverse_InvertibleMatrix_EqualsInverse";
    try {
        Matrix<double> A = createMatrix({ {4.0, 7.0}, {2.0, 6.0} });
        Matrix<double> pinv = A.pseudoInverse();
        Matrix<double> expected = createMatrix({ {0.6, -0.7}, {-0.2, 0.4} });
        bool passed = matricesEqual(pinv, expected, 1e-5);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: pseudoInverse of an orthogonal matrix equals its transpose.
 * Inputs:
 *   Matrix A = [[0, 1],
 *             [1, 0]]
 * Expected Output:
 *   Pseudo-Inverse = [[0, 1],
 *                    [1, 0]]
 */
void testPseudoInverse_OrthogonalMatrix_EqualsTranspose() 
{
    std::string testName = "PseudoInverse_OrthogonalMatrix_EqualsTranspose";
    try {
        Matrix<double> A = createMatrix({ {0.0, 1.0}, {1.0, 0.0} });
        Matrix<double> pinv = A.pseudoInverse();
        Matrix<double> expected = A.transpose();
        bool passed = matricesEqual(pinv, expected, 1e-5);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: pseudoInverse of a non-invertible, non-orthogonal matrix computes correctly.
 * Inputs:
 *   Matrix A = [[1, 2],
 *             [3, 4],
 *             [5, 6]]
 * Expected Output:
 *   Pseudo-Inverse is computed as A^T(AA^T)^-1
 */
void testPseudoInverse_NonInvertibleNonOrthogonalMatrix_ComputesCorrectly() 
{
    std::string testName = "PseudoInverse_NonInvertibleNonOrthogonalMatrix_ComputesCorrectly";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0} });
        Matrix<double> At = A.transpose();
        Matrix<double> AtA = A * At;
        Matrix<double> AtA_inv = AtA.inverse();
        Matrix<double> expected = At * AtA_inv;
        Matrix<double> pinv = A.pseudoInverse();
        bool passed = matricesEqual(pinv, expected, 1e-5);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: pseudoInverse throws an exception for a matrix that cannot be pseudoinverted.
 * Inputs:
 *   Matrix A = [[1, 2, 3],
 *             [4, 5, 6]]
 * Expected Output:
 *   Runtime error with message "Cannot compute pseudoinverse of given matrix!"
 */
void testPseudoInverse_CannotCompute_ThrowsException() 
{
    std::string testName = "PseudoInverse_CannotCompute_ThrowsException";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} });
        Matrix<double> pinv = A.pseudoInverse();
        // If no exception is thrown, the test fails
        reportTestResult(testName, false);
    }
    catch (const std::runtime_error& e) {
        std::string expectedMsg = "Cannot compute pseudoinverse of given matrix!";
        bool passed = std::string(e.what()).find(expectedMsg) != std::string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// Tests for isSemiOrthogonal() Function
// /////////////////////////////////////////////////////////////////////////////

/*
 * Test: isSemiOrthogonal returns true for a semi-orthogonal matrix (A^T*A = I).
 * Inputs:
 *   Matrix A = [[1, 0],
 *             [0, 1],
 *             [0, 0]]
 * Expected Output:
 *   true
 */
void testIsSemiOrthogonal_SemiOrthogonalMatrix_ReturnsTrue() 
{
    std::string testName = "IsSemiOrthogonal_SemiOrthogonalMatrix_ReturnsTrue";
    try {
        Matrix<double> A = createMatrix({ {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0} });
        bool result = A.isSemiOrthogonal();
        bool expected = true;
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: isSemiOrthogonal returns true for a semi-orthogonal matrix (A*A^T = I).
 * Inputs:
 *   Matrix A = [[1, 0, 0],
 *             [0, 1, 0]]
 * Expected Output:
 *   true
 */
void testIsSemiOrthogonal_SemiOrthogonalMatrix_AAtEqualsI_ReturnsTrue() 
{
    std::string testName = "IsSemiOrthogonal_SemiOrthogonalMatrix_AAtEqualsI_ReturnsTrue";
    try {
        Matrix<double> A = createMatrix({ {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0} });
        bool result = A.isSemiOrthogonal();
        bool expected = true;
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: isSemiOrthogonal returns false for a non-semi-orthogonal matrix.
 * Inputs:
 *   Matrix A = [[1, 2],
 *             [3, 4]]
 * Expected Output:
 *   false
 */
void testIsSemiOrthogonal_NonSemiOrthogonalMatrix_ReturnsFalse() 
{
    std::string testName = "IsSemiOrthogonal_NonSemiOrthogonalMatrix_ReturnsFalse";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0}, {3.0, 4.0} });
        bool result = A.isSemiOrthogonal();
        bool expected = false;
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// Tests for qrDecomposition() Function
// /////////////////////////////////////////////////////////////////////////////

/*
 * Test: qrDecomposition correctly decomposes a full-rank square matrix.
 * Inputs:
 *   Matrix A = [[12, -51, 4],
 *             [6, 167, -68],
 *             [-4, 24, -41]]
 * Expected Output:
 *   Q and R matrices such that A = Q * R
 */
void testQRDecomposition_FullRankSquareMatrix_DecomposesCorrectly() 
{
    std::string testName = "QRDecomposition_FullRankSquareMatrix_DecomposesCorrectly";
    try {
        Matrix<double> A = createMatrix({ {12.0, -51.0, 4.0}, {6.0, 167.0, -68.0}, {-4.0, 24.0, -41.0} });
        Matrix<double> Q, R;
        A.qrDecomposition(Q, R);
        Matrix<double> reconstructed = Q * R;
        bool passed = matricesEqual(A, reconstructed, 1e-4);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: qrDecomposition throws an exception for a non-full-rank matrix.
 * Inputs:
 *   Matrix A = [[1, 2],
 *             [2, 4]]
 * Expected Output:
 *   Runtime error if QR decomposition cannot be performed
 */
void testQRDecomposition_NonFullRankMatrix_ThrowsException() 
{
    std::string testName = "QRDecomposition_NonFullRankMatrix_ThrowsException";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0}, {2.0, 4.0} });
        Matrix<double> Q, R;
        A.qrDecomposition(Q, R);
        // If no exception is thrown, the test fails
        reportTestResult(testName, false);
    }
    catch (const std::runtime_error& e) {
        std::string expectedMsg = "QR decomposition failed due to non-full rank.";
        bool passed = std::string(e.what()).find(expectedMsg) != std::string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// Tests for eigenvaluesVector() Function
// /////////////////////////////////////////////////////////////////////////////

/*
 * Test: eigenvaluesVector correctly computes eigenvalues of a diagonal matrix.
 * Inputs:
 *   Matrix A = [[5, 0],
 *             [0, 3]]
 * Expected Output:
 *   Eigenvalues = [5, 3]
 */
void testEigenvaluesVector_DiagonalMatrix_ComputesCorrectly() 
{
    std::string testName = "EigenvaluesVector_DiagonalMatrix_ComputesCorrectly";
    try {
        Matrix<double> A = createMatrix({ {5.0, 0.0}, {0.0, 3.0} });
        std::vector<double> eigVals = A.eigenvaluesVector();
        std::vector<double> expected = { 5.0, 3.0 };
        bool passed = true;
        for (size_t i = 0; i < eigVals.size(); ++i) {
            if (std::fabs(eigVals[i] - expected[i]) > 1e-5) {
                passed = false;
                break;
            }
        }
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: eigenvaluesVector throws an exception for a non-square matrix.
 * Inputs:
 *   Matrix A = [[1, 2, 3],
 *             [4, 5, 6]]
 * Expected Output:
 *   Runtime error with message "Matrix must be square to compute eigenvalues."
 */
void testEigenvaluesVector_NonSquareMatrix_ThrowsException() 
{
    std::string testName = "EigenvaluesVector_NonSquareMatrix_ThrowsException";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} });
        std::vector<double> eigVals = A.eigenvaluesVector();
        // If no exception is thrown, the test fails
        reportTestResult(testName, false);
    }
    catch (const std::runtime_error& e) {
        std::string expectedMsg = "Matrix must be square to compute eigenvalues.";
        bool passed = std::string(e.what()).find(expectedMsg) != std::string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: eigenvaluesVector computes eigenvalues for a symmetric matrix.
 * Inputs:
 *   Matrix A = [[2, 1],
 *             [1, 2]]
 * Expected Output:
 *   Eigenvalues = [3, 1]
 */
void testEigenvaluesVector_SymmetricMatrix_ComputesCorrectly() 
{
    std::string testName = "EigenvaluesVector_SymmetricMatrix_ComputesCorrectly";
    try {
        Matrix<double> A = createMatrix({ {2.0, 1.0}, {1.0, 2.0} });
        std::vector<double> eigVals = A.eigenvaluesVector();
        std::vector<double> expected = { 3.0, 1.0 };
        bool passed = true;
        for (size_t i = 0; i < eigVals.size(); ++i) {
            if (std::fabs(eigVals[i] - expected[i]) > 1e-5) {
                passed = false;
                break;
            }
        }
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// Tests for eigenvectors() Function
// /////////////////////////////////////////////////////////////////////////////

/*
 * Test: eigenvectors correctly computes eigenvectors of a diagonal matrix.
 * Inputs:
 *   Matrix A = [[5, 0],
 *             [0, 3]]
 * Expected Output:
 *   Eigenvectors = [[1, 0],
 *                  [0, 1]]
 */
void testEigenvectors_DiagonalMatrix_ComputesCorrectly() 
{

    std::string testName = "Eigenvectors_DiagonalMatrix_ComputesCorrectly";
    try {
        Matrix<double> A = createMatrix({ {5.0, 0.0}, {0.0, 3.0} });
        Matrix<double> eigVec = A.eigenvectors();
        Matrix<double> expected = createMatrix({ {1.0, 0.0}, {0.0, 1.0} });
        bool passed = matricesEqual(eigVec, expected, 1e-5);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: eigenvectors throws an exception for a non-square matrix.
 * Inputs:
 *   Matrix A = [[1, 2, 3],
 *             [4, 5, 6]]
 * Expected Output:
 *   Runtime error with message "Matrix must be square to compute eigenvectors."
 */
void testEigenvectors_NonSquareMatrix_ThrowsException() 
{
    std::string testName = "Eigenvectors_NonSquareMatrix_ThrowsException";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} });
        Matrix<double> eigVec = A.eigenvectors();
        // If no exception is thrown, the test fails
        reportTestResult(testName, false);
    }
    catch (const std::runtime_error& e) {
        std::string expectedMsg = "Matrix must be square to compute eigenvectors.";
        bool passed = std::string(e.what()).find(expectedMsg) != std::string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: eigenvectors computes eigenvectors for a symmetric matrix.
 * Inputs:
 *   Matrix A = [[2, 1],
 *             [1, 2]]
 * Expected Output:
 *   Eigenvectors = [[1/sqrt(2), -1/sqrt(2)],
 *                  [1/sqrt(2), 1/sqrt(2)]]
 */
void testEigenvectors_SymmetricMatrix_ComputesCorrectly() 
{
    std::string testName = "Eigenvectors_SymmetricMatrix_ComputesCorrectly";
    try {
        Matrix<double> A = createMatrix({ {2.0, 1.0}, {1.0, 2.0} });
        Matrix<double> eigVec = A.eigenvectors();
        Matrix<double> expected = createMatrix({
            {1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0)},
            {1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)}
            });
        bool passed = matricesEqual(eigVec, expected, 1e-5);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// Tests for choleskyDecomposition() Function
// /////////////////////////////////////////////////////////////////////////////

/*
 * Test: choleskyDecomposition correctly decomposes a positive definite matrix.
 * Inputs:
 *   Matrix A = [[4, 12, -16],
 *             [12, 37, -43],
 *             [-16, -43, 98]]
 * Expected Output:
 *   Lower Triangular Matrix L = [[2, 0, 0],
 *                               [6, 1, 0],
 *                               [-8, 5, 3]]
 */
void testCholeskyDecomposition_PositiveDefiniteMatrix_DecomposesCorrectly() 
{
    std::string testName = "CholeskyDecomposition_PositiveDefiniteMatrix_DecomposesCorrectly";
    try {
        Matrix<double> A = createMatrix({
            {4.0, 12.0, -16.0},
            {12.0, 37.0, -43.0},
            {-16.0, -43.0, 98.0}
            });
        Matrix<double> L = A.choleskyDecomposition();
        Matrix<double> expected = createMatrix({
            {2.0, 0.0, 0.0},
            {6.0, 1.0, 0.0},
            {-8.0, 5.0, 3.0}
            });
        bool passed = matricesEqual(L, expected, 1e-5);
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: choleskyDecomposition throws an exception for a non-positive definite matrix.
 * Inputs:
 *   Matrix A = [[1, 2],
 *             [2, 1]]
 * Expected Output:
 *   Runtime error with message "Cannot perform Cholesky decomposition for non positive semi definite matrix!"
 */
void testCholeskyDecomposition_NonPositiveDefiniteMatrix_ThrowsException() 
{
    std::string testName = "CholeskyDecomposition_NonPositiveDefiniteMatrix_ThrowsException";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0}, {2.0, 1.0} });
        Matrix<double> L = A.choleskyDecomposition();
        // If no exception is thrown, the test fails
        reportTestResult(testName, false);
    }
    catch (const std::runtime_error& e) {
        std::string expectedMsg = "Cannot perform Cholesky decomposition for non positive semi definite matrix!";
        bool passed = std::string(e.what()).find(expectedMsg) != std::string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: choleskyDecomposition throws an exception for a non-square matrix.
 * Inputs:
 *   Matrix A = [[1, 2, 3],
 *             [4, 5, 6]]
 * Expected Output:
 *   Runtime error with message "Matrix must be square."
 */
void testCholeskyDecomposition_NonSquareMatrix_ThrowsException() 
{
    std::string testName = "CholeskyDecomposition_NonSquareMatrix_ThrowsException";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} });
        Matrix<double> L = A.choleskyDecomposition();
        // If no exception is thrown, the test fails
        reportTestResult(testName, false);
    }
    catch (const std::runtime_error& e) {
        std::string expectedMsg = "Matrix must be square.";
        bool passed = std::string(e.what()).find(expectedMsg) != std::string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// Additional Tests for Other Functions
// /////////////////////////////////////////////////////////////////////////////

/*
 * Test: frobeniusNorm correctly computes the Frobenius norm of a matrix.
 * Inputs:
 *   Matrix A = [[1, 2],
 *             [3, 4]]
 * Expected Output:
 *   Frobenius Norm = sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30) ≈ 5.47723
 */
void testFrobeniusNorm_ComputesCorrectly() 
{
    std::string testName = "FrobeniusNorm_ComputesCorrectly";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0}, {3.0, 4.0} });
        double norm = A.frobeniusNorm();
        double expected = std::sqrt(30.0);
        bool passed = std::fabs(norm - expected) < 1e-5;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: frobeniusNorm throws an exception for an empty matrix.
 * Inputs:
 *   Matrix A = []
 * Expected Output:
 *   Runtime error with message "Cannot calculate norm for an empty matrix."
 */
void testFrobeniusNorm_EmptyMatrix_ThrowsException() 
{
    std::string testName = "FrobeniusNorm_EmptyMatrix_ThrowsException";
    try {
        Matrix<double> A(0, 0); // Empty matrix
        double norm = A.frobeniusNorm();
        // If no exception is thrown, the test fails
        reportTestResult(testName, false);
    }
    catch (const std::runtime_error& e) {
        std::string expectedMsg = "Cannot calculate norm for an empty matrix!";
        bool passed = std::string(e.what()).find(expectedMsg) != std::string::npos;
        reportTestResult(testName, passed);
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: isIdentity correctly identifies an identity matrix.
 * Inputs:
 *   Matrix A = [[1, 0],
 *             [0, 1]]
 * Expected Output:
 *   true
 */
void testIsIdentity_IdentityMatrix_ReturnsTrue() 
{
    std::string testName = "IsIdentity_IdentityMatrix_ReturnsTrue";
    try {
        Matrix<double> A = createMatrix({ {1.0, 0.0}, {0.0, 1.0} });
        bool result = A.isIdentity();
        bool expected = true;
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: isIdentity correctly identifies a non-identity matrix.
 * Inputs:
 *   Matrix A = [[1, 2],
 *             [0, 1]]
 * Expected Output:
 *   false
 */
void testIsIdentity_NonIdentityMatrix_ReturnsFalse() 
{
    std::string testName = "IsIdentity_NonIdentityMatrix_ReturnsFalse";
    try {
        Matrix<double> A = createMatrix({ {1.0, 2.0}, {0.0, 1.0} });
        bool result = A.isIdentity();
        bool expected = false;
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

/*
 * Test: isIdentity throws false for a non-square matrix.
 * Inputs:
 *   Matrix A = [[1, 0, 0],
 *             [0, 1, 0]]
 * Expected Output:
 *   false
 */
void testIsIdentity_NonSquareMatrix_ReturnsFalse() 
{
    std::string testName = "IsIdentity_NonSquareMatrix_ReturnsFalse";
    try {
        Matrix<double> A = createMatrix({ {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0} });
        bool result = A.isIdentity();
        bool expected = false;
        reportTestResult(testName, (result == expected));
    }
    catch (...) {
        reportTestResult(testName, false);
    }
}

// /////////////////////////////////////////////////////////////////////////////
// Function to run the unit tests
// /////////////////////////////////////////////////////////////////////////////

int unitTests() {
    cout << "Starting Matrix Class Unit Tests...\n\n";

    // operator+ tests
    testOperatorPlus_SameSize_AddsCorrectly();
    testOperatorPlus_DifferentSize_ThrowsException();

    // operator- tests
    testOperatorMinus_SameSize_SubtractsCorrectly();
    testOperatorMinus_DifferentSize_ThrowsException();

    // operator* (Matrix szorzás) tests
    testOperatorMultiply_CompatibleSizes_MultipliesCorrectly();
    testOperatorMultiply_IncompatibleSizes_ThrowsException();
    
    // operator* (skálárszorzás) tests
    testOperatorMultiplyScalar_MultipliesCorrectly();

    // operator/= tests
    testOperatorDivideScalar_DividesCorrectly();
    testOperatorDivideScalar_DivideByZero_ThrowsException();

    // swapRows tests
    testSwapRows_ValidIndices_SwapsCorrectly();
    testSwapRows_InvalidIndices_ThrowsException();

    // swapCols tests
    testSwapCols_ValidIndices_SwapsCorrectly();
    testSwapCols_InvalidIndices_ThrowsException();

    // normalizeRowForPosition tests
    testNormalizeRowForPosition_ValidIndices_NormalizesCorrectly();
    testNormalizeRowForPosition_InvalidIndices_ThrowsException();
    testNormalizeRowForPosition_DivideByZero_ThrowsException();

    // scalarMultiplyRow tests
    testScalarMultiplyRow_ValidRow_MultipliesCorrectly();
    testScalarMultiplyRow_InvalidRow_ThrowsException();

    // scalarMultiplyCol tests
    testScalarMultiplyCol_ValidCol_MultipliesCorrectly();
    testScalarMultiplyCol_InvalidCol_ThrowsException();

    // addRow (row index) tests
    testAddRow_RowIndices_AddsCorrectly();
    testAddRow_RowIndices_InvalidRowIndices_ThrowsException();

    // addRow (vektor based) tests
    testAddRow_Vector_AddsCorrectly();
    testAddRow_Vector_SizeMismatch_ThrowsException();

    // dotProduct (row index based) tests
    testDotProduct_Rows_CalculatesCorrectly();
    testDotProduct_Rows_InvalidIndices_ThrowsException();
    testDotProduct_SameRow_ReturnsZero();

    // dotProduct (vektor és sor) tests
    testDotProduct_Vector_Row_CalculatesCorrectly();
    testDotProduct_Vector_Row_SizeMismatch_ThrowsException();
    testDotProduct_Vector_Row_InvalidRowIndex_ThrowsException();

    // rowAbs tests
    testRowAbs_ValidRow_CalculatesCorrectly();
    testRowAbs_InvalidRowIndex_ThrowsException();

    // transpose tests
    testTranspose_ReturnsCorrectTranspose();
    testTranspose_SymmetricMatrix_ReturnsSameMatrix();

    // trace tests
    testTrace_SquareMatrix_CalculatesCorrectly();
    testTrace_NonSquareMatrix_ThrowsException();

    // setToIdentity és identity tests
    testSetToIdentity_SquareMatrix_SetsCorrectly();
    testSetToIdentity_NonSquareMatrix_ThrowsException();
    testIdentity_CreatesCorrectIdentityMatrix();

    // isSymmetric tests
    testIsSymmetric_SymmetricMatrix_ReturnsTrue();
    testIsSymmetric_NonSymmetricMatrix_ReturnsFalse();
    testIsSymmetric_NonSquareMatrix_ReturnsFalse();

    // isThisOrthogonal és isThisOrthonormed tests
    testIsThisOrthogonal_OrthogonalMatrix_ReturnsTrue();
    testIsThisOrthogonal_NonOrthogonalMatrix_ReturnsFalse();
    testIsThisOrthonormed_OrthonormalMatrix_ReturnsTrue();
    testIsThisOrthonormed_OrthogonalNotOrthonormal_ReturnsFalse();
    testIsThisOrthonormed_NonOrthogonalMatrix_ReturnsFalse();

    // determinant tests
    testDeterminant_SquareMatrix_CalculatesCorrectly();
    testDeterminant_NonSquareMatrix_ThrowsException();

    // inverse tests
    testInverse_InvertibleMatrix_CalculatesCorrectly();
    testInverse_SingularMatrix_ThrowsException();
    testInverse_NonSquareMatrix_ThrowsException();

    // Tests for span()
    testSpan_FullRank_ReturnsCorrectBasis();
    testSpan_RankDeficient_ReturnsReducedBasis();

    // Tests for pseudoInverse()
    testPseudoInverse_InvertibleMatrix_EqualsInverse();
    testPseudoInverse_OrthogonalMatrix_EqualsTranspose();
    testPseudoInverse_NonInvertibleNonOrthogonalMatrix_ComputesCorrectly();
    testPseudoInverse_CannotCompute_ThrowsException();

    // Tests for isSemiOrthogonal()
    //testIsSemiOrthogonal_SemiOrthogonalMatrix_ReturnsTrue();
    testIsSemiOrthogonal_SemiOrthogonalMatrix_AAtEqualsI_ReturnsTrue();
    testIsSemiOrthogonal_NonSemiOrthogonalMatrix_ReturnsFalse();

    // Tests for qrDecomposition()
    testQRDecomposition_FullRankSquareMatrix_DecomposesCorrectly();
    testQRDecomposition_NonFullRankMatrix_ThrowsException();

    // Tests for eigenvaluesVector()
    testEigenvaluesVector_DiagonalMatrix_ComputesCorrectly();
    testEigenvaluesVector_NonSquareMatrix_ThrowsException();
    testEigenvaluesVector_SymmetricMatrix_ComputesCorrectly();

    // Tests for eigenvectors()
    testEigenvectors_DiagonalMatrix_ComputesCorrectly();
    testEigenvectors_NonSquareMatrix_ThrowsException();
    testEigenvectors_SymmetricMatrix_ComputesCorrectly();

    // Tests for choleskyDecomposition()
    testCholeskyDecomposition_PositiveDefiniteMatrix_DecomposesCorrectly();
    testCholeskyDecomposition_NonPositiveDefiniteMatrix_ThrowsException();
    testCholeskyDecomposition_NonSquareMatrix_ThrowsException();

    // Additional Tests
    testFrobeniusNorm_ComputesCorrectly();
    testFrobeniusNorm_EmptyMatrix_ThrowsException();
    testIsIdentity_IdentityMatrix_ReturnsTrue();
    testIsIdentity_NonIdentityMatrix_ReturnsFalse();
    testIsIdentity_NonSquareMatrix_ReturnsFalse();

    // Summary
    cout << "\nUnit Tests Completed.\n";
    cout << "Passed: " << testsPassed << "\n";
    cout << "Failed: " << testsFailed << "\n";

    if (testsFailed == 0) {
        cout << "All tests passed successfully!\n";
        return 0;
    }
    else {
        cout << "Some tests failed. Please review the failures.\n";
        return 1;
    }
}
