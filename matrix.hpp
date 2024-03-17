#pragma once

#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <vector>
#include <iostream>
#include <exception>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <iterator>
#include <map>

using std::vector;
using MatrixDouble = std::vector<vector<double>>;
using std::tuple;
using std::pair;

/***********************
*	|-----col1-------|
*   r1				 rn
*   |-----coln-------|
*/

template<typename numericalType>
class Matrix
{
private:
	numericalType** matrix;	// Represents the matrix
	size_t _row;			// Number of rows in the matrix
	size_t _col;			// Number of columns in the matrix

	bool _symmetric;		// Wether the matrix is symmetric or not, helps speed up some functions
public:
	// Constructors
	// Default constructor: Initializes an empty matrix.
	// Parameters: None
	// Returns: An instance of a Matrix object with no elements.
	Matrix() : matrix(nullptr), _row(0), _col(0), _symmetric(false) {}
	
	// Parameterized constructor: Initializes a matrix of size r x c, filled with default values of numericalType.
	// Parameters:
	// - r: Number of rows in the matrix.
	// - c: Number of columns in the matrix.
	// Returns: An instance of a Matrix object with specified dimensions, filled with default values.
	Matrix(const size_t& r, const size_t& c) : _row(r), _col(c), _symmetric(false)
	{
		matrix = new numericalType*[_row];
		for (unsigned i = 0; i < _row; i++)
			matrix[i] = new numericalType[_col];

		for(unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)		// Initialize to zero or some default value
			matrix[i][j] = numericalType{};		// Make sure numericalType can be value-initialized
	}
	
	// Initializer list constructor: Initializes a matrix with the specified values.
	// Parameters:
	// - init: An initializer list of initializer lists containing elements of type numericalType.
	// Returns: An instance of a Matrix object initialized with the specified values.
	Matrix(std::initializer_list<std::initializer_list<numericalType>> init) 
	{
		_row = init.size();
		_col = (init.begin())->size();

		matrix = new numericalType*[_row];
		for (size_t i = 0; i < _row; i++)
			matrix[i] = new numericalType[_col];

		size_t i = 0;
		for (auto& rowList : init) 
		{
			if (rowList.size() != _col)
				throw std::invalid_argument("All rows must have the same number of columns");
			size_t j = 0;
			for (auto& elem : rowList) 
				matrix[i][j++] = elem;
			++i;
		}
		if (isSymmetric())_symmetric = true;
	}
	
	// Copy constructor: Creates a new matrix as a deep copy of an existing matrix.
	// Parameters:
	// - other: A reference to the Matrix object to be copied.
	// Returns: An instance of a Matrix object that is a deep copy of the specified matrix.
	Matrix(const Matrix& other) : _row(other._row), _col(other._col), _symmetric(other._symmetric)
	{
		// Allocating memory
		matrix = new numericalType * [_row];
		for (size_t i = 0; i < _row; ++i) 
		{
			for (unsigned i = 0; i < _row; i++)
				matrix[i] = new numericalType[_col];
		}

		// Copiing data
		for (unsigned i = 0; i < _row; i++)
			for (unsigned j = 0; j < _col; j++)
				matrix[i][j] = other[i][j];
	}
	
	// Destructor: Cleans up the resources used by the matrix, deallocating any dynamic memory.
	// Parameters: None
	// Returns: None
	~Matrix()
	{
		for (unsigned i = 0; i < _row; i++)
			delete[] matrix[i];
		delete[] matrix;
	}

	// getMatrix: Retrieves the raw pointer to the matrix's underlying array.
	// Parameters: None
	// Returns: A double pointer to numericalType, representing the raw 2D array of the matrix's elements.
	numericalType** getMatrix() const { return matrix; }

	// setMatrix: Sets the matrix's underlying array to the specified raw pointer. (Use with caution; can lead to memory leaks if not handled properly)
	// Parameters:
	// - pMatrix: A double pointer to numericalType, representing a new 2D array to be used as the matrix's elements.
	// Returns: None
	void setMatrix(numericalType** pMatrix) { matrix = pMatrix; }
	
	// getRow: Retrieves a pointer to the specified row.
	// Parameters:
	// - rowNum: The index of the row to retrieve.
	// Returns: A pointer to numericalType, representing the specified row's elements.
	numericalType* getRow(const size_t& rowNum) 
	{
		if (rowNum > _row)
			return nullptr; 
		else 
			return matrix[rowNum];
	}
	
	// getRowVector: Constructs and returns a vector representing the specified row.
	// Parameters:
	// - rowNum: The index of the row to convert to a vector.
	// Returns: A vector of numericalType, containing the elements of the specified row.
	vector<numericalType> getRowVector(const size_t& rowNum)
	{
		if(rowNum > _row)
			throw std::runtime_error("Cannot return row, since it is out of bounds!");

		vector<numericalType> retVec;
		for (unsigned i = 0; i < _col; i++)
			retVec.push_back(matrix[rowNum][i]);
		return retVec;
	}

	// getCol: Retrieves a pointer to the specified column. (Note: This function's behavior might not align with expectations due to row-major storage)
	// Parameters:
	// - colNum: The index of the column to retrieve.
	// Returns: A pointer to numericalType, representing the start of the specified column.
	numericalType* getCol(const size_t& colNum) 
	{
		if (colNum > _col)
			throw std::runtime_error("Cannot return col, since it is out of bounds!");

		return &matrix[0][colNum];
	}

	// getColVector: Constructs and returns a vector representing the specified column.
	// Parameters:
	// - colNum: The index of the column to convert to a vector.
	// Returns: A vector of numericalType, containing the elements of the specified column.
	vector<numericalType> getColVector(const size_t& colNum) const 
	{
		if (colNum >= _col)
			throw std::runtime_error("Cannot return col, since it is you of bounds!");

		vector<numericalType> retVec;
		for (unsigned i = 0; i < _row; i++)
			retVec.push_back(matrix[i][colNum]);
		return retVec;
	}

	// setRow: Sets the specified row to the given values using a raw pointer array. Returns true if successful.
	// Parameters:
	// - newRow: A pointer to an array of numericalType, representing the new values for the row.
	// - rowNum: The index of the row to be set.
	// - rowSize: The size of newRow array; should match the number of columns in the matrix.
	// Returns: A boolean indicating success (true) or failure (false) of the operation.
	bool setRow(numericalType* newRow, const size_t& rowNum, const size_t& rowSize) 
	{
		if (rowSize != _col)
		{
			throw std::runtime_error("Cannot change row if not the same size");
			return false;
		}
		else if (rowNum >= _row) throw std::runtime_error("Cannot set row, index out of bounds!");
		else 
		{ 
			// Overwriting matrix's row
			for (unsigned i = 0; i < _col; i++)
				matrix[rowNum][i] = newRow[i];
			return true; 
		}
	}

	// setRow: Sets the specified row to the values in the given vector. Returns true if successful.
	// Parameters:
	// - newRow: A vector of numericalType, representing the new values for the row.
	// - rowNum: The index of the row to be set.
	// Returns: A boolean indicating success (true) or failure (false) of the operation.
	bool setRow(const vector<numericalType>& newRow, const size_t& rowNum)
	{ 
		if (newRow.size() != _row) 
		{
			throw std::runtime_error("Cannot change row if not the same size"); 
			return false;
		}
		else
		{
			// Copiing vector elements
			numericalType* nRow = new numericalType[_row];
			for (unsigned i = 0; i < _row; i++)
				nRow[i] = newRow[i];

			return setRow(nRow, rowNum, _row);
		}
	}

	// setCol: Sets a specific column to the values specified in a vector.
	// Parameters:
	// - colIdx: Index of the column to be modified.
	// - vec: Vector containing the new values for the column.
	// Returns: None.
	void setCol(const size_t& colIdx, const vector<numericalType> vec)
	{
		if (colIdx >= _col) throw std::runtime_error("Column index out of bounds, cannot set columns!");

		for (unsigned i = 0; i < _row; i++)
			matrix[i][colIdx] = vec[i];
	}

	// setCol: Sets a specific column to the values specified in an array.
	// Parameters:
	// - vec: Array containing the new values for the column.
	// - size: Number of elements in the vec array.
	// - colIdx: Index of the column to be modified.
	// Returns: None.
	void setCol(numericalType* vec, const size_t& size, const size_t& colIdx)
	{
		if (size >= _row) throw std::runtime_error("Vector too long, cannot insert new column!");

		for (unsigned i = 0; i < _row; i++)
			matrix[i][colIdx] = vec[i];
	}

	// areOrthogonal: Checks if two rows are orthogonal to each other.
	// Parameters:
	// - row1: Index of the first row.
	// - row2: Index of the second row.
	// Returns: True if the rows are orthogonal, false otherwise.
	bool areOrthogonal(const size_t& row1, const size_t& row2) const { return (0 == dotProduct(row1, row2)); }

	// isThisOrthogonal (array version): Checks if the matrix is orthogonal.
	// Parameters: 
	// - row: An array of the first row, that should be compared.
	// - size: Size of this row.
	// - rowIdx: The index of the row in the matrix that should be comapred.
	// Returns: True if the two rows are orthogonal false otherwise.
	bool areOrthogonal(numericalType* row, const size_t& size, const size_t& rowIdx) const { return (0 == dotProduct(row, size, rowIdx)); }

	// areOrthogonal (vector version): Checks if the given row vector is orthogonal to a specific row in the matrix.
	// Parameters:
	// - row: A vector of numericalType representing the row vector for the orthogonality check.
	// - rowIdx: The index of the row within the matrix to check for orthogonality against the provided vector.
	// Returns: True if the given vector is orthogonal to the specified matrix row, false otherwise.
	bool areOrthogonal(const vector<numericalType>& row, const size_t& rowIdx) const { return (0 == dotProduct(row, rowIdx)); }

	// isThisOrthogonal: Checks if the matrix is orthogonal.
	// Parameters: None.
	// Returns: True if the matrix is orthogonal, false otherwise.
	bool isThisOrthogonal() const;

	// isThisOrthonormed: Checks if the matrix is orthonormal, i.e., its rows are orthogonal and of unit length.
	// Parameters: None.
	// Returns: True if the matrix is orthonormal, false otherwise.
	bool isThisOrthonormed() const;

	// isAlike: Checks if two matrices are similar in terms of rank and null spaces.( A ~ B)
	// Parameters:
	// - other: The matrix to compare with.
	// Returns: True if the matrices are alike, false otherwise.
	bool isAlike(const Matrix<numericalType>& other) const;

	// isSymmetric: Checks if the matrix is symmetric.
	// Parameters: None.
	// Returns: True if the matrix is symmetric, false otherwise.
	bool isSymmetric() const;

	// isNegativeSymmetric: Checks if the matrix is negatively symmetric.
	// Parameters: None.
	// Returns: True if the matrix is negatively symmetric, false otherwise.
	bool isNegativeSymmetric() const;

	// isDiagonal: Checks if the matrix is diagonal.
	// Parameters: None.
	// Returns: True if the matrix is diagonal, false otherwise.
	bool isDiagonal() const;

	// euclidian: Checks if the matrix can stretch an Euclidean space.
	// Parameters: None.
	// Returns: True if the matrix can stretch an Euclidean space, false otherwise.
	bool euclidian() const;

	// isFullColumnRank: Checks if the matrix has full column rank.
	// Parameters: None.
	// Returns: True if the matrix has full column rank, false otherwise.
	bool isFullColumnRank() const { return (rank() == _col); }

	// isIdempotent: Checks if the matrix is idempotent.
	// Parameters: None.
	// Returns: True if the matrix is idempotent, false otherwise.
	bool isIdempotent() const;

	// isOrthogonalProjectionMatrix: Checks if the matrix is an orthogonal projection matrix.
	// Parameters: None.
	// Returns: True if the matrix is an orthogonal projection matrix, false otherwise.
	bool isOrthogonalProjectionMatrix() const;

	// isSemiOrthogonal: Checks if the matrix is semi-orthogonal.
	// Parameters: None.
	// Returns: True if the matrix is semi-orthogonal, false otherwise.
	bool isSemiOrthogonal() const;

	// isPositiveDefinite: Checks if the matrix is positive definite.
	// Parameters: None.
	// Returns: True if the matrix is positive definite, false otherwise.
	bool isPositiveDefinite() const;

	// isPositiveSemiDefinite: Checks if the matrix is positive semi-definite.
	// Parameters: None.
	// Returns: True if the matrix is positive semi-definite, false otherwise.
	bool isPositiveSemiDefinite() const;

	// isUpperTriangle: Checks if the matrix is an upper triangle matrix.
	// Parameters: None.
	// Returns: True if the matrix is an upper triangle matrix, false otherwise.
	bool isUpperTriangle() const;

	// isLowerTriangle: Checks if the matrix is a lower triangle matrix.
	// Parameters: None.
	// Returns: True if the matrix is a lower triangle matrix, false otherwise.
	bool isLowerTriangle() const;

	// isNilPotent: Checks if the matrix is nilpotent, meaning all eigenvalues are zeroes.
	// Parameters: None.
	// Returns: True if the matrix is nilpotent, false otherwise.
	bool isNilPotent() const;
	
	// canBeDiagonalized: Checks if the matrix can be diagonalized by checking if it has at least n linearly independent eigenvector.
	// Parameters: None.
	// Returns: True if the matrix can be diagonalized, false otherwise.
	bool canBeDiagonalized() const;

	// isColumnVector (vector version): Checks if a given vector is in the column space of the matrix.
	// Parameters:
	// - vec: A vector of numericalType representing the column vector to check.
	// Returns: True if vec is in the column space of the matrix, false otherwise.
	bool isColumnVector(const vector<numericalType>& vec) const;

	// isColumnVector (array version): Checks if a given array represents a column vector in the column space of the matrix.
	// Parameters:
	// - vec: A pointer to an array of numericalType representing the column vector to check.
	// - size: The size of the array pointed to by vec.
	// Returns: True if vec is in the column space of the matrix, false otherwise.
	bool isColumnVector(numericalType* vec, const size_t& size) const;

	// isRowVector (vector version): Checks if a given vector is in the row space of the matrix.
	// Parameters:
	// - vec: A vector of numericalType representing the row vector to check.
	// Returns: True if vec is in the row space of the matrix, false otherwise.
	bool isRowVector(const vector<numericalType>& vec) const;

	// isRowVector (array version): Checks if a given array represents a row vector in the row space of the matrix.
	// Parameters:
	// - vec: A pointer to an array of numericalType representing the row vector to check.
	// - size: The size of the array pointed to by vec.
	// Returns: True if vec is in the row space of the matrix, false otherwise.
	bool isRowVector(numericalType* vec, const size_t& size) const;

	// isLinearlyIndependent (row index version): Checks if two specified rows in the matrix are linearly independent.
	// Parameters:
	// - row1Idx: Index of the first row.
	// - row2Idx: Index of the second row.
	// Returns: True if the rows are linearly independent, false otherwise.
	bool isLinearlyIndependent(const size_t& row1Idx, const size_t& row2Idx) const;

	// isLinearlyIndependent (array version): Checks if the specified row in the matrix and an external row represented as an array are linearly independent.
	// Parameters:
	// - row: Array representing an external row.
	// - size: Size of the array representing the external row.
	// - rowIdx: Index of the row in the matrix.
	// Returns: True if the matrix row and the external row are linearly independent, false otherwise.
	bool isLinearlyIndependent(numericalType* row, const size_t& size, const size_t& rowIdx) const;

	// isLinearlyIndependent (vector version): Checks if the specified row in the matrix and an external row represented as a vector are linearly independent.
	// Parameters:
	// - row: Vector representing an external row.
	// - rowIdx: Index of the row in the matrix.
	// Returns: True if the matrix row and the external row are linearly independent, false otherwise.
	bool isLinearlyIndependent(const vector<numericalType>& row, const size_t& rowIdx) const;

	// isLinearlyIndependent (array version): Checks if two arrays representing vectors are linearly independent.
	// Parameters:
	// - row1: Pointer to the first array of numericalType representing the first vector.
	// - row1Size: The size of the first vector.
	// - row2: Pointer to the second array of numericalType representing the second vector.
	// - row2Size: The size of the second vector.
	// Returns: True if the vectors are linearly independent, false otherwise.
	bool isLinearlyIndependent(numericalType* row1, const size_t& row1Size, numericalType* row2, const size_t& row2Size) const;

	// isLinearlyIndependent (vector version): Checks if two vectors are linearly independent.
	// Parameters:
	// - row1: The first vector as a vector of numericalType.
	// - row2: The second vector as a vector of numericalType.
	// Returns: True if the vectors are linearly independent, false otherwise.
	bool isLinearlyIndependent(const vector<numericalType>& row1, const vector<numericalType>& row2) const;

	// swap: Swaps the contents of two Matrix objects.
	// This function is a friend of the Matrix class, allowing it to access private members of Matrix objects.
	// The swap function is useful for implementing the copy-and-swap idiom, particularly in the assignment operator,
	// to ensure strong exception safety and handle self-assignment gracefully.
	// Parameters:
	// - first: A reference to the first Matrix object involved in the swap.
	// - second: A reference to the second Matrix object involved in the swap.
	// Returns: None. This function modifies the input matrices in place, swapping their contents.
	// Exception Guarantee: Nothrow. This function does not throw exceptions as it only involves swapping primitive types and pointers,
	// which are noexcept operations.
	friend void swap(Matrix& first, Matrix& second)noexcept 
	{
		using std::swap;
		swap(first._row, second._row);
		swap(first._col, second._col);
		swap(first.matrix, second.matrix);
	}

	// swapRows: Swaps two rows in the matrix.
	// Parameters:
	// - row1Idx, row2Idx: The indexes of the rows to swap.
	// Returns: None.
	void swapRows(const size_t& row1Idx, const size_t& row2Idx);

	// swapCols: Swaps two columns in the matrix.
	// Parameters:
	// - col1Idx, col2Idx: The indexes of the columns to swap.
	// Returns: None.
	void swapCols(const size_t& col1Idx, const size_t& col2Idx);

	// normalizeRowForPosition: Normalizes a row so that a specified element becomes 1.
	// Parameters:
	// - rowIdx: The index of the row to normalize.
	// - oneIdx: The index within the row of the element to normalize to 1.
	// Returns: The common denominator used for normalization.
	numericalType normalizeRowForPosition(const size_t& rowIdx, const size_t& oneIdx);

	// scalarMultiplyRow: Multiplies all elements of a specified row by a scalar value.
	// Parameters:
	// - scalar: The scalar value to multiply the row elements by.
	// - rowIdx: The index of the row to be scaled.
	// Returns: None.
	void scalarMultiplyRow(const numericalType& scalar, const size_t& rowIdx);

	// scalarMultiplyCol: Multiplies all elements of a specified column by a scalar value.
	// Parameters:
	// - scalar: The scalar value to multiply the column elements by.
	// - colIdx: The index of the column to be scaled.
	// Returns: None.
	void scalarMultiplyCol(const numericalType& scalar, const size_t& colIdx);

	// addRow: Adds one row to another, possibly with a scalar multiplier and sign change.
	// Parameters:
	// - row: The index of the row to be modified.
	// - rowToAdd: The index of the row to add to the first row.
	// - sign: The sign of the operation (1 for addition, -1 for subtraction), so it works with subtraction aswell.
	// - howManyTimes: A scalar multiplier for the row being added.
	// Returns: None.
	void addRow(const size_t& row, const size_t& rowToAdd, const numericalType sign = 1, const numericalType howManyTimes = 1);

	// addRow: Adds or subtracts (depending on the sign) a given row vector to/from another row in the matrix, optionally multiple times (scaled by howManyTimes).
	// Parameters:
	// - row: The index of the row in the matrix to be modified.
	// - rowToAdd: The vector to be added or subtracted.
	// - sign: Determines if the operation is addition (1) or subtraction (-1).
	// - howManyTimes: The scalar by which the rowToAdd vector is scaled before the operation.
	// Returns: None. Modifies the matrix in place.
	void addRow(const size_t& row, vector<numericalType> rowToAdd, const numericalType sign = 1, const numericalType howManyTimes = 1);

	// subtractRow (row index version): Subtracts one row from another within the matrix, optionally multiple times (scaled by howManyTimes).
	// Parameters:
	// - row: The index of the row in the matrix to be modified.
	// - rowToSub: The index of the row to be subtracted.
	// - howManyTimes: The scalar by which the rowToSub is scaled before the subtraction.
	// Note: Utilizes addRow internally for the operation.
	void subractRow(const size_t& row, const size_t& rowToSub, const numericalType howManyTimes = 1) { addRow(row, rowToSub, -1, howManyTimes); }

	// subtractRow (vector version): Subtracts a given row vector from another row in the matrix, optionally multiple times (scaled by howManyTimes).
	// Parameters:
	// - row: The index of the row in the matrix to be modified.
	// - rowToSub: The vector to be subtracted.
	// - howManyTimes: The scalar by which the rowToSub vector is scaled before the subtraction.
	// Note: Utilizes addRow internally for the operation.
	void subractRow(const size_t& row, vector<numericalType> rowToSub, const numericalType howManyTimes = 1) { addRow(row, rowToSub, -1, howManyTimes); }

	// subtractRow (array version): Subtracts one row from another, both represented as arrays.
	// Parameters:
	// - subFrom: The array representing the row from which to subtract.
	// - size1: The size of the subFrom array.
	// - subThis: The array representing the row to subtract.
	// - size2: The size of the subThis array.
	// Returns: A pointer to a dynamically allocated array of numericalType containing the result of the subtraction.
	// Note: The caller is responsible for deleting the returned array to avoid memory leaks.
	numericalType* subtractRow(numericalType* subFrom, const size_t& size1, numericalType* subThis, const size_t& size2) const;

	// subtractRow (vector version): Subtracts one row vector from another.
	// Parameters:
	// - subFrom: The vector representing the row from which to subtract.
	// - subThis: The vector representing the row to subtract.
	// Returns: A pointer to a dynamically allocated array of numericalType containing the result of the subtraction.
	// Note: The caller is responsible for deleting the returned array to avoid memory leaks.
	numericalType* subtractRow(const vector<numericalType>& subFrom, const vector<numericalType>& subThis) const;

	// subtractRowVector (array version): Subtracts one row from another, both represented as arrays, and returns the result as a vector.
	// Parameters:
	// - subFrom: The array representing the row from which to subtract.
	// - size1: The size of the subFrom array.
	// - subThis: The array representing the row to subtract.
	// - size2: The size of the subThis array.
	// Returns: A vector of numericalType containing the result of the subtraction.
	vector<numericalType> subtractRowVector(numericalType* subFrom, const size_t& size1, numericalType* subThis, const size_t& size2) const;

	// subtractRowVector (vector version): Subtracts one row vector from another and returns the result as a vector.
	// Parameters:
	// - subFrom: The vector representing the row from which to subtract.
	// - subThis: The vector representing the row to subtract.
	// Returns: A vector of numericalType containing the result of the subtraction.
	vector<numericalType> subtractRowVector(const vector<numericalType>& subFrom, const vector<numericalType>& subThis) const;

	// dotProduct (row index version): Calculates the dot product between two rows within the matrix.
	// Parameters:
	// - row1: Index of the first row.
	// - row2: Index of the second row.
	// Returns: The dot product of the two specified rows as a numericalType value.
	numericalType dotProduct(const size_t& row1, const size_t& row2) const;

	// dotProduct (row and array version): Calculates the dot product between a matrix row and an external row represented as an array.
	// Parameters:
	// - vec: Array representing an external row.
	// - size: Size of the array representing the external row.
	// - rowIdx: Index of the row in the matrix.
	// Returns: The dot product as a numericalType value.
	numericalType dotProduct(numericalType* vec, const size_t& size, const size_t& rowIdx) const;

	// dotProduct (row and vector version): Calculates the dot product between a matrix row and an external row represented as a vector.
	// Parameters:
	// - vec: Vector representing an external row.
	// - rowIdx: Index of the row in the matrix.
	// Returns: The dot product as a numericalType value.
	numericalType dotProduct(const vector<numericalType>& vec, const size_t& rowIdx) const;

	// dotProduct (vector version): Calculates the dot product between two vectors.
	// Parameters:
	// - vec1: The first vector as a vector of numericalType.
	// - vec2: The second vector as a vector of numericalType.
	// Returns: The dot product of the two vectors as a numericalType value.
	numericalType dotProduct(const vector<numericalType>& vec1, const vector<numericalType>& vec2) const;

	// dotProduct (array version): Calculates the dot product between two arrays representing vectors.
	// Parameters:
	// - vec1: Pointer to the first array of numericalType representing the first vector.
	// - size1: The size of the first vector.
	// - vec2: Pointer to the second array of numericalType representing the second vector.
	// - size2: The size of the second vector.
	// Returns: The dot product of the two vectors as a numericalType value.
	numericalType dotProduct(numericalType* vec1, const size_t& size1, numericalType* vec2, const size_t& size2) const;

	// rowAbs: Calculates the Euclidean norm of a specified row.
	// Parameters:
	// - idx: The index of the row.
	// Returns: The Euclidean norm of the row.
	numericalType rowAbs(const size_t& idx) const;

	// rowAbs (array version): Calculates the Euclidean norm of a specified row.
	// Parameters:
	// - row: The vector that you want to calculate norm of.
	// - size: Size of the vector
	// Returns: The Euclidean norm of the row.
	numericalType rowAbs(numericalType* row, const size_t& size) const;

	// rowAbs (vector version): Calculates the Euclidean norm of a specified row.
	// Parameters:
	//  - row: The vector that you want to calculate norm of.
	// Returns: The Euclidean norm of the row.
	numericalType rowAbs(vector<numericalType> row) const;

	// colAbs: Calculates the Euclidean norm of a specified column.
	// Parameters:
	// - idx: The index of the column.
	// Returns: The Euclidean norm of the column.
	numericalType colAbs(const size_t& idx) const;

	// distanceFromRow: Calculates the distance from a given matrix row to another specified row.
	// Parameters:
	// - distanceFromIdx: The index of the row from which the distance is calculated.
	// - idx: The index of the row to which the distance is calculated.
	// Returns: The distance as a numericalType value between the two specified rows.
	numericalType distanceFromRow(const size_t& distanceFromIdx, const size_t& idx) const;

	// distanceFromRow (array version): Calculates the distance from a given matrix row to an external row represented as an array.
	// Parameters:
	// - otherRow: An array representing the row vector to calculate distance to.
	// - rowSize: The size of the array representing the other row.
	// - distanceFromIdx: The index of the matrix row from which the distance is calculated.
	// Returns: The distance as a numericalType value between the matrix row and the external row vector.
	numericalType distanceFromRow(numericalType* otherROw, const size_t& rowSize, const size_t& distanceFromIdx) const;

	// distanceFromRow (vector version): Calculates the distance from a given matrix row to an external row represented as a vector.
	// Parameters:
	// - otherRow: A vector representing the row vector to calculate distance to.
	// - distanceFromIdx: The index of the matrix row from which the distance is calculated.
	// Returns: The distance as a numericalType value between the matrix row and the external row vector.
	numericalType distanceFromRow(const vector<numericalType>& otherRow, const size_t& distanceFromIdx) const;

	// distanceFromCol: Calculates the distance from a given matrix column to another specified column.
	// Parameters:
	// - distanceFromIdx: The index of the column from which the distance is calculated.
	// - idx: The index of the column to which the distance is calculated.
	// Returns: The distance as a numericalType value between the two specified columns.
	numericalType distanceFromCol(const size_t& distanceFromIdx, const size_t& idx) const;

	// distanceFromCol (array version): Calculates the distance from a given matrix column to an external column represented as an array.
	// Parameters:
	// - otherCol: An array representing the column vector to calculate distance to.
	// - colSize: The size of the array representing the other column.
	// - distanceFromIdx: The index of the matrix column from which the distance is calculated.
	// Returns: The distance as a numericalType value between the matrix column and the external column vector.
	numericalType distanceFromCol(numericalType* otherCol, const size_t& colSize, const size_t& distanceFromIdx) const;

	// distanceFromCol (vector version): Calculates the distance from a given matrix column to an external column represented as a vector.
	// Parameters:
	// - otherCol: A vector representing the column vector to calculate distance to.
	// - distanceFromIdx: The index of the matrix column from which the distance is calculated.
	// Returns: The distance as a numericalType value between the matrix column and the external column vector.
	numericalType distanceFromCol(const vector<numericalType>& otherCol, const size_t& distanceFromIdx) const;

	// distanceFromMatrix: Calculates the Frobenius norm-based distance between the current matrix and another matrix.
	// Parameters:
	// - other: The matrix to calculate the distance to.
	// Returns: The distance as a numericalType value between the two matrices.
	numericalType distanceFromMatrix(const Matrix<numericalType>& other) const;

	// Operator overloads

	// operator[] (non-const): Provides direct access to a specific row of the matrix.
	// Parameters:
	// - index: The index of the row to access.
	// Returns: A pointer to the row allowing modification.
	numericalType* operator[](const size_t& index) { return matrix[index];}				// Operator[] to return row

	// operator[] (const): Provides read-only access to a specific row of the matrix.
	// Parameters:
	// - index: The index of the row to access.
	// Returns: A const pointer to the row, preventing modification.
	const numericalType* operator[](const size_t& index) const { return matrix[index];}	// Const version of operator[] for read-only access

	// operator=: Assigns the contents of another matrix to this matrix using the copy-and-swap idiom.
	// Parameters:
	// - other: The matrix to copy from.
	// Returns: A reference to *this.
	Matrix& operator=(Matrix other) {
		swap(*this, other);
		return *this;
	}											// Assignment Operator using Copy-and-Swap Idiom

	// operator+: Calculates the sum of this matrix with another matrix.
	// Parameters:
	// - rhs: The right-hand side matrix to add.
	// Returns: A new Matrix object representing the sum of the two matrices.
	Matrix<numericalType> operator+(const Matrix<numericalType>& rhs);

	// operator-: Calculates the difference between this matrix and another matrix.
	// Parameters:
	// - rhs: The right-hand side matrix to subtract.
	// Returns: A new Matrix object representing the difference between the two matrices.
	Matrix<numericalType> operator-(const Matrix<numericalType>& rhs);

	// operator*: Performs matrix multiplication with another matrix.
	// Parameters:
	// - rhs: The right-hand side matrix to multiply with.
	// Returns: A new Matrix object representing the product of the two matrices.
	Matrix<numericalType> operator*(const Matrix<numericalType>& rhs);					

	// operator*: Performs scalar multiplication of this matrix.
	// Parameters:
	// - scalar: The scalar value to multiply each element of the matrix by.
	// Returns: A new Matrix object representing the scalar multiplication result.
	Matrix<numericalType> operator*(const numericalType& scalar);		

	// operator-=: Subtracts another matrix from this matrix and updates this matrix with the result.
	// Parameters:
	// - rhs: The right-hand side matrix to subtract.
	// Returns: A reference to *this, updated with the subtraction result.
	Matrix<numericalType> operator-=(const Matrix<numericalType>& rhs);

	// operator/: Divides each element of this matrix by a scalar value.
	// Parameters:
	// - scalar: The scalar value to divide each matrix element by.
	// Returns: A new Matrix object representing the division result.
	Matrix<numericalType> operator/(const numericalType& scalar);

	// operator==: Checks if this matrix is equal to another matrix.
	// Parameters:
	// - other: The matrix to compare with.
	// Returns: True if both matrices are equal, false otherwise.
	bool operator==(const Matrix<numericalType>& other)
	{
		if (other.row() != _row || other.col() != _col)
			return false;

		for (unsigned i = 0; i < _row; i++)
			for (unsigned j = 0; j < _col; j++)
				if (matrix[i][j] != other[i][j])
					return false;

		return true;
	}

	// at: Accesses the element at the specified row and column of the matrix.
	// Parameters:
	// - row: The row index of the element to access.
	// - col: The column index of the element to access.
	// Returns: The value of the element at the specified row and column, no out of bounds check!
	numericalType at(const size_t& row, const size_t& col) const  {return matrix[row][col];}

	// push_back (array version): Adds a new row to the matrix from an array.
	// Parameters:
	// - row: An array representing the new row to add.
	// - size: The size of the array representing the new row.
	// Returns: None.
	void push_back(numericalType* row, const size_t& size);

	// push_back (vector version): Adds a new row to the matrix from a vector.
	// Parameters:
	// - row: A vector representing the new row to add.
	// Returns: None.
	void push_back(vector<numericalType> row);

	// printToStdOut: Prints the matrix to standard output.
	// Parameters: None.
	// Returns: None.
	void printToStdOut() const;

	// trace: Calculates the trace of the matrix (the sum of the diagonal elements).
	// Parameters: None.
	// Returns: The trace of the matrix as a numericalType value.
	numericalType trace() const;

	// transpose: Calculates the transpose of the matrix.
	// Parameters: None.
	// Returns: A new Matrix object representing the transpose of the matrix.
	Matrix<numericalType> transpose() const;

	// col: Returns the number of columns in the matrix.
	// Parameters: None.
	// Returns: The number of columns as a size_t value.
	size_t col() const { return _col; }

	// row: Returns the number of rows in the matrix.
	// Parameters: None.
	// Returns: The number of rows as a size_t value.
	size_t row() const { return _row; }

	// Basic matrix types

	// setToHomogenous: Sets all elements of the matrix to a specified homogeneous value.
	// Parameters:
	// - hom: The value to set all elements of the matrix to. Defaults to numericalType{} + 1.
	// Returns: None. Modifies the matrix in place.
	void setToHomogenous(const numericalType& hom = ({} + 1));

	// setToIdentity: Sets the matrix to an identity matrix if it is square.
	// Parameters: None.
	// Returns: None. Modifies the matrix in place, setting it to an identity matrix if square, otherwise no action is taken.
	void setToIdentity();

	// setToZeroMatrix-zeroes: Sets all elements of the matrix to zero.
	// Parameters: None.
	// Returns: None. Modifies the matrix in place, setting all elements to zero.
	void setToZeroMatrix();
	void zeroes() { setToZeroMatrix(); }
	
	// luDecomposition: Performs LU decomposition on the matrix, where the L and U matrices are stored in the places of the original matrix.
	// Parameters:
	// - L: Reference to a Matrix object where the L matrix will be stored.
	// - U: Reference to a Matrix object where the U matrix will be stored.
	// Returns: None. The L and U matrices are output parameters that are filled by the function.
	void luDecomposition(Matrix<numericalType>& L, Matrix<numericalType>& U) const;

	// organizeDecreasing: Reorganizes the matrix rows so that they are in decreasing order based on a specified criterion.
	// Parameters: None.
	// Returns: A new Matrix object that represents the original matrix with rows organized in decreasing order.
	Matrix<numericalType> organizeDecreasing() const;

	// poorGaussian (array version): Performs a basic form of Gaussian elimination on the matrix.
	// Parameters:
	// - rhs: A vector of numericalType representing the right-hand side of the equation system.
	// - size: Size of the vector.
	// Returns: A Matrix object representing the result of applying Gaussian elimination.
	Matrix<numericalType> poorGaussian(numericalType* rhs, const size_t& size) const;

	// poorGaussian (vector version): Performs a basic form of Gaussian elimination on the matrix.
	// Parameters:
	// - rhs: A vector of numericalType representing the right-hand side of the equation system.
	// Returns: A Matrix object representing the result of applying Gaussian elimination.
	Matrix<numericalType> poorGaussian(vector<numericalType> rhs) const;

	// gaussJordanElimination: Performs Gauss-Jordan elimination on the matrix to achieve reduced row echelon form.
	// Parameters: None.
	// Returns: A Matrix object representing the matrix in reduced row echelon form after applying Gauss-Jordan elimination.
	Matrix<numericalType> gaussJordanElimination() const;

	// kernel: Computes the kernel (null space) of the matrix.
	// Parameters: None.
	// Returns: A Matrix object representing the kernel of the original matrix.
	Matrix<numericalType> kernel() const;

	// image: Computes the image (column space) of the matrix.
	// Parameters: None.
	// Returns: A Matrix object representing the image of the original matrix.
	Matrix<numericalType> image() const;

	// rank: Computes the rank of the matrix, defined as the number of linearly independent rows or columns.
	// Parameters: None.
	// Returns: An unsigned integer representing the rank of the matrix.
	unsigned rank() const;

	// determinant: Calculates the determinant of the matrix.
	// Parameters: None.
	// Returns: A numericalType value representing the determinant of the matrix, determinant sign might mismatch!
	numericalType determinant() const;

	// inverse: Computes the inverse of the matrix, if it exists.
	// Parameters: None.
	// Returns: A Matrix object representing the inverse of the original matrix.
	Matrix<numericalType> inverse() const;

	// getNormalVector: Computes a vector that is orthogonal to all basis vectors in the null space of the matrix. Skipping {0} as it is the trivial solution.
	// Parameters: None.
	// Returns: A vector of numericalType representing the normal vector.
	vector<numericalType> getNormalVector() const;

	// outerProduct (array versio): Computes the outer product of two vectors.
	// Parameters:
	// - vector1: The first vector for the outer product calculation, either as an array or vector.
	// - size1: The size of the first vector if provided as an array.
	// - vector2: The second vector for the outer product calculation, either as an array or vector.
	// - size2: The size of the second vector if provided as an array.
	// Returns: A Matrix object representing the outer product of the two vectors.
	Matrix<numericalType> outerProduct(numericalType* vector1, const size_t& size1, numericalType* vector2, const size_t& size2) const;

	// outerProduct (vector version): Computes the outer product of two vectors.
	// Parameters:
	// - vector1: The first vector for the outer product calculation, either as an array or vector.
	// - vector2: The second vector for the outer product calculation, either as an array or vector.
	// Returns: A Matrix object representing the outer product of the two vectors.
	Matrix<numericalType> outerProduct(vector<numericalType> vector1, vector<numericalType> vector2) const;

	// projectToStraightLine (array version): Computes the projection matrix for projecting vectors onto a straight line defined by a given direction vector.
	// Parameters:
	// - lineVector: The direction vector defining the straight line, provided as either an array with its size or a vector.
	// - size: Size of the vector.
	// Returns: A Matrix object representing the projection matrix onto the line.
	Matrix<numericalType> projectToStarightLine(numericalType* lineVector, const size_t& size) const;

	// projectToStraightLine (vector version): Computes the projection matrix for projecting vectors onto a straight line defined by a given direction vector.
	// Parameters:
	// - lineVector: The direction vector defining the straight line, provided as either an array with its size or a vector.
	// Returns: A Matrix object representing the projection matrix onto the line.
	Matrix<numericalType> projectToStarightLine(const vector<numericalType> lineVector) const;

	// projectToHyperPlane (array version): Computes the projection matrix for projecting vectors onto a hyperplane defined by its normal vector.
	// Parameters:
	// - normalVector: The normal vector defining the hyperplane, provided as either an array with its size or a vector.
	// size: Size of the vector.
	// Returns: A Matrix object representing the projection matrix onto the hyperplane.
	Matrix<numericalType> projectToHyperPlane(numericalType* normalVector, const size_t& size) const;

	// projectToHyperPlane (vector version): Computes the projection matrix for projecting vectors onto a hyperplane defined by its normal vector.
	// Parameters:
	// - normalVector: The normal vector defining the hyperplane, provided as either an array with its size or a vector.
	// Returns: A Matrix object representing the projection matrix onto the hyperplane.
	Matrix<numericalType> projectToHyperPlane(const vector<numericalType> normalVector) const;

	// mirrorToHyperPlane (array version): Computes the mirroring matrix relative to a hyperplane defined by its normal vector.
	// Parameters:
	// - normalVector: The normal vector defining the hyperplane, provided as either an array with its size or a vector.
	// - size: Size of the vector.
	// Returns: A Matrix object representing the mirroring matrix relative to the hyperplane.
	Matrix<numericalType> mirrorToHyperPlane(numericalType* normalVector, const size_t& size) const;

	// mirrorToHyperPlane (vector version): Computes the mirroring matrix relative to a hyperplane defined by its normal vector.
	// Parameters:
	// - normalVector: The normal vector defining the hyperplane, provided as either an array with its size or a vector.
	// Returns: A Matrix object representing the mirroring matrix relative to the hyperplane.
	Matrix<numericalType> mirrorToHyperPlane(const vector<numericalType>& normalVector) const;

	// projectToW: Computes the projection matrix onto a subspace W, assuming the matrix A has full column rank and represents the subspace.
	// Parameters: None.
	// Returns: A Matrix object representing the projection matrix onto the subspace W.
	Matrix<numericalType> projectToW() const;

	// span: Computes the span of the matrix, represented by its basis vectors.
	// Parameters: None.
	// Returns: A Matrix object representing the span of the original matrix.
	Matrix<numericalType> span() const;

	// A^+ is the pseudoinverse notation
	// pseudoInverse: Computes the Moore-Penrose pseudoinverse of the matrix.
	// Parameters: None.
	// Returns: A Matrix object representing the pseudoinverse of the original matrix.
	Matrix<numericalType> pseudoInverse() const;

	// qrDecomposition: Performs QR decomposition of the matrix into an orthogonal matrix Q and an upper triangular matrix R.
	// Parameters:
	// - Q: Reference to a Matrix object where the Q matrix will be stored.
	// - R: Reference to a Matrix object where the R matrix will be stored.
	// Returns: None. The Q and R matrices are output parameters that are filled by the function.
	void qrDecomposition(Matrix<numericalType>& Q, Matrix<numericalType>& R) const;

	// reducedQRDecomposition: Performs a reduced QR decomposition of the matrix, applicable for non-square matrices as well.
	// Parameters:
	// - Q: Reference to a Matrix object where the reduced Q matrix will be stored.
	// - R: Reference to a Matrix object where the reduced R matrix will be stored.
	// Returns: None. The reduced Q and R matrices are output parameters filled by the function.
	void reducedQRDecomposition(Matrix<numericalType>& Q, Matrix<numericalType>& R) const;

	// eigenvaluesVector: Calculates the eigenvalues of the matrix.
	// Parameters:
	// - maxIterations: Maximum number of iterations for the eigenvalue calculation algorithm.
	// - tol: Tolerance for determining convergence of the algorithm.
	// Returns: A vector of numericalType containing the eigenvalues of the matrix.
	vector<numericalType> eigenvaluesVector(int maxIterations = 1000, numericalType tol = 1e-9) const;
	vector<numericalType> spectrumVector(int maxIterations = 1000, numericalType tol = 1e-9) const { return eigenvaluesVector(maxIterations, tol); }

	// eigenvalues: Calculates the eigenvalues of the matrix.
	// Parameters:
	// - maxIterations: Maximum number of iterations for the eigenvalue calculation algorithm.
	// - tol: Tolerance for determining convergence of the algorithm.
	// Returns: A array of numericalType containing the eigenvalues of the matrix.
	numericalType* eigenvalues(int maxIterations = 1000, numericalType tol = 1e-9) const;
	numericalType* spectrum(int maxIterations = 1000, numericalType tol = 1e-9) const { return eigenvalues(maxIterations, tol); }

	// eigenvectors: Computes the eigenvectors of the matrix.
	// Parameters: None.
	// Returns: A Matrix object where each column represents an eigenvector of the original matrix.
	Matrix<numericalType> eigenvectors() const;

	// getSubMatrix: Extracts a submatrix from the current matrix.
	// Parameters:
	// - rowStart, rowEnd, colStart, colEnd: The starting and ending row and column indexes defining the submatrix.
	// Returns: A Matrix object representing the specified submatrix of the original matrix.
	Matrix<numericalType> getSubMatrix(const size_t& rowEnd, const size_t& colEnd) const;
	Matrix<numericalType> getSubMatrix(const size_t& rowStart, const size_t& rowEnd, const size_t& colStart, const size_t& colEnd) const;

	// getLeadingPrincipalMinor: Computes the leading principal minor of the matrix up to a specified order k.
	// Parameters:
	// - k: The order up to which the leading principal minor is computed.
	// Returns: A numericalType value representing the leading principal minor of order k.
	numericalType getLeadingPrincipalMinor(size_t k) const;

	// Returns the own eigen pairs of the matrix
	vector<pair<numericalType, numericalType*>> ownEigenPairs() const;
	vector<pair<numericalType, vector<numericalType>>> ownEigenPairsVector() const;

	// characteristics: Computes the characteristic polynomial of the matrix. Calculated with using x(lambda) = det(A - lamdba * I).
	// Parameters: None.
	// Returns: A Matrix object representing the coefficients of the characteristic polynomial.
	Matrix<numericalType> characteristics() const;

	// applySVD: Applies Singular Value Decomposition (SVD) to the matrix, decomposing it into U, Sigma, and VT matrices.
	// Parameters:
	// - U: Reference to a Matrix object to store the left singular vectors.
	// - Sigma: Reference to a Matrix object to store the singular values.
	// - VT: Reference to a Matrix object to store the right singular vectors transposed.
	// Returns: None. The U, Sigma, and VT matrices are output parameters filled by the function.
	void applySVD(Matrix<numericalType>& U, Matrix<numericalType>& Sigma, Matrix<numericalType>& VT) const;

	// frobeniusNorm: Calculates the Frobenius norm of the matrix.
	// Parameters: None.
	// Returns: A numericalType value representing the Frobenius norm of the matrix.
	numericalType frobeniusNorm() const;

	// l1Norm: Calculates the L1 norm of the matrix, which is the maximum absolute column sum.
	// Parameters: None.
	// Returns: A numericalType value representing the L1 norm of the matrix.
	numericalType l1Norm() const;

	// matrixExponential: Calculates the matrix exponential e^A using a series expansion I + A + A^2/(2!) + A^3/(3!) + A^4/(4!) +... .
	// Parameters:
	// - truncSize: The number of terms to include in the series expansion. By default set to 20.
	// Returns: A Matrix object representing e^A calculated up to truncSize terms.
	Matrix<numericalType> matrixExponential(const size_t& truncSize = 20) const;

	// lowerBandwidth: Computes the lower bandwidth of the matrix.
	// Parameters: None.
	// Returns: The size_t value representing the lower bandwidth of the matrix.
	size_t lowerBandwidth() const;

	// upperBandwidth: Computes the upper bandwidth of the matrix.
	// Parameters: None.
	// Returns: The size_t value representing the upper bandwidth of the matrix.
	size_t upperBandwidht() const;

	// bandwidth: Computes the total bandwidth of the matrix.
	// Parameters: None.
	// Returns: The size_t value representing the total bandwidth of the matrix.
	size_t bandwidth() const;

	// choleskyDecomposition: Performs Cholesky decomposition on the matrix, assuming it is symmetric and positive definite.
	// Parameters: None.
	// Returns: A Matrix object representing the Cholesky decomposition of the original matrix.
	Matrix<numericalType> choleskyDecomposition() const;

	// ownSubSpace: Computes the eigenspace of the matrix, including the zero vector.
	// Parameters: None.
	// Returns: A Matrix object representing the eigenspace of the matrix.
	Matrix<numericalType> ownSubSpace() const;

	// diagonalize: Diagonalizes the matrix if possible, producing matrices Pand D such that A = PDP ^ (-1).
	// Parameters:
	// - P: Reference to a Matrix object to store the matrix of eigenvectors.
	// - D: Reference to a Matrix object to store the diagonal matrix of eigenvalues.
	// Returns: None. The P and D matrices are output parameters filled by the function.
	void diagonalize(Matrix<numericalType>& P, Matrix<numericalType>& D) const;

	// angleBetween (row index version): Calculates the angle between two rows within the matrix.
	// Parameters:
	// - row1Idx: Index of the first row.
	// - row2Idx: Index of the second row.
	// Returns: The angle in radians between the two specified rows as a double.
	double angleBetween(const size_t& row1Idx, const size_t& row2Idx) const;

	// angleBetween (array and row index version): Calculates the angle between an external row represented as an array and a specific row in the matrix.
	// Parameters:
	// - row: Array representing an external row.
	// - size: The size of the array representing the external row.
	// - rowIdx: The index of the row within the matrix.
	// Returns: The angle in radians between the external row and the specified matrix row as a double.
	double angleBetween(numericalType* row, const size_t& size, const size_t& rowIdx) const;

	// angleBetween (vector and row index version): Calculates the angle between an external row represented as a vector and a specific row in the matrix.
	// Parameters:
	// - row: Vector representing an external row.
	// - rowIdx: The index of the row within the matrix.
	// Returns: The angle in radians between the external row and the specified matrix row as a double.
	double angleBetween(const vector<numericalType>& row, const size_t& rowIdx) const;

	// angleBetween (array version): Calculates the angle between two rows represented as arrays.
	// Parameters:
	// - row1: Array representing the first row.
	// - size1: The size of the first array.
	// - row2: Array representing the second row.
	// - size2: The size of the second array.
	// Returns: The angle in radians between the two rows as a double.
	double angleBetween(numericalType* row1, const size_t& size1, numericalType* row2, const size_t& size2) const;

	// angleBetween (vector version): Calculates the angle between two rows represented as vectors.
	// Parameters:
	// - row1: Vector representing the first row.
	// - row2: Vector representing the second row.
	// Returns: The angle in radians between the two rows as a double.
	double angleBetween(const vector<numericalType>& row1,const vector<numericalType>& row2) const;

	// projectToVector (vector version): Projects one vector onto another and returns the projection as a new vector.
	// Parameters:
	// - projectThisTo: The vector that is being projected.
	// - toThis: The vector onto which projectThisTo is projected.
	// Returns: A vector of numericalType representing the projection of projectThisTo onto toThis.
	vector<numericalType> projectToVector(const vector<numericalType>& projectThisTo, const vector<numericalType>& toThis) const;

	// projectToVector (array version): Projects one vector represented as an array onto another vector also represented as an array, and returns the projection as a new vector.
	// Parameters:
	// - projectThisTo: Array representing the vector that is being projected.
	// - size1: The size of projectThisTo array.
	// - toThis: Array representing the vector onto which projectThisTo is projected.
	// - size2: The size of toThis array.
	// Returns: A vector of numericalType representing the projection of projectThisTo onto toThis.
	vector<numericalType> projectToVector(numericalType* projectThisTo, const size_t& size1, numericalType* toThis, const size_t& size2) const;

	// projectTo (vector version): Projects one vector onto another and returns the projection as a new dynamically allocated array.
	// Parameters:
	// - projectThisTo: The vector that is being projected.
	// - toThis: The vector onto which projectThisTo is projected.
	// Returns: A pointer to a dynamically allocated array of numericalType representing the projection of projectThisTo onto toThis.
	// Note: The caller is responsible for deleting the returned array to avoid memory leaks.
	numericalType* projectTo(const vector<numericalType>& projectThisTo, const vector<numericalType>& toThis) const;

	// projectTo (array version): Projects one vector represented as an array onto another vector also represented as an array, and returns the projection as a new dynamically allocated array.
	// Parameters:
	// - projectThisTo: Array representing the vector that is being projected.
	// - size1: The size of projectThisTo array.
	// - toThis: Array representing the vector onto which projectThisTo is projected.
	// - size2: The size of toThis array.
	// Returns: A pointer to a dynamically allocated array of numericalType representing the projection of projectThisTo onto toThis.
	// Note: The caller is responsible for deleting the returned array to avoid memory leaks.
	numericalType* projectTo(numericalType* projectThisTo, const size_t& size1, numericalType* toThis, const size_t& size2) const;

	// gramSchmidAlgorithm: Applies the Gram-Schmidt process to the matrix to orthogonalize its rows.
	// Parameters: None.
	// Returns: A new Matrix object where the rows are the result of applying the Gram-Schmidt orthogonalization process to the original matrix's rows.
	Matrix<numericalType> gramSchmidAlgorithm() const;

	// leastSquares (vector version): Solves the least squares problem using the current matrix as the coefficient matrix and a given vector 'b' as the observation vector.
	// This method is used to find the vector 'x' that minimizes the squared differences between the observed outcomes in 'b' and those predicted by the linear model defined by the matrix.
	// Parameters:
	// - b: A vector of numericalType representing the observation or dependent variable values.
	// Returns: A Matrix object representing the solution vector 'x' that minimizes the equation ||Ax - b||^2, where A is the current matrix.
	Matrix<numericalType> leastSquares(const vector<numericalType>& b) const;

	// leastSquares (array version): Solves the least squares problem using the current matrix as the coefficient matrix and a given array 'b' as the observation vector.
	// This method aims to find the vector 'x' that minimizes the squared differences between the observed outcomes in 'b' and those predicted by the linear model defined by the matrix.
	// Parameters:
	// - b: A pointer to an array of numericalType representing the observation or dependent variable values.
	// - size: The size of the array 'b', which should match the number of rows in the matrix for the calculation to be valid.
	// Returns: A Matrix object representing the solution vector 'x' that minimizes the equation ||Ax - b||^2, where A is the current matrix.
	// Note: It is the caller's responsibility to ensure that the size of 'b' matches the number of rows in the matrix.
	Matrix<numericalType> leastSquares(numericalType* b, const size_t& size) const;
};

#endif /*_MATRIX_HPP*/