#include "matrix.hpp"

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::operator+(const Matrix<numericalType>& rhs)
{
	if (_row != rhs.row() || _col != rhs.col())
		throw std::runtime_error("Cannot add matrices of not the same size!");

	Matrix<numericalType> cpy(_row, _col);

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			cpy[i][j] = matrix[i][j] + rhs[i][j];

	return cpy;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::operator-(const Matrix<numericalType>& rhs)
{
	if (_row != rhs.row() || _col != rhs.col())
		throw std::runtime_error("Cannot subtract matrices of not the same size!");

	Matrix<numericalType> cpy(_row, _col);

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			cpy[i][j] = matrix[i][j] - rhs[i][j];

	return cpy;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::operator*(const Matrix<numericalType>& rhs)
{
	// First, check if the number of columns in the first matrix (this->col)
	// matches the number of rows in the second matrix (rhs.row)
	// If not, matrix multiplication cannot be performed.
	if (_col != rhs.row())
		throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
	
	// Create a result matrix with appropriate size (rows of the first matrix, cols of the second matrix)
	Matrix<numericalType> result(_row, rhs.col());

	// Iterate over each row of the first matrix
	for (size_t i = 0; i < _row; ++i) 
	{
		// Inside each row of the first matrix, iterate over each column of the second matrix
		for (size_t j = 0; j < rhs.col(); ++j) 
		{
			// Initialize a variable to accumulate the sum of the product of elements
			numericalType sum = {};

			// Now, iterate over each element of the current row of the first matrix
			// and each element of the current column of the second matrix
#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:sum)
#endif
			for (size_t k = 0; k < _col; ++k) 
			{
				// Multiply each element of the row by the corresponding element of the column
				// and add it to the sum. This follows the rule:
				// result[i][j] += matrix[i][k] * rhs.matrix[k][j];
				sum += matrix[i][k] * rhs.matrix[k][j];
			}

			// Once the sum for the current element is calculated, assign it to the corresponding
			// element in the result matrix.
			result[i][j] = sum;
		}
	}
	// After filling the result matrix with the correct values, return it.
	return result;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::operator*(const numericalType& scalar)
{
	Matrix<numericalType> retM = *this;

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2) 
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			retM[i][j] *= scalar;

	return retM;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::operator-=(const Matrix<numericalType>& rhs)
{
	Matrix<numericalType> deepCpy = *this;
#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			deepCpy[i][j] -= rhs[i][j];
	return deepCpy;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::operator/(const numericalType& scalar)
{
	Matrix<numericalType> retM = *this;	// Deep copy of this
#ifdef _USING_OMP_
#pragma omp paralell for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			retM[i][j] /= scalar;
	return retM;
}

template<typename numericalType>
void Matrix<numericalType>::swapRows(const size_t& row1Idx, const size_t& row2Idx)
{
	if (row1Idx > _row || row2Idx > _row)
		throw std::runtime_error("Invalid row indexes for swapping!");

	// Swap without new allocation
	numericalType* temp = matrix[row1Idx];
	matrix[row1Idx] = matrix[row2Idx];
	matrix[row2Idx] = temp;
}

template<typename numericalType>
void Matrix<numericalType>::swapCols(const size_t& col1Idx, const size_t& col2Idx)
{
	if (col1Idx >= _col || col2Idx >= _col) 
		throw std::runtime_error("Column index out of bounds.");

	for (size_t i = 0; i < _row; ++i) 
		std::swap(matrix[i][col1Idx], matrix[i][col2Idx]);
}

template<typename numericalType>
numericalType Matrix<numericalType>::normalizeRowForPosition(const size_t& rowIdx, const size_t& oneIdx)
{
	if (rowIdx > _row || oneIdx > _col)
		throw std::runtime_error("Cannot normalize row, indexes out of bounds!");

	numericalType commonDenominator = matrix[rowIdx][oneIdx];

	if (commonDenominator == 0)
		throw std::runtime_error("Cannot divide by zero, at normalizing row!");

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < _col; i++)
		matrix[rowIdx][i] /= commonDenominator;

	return commonDenominator;
}

template<typename numericalType>
void Matrix<numericalType>::scalarMultiplyRow(const numericalType& scalar, const size_t& rowIdx)
{
	if (rowIdx >= _row)
		throw std::runtime_error("Cannot scalar multiply row, row index out of bounds!");
#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < _col; i++)
		matrix[rowIdx][i] *= scalar;
}

template<typename numericalType>
void Matrix<numericalType>::scalarMultiplyCol(const numericalType& scalar, const size_t& colIdx)
{
	if (colIdx >= _col)
		throw std::runtime_error("Cannot scalar multiply col, col index out if bounds!");
#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < _row; i++)
		matrix[i][colIdx] *= scalar;
}

template<typename numericalType>
void Matrix<numericalType>::addRow(const size_t& row, const size_t& rowToAdd, const numericalType sign, const numericalType howManyTimes)
{
	if (row > _row || rowToAdd > _row)
		throw std::runtime_error("Cannot add rows, since they are out of bounds!");

	// Adding row
#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < _col; i++)
		matrix[row][i] = matrix[row][i] + howManyTimes * sign * 1 * matrix[rowToAdd][i];
}

template<typename numericalType>
void Matrix<numericalType>::addRow(const size_t& row, vector<numericalType> rowToAdd, const numericalType sign, const numericalType howManyTimes)
{
	if (row > _row || rowToAdd.size() > _col)
		throw std::runtime_error("Cannot add row since it is out of bounds, or size doesnt match!");

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < _col; i++)
		matrix[row][i] = matrix[row][i] + howManyTimes * sign * 1 * rowToAdd[i];
}

template<typename numericalType>
numericalType* Matrix<numericalType>::subtractRow(numericalType* subFrom, const size_t& size1, numericalType* subThis, const size_t& size2) const
{
	if (size1 != size2)
		throw std::runtime_error("Cannot subtract rows which are not the same size!");

	numericalType* retRow = new numericalType[size1];

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < size1; i++)
		retRow[i] = subFrom[i] - subThis[i];

	return retRow;
}

template<typename numericalType>
numericalType* Matrix<numericalType>::subtractRow(const vector<numericalType>& subFrom, const vector<numericalType>& subThis) const
{
	size_t size1 = subFrom.size();
	size_t size2 = subThis.size();

	numericalType* sf = new numericalType[size1];
	numericalType* st = new numericalType[size2];

	numericalType* retVec = subtractRow(sf, size1, st, size2);
	delete[] sf;
	delete[] st;
	return retVec;
}

template<typename numericalType>
vector<numericalType> Matrix<numericalType>::subtractRowVector(numericalType* subFrom, const size_t& size1, numericalType* subThis, const size_t& size2) const
{
	numericalType* rv = subtractRow(subFrom, size1, subThis, size2);
	vector<numericalType> retVec;
	for (unsigned i = 0; i < size1; i++)
		retVec.push_back(rv[i]);
	delete[] rv;
	return retVec;
}

template<typename numericalType>
vector<numericalType> Matrix<numericalType>::subtractRowVector(const vector<numericalType>& subFrom, const vector<numericalType>& subThis) const
{
	numericalType* rv = subtractRow(subFrom, subThis);
	vector<numericalType> retVec;
	for (unsigned i = 0; i < subFrom.size(); i++)
		retVec.push_back(rv[i]);
	delete[] rv;
	return retVec;
}

template<typename numericalType>
numericalType Matrix<numericalType>::dotProduct(const size_t& row1, const size_t& row2) const
{
	if (row1 >= _row || row2 >= _row)
		throw std::runtime_error("Cannot calulate dotproduct because indexes are out of bounds!");

	if (row1 == row2)
		return {};

	numericalType prod = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:prod)
#endif
	for (unsigned i = 0; i < _col; i++)
		prod += matrix[row1][i] * matrix[row2][i];

	return prod;
}

template<typename numericalType>
numericalType Matrix<numericalType>::dotProduct(numericalType* vec, const size_t& size, const size_t& rowIdx) const
{
	if (size != _col)
		throw std::runtime_error("Cannot perform dot product, size doesnt match, vector out of bounds!");

	if (rowIdx >= _row)
		throw std::runtime_error("Cannot perform dot product, row index out of bounds!");

	numericalType prod = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:prod)
#endif
	for (unsigned i = 0; i < size; i++)
		prod += matrix[rowIdx][i] * vec[i];

	return prod;
}

template<typename numericalType>
numericalType Matrix<numericalType>::dotProduct(const vector<numericalType>& vec, const size_t& rowIdx) const
{
	if(vec.size() != _col)
		throw std::runtime_error("Cannot perform dot product, size doesnt match, vector out of bounds!");

	if (rowIdx >= _row)
		throw std::runtime_error("Cannot perform dot product, row index out of bounds!");

	numericalType prod = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:prod)
#endif
	for (unsigned i = 0; i < _col; i++)
		prod += matrix[rowIdx][i] * vec[i];

	return prod;
}

template<typename numericalType>
numericalType Matrix<numericalType>::dotProduct(const vector<numericalType>& vec1, const vector<numericalType>& vec2) const
{
	if (vec1.size() != vec2.size())
		throw std::runtime_error("Cannot calcualte dor product of different sized vetors!");

	numericalType prod = {};

	unsigned size = vec1.size();

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:prod)
#endif
	for (unsigned i = 0; i < size; i++)
		prod += vec1[i] * vec2[1];

	return prod;
}

template<typename numericalType>
numericalType Matrix<numericalType>::dotProduct(numericalType* vec1, const size_t& size1, numericalType* vec2, const size_t& size2) const
{
	if(size1 != size2)
		throw std::runtime_error("Cannot calcualte dor product of different sized vetors!");

	numericalType prod = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:prod)
#endif
	for (unsigned i = 0; i < size1; i++)
		prod += vec1[i] * vec2[i];

	return prod;
}

template<typename numericalType>
numericalType Matrix<numericalType>::rowAbs(const size_t& idx) const
{
	if (idx >= _row)
		throw std::runtime_error("Row index out of bounds, cannot calculate absoullte!");

	numericalType abs = {};

	// Sum up all the elements squared
#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:abs)
#endif
	for (unsigned i = 0; i < _col; i++)
		abs += matrix[idx][i] * matrix[idx][i];

	return static_cast<numericalType>(std::sqrt(abs));
}

template<typename numericalType>
numericalType Matrix<numericalType>::rowAbs(numericalType* row, const size_t& size) const
{
	numericalType abs = {};

	// Sum up all the elements squared
#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:abs)
#endif
	for (unsigned i = 0; i < _col; i++)
		abs += row[i] * row[i];

	return static_cast<numericalType>(std::sqrt(abs));
}

template<typename numericalType>
numericalType Matrix<numericalType>::rowAbs(vector<numericalType> row) const
{
	numericalType abs = {};

	// Sum up all the elements squared
#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:abs)
#endif
	for (unsigned i = 0; i < _col; i++)
		abs += row[i] * row[i];

	return static_cast<numericalType>(std::sqrt(abs));
}

template<typename numericalType>
numericalType Matrix<numericalType>::colAbs(const size_t& idx) const
{
	if (idx >= _col)
		throw std::runtime_error("Columns index out of bounds, cannot calculate absolute!");

	numericalType abs = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:abs)
#endif
	for (unsigned i = 0; i < _row; i++)
		abs += matrix[i][idx] * matrix[i][idx];

	return static_cast<numericalType>(std::sqrt(abs));
}

template<typename numericalType>
numericalType Matrix<numericalType>::distanceFromRow(const size_t& distanceFromIdx, const size_t& idx) const
{
	if (distanceFromIdx >= _row || idx >= _row)
		throw std::runtime_error("Cannout calculate distance indexes out of bounds!");

	if (distanceFromIdx == idx) return 0;

	numericalType dist = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:dist)
#endif
	for (unsigned i = 0; i < _col; i++)
		dist += (matrix[distanceFromIdx][i] - matrix[idx][i]) * (matrix[distanceFromIdx][i] - matrix[idx][i]);

	return static_cast<numericalType>(std::sqrt(dist));
}

template<typename numericalType>
numericalType Matrix<numericalType>::distanceFromRow(numericalType* otherROw, const size_t& rowSize, const size_t& distanceFromIdx) const
{
	if (rowSize >= _col)
		throw std::runtime_error("Row too long, cannot compute distance!");

	if (distanceFromIdx >= _row)
		throw std::runtime_error("Cannot calculate distance from row, index out of bounds!");

	numericalType dist = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:dist)
#endif
	for (unsigned i = 0; i < _col; i++)
		dist += (matrix[distanceFromIdx][i] - otherROw[i]) * (matrix[distanceFromIdx][i] - otherROw[i]);

	return static_cast<numericalType>(std::sqrt(dist));
}

template<typename numericalType>
numericalType Matrix<numericalType>::distanceFromRow(const vector<numericalType>& otherRow, const size_t& distanceFromIdx) const
{
	if (otherRow.size() != _col)
		throw std::runtime_error("Vector too long, cannot calculate distance!");

	if (distanceFromIdx >= _row)
		throw std::runtime_error("Cannot calculate distance, index out of bounds!");

	// Copying data
	const unsigned len = otherRow.size();
	numericalType* other_row = new numericalType[len];

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < len; i++)
		other_row[i] = otherRow[i];

	numericalType dist = distanceFromRow(other_row, _col, distanceFromIdx);

	// Free allocated memory
	delete[] other_row;

	return dist;
}

template<typename numericalType>
numericalType Matrix<numericalType>::distanceFromCol(const size_t& distanceFromIdx, const size_t& idx) const
{
	if (distanceFromIdx >= _col || idx >= _col)
		throw std::runtime_error("Cannot calculate distance between rows, since indexes are out of bounds!");

	numericalType dist = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:dist)
#endif
	for (unsigned i = 0; i < _row; i++)
		dist += (matrix[i][distanceFromIdx] - matrix[i][idx]) * (matrix[i][distanceFromIdx] - matrix[i][idx]);

	return static_cast<numericalType>(std::sqrt(dist));
}

template<typename numericalType>
numericalType Matrix<numericalType>::distanceFromCol(numericalType* otherCol, const size_t& colSize, const size_t& distanceFromIdx) const
{
	if (colSize != _row)
		throw std::runtime_error("Distance from column cannot be calculated, vector too long!");

	if (distanceFromIdx >= _col)
		throw std::runtime_error("Cannot calculate distance, since index is out of bounds!");

	numericalType dist = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:dist)
#endif
	for (unsigned i = 0; i < _row; i++)
		dist += (matrix[i][distanceFromIdx] - otherCol[i]);

	return static_cast<numericalType>(std::sqrt(dist));
}

template<typename numericalType>
numericalType Matrix<numericalType>::distanceFromCol(const vector<numericalType>& otherCol, const size_t& distanceFromIdx) const
{
	if (otherCol.size() >= _row)
		throw std::runtime_error("Vector too long, cannot calculate distance!");

	if (distanceFromIdx >= _col)
		throw std::runtime_error("Cannot calculate distance, since index is out of bounds!");

	numericalType* other_col = new numericalType[_row];

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < _row; i++)
		other_col[i] = otherCol[i];

	numericalType dist = distanceFromCol(other_col, _row, distanceFromIdx);

	delete[] other_col;

	return dist;
}

template<typename numericalType>
numericalType Matrix<numericalType>::distanceFromMatrix(const Matrix<numericalType>& other) const
{
	if (_row != other.row() || _col != other.col())
		throw std::runtime_error("Cannot calulate distance between matrices of no same size!");

	numericalType dist = {};

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2) reduction(+:dist)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			dist += (matrix[i][j] - other[i][j]) * (matrix[i][j] - other[i][j]);

	return static_cast<numericalType>(std::sqrt(dist));
}

template<typename numericalType>
bool Matrix<numericalType>::isThisOrthogonal() const
{
	// Iterating through each row, and checking if they are all orthogonal for each other
	for (unsigned i = 0; i < _row - 1; i++)
		for (unsigned j = i + 1; j < _row; j++)
			if (!areOrthogonal(i, j))
				return false;

	// If we didnt return so far, the matrix is orthogonal, and hence returning true
	return true;
}

template<typename numericalType>
bool Matrix<numericalType>::isThisOrthonormed() const
{
	// Check if the matrix is orthogonal
	if (!isThisOrthogonal())
		return false;

	// Check is the vectors, are normed
	for (unsigned i = 0; i < _row; i++)
		if (rowAbs(i) != 1)
			return false;

	// If we didnt return so far, the matrix is orthogonal, and hence returning true
	return true;
}

template<typename numericalType>
void Matrix<numericalType>::push_back(numericalType* row, const size_t& size)
{
	if (size != _col)
		throw std::runtime_error("Cannot push back row, since it is not the same length as others!");

	vector<numericalType> tmpVec;
	for (unsigned i = 0; i < size; i++)
		tmpVec.push_back(row[i]);

	push_back(tmpVec);
}

template<typename numericalType>
void Matrix<numericalType>::push_back(vector<numericalType> row)
{
	if (row.size() != _col)
		throw std::runtime_error("Row is not the same size as other rows, cannot push back!");

	// Allocating and copiing memory
	numericalType* newRow = new numericalType[_col];
	for (unsigned i = 0; i < _col; i++)
		newRow[i] = row[i];

	numericalType** newMatrix = new numericalType * [_row + 1];
	for (unsigned i = 0; i < _row; i++)
		newMatrix[i] = new numericalType[_col];

	// Copy old data
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			newMatrix[i][j] = matrix[i][j];

	// Adding new row
	newMatrix[_row] = newRow;

	// Free allocated memory
	for (unsigned i = 0; i < _row; i++)
		delete[] matrix[i];
	delete[]matrix;

	// Assigning matrix to be the newly allocated and copied and extended matrix
	matrix = newMatrix;

	_row = _row + 1;
}

template<typename numericalType>
void Matrix<numericalType>::printToStdOut() const 
{
	std::cout << "[";
	for (unsigned i = 0; i < this->row(); i++)
	{
		for (unsigned j = 0; j < this->col(); j++)
			std::cout << matrix[i][j] << "\t";
		if (i == this->col() - 1)
		{
			std::cout << "]\n";
			return;
		}
		else
			std::cout << "\n";
	}
}

template<typename numericalType>
numericalType Matrix<numericalType>::trace() const 
{
	if (_row != _col)
		throw std::runtime_error("Trace cannot be calculated for non square matrices!");

	numericalType sum = {};

	// Summing up all elements in the diagonal of the matrix
#ifdef _USING_OMP_
#pragma omp parallel for collapse(2) reduction(+:sum)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			if (i == j)
				sum += matrix[i][j];

	return sum;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::transpose() const
{
	if (_symmetric) return *this;	// If matrix is symmetrix, no need for tranpsoing

	Matrix<numericalType> result(_col, _row); // Note swapped dimensions

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (size_t i = 0; i < _row; ++i) 
		for (size_t j = 0; j < _col; ++j) 
			result[j][i] = matrix[i][j]; // Swap indices for transpose
		
	return result;
}

template<typename numericalType>
void Matrix<numericalType>::setToIdentity()
{
	if (_row != _col)
		throw std::runtime_error("Cannot set non square matrix to identity!");

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
	{
		for (unsigned j = 0; j < _col; j++)
		{
			if (i == j)
				matrix[i][j] = 1;
			else
				matrix[i][j] = {};
		}
	}
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::identity(const size_t& n) const
{
	Matrix<numericalType> retM(n, n);
	retM.setToIdentity();
	return retM;
}

template<typename numericalType>
void Matrix<numericalType>::setToZeroMatrix()
{
#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			matrix[i][j] = {};
}

template<typename numericalType>
void Matrix<numericalType>::setToHomogenous(const numericalType& hom)
{
#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			matrix[i][j] = hom;
}

template<typename numericalType>
void Matrix<numericalType>::luDecomposition(Matrix<numericalType>& L, Matrix<numericalType>& U) const
{
	Matrix<numericalType> deepCpy = *this;	// Creating deep copy of matrix for which from LU decomposition will be performed

	Matrix<numericalType> pL(_row, _row), pU(_row, _row);			// L and U direction matrices

	if (_row != _col)							// Checking for square matrix
		throw std::runtime_error("Cannot perform LU decomposition on non-square matrix!");

	unsigned n = _row;						// n is the size of the nxn matrix

	pL.setToIdentity();						// Setting L to identity, U matrix to zeroes
	pU.setToZeroMatrix();

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < n; i++)
	{
		// Upper triangle
		for (unsigned j = i; j < n; j++)
		{
			numericalType sum = {};
			for (unsigned k = 0; k < i; k++)
				sum += pL[i][k] * pU[k][j];
			
			pU[i][j] = deepCpy[i][j] - sum;
		}
		// Lower triangle
		for (unsigned j = i; j < n; j++)
		{
			if (i == j)
				pL[i][i] = 1;	// Diagonal of L is set to 1
			else
			{
				numericalType sum = {};
				for (unsigned k = 0; k < i; k++)
					sum += pL[j][k] * pU[k][i];

				pL[j][i] = (deepCpy[j][i] - sum) / pU[i][i];
			}
		}
	}

	L = pL;
	U = pU;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::organizeDecreasing() const
{
	// Find leading elements of each row
	numericalType* leadingIndexes = new numericalType[_row];
	for (unsigned i = 0; i < _row; i++)
	{
		unsigned j = 0;
		for (; j < _col; j++)
		{
			if (matrix[i][j] != 0)
			{
				leadingIndexes[i] = static_cast<numericalType>(j);
				break;
			}
		}
		if (j == _col)leadingIndexes[i] = -1;
	}

	unsigned nonZeroRows = 0;
	for (unsigned i = 0; i < _row; i++)
		if (leadingIndexes[i] != -1)
			nonZeroRows++;

	// Now leading indexes look like something like this: { 2, 1, 0, 0, 3, 6, 5, 4, 2, 1, 0}
	// All zero rows should have a -1 value
	// Reorganize matrix based on leading elements
	Matrix<numericalType> organizedMatrix(_row, _col);

	unsigned rowCnt = 0;

	for (unsigned i = 0; i < _row; i++)
	{
		for (unsigned j = 0; j < _row; j++)
		{
			if (rowCnt == nonZeroRows)
				return organizedMatrix;
			if (leadingIndexes[j] == i)
			{
				organizedMatrix.setRow(matrix[j], i, _col);
				rowCnt++;
			}
		}
	}

	organizedMatrix.printToStdOut();

	return organizedMatrix;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::poorGaussian(numericalType* rhs, const size_t& size) const
{
	if (size != _row) throw std::runtime_error("Cannot solve with poor gaussian, vector not right size!");

	// Organize matrix to decreasing order
	Matrix<numericalType> organized = organizeDecreasing();

	// Iterating through leading elements, convertin them to 1s
	for (unsigned i = 0; i < _row; i++)
	{
		// Find the first non zero element, then divide by it
		numericalType divisor = {};
		unsigned j = 0;
		for (; j < _col; j++)
		{
			if (organized[i][j] != 0)
			{
				divisor = organized[i][j];
				break;
			}
		}

		for (; j < _col; j++)
			organized[i][j] /= divisor;

		rhs[i] /= divisor;
	}


	for (unsigned i = 0; i < _row; i++)
	{
		for (unsigned j = 0; j < _col; j++)
		{
			if (organized[i][j] == 1)	// IF found leadin zero, subtracting them for all of the rows above
			{
				for (unsigned k = i; k < _row; k++)
				{
					if (k != i && organized[k][j] != 0)
					{
						numericalType divisor = organized[k][j] / organized[i][j];

						organized.subractRow(k, j, divisor);	// Subtracting rows n times

						rhs[k] = rhs[k] - divisor * rhs[k];		// Subtracting from destionation vector

						break;
					}
				}
			}
		}
	}

	return organized;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::poorGaussian(vector<numericalType> rhs) const
{
	if(rhs.size() != _row) throw std::runtime_error("Cannot solve with poor gaussian, vector not right size!");

	numericalType* tmpRhs = new numericalType[_row];

	for (unsigned i = 0; i < _row; i++)
		tmpRhs[i] = rhs[i];

	Matrix<numericalType> retM = this->poorGaussian(tmpRhs, _row);

	for (unsigned i = 0; i < _row; i++)
		rhs[i] = tmpRhs[i];

	delete[] tmpRhs;

	return retM;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::gaussJordanElimination() const
{
	size_t rowCount = _row;
	size_t colCount = _col;

	Matrix<numericalType> retM(rowCount, colCount);
	for (unsigned i = 0; i < rowCount; i++)
		for (unsigned j = 0; j < colCount; j++)
			retM[i][j] = matrix[i][j];

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t col = 0; col < colCount; ++col) 
	{
		// Step 1: Find a pivot for the current column
		size_t pivotRow = col;
		while (pivotRow < rowCount && retM[pivotRow][col] == 0) 
			++pivotRow;
		if (pivotRow == rowCount) continue; // Skip if no pivot found

		// Step 2: Swap the pivot row with the current row, if necessary
		if (pivotRow != col) 
			retM.swapRows(pivotRow, col);

		// Step 3: Normalize the pivot row
		numericalType pivotValue = retM[col][col];
		for (size_t j = 0; j < colCount; ++j) 
			retM[col][j] /= pivotValue;

		// Step 4: Eliminate all other elements in the current column
		for (size_t i = 0; i < rowCount; ++i) 
		{
			if (i != col) 
			{ // Skip the pivot row itself
				numericalType factor = retM[i][col];
				for (size_t j = 0; j < colCount; ++j) 
					retM[i][j] -= factor * retM[col][j];
			}
		}
	}
	return retM;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::kernel() const 
{
	// The kernel or nullity of the matrix is the #colvectors - #colvectors with leading 1 in rref form
	Matrix<numericalType> rrefMatrix = gaussJordanElimination();

	size_t rowCount = _row;
	size_t colCount = _col;

	Matrix<numericalType> basis(_row, _col);

	// Identify free variables and create basis vectors
#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t col = 0; col < colCount; ++col) 
	{
		bool isFreeVariable = true;
		for (size_t row = 0; row < rowCount && isFreeVariable; ++row) 
		{
			if (rrefMatrix[row][col] == 1)  // This column has a leading 1, so it's not a free variable
				isFreeVariable = false;
		}
		if (isFreeVariable) 
		{
			vector<numericalType> basisVector(colCount, 0);
			basisVector[col] = 1; // Set the free variable to 1
			// Set other entries based on the RREF form to ensure the whole vector is in the kernel
			for (size_t row = 0; row < rowCount; ++row) 
			{
				if (rrefMatrix[row][col] != 0)
				{
					for (size_t otherCol = 0; otherCol < colCount; ++otherCol) 
					{
						if (rrefMatrix[row][otherCol] == 1)
						{
							basisVector[otherCol] = static_cast<numericalType>(-1.0 * rrefMatrix[row][col]);
							break;
						}
					}
				}
			}
			basis.push_back(basisVector);
		}
	}

	return basis;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::image() const 
{
	Matrix<numericalType> rrefMatrix = gaussJordanElimination();

	Matrix<numericalType> basis;

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t row = 0; row < _row; ++row) 
	{
		for (size_t col = 0; col < _col; ++col) 
		{
			if (rrefMatrix[row][col] == 1) 
			{
				// This column in the original matrix forms part of the basis of the image
				basis.push_back(this->getColVector(col));
				break; // Move to the next row since we found our leading 1
			}
		}
	}

	return basis;
}

template<typename numericalType>
unsigned Matrix<numericalType>::rank() const 
{
	// The rank of the matrix is the dimension of the columnspace which
	// is equal to the dimension of the rowspace of the matrix
	// which translates: rank = rank(A) = rank(A^T) = dim(col(A)) = dim(row(A))
	unsigned r = 0;

	// Get the rref form of the matrix
	Matrix<numericalType> rrefM = gaussJordanElimination();

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < _row; i++)
	{
		unsigned j = 0;
		for (; j < _col; j++)
			if (rrefM[i][j] != 0)
				break;
	
		// If the whole row is zeroes then it is not linearly independent
		if (j == _col)
			continue;
		else
			r++;	// Else incrementing rank variable
	}
		
	return r;
}

template<typename numericalType>
numericalType Matrix<numericalType>::determinant() const 
{
	if (_row != _col) 
		throw std::runtime_error("Determinant can only be calculated for square matrices.");
	
	// If the matirx is an upper triangle, then the product of the diagonal elements is the determinant
	if (isUpperTriangle())
	{
		unsigned cntr = 0;
		numericalType det = 1;
		for (unsigned i = 0; i < _row; i++)
		{
			for (unsigned j = 0; j < _col; j++)
			{
				det *= matrix[i][j];
				cntr++;
			}
		}
		det *= (cntr % 2) ? -1 : 1;
		return det;
	}

	Matrix<numericalType> L, U;
	luDecomposition(L, U); // Feltételezve, hogy ez a függvény nem változtatja meg az eredeti mátrixot

	numericalType det = 1;
#ifdef _USING_OMP_
#pragma omp parallel for redustion(*:det)
#endif
	for (size_t i = 0; i < U.row(); ++i) 
		det *= U[i][i];

	return det;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::inverse() const 
{
	if (_row != _col) 
		throw std::runtime_error("Inverse can only be calculated for square matrices.");

	// Step 1: Check if the matrix is invertible (non-zero determinant)
	numericalType det = determinant();
	if (det == 0) 
		throw std::runtime_error("Matrix is not invertible.");

	// Step 2: Form the augmented matrix [this | I]
	size_t n = _row;
	Matrix<numericalType> augmented(n, 2 * n);

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t i = 0; i < n; ++i) 
	{
		for (size_t j = 0; j < n; ++j) 
			augmented[i][j] = matrix[i][j]; // Copy original matrix
		
		augmented[i][n + i] = 1; // Set identity matrix on the right
	}

	// Step 3: Apply Gauss-Jordan Elimination on the augmented matrix
	augmented = augmented.gaussJordanElimination();

	// Step 4: Extract the inverse matrix from the augmented matrix
	Matrix<numericalType> inverse(n, n);

	// After Gauss-elimination the matrix looks like: [I | inverse]
	// Copy the right half
#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (size_t i = 0; i < n; ++i) 
		for (size_t j = 0; j < n; ++j) 
			inverse[i][j] = augmented[i][n + j];

	return inverse;
}

template<typename numericalType>
bool Matrix<numericalType>::isAlike(const Matrix<numericalType>& other) const
{
	// 1. rank(A) = rank(B)
	unsigned rA = rank();
	unsigned rOther = other.rank();

	if (rA != rOther)
		return false;

	// 2. dim(N(A)) = dim(N(B))
	Matrix<numericalType> nullSpaceA = kernel();
	Matrix<numericalType> nullSpaceOther = other.kernel();

	unsigned dimNullA = nullSpaceA.row();
	unsigned dimNullOther = nullSpaceOther.row();

	if (dimNullA != dimNullOther)
		return false;

	// 3. det(A) = det(B)
	numericalType detA = determinant();
	numericalType detOther = other.determinant();

	if (detA != detOther)
		return false;

	// 4. trace(A) = trace(B)
	numericalType traceA = trace();
	numericalType traceB = other.trace();

	if (traceA != traceB)
		return false;

	return true;
}

template<typename numericalType>
bool Matrix<numericalType>::isSymmetric() const
{
	if (_row != _col)
		return false;

	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			if (matrix[i][j] != matrix[j][i] && i != j)
				return false;

	return true;
}

template<typename numericalType>
bool Matrix<numericalType>::isNegativeSymmetric() const
{
	if (_row != _col)return false;

	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			if (i != j && matrix[i][j] != -1 * matrix[j][i])
				return false;
	return true;
}

template<typename numericalType>
bool Matrix<numericalType>::isDiagonal() const
{
	bool notFull = false;
	for (unsigned i = 0; i < _row; i++)
	{
		for (unsigned j = 0; j < _col; j++)
		{
			if (notFull && i == j && matrix[i][j] != 0)	// Check case if diagonal elements contain zero
				return false;
			else if (i != j && matrix[i][j] != 0)	// If element is not on the diagonal returning false
				return false;
			else if (i == j && matrix[i][j] == 0)
				notFull = true;
		}
	}
	return true;
}

template<typename numericalType>
bool Matrix<numericalType>::euclidian() const 
{
	// Check if the matrix is square
	if (_row != _col) 
		return false; // Transformation must not change the dimensionality of the space

	// Calculate A^T (transpose)
	Matrix<numericalType> transpose = this->transpose();

	// Calculate A^T A
	Matrix<numericalType> product = transpose * (*this);

	// Check if A^T A is the identity matrix
	for (unsigned i = 0; i < _row; ++i) 
	{
		for (unsigned j = 0; j < _col; ++j) 
		{
			if (i == j) 
				if (product[i][j] != 1) return false;	// Diagonal elements should be 1
			else 
				if (product[i][j] != 0) return false;	// Off-diagonal elements should be 0
		}
	}

	// Passed all checks, the matrix preserves the Euclidean structure
	return true;
}

template<typename numericalType>
vector<numericalType> Matrix<numericalType>::getNormalVector() const 
{
	// First, transpose the matrix to work with row vectors
	Matrix<numericalType> transposed = this->transpose();

	// Then, find the kernel (null space) of the transposed matrix
	Matrix<numericalType> nullSpaceMatrix = transposed.kernel();

	// Check if the null space is non-trivial (has at least one non-zero vector)
	if (nullSpaceMatrix.row() == 0 || nullSpaceMatrix.col() == 0) 
		throw std::runtime_error("No normal vector found: the null space is trivial.");

	// Return the first non-zero vector in the null space as the normal vector
	vector<numericalType> normalVector(nullSpaceMatrix.col());

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t j = 0; j < nullSpaceMatrix.col(); ++j) 
		normalVector[j] = nullSpaceMatrix[0][j];		// Taking the first row for simplicity

	return normalVector;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::outerProduct(numericalType* vector1, const size_t& size1, numericalType* vector2, const size_t& size2) const
{
	// Ensure the second vector is the transpose of the first
	if (size1 != size2)
		throw std::invalid_argument("Vectors must be of the same size, cannot perform outer product!");

	size_t n = size1;
	Matrix<numericalType> m(n, n);

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			m[i][j] = vector1[i] * vector2[j];

	return m;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::outerProduct(vector<numericalType> vector1, vector<numericalType> vector2) const
{
	// Ensure the second vector is the transpose of the first
	if (vector1.size() != vector2.size()) 
		throw std::invalid_argument("Vectors must be of the same size, cannot perform outer product!");

	size_t n = vector1.size();
	Matrix<numericalType> m(n, n);

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (size_t i = 0; i < n; ++i) 
		for (size_t j = 0; j < n; ++j) 
			m[i][j] = vector1[i] * vector2[j];

	return m;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::projectToStarightLine(numericalType* lineVector, const size_t& size) const
{
	return outerProduct(lineVector, size, lineVector, size);
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::projectToStarightLine(const vector<numericalType> lineVector) const
{
	return outerProduct(lineVector, lineVector);
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::projectToHyperPlane(numericalType* normalVector, const size_t& size) const
{
	Matrix<numericalType> identity(size, size);
	identity.setToIdentity();

	Matrix<numericalType> nnT = outerProduct(normalVector, size, normalVector, size);

	return (identity - nnT);
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::projectToHyperPlane(const vector<numericalType> normalVector) const
{
	Matrix<numericalType> identity(normalVector.size(), normalVector.size());
	identity.setToIdentity();

	return (identity - outerProduct(normalVector, normalVector));
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::mirrorToHyperPlane(numericalType* normalVector, const size_t& size) const
{
	Matrix<numericalType> identity(size, size);
	identity.setToIdentity();

	// normal * normal^transpose
	Matrix<numericalType> nnT = outerProduct(normalVector, size, normalVector, size);
	nnT = nnT * 2;

	return (identity - nnT);
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::mirrorToHyperPlane(const vector<numericalType>& normalVector) const
{
	Matrix<numericalType> identity(normalVector.size(), normalVector.size());
	identity.setToIdentity();

	// normal * normal^transpose
	Matrix<numericalType> nnT = outerProduct(normalVector, normalVector);
	nnT = nnT * 2;

	return (identity - nnT);
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::projectToW() const
{
	Matrix<numericalType> projM = *this;
	Matrix<numericalType> A = *this;
	Matrix<numericalType> transposeA = A.transpose();

	// Projw = A(A^TA)^(-1)A^T
	projM = transposeA * A;			// (A^TA)
	projM = projM.inverse();		// (A^TA)^(-1)
	projM = A * projM;				// A(A^TA)^(-1)
	projM = projM * transposeA;		// A(A^TA)^(-1)A^T

	return projM;
}

template<typename numericalType>
bool Matrix<numericalType>::isIdempotent() const
{
	Matrix<numericalType> P = *this;
	Matrix<numericalType> Psquare = *this;

	// It is idempotent if P^2=P
	Psquare = Psquare * Psquare;

	return (P == Psquare);
}

template<typename numericalType>
bool Matrix<numericalType>::isOrthogonalProjectionMatrix() const
{
	// If the matrix is idempotnet and symmetric it is an orthogonal projection matrix
	// P = P^T = P^2
	return (isIdempotent() && isSymmetric());
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::span() const
{
	// Perform Gauss-Jordan Elimination to get RREF
	Matrix<numericalType> rrefMatrix = this->gaussJordanElimination();

	Matrix<numericalType> basisVectors;
	for (size_t i = 0; i < rrefMatrix.row(); ++i) 
	{
		// Check for pivot elements in each row
		for (size_t j = 0; j < rrefMatrix.col(); ++j) 
		{
			if (rrefMatrix[i][j] == 1)  // Pivot elements are 1
			{	
				// Extract the original column corresponding to the pivot
				std::vector<numericalType> basisVector = this->getColVector(j);
				basisVectors.push_back(basisVector);
				break; // Move to the next row after finding a pivot
			}
		}
	}
	return basisVectors;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::pseudoInverse() const
{
	// If inv(A) exists: A^+=A^(-1)
	if (determinant() != 0) return inverse();

	// If matrix is orthogonal then O^+=O^T
	if (isThisOrthogonal()) return transpose();

	// If the matrix is dagonal and aii != 0
	// Then each element should be rasied to (-1) power
	if (isDiagonal())
	{
		Matrix<numericalType> retM(_row, _col);
		for (unsigned i = 0; i < _row; i++)
			for (unsigned j = 0; j < _col; j++)
				retM[i][j] = 1 / matrix[i][j];
		return retM;
	}

	// If A is real and full column rank, then the pseudoinverse is calcualted
	// A^+=A^T(AA^T)^(-1)
	if (isFullColumnRank())
	{
		Matrix<numericalType> aT = transpose();	// A^T
		Matrix<numericalType> a = *this;		// A
		Matrix pseudoinverse = *this;			// A
		a = a * aT;								// AA^T
		a = a.inverse();						// (AA^T)^(-1)
		pseudoinverse = pseudoinverse * a;		// A(AA^T)^(-1)
		return pseudoinverse;
	}
	else
	{
		std::cerr << "Cannot compute pseudoinverse of given matrix!\n";
		return Matrix(0, 0);
	}

}

template<typename numericalType>
bool Matrix<numericalType>::isSemiOrthogonal() const 
{
	// Calculate A^T * A and A * A^T
	Matrix<numericalType> transpose = this->transpose();
	Matrix<numericalType> AtA = transpose;
	AtA = AtA * (*this);
	Matrix<numericalType> AAt = (*this);
	AAt = AAt * transpose;

	// Check if A^T * A is the identity matrix
	bool AtAisIdentity = true;
	for (size_t i = 0; i < AtA.row(); ++i) 
	{
		for (size_t j = 0; j < AtA.col(); ++j) 
		{
			numericalType abs = static_cast<numericalType>(std::fabs(static_cast<double>(AtA[i][j])));
			if (i == j &&  ((abs - 1) > 1e-6))
			{
				AtAisIdentity = false;
				break;
			}
			else if (i != j && (abs > 1e-6))
			{
				AtAisIdentity = false;
				break;
			}
		}
		if (!AtAisIdentity) break;
	}

	// Check if A * A^T is the identity matrix
	bool AAtisIdentity = true;
	for (size_t i = 0; i < AAt.row(); ++i) 
	{
		for (size_t j = 0; j < AAt.col(); ++j) 
		{
			numericalType abs = static_cast<numericalType>(std::fabs(static_cast<double>(AtA[i][j])));
			if (i == j && ((abs - 1) > 1e-6))
			{
				AAtisIdentity = false;
				break;
			}
			else if (i != j && (abs > 1e-6))
			{
				AAtisIdentity = false;
				break;
			}
		}
		if (!AAtisIdentity) break;
	}

	// The matrix is semi-orthogonal if either A^T*A or A*A^T is the identity matrix
	return AtAisIdentity || AAtisIdentity;
}

template<typename numericalType>
void Matrix<numericalType>::qrDecomposition(Matrix<numericalType>& Q, Matrix<numericalType>& R) const
{
	size_t n = _col;
	size_t m = _row;

	// Initialize Q and R
	Q = Matrix<numericalType>(m, n);
	R = Matrix<numericalType>(n, n);

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t i = 0; i < n; ++i) 
	{
		vector<numericalType> ai = this->getColVector(i);

		// Orthogonalization
		vector<numericalType> ui = ai;
		for (size_t j = 0; j < i; ++j) 
		{
			vector<numericalType> qj = Q.getColVector(j);
			numericalType dot = dotProduct(qj, ai);
			R[j][i] = dot;
			for (size_t k = 0; k < m; ++k) 
				ui[k] -= dot * qj[k];
		}

		// Normalization
		numericalType norm = static_cast<numericalType>(sqrt(dotProduct(ui, ui)));
		R[i][i] = norm;
		for (size_t k = 0; k < m; ++k) 
			Q[k][i] = ui[k] / norm;
	}
}

template<typename numericalType>
void Matrix<numericalType>::reducedQRDecomposition(Matrix<numericalType>& Q, Matrix<numericalType>& R) const 
{
	// Determine the dimensions of the original matrix
	size_t m = _row;
	size_t n = _col;
	size_t r = rank();

	// Initialize Q and R matrices
	// Q: Orthogonal matrix with dimensions m x r (columns are orthonormal vectors)
	Q = Matrix<numericalType>(m, r);
	// R: Upper triangular matrix with dimensions r x n
	R = Matrix<numericalType>(r, n);

	// Apply the Gram-Schmidt process to orthogonalize the columns of the original matrix
#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t i = 0; i < r; ++i)  // Iterate up to the rank 'r'
	{
		vector<numericalType> ai = this->getColVector(i); // Get the i-th column of A

		// Step 1: Orthogonalization
		// Compute the orthogonal component 'ui' of 'ai' relative to previous vectors
		vector<numericalType> ui = ai; // Start with 'ai' itself
		for (size_t j = 0; j < i; ++j)  // Subtract projection onto each previously computed 'qj'
		{
			vector<numericalType> qj = Q.getColVector(j); // j-th column of Q
			numericalType dot = dotProduct(qj, ai); // Project 'ai' onto 'qj'
			// Subtract this projection from 'ui'
			for (size_t k = 0; k < m; ++k) 
				ui[k] -= dot * qj[k];
		}

		// Step 2: Normalization
		// Normalize 'ui' to form the i-th column of Q
		numericalType norm = static_cast<numericalType>(sqrt(dotProduct(ui, ui))); // Compute the norm of 'ui'
		for (size_t k = 0; k < m; ++k) 
			Q[k][i] = ui[k] / norm; // Normalize and set as i-th column of Q

		// Step 3: Construct R
		// Now that 'qi' is computed, fill the corresponding row of R
		for (size_t j = i; j < n; ++j)  // Only fill upper triangular part
		{
			vector<numericalType> aj = this->getColVector(j); // j-th column of A
			// Set R[i][j] as the dot product of 'qi' and 'aj'
			R[i][j] = dotProduct(Q.getColVector(i), aj);
		}
	}
}

template<typename numericalType>
vector<numericalType> Matrix<numericalType>::eigenvaluesVector(int maxIterations, numericalType tol) const
{
	if (_row != _col) 
		throw std::runtime_error("Matrix must be square to compute eigenvalues.");

	size_t n = _row;
	Matrix<numericalType> A = *this; // Copy of the matrix, as the process is destructive
	Matrix<numericalType> Q(n, n), R(n, n);

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (int iter = 0; iter < maxIterations; ++iter) 
	{
		// QR decomposition of A
		A.qrDecomposition(Q, R); // Ensure this modifies Q and R correctly

		// Form the next matrix A = RQ
		A = R * Q;

		// Check for convergence (simplified check: if off-diagonal elements are below tolerance)
		bool converged = true;
		for (size_t i = 0; i < n && converged; ++i) 
		{
			for (size_t j = 0; j < n; ++j) 
			{
				if (i != j && (static_cast<numericalType>(std::fabs(static_cast<double>(A[i][j]))) > tol))
				{
					converged = false;
					break;
				}
			}
		}

		if (converged) 
			break;
	}

	// Extract the diagonal elements as the eigenvalues
	std::vector<numericalType> eigenvalues(n);
	for (size_t i = 0; i < n; ++i) 
		eigenvalues[i] = A[i][i];

	return eigenvalues;
}

template<typename numericalType>
numericalType* Matrix<numericalType>::eigenvalues(int maxIterations, numericalType tol) const
{
	vector<numericalType> retVec = eigenvaluesVector(maxIterations, tol);

	// Copiing each element
	numericalType* ret = new numericalType[_row];

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < _row; i++)
		ret[i] = retVec[i];

	return ret;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::eigenvectors() const
{
	vector<numericalType> eigenvalues = eigenvaluesVector();
	Matrix<numericalType> eigenvectors(_row, eigenvalues.size());

	Matrix<numericalType> I(_row, _row);
	I.setToIdentity();

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t i = 0; i < eigenvalues.size(); ++i) 
	{
		numericalType lambda = eigenvalues[i];
		Matrix<numericalType> lambdaI = I * lambda;
		Matrix<numericalType> A_lambdaI = *this; // A - lambda*I
		A_lambdaI -= lambdaI;

		// Method to solve (A - lambda*I)x = b using, e.g., LU decomposition
		std::vector<numericalType> b(_row, 1); // Initial guess vector
		std::vector<numericalType> x = A_lambdaI.eigenvaluesVector();

		// Normalize x
		numericalType norm = static_cast<numericalType>(sqrt(std::accumulate(x.begin(), x.end(), 0.0, [](numericalType acc, numericalType xi) { return acc + xi * xi; })));
		std::transform(x.begin(), x.end(), x.begin(), [norm](numericalType xi) { return xi / norm; });

		// Set x as a column in the eigenvectors matrix
		for (size_t j = 0; j < this->row(); ++j) 
			eigenvectors[j][i] = x[j];
	}

	return eigenvectors;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::getSubMatrix(const size_t& rowEnd, const size_t& colEnd) const
{
	if (rowEnd >= _row || colEnd >= _col)
		throw std::runtime_error("Cannot return submatrix, indexes out of bounds!");

	Matrix<numericalType> subM(rowEnd, colEnd);

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < rowEnd; i++)
		for (unsigned j = 0; j < colEnd; j++)
			subM[i][j] = matrix[i][j];

	return subM;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::getSubMatrix(const size_t& rowStart, const size_t& rowEnd, const size_t& colStart, const size_t& colEnd) const
{
	if (rowStart < 0 || colStart < 0)
		throw std::runtime_error("Cannot index matrix with negative indexes, hence cannot return submatrix!");

	if (rowEnd >= _row || colEnd >= _col)
		throw std::runtime_error("Cannot return submatrix indexes out of bounds!");

	Matrix<numericalType> subM(rowEnd - rowStart, colEnd - colStart);

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = rowStart; i < rowEnd; i++)
		for (unsigned j = colStart; j < colEnd; j++)
			subM[i - rowStart][j - colStart] = matrix[i][j];

	return subM;
}

template<typename numericalType>
numericalType Matrix<numericalType>::getLeadingPrincipalMinor(size_t k) const
{
	if (k > _row || k > _col)
		throw std::invalid_argument("k is out of bounds, cannot return leading principal minor!");
	
	// Extract the submatrix from the top-left corner up to (k, k)
	Matrix sub = getSubMatrix(k, k);

	// Calculate and return the determinant of this submatrix
	return sub.determinant();
}

template<typename numericalType>
bool Matrix<numericalType>::isPositiveDefinite() const 
{
	if (_row != _col) 
		return false;	// Matrix must be square
	
	for (size_t k = 1; k <= _row; ++k) 
		if (this->getLeadingPrincipalMinor(k) <= 0) 
			return false;	// If any leading principal minor is not positive, the matrix is not positive definite
	
	return true; // All leading principal minors are positive
}

template<typename numericalType>
bool Matrix<numericalType>::isPositiveSemiDefinite() const 
{
	if (_row != _col) 
		return false;		// Matrix must be square
	
	for (size_t k = 1; k <= _row; ++k) 
		if (this->getLeadingPrincipalMinor(k) < 0) 
			return false;	// If any leading principal minor is negative, the matrix is not positive semi-definite
		
	return true; // All leading principal minors are non-negative
}

template<typename numericalType>
vector<pair<numericalType, numericalType*>> Matrix<numericalType>::ownEigenPairs() const
{
	Matrix<numericalType> eigVectors = eigenvectors();
	numericalType* eigValVector = eigenvalues();

	vector<pair<numericalType, numericalType*>> retVec;
	
	for (unsigned i = 0; i < eigVectors.row(); i++)
	{
		pair<numericalType, numericalType*> retPair;
		retPair.first = eigValVector[i];
		retPair.second = eigVectors[i];
		retVec.push_back(retPair);
	}

	return retVec;
}

template<typename numericalType>
vector<pair<numericalType, vector<numericalType>>> Matrix<numericalType>::ownEigenPairsVector() const
{
	Matrix<numericalType> eigVectors = eigenvectors();
	vector<numericalType> eigValVector = eigenvaluesVector();

	vector<pair<numericalType, vector<numericalType>>> retVec;

	for (unsigned i = 0; i < eigVectors.row(); i++)
	{
		pair<numericalType, vector<numericalType>> retPair;
		retPair.first = eigValVector[i];
		
		vector<numericalType> currentEigVec;

		for (unsigned j = 0; j < eigVectors.row(); j++)
			currentEigVec[j] = eigVectors[i][j];

		retPair.second = currentEigVec;
		retVec.push_back(retPair);

	}

	return retVec;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::characteristics() const
{
	if (_row != _col)
		throw std::runtime_error("Cannot calulate characteristics for non square matrix!");

	Matrix<numericalType> identity(_row, _row);
	identity.setToIdentity();

	numericalType* eigValues = eigenvalues();

	// Creating lambda*I
	for (unsigned i = 0; i < _row; i++)
		identity.scalarMultiplyRow(eigValues[i], i);

	delete[] eigValues;

	// Calculating A - lambda*I
	Matrix<numericalType> A = *this;
	A -= identity;

	return A;
}

template<typename numericalType>
bool Matrix<numericalType>::isUpperTriangle() const
{
	// Iterating through lower triangle to check if there are any elements
	for (unsigned i = 1; i < _row; i++)
		for (unsigned j = 0; j < i; j++)
			if (matrix[i][j] != 0)return false;

	return true;
}

template<typename numericalType>
bool Matrix<numericalType>::isLowerTriangle() const
{
	// Iterating through upper triangle, to check for non zero elements
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 1; j < _col; j++)
			if (matrix[i][j] != 0)return false;

	return true;
}

template<typename numericalType>
bool Matrix<numericalType>::isNilPotent() const
{
	vector<numericalType> eigVals = eigenvaluesVector();
	for (unsigned i = 0; i < eigVals.size(); i++)
		if (eigVals[i] != 0)	// Check if there is an eigen values which is non zero
			return false;

	return true;
}

template<typename numericalType>
bool Matrix<numericalType>::canBeDiagonalized() const
{
	Matrix<numericalType> eigVectors = eigenvectors();
	size_t n = eigVectors.row();

	unsigned maxLinearlyIndependentCount = 0;
	unsigned linearlyIndependentCount = 0;

	for (unsigned i = 0; i < n - 1; i++)
	{
		for (unsigned j = i + 1; j < n; j++)
		{
			if (dotProduct(eigVectors[i], n, eigVectors[j], n) == 0)
				linearlyIndependentCount++;
		}
		if (maxLinearlyIndependentCount < linearlyIndependentCount)
			maxLinearlyIndependentCount = linearlyIndependentCount;
		linearlyIndependentCount = 0;
	}

	return maxLinearlyIndependentCount == _row;
}

template<typename numericalType>
void Matrix<numericalType>::applySVD(Matrix<numericalType>& U, Matrix<numericalType>& Sigma, Matrix<numericalType>& VT) const
{

}

template<typename numericalType>
numericalType Matrix<numericalType>::frobeniusNorm() const 
{
	numericalType sum = 0;

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2) reduction(+:sum)
#endif
	for (size_t i = 0; i < _row; ++i) 
		for (size_t j = 0; j < _col; ++j) 
			sum += matrix[i][j] * matrix[i][j];
		
	return static_cast<numericalType>(std::sqrt(sum));
}

template<typename numericalType>
numericalType Matrix<numericalType>::l1Norm() const 
{
	numericalType maxSum = 0;

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t j = 0; j < _col; ++j) 
	{
		numericalType sum = 0;
		for (size_t i = 0; i < _row; ++i) 
			sum += static_cast<numericalType>(std::fabs(static_cast<double>(matrix[i][j])));
		
		if (sum > maxSum) 
			maxSum = sum;
		
	}
	return maxSum;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::matrixExponential(const size_t& truncSize) const 
{
	if (_row != _col) 
		throw std::runtime_error("Matrix must be square., cannot raise to power e^A!");

	Matrix<numericalType> result(_row, _col);
	result.setToIdentity(); // Start with the identity matrix

	Matrix<numericalType> term(_row, _col);
	term.setToIdentity(); // First term is also I

	size_t factorial = 1;
	for (size_t n = 1; n <= truncSize; ++n) // Truncated at 20 terms for simplicity, can be changed when calling
	{ 
		factorial *= n;
		term = term * (*this) / static_cast<numericalType>(factorial); // Each term A^n / n!
		result = result + term;
	}

	return result;
}

template<typename numericalType>
size_t Matrix<numericalType>::lowerBandwidth() const
{
	size_t lower = 0;

	for (size_t i = 0; i < _row; ++i) 
	{
		for (size_t j = 0; j < _col; ++j) 
		{
			if (matrix[i][j] != 0) 
			{  
				if (i > j) 
					lower = std::max(lower, i - j);
			}
		}
	}

	return lower;
}

template<typename numericalType>
size_t Matrix<numericalType>::upperBandwidht() const
{
	size_t upper = 0;

	for (size_t i = 0; i < _row; ++i)
	{
		for (size_t j = 0; j < _col; ++j)
		{
			if (matrix[i][j] != 0)
			{
				if (i > j)
					upper = std::max(upper, j - i);
			}
		}
	}

	return upper;
}

template<typename numericalType>
size_t Matrix<numericalType>::bandwidth() const
{
	size_t upper = upperBandwidht();
	size_t lower = lowerBandwidth();
	return (upper > lower) ? upper : lower;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::choleskyDecomposition() const 
{
	if (_row != _col) 
		throw std::runtime_error("Matrix must be square.");
	
	if (!isSymmetric()) throw std::runtime_error("Cannot perfrom Cholesky decomposition for non symmetric matrix!");
	if (!isPositiveSemiDefinite()) throw std::runtime_error("Cannot perform Cholesky decomposition for non positive semi definite matrix!");

	Matrix<numericalType> L(_row, _col);

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t i = 0; i < _row; ++i) 
	{
		for (size_t j = 0; j <= i; ++j) 
		{
			numericalType sum = 0;
			for (size_t k = 0; k < j; ++k) 
				sum += L[i][k] * L[j][k];
	
			if (i == j) 
				L[i][j] = static_cast<numericalType>(std::sqrt(matrix[i][i] - sum));
			else 
				L[i][j] = (matrix[i][j] - sum) / L[j][j];
		}
	}

	return L;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::ownSubSpace() const
{
	Matrix<numericalType> eigVec = eigenvectors();

	// Creating zero vector
	vector<numericalType> zero;
	for (unsigned i = 0; i < _col; i++)
		zero.push_back({});

	// Adding zero vector
	eigVec.push_back(zero);
	return eigVec;
}

template<typename numericalType>
void Matrix<numericalType>::diagonalize(Matrix<numericalType>& P, Matrix<numericalType>& D) const
{
	// Step 1: Compute eigenvalues and eigenvectors
	auto eigenPairs = ownEigenPairsVector(); 

	// Not enough independent eigenvectors to diagonalize
	if (eigenPairs.size() < _row)
	{
		std::cerr << "Cannot diagonalize matrix, not enough independent eigenvectors!\n";
		return;	
	}

	// Step 2: Form the matrix P with eigenvectors as columns
	P = Matrix<numericalType>(_row, _row); // Initialize P with the appropriate size
	for (size_t i = 0; i < eigenPairs.size(); ++i) 
		P.setCol(i, eigenPairs[i].second); // Sets the i-th col of P to given eigenvector

	// Step 3: Form the diagonal matrix D with eigenvalues
	D = Matrix<numericalType>(_row, _row); // Initialize D with zeroes

	for (size_t i = 0; i < eigenPairs.size(); ++i) 
		D[i][i] = eigenPairs[i].first; // Set the diagonal elements to be the eigenvalues

	return; // The matrix was successfully diagonalized
}

template<typename numericalType>
bool Matrix<numericalType>::isColumnVector(const vector<numericalType>& vec) const
{
	if (vec.size() != _row)
		throw std::runtime_error("Cannot compare rows, since the given vector is not the right size!");

	// Check if the given vector is 1:1 in the column space
	for (unsigned i = 0; i < _row; i++)
	{
		if (matrix[0][i] != vec[0])continue;
		else
		{
			unsigned j = 0;
			for (; j < _col; j++)
				if (matrix[j][i] != vec[j]) break;

			if (j == _col)return true;
		}
	}

	// Checki if the given vector is scaled in the columns space
	for (unsigned i = 0; i < _row; i++)
	{
		numericalType commonFactor = vec[0] / matrix[0][i];
		unsigned j = 0;
		for (; j < _col; j++)
			if (vec[j] * commonFactor != matrix[i][j])break;
		
		if (j == _col) return true;
	}
	return false;
}

template<typename numericalType>
bool Matrix<numericalType>::isColumnVector(numericalType* vec, const size_t& size) const
{
	if (size != _col)
		throw std::runtime_error("Vector size doesnt match, cannot compare columns!");
	vector<numericalType> col;
	for (unsigned i = 0; i < size; i++)
		col.push_back(vec[i]);

	return isColumnVector(col);
}

template<typename numericalType>
bool Matrix<numericalType>::isRowVector(const vector<numericalType>& vec) const
{
	if (vec.size() != _row)
		throw std::runtime_error("Cannot compare rows, vector size doesnt match!");

	// Check if the row is 1:1 
	for (unsigned i = 0; i < _row; i++)
	{
		if (vec[i] != matrix[i][0]) break;
		else
		{
			unsigned j = 0;
			for (; j < _col; j++)
				if (matrix[i][j] != vec[j])
					break;
			if (j == _col) return true;
		}
	}
	
	// Check if the given row is a scalar multiple of other rows
	for (unsigned i = 0; i < _row; i++)
	{
		numericalType commonFactor = vec[0] / matrix[i][0];
		unsigned j = 0;
		for (; j < _col; j++)
			if (vec[j] * commonFactor != matrix[i][j])
				break;
		
		if (j == _col)return true;
	}
	return false;
}

template<typename numericalType>
bool Matrix<numericalType>::isRowVector(numericalType* vec, const size_t& size) const
{
	if (size != _col)
		throw std::runtime_error("Cannot compare rows, since vector not the same size!");
	vector<numericalType> rowVec;
	for (unsigned i = 0; i < size; i++)
		rowVec.push_back(vec[i]);

	return isRowVector(rowVec);
}

template<typename numericalType>
bool Matrix<numericalType>::isLinearlyIndependent(const size_t& row1Idx, const size_t& row2Idx) const
{
	if (row1Idx >= _row || row2Idx >= _row)
		throw std::runtime_error("Cannot determine linear independency, row index out of bounds!");

	return isLinearlyIndependent(matrix[row1Idx], _col, matrix[row2Idx], _col);
}

template<typename numericalType>
bool Matrix<numericalType>::isLinearlyIndependent(numericalType* row, const size_t& size, const size_t& rowIdx) const
{
	if (size != _col)
		throw std::runtime_error("Cannot determine linear independency of different size vectors!");
	if (rowIdx >= _row)
		throw std::runtime_error("Cannot determine linear independency, row index out of bounds!");

	return isLinearlyIndependent(row, size, matrix[rowIdx], _col);
}

template<typename numericalType>
bool Matrix<numericalType>::isLinearlyIndependent(const vector<numericalType>& row, const size_t& rowIdx) const
{
	if (row.size() != _col)
		throw std::runtime_error("Cannot determine linear independency of different size vectors!");
	if (rowIdx >= _row)
		throw std::runtime_error("Cannot determine linear independency, row index out of bounds!");

	numericalType* row2 = matrix[rowIdx];

	vector<numericalType> r2;
	for (unsigned i = 0; i < _col; i++)
		r2.push_back(row2[i]);
	
	return isLinearlyIndependent(row, r2);
}

template<typename numericalType>
bool Matrix<numericalType>::isLinearlyIndependent(numericalType* row1, const size_t& row1Size, numericalType* row2, const size_t& row2Size) const
{
	if(row1Size != row2Size)
		throw std::runtime_error("Cannot determine linear independency of different size vectors!");

	vector<numericalType> r1, r2;
	for (unsigned i = 0; i < row1Size; i++)
	{
		r1.push_back(row1[i]);
		r2.push_back(row2[i]);
	}

	return isLinearlyIndependent(r1, r2);
}

template<typename numericalType>
bool Matrix<numericalType>::isLinearlyIndependent(const vector<numericalType>& row1, const vector<numericalType>& row2) const
{
	if (row1.size() != row2.size())
		throw std::runtime_error("Cannot determine linear independency of different size vectors!");

	const unsigned n = row1.size();
	unsigned i = 0;

	// Getting common divisor
	numericalType commonDivisor = {};
	for (; i < n; i++)
	{
		if (row1[i] != 0)
		{
			if (row2[i] != 0)
			{
				commonDivisor = row1[i];
				break;
			}
			else return false;
		}
	}
	numericalType commonRemainder = row2[0] / commonDivisor;

	for (i = 1; i < n; i++)
		if (row2[i] / commonDivisor != commonRemainder)	// If r2[i] mod divisor != const 
			return false;
	return true;
}

template<typename numericalType>
double Matrix<numericalType>::angleBetween(const size_t& row1Idx, const size_t& row2Idx) const
{
	return angleBetween(matrix[row1Idx], _col, matrix[row2Idx], _col);
}

template<typename numericalType>
double Matrix<numericalType>::angleBetween(numericalType* row, const size_t& size, const size_t& rowIdx) const
{
	return angleBetween(row, size, matrix[rowIdx], _col);
}

template<typename numericalType>
double Matrix<numericalType>::angleBetween(const vector<numericalType>& row, const size_t& rowIdx) const
{
	if(row.size() != _col)
		throw std::runtime_error("Cannot calculate angle, vectors are not the same size!");

	numericalType* r = new numericalType[_col];
	for (unsigned i = 0; i < _col; i++)
		r[i] = row[i];

	double result = angleBetween(r, _col, matrix[rowIdx], _col);

	delete[] r;

	return result;
}

template<typename numericalType>
double Matrix<numericalType>::angleBetween(numericalType* row1, const size_t& size1, numericalType* row2, const size_t& size2) const
{
	if (size1 != size2)
		throw std::runtime_error("Cannot calculate angle, vectors are not the same size!");

	double cosTheta = 0.0f;

	// Calculating a*b
#ifdef _USING_OMP_
#pragma omp parallel for redution(+:cosTheta)
#endif
	for (unsigned i = 0; i < size1; i++)
		cosTheta += static_cast<double>(row1[i] * row2[i]);

	// Calculating ||a||*||b||
	cosTheta /= (rowAbs(row1, size1) * rowAbs(row2, size2));

	return std::acos(cosTheta);
}

template<typename numericalType>
double Matrix<numericalType>::angleBetween(const vector<numericalType>& row1, const vector<numericalType>& row2) const
{
	if(row1.size() != row2.size())
		throw std::runtime_error("Cannot calculate angle, vectors are not the same size!");

	unsigned n = row1.size();

	numericalType* r1 = new numericalType[n];
	numericalType* r2 = new numericalType[n];

	for (unsigned i = 0; i < n; i++)
	{
		r1[i] = row1[i];
		r2[i] = row2[i];
	}

	double result = angleBetween(r1, n, r2, n);

	delete[] r1;
	delete[] r2;

	return result;
}

template<typename numericalType>
vector<numericalType> Matrix<numericalType>::projectToVector(const vector<numericalType>& projectThisTo, const vector<numericalType>& toThis) const
{
	numericalType* proj = projectTo(projectThisTo, toThis);
	vector<numericalType> projVec;
	for (unsigned i = 0; i < toThis.size(); i++)
		projVec.push_back(proj[i]);
	delete[] proj;
	return projVec;
}

template<typename numericalType>
vector<numericalType> Matrix<numericalType>::projectToVector(numericalType* projectThisTo, const size_t& size1, numericalType* toThis, const size_t& size2) const
{
	numericalType* proj = projectTo(projectThisTo, size1, toThis, size2);
	vector<numericalType> projVec;
	for (unsigned i = 0; i < size2; i++)
		projVec.push_back(proj[i]);
	delete[] proj;
	return projVec;
}

template<typename numericalType>
numericalType* Matrix<numericalType>::projectTo(const vector<numericalType>& projectThisTo, const vector<numericalType>& toThis) const
{
	size_t size1 = projectThisTo.size();
	size_t size2 = toThis.size();

	if(size1 != size2)
		throw std::runtime_error("Cannot project vectors of different sizes!");

	numericalType* ptt = new numericalType[size1];
	numericalType* tt = new numericalType[size2];

	for (unsigned i = 0; i < size1; i++)
	{
		ptt[i] = projectThisTo[i];
		tt[i] = toThis[i];
	}

	numericalType* retVec = projectTo(ptt, size1, tt, size2);

	delete[] ptt;
	delete[] tt;

	return retVec;
}

template<typename numericalType>
numericalType* Matrix<numericalType>::projectTo(numericalType* projectThisTo, const size_t& size1, numericalType* toThis, const size_t& size2) const
{
	if (size1 != size2)
		throw std::runtime_error("Cannot project vectors of different sizes!");

	// Calculating u*v
	numericalType upper = 0;

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:upper)
#endif
	for (unsigned i = 0; i < size1; i++)
		upper += toThis[i] * projectThisTo[i];

	// Calculating u*u
	numericalType lower = 0;
#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:lower)
#endif
	for (unsigned i = 0; i < size1; i++)
		lower += toThis[i] * toThis[i];

	numericalType scalar = upper / lower;
	// Calculating (u*v)/(u*u) * u
	numericalType* projected = new numericalType[size1];

#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (unsigned i = 0; i < size1; i++)
		projected[i] = scalar * toThis[i];

	return projected;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::gramSchmidAlgorithm() const
{
	// Create a copy of the matrix to work on
	Matrix<numericalType> orthoMatrix(*this);
	size_t numRows = orthoMatrix.row();
	size_t numCols = orthoMatrix.col();

	// Process each row
#ifdef _USING_OMP_
#pragma omp parallel for
#endif
	for (size_t i = 0; i < numRows; ++i) 
	{
		vector<numericalType> vi = orthoMatrix.getRowVector(i);

		// Subtract the projection of vi onto all previous rows vj
		for (size_t j = 0; j < i; ++j) 
		{
			vector<numericalType> vj = orthoMatrix.getRowVector(j);
			vector<numericalType> proj = projectToVector(vi, vj); // ProjectToVector returns the projection of vi onto vj
			vi = subtractRowVector(vi, proj); // SubtractRowVector subtracts the second vector from the first and returns the result
		}

		// Normalize the result to make it a unit vector
		numericalType norm = static_cast<numericalType>(std::sqrt(dotProduct(vi, vi))); // dotProduct calculates the dot product of the vector with itself
		for (auto& element : vi) 
			element /= norm;

		// Set the processed row back into the matrix
		orthoMatrix.setRow(vi, i);
	}

	return orthoMatrix;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::leastSquares(const vector<numericalType>& b) const
{
	if (b.size() != _row)
		throw std::runtime_error("Cannot solve least sqaures problem, matrix size doesnt match vector size!");

	size_t n = b.size();

	numericalType* bArr = new numericalType[n];
	for (unsigned i = 0; i < n; i++)
		bArr[i] = b[i];

	Matrix<numericalType> sol = leastSquares(bArr, n);

	delete[] bArr;

	return sol;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::leastSquares(numericalType* b, const size_t& size) const
{
	if (size != _row) 
		throw std::invalid_argument("Size of b must match the number of rows in the matrix.");

	// We want to solve the A^TAx=A^Tb

	// Step 1: Convert the 'b' array into a matrix
	Matrix<numericalType> B(_row, 1);
	for (size_t i = 0; i < size; ++i) 
		B[i][0] = b[i];

	// Step 2: Compute A^T
	Matrix<numericalType> At = this->transpose();

	// Step 3: Compute A^TA
	Matrix<numericalType> AtA = At * (*this);

	// Step 4: Compute the inverse of A^TA
	Matrix<numericalType> AtA_inv = AtA.inverse(); // Assuming inverse() method is implemented and handles singularity

	// Step 5: Compute A^TB
	Matrix<numericalType> AtB = At * B;

	// Step 6: Compute the pseudoinverse of A: (A^TA)^{-1}A^T
	Matrix<numericalType> pseudoinverse = AtA_inv * At;

	// Step 7: Compute the least squares solution: x = pseudoinverse * B
	Matrix<numericalType> x = pseudoinverse * B;

	return x;
}

template<typename numericalType>
numericalType Matrix<numericalType>::mean() const
{
	if (_col == 0 || _row == 0)
		throw std::runtime_error("Cannot calulate mean for matrix with 0 dimesnsions!");

	numericalType _mean = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:mean)
#endif
	for (unsigned i = 0; i < _row; i++)
		_mean += meanRow(i);

	return _mean / _row;
}

template<typename numericalType>
numericalType Matrix<numericalType>::meanRow(const size_t& rowIdx) const
{
	if (rowIdx > _row)
		throw std::runtime_error("Cannot calulate mean for row, since index is out of bounds!");

	if (_row == 0)
		throw std::runtime_error("Cannot calculate mean, if row dimension is 0.");

	numericalType mean = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:mean)
#endif
	for (unsigned i = 0; i < _col; i++)
		mean += matrix[rowIdx][i];

	return mean / _col;
}

template<typename numericalType>
numericalType Matrix<numericalType>::meanCol(const size_t& colIdx) const
{
	if (colIdx > _col)
		throw std::runtime_error("Cannot calulate mean for col, since index is out of bounds!");

	if (_col == 0)
		throw std::runtime_error("Cannot calculate mean, if row dimension is 0.");

	numericalType mean = {};

#ifdef _USING_OMP_
#pragma omp parallel for reduction(+:mean)
#endif
	for (unsigned i = 0; i < _row; i++)
		mean += matrix[i][colIdx];

	return mean / _row;
}

template<typename numericalType>
numericalType Matrix<numericalType>::max() const
{
	numericalType _max = std::numeric_limits<numericalType>::min();

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			if (_max < matrix[i][j])
				_max = matrix[i][j];
	return _max;
}

template<typename numericalType>
pair<size_t, size_t> Matrix<numericalType>::maxIdx() const
{
	size_t iIdx = 0, jIdx = 0;
	numericalType _max = std::numeric_limits<numericalType>::min();

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
	{
		for (unsigned j = 0; j < _col; j++)
		{
			if (_max < matrix[i][j])
			{
				_max = matrix[i][j];
				iIdx = i;
				jIdx = j;
			}
		}
	}
	pair<size_t, size_t> retPair;
	retPair.first = iIdx;
	retPair.second = jIdx;
	return retPair;
}

template<typename numericalType>
numericalType Matrix<numericalType>::min() const
{
	numericalType _min = std::numeric_limits<numericalType>::max();

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			if (_min < matrix[i][j])
				_min = matrix[i][j];
	return _min;
}

template<typename numericalType>
pair<size_t, size_t> Matrix<numericalType>::minIdx() const
{
	size_t iIdx = 0, jIdx = 0;
	numericalType _min = std::numeric_limits<numericalType>::max();

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
	{
		for (unsigned j = 0; j < _col; j++)
		{
			if (_min > matrix[i][j])
			{
				_min = matrix[i][j];
				iIdx = i;
				jIdx = j;
			}
		}
	}
	pair<size_t, size_t> retPair;
	retPair.first = iIdx;
	retPair.second = jIdx;
	return retPair;
}

template<typename numericalType>
pair<size_t, size_t> Matrix<numericalType>::find(const numericalType& f) const
{
	pair<size_t, size_t> retPair;
	retPair.first = -1;
	retPair.second = -1;

	for (size_t i = 0; i < _row; i++)
	{
		for (size_t j = 0; j < _col; j++)
		{
			if (matrix[i][j] == f)	// If number is found then returning the first instance
			{
				retPair.first = i;
				retPair.second = j;
				return retPair;
			}
		}
	}

	// Returning {-1, -1}
	return retPair;
}

template<typename numericalType>
unsigned Matrix<numericalType>::count(const numericalType& num) const
{
	unsigned cnt = 0;
#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
	for (unsigned i = 0; i < _row; i++)
		for (unsigned j = 0; j < _col; j++)
			if (matrix[i][j] == num)
				cnt++;
	return cnt;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::rand(const size_t& rowCnt, const size_t& colCnt, const double& lowerLimit, const double& upperLimit) const
{
	// Uses constexpr, which mean the if statements will be evaluated at compile time
	// It checks if the numericalType is a floating point variable, or integral type

	Matrix<numericalType> result(rowCnt, colCnt);

	// Initialize a random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Use different distributions based on the numericalType
	if constexpr (std::is_floating_point<numericalType>::value) 
	{
		std::uniform_real_distribution<numericalType> dis(lowerLimit, upperLimit);
		for (size_t i = 0; i < rowCnt; ++i) 
			for (size_t j = 0; j < colCnt; ++j) 
				result.matrix[i][j] = dis(gen);
			
	}
	else if constexpr (std::is_integral<numericalType>::value) 
	{
		// Since the uniform_int_distribution constructor does not accept floating-point types,
		// static_cast them to the numericalType, ensuring the function parameters are of numericalType.
		std::uniform_int_distribution<numericalType> dis(static_cast<numericalType>(lowerLimit), static_cast<numericalType>(upperLimit));
		for (size_t i = 0; i < rowCnt; ++i) 
			for (size_t j = 0; j < colCnt; ++j) 
				result.matrix[i][j] = dis(gen);
	}

	return result;
}

template<typename numericalType>
Matrix < numericalType> Matrix<numericalType>::rotationMatrix2D(const double& theta) const
{
	Matrix<numericalType> rotM(2, 2);

	rotM[0][0] = static_cast<numericalType>(std::cos(theta));
	rotM[0][1] = -1 * static_cast<numericalType>(std::sin(theta));
	rotM[1][0] = static_cast<numericalType>(std::sin(theta));
	rotM[1][1] = static_cast<numericalType>(std::cos(theta));

	// Rotation matrix should look like something like this:
	//	[ cos(theta)	-sin(theta)	]
	//	[ sin(theta)	cos(theta)	]

	return rotM;
}

template<typename numericalType>
Matrix<numericalType> Matrix<numericalType>::rotationMatrix3D(const double& alpha, const double& beta, const double& gamma) const
{
	Matrix<numericalType> rotM(3, 3);

	rotM[0][0] = static_cast<numericalType>(std::cos(beta) * std::cos(gamma));
	rotM[0][1] = static_cast<numericalType>(std::sin(alpha) * std::sin(beta)*std::cos(gamma) - std::cos(alpha) * std::sin(gamma));
	rotM[0][2] = static_cast<numericalType>(std::cos(alpha) * std::sin(beta)*std::cos(gamma) + std::sin(alpha) * std::sin(gamma));
	rotM[1][0] = static_cast<numericalType>(std::cos(beta) * std::sin(gamma));
	rotM[1][1] = static_cast<numericalType>(std::sin(alpha) * std::sin(beta) * std::sin(gamma) + std::cos(alpha) * std::cos(gamma));
	rotM[1][2] = static_cast<numericalType>(std::cos(alpha) * std::sin(beta) * std::sin(gamma) - std::sin(alpha) * std::cos(gamma));
	rotM[2][0] = static_cast<numericalType>(-1 * std::sin(beta));
	rotM[2][1] = static_cast<numericalType>(std::sin(alpha) * std::cos(beta));
	rotM[2][2] = static_cast<numericalType>(std::cos(alpha) * std::cos(beta));

	return rotM;
}

template<typename numericalType>
void Matrix<numericalType>::resize(const size_t& rowNum, const size_t& colNum, bool fillWithOld)
{
	Matrix<numericalType> newM(rowNum, colNum);

	// If fill with old is not selected, then returning the new matrix, with all zeroes
	if (!fillWithOld) 
	{
		*this = newM;
	}
	else	// Filling in old matrix data, until it is possible
	{
		unsigned maxRow = (rowNum > this->_row) ? this->_row : rowNum;
		unsigned maxCol = (colNum > this->_col) ? this->_col : colNum;

#ifdef _USING_OMP_
#pragma omp parallel for collapse(2)
#endif
		for (unsigned i = 0; i < maxRow; i++)
			for (unsigned j = 0; j < maxCol; j++)
				newM[i][j] = this->matrix[i][j];
		*this = newM;
	}
}


template class Matrix<unsigned>;
template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;