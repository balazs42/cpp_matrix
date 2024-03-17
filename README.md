# C++ Matrix Library

Welcome to the GitHub repository of my C++ Matrix Library, a comprehensive, efficient, and easy-to-use library designed for performing a wide range of matrix operations. This library is developed with the intention to provide a solid foundation for any project requiring linear algebra computations, including but not limited to, mathematical modeling, simulations, and data analysis.

## Features

- **Dynamic Matrix Creation**: Create matrices dynamically with any numerical type, supporting operations such as addition, subtraction, multiplication, and more.
- **Memory Management**: Carefully designed to manage memory efficiently and prevent leaks.
- **Advanced Operations**: Supports advanced matrix operations including determinant calculation, LU decomposition, inverse, transpose, and eigenvalue computation.
- **Linear Algebra Essentials**: Offers functionalities to check matrix properties (symmetric, orthogonal, etc.), perform eigen decomposition, and more.
- **Exception Safety**: Implements thorough error checking and exception handling to ensure robustness and reliability.
- **Type conversions**: You can use perviously defined vector<vector<>> as matrixes, the class will handle it. Also you can use dynamic memory allocated arrays as vectors, or you can use std::vector<> aswell.

## Getting Started

Clone this repository and include the `matrix.hpp` and `matrix.cpp` files in your C++ project. The library is template-based, allowing for flexibility with different numerical types.

```cpp
#include "matrix.hpp"
// Example usage
int main() {
    Matrix<double> mat(3, 3); // Create a 3x3 double matrix
    // Perform operations...
    return 0;
}
