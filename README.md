# Matrix Library for C++

A versatile, template-based C++ library for matrix operations, designed to support both basic and advanced matrix manipulations. Whether you're working on mathematical problems, data analysis, developing algorithms that require matrix operations, or implementing neural network models, this library provides a robust set of features to facilitate your work.

The implementation draws inspiration from my linear algebra classes, brilliant.org linear algebra classes, and various educational YouTube videos. In the future, I plan to add `CUDA` and expand `OMP` parallelization options to enhance performance further. Currently, the library supports `Open MP` for efficient CPU parallelization. Basic operations and algorithms have been tested and are functional; feel free to integrate them into your projects. Should you encounter any bugs, errors, or inaccuracies, please do not hesitate to reach out through this repository.

This project is licensed under the GPL 3.0 license, ensuring that all derivatives of this code remain open-source and accessible.

## Features

- **Basic Operations:** Addition, subtraction, multiplication, and scalar operations.
- **Advanced Manipulations:** Transpose, determinant calculation, inverse, LU decomposition, and more.
- **Neural Network Support:** Includes pooling (max and min), and convolution functions tailored for neural network operations such as filtering.
- **Specialized Functions:** Eigenvalues and eigenvectors calculation, matrix exponentiation, least squares solving, and QR decomposition.
- **Utility Functions:** Mean, max, min, and various norms (e.g., Frobenius, L1) calculations.
- **Parallel Processing Support:** Utilizes OpenMP for efficient CPU parallelization to speed up computations on multicore processors.
- **Flexibility:** Template-based implementation supports various numerical types, including `int`, `float`, and `double`.
- **Ease of Use:** Function names directly reflect their operations, minimizing the likelihood of confusion. Numerous function overrides are available, accommodating a wide range of data types for matrix operations.

## Getting Started

### Prerequisites

Ensure your compiler supports C++17 (or later) and OpenMP for the best experience with this library. To use OMP you should also `#define _USING_OMP_` at the top of the matrix.hpp file.

### Including the Library

Copy `matrix.hpp` and `matrix.cpp` into your project directory. Include the header file in your project as shown below:

```cpp
#include "matrix.hpp"
```

### Basic Usage

Here's how you can perform some basic operations with the Matrix library:

```cpp
#include "matrix.hpp"

int main() {
    // Create matrices
    Matrix<double> A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Matrix<double> B = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};

    // Matrix addition
    auto C = A + B;
    C.printToStdOut();

    // Matrix multiplication
    auto D = A * B;
    D.printToStdOut();

    // Transpose
    auto At = A.transpose();
    At.printToStdOut();

    // Determinant
    std::cout << "Determinant of A: " << A.determinant() << std::endl;

    return 0;
}
```

### Advanced Operations

The library also supports more complex operations, such as computing eigenvalues and solving for least squares:

```cpp
// Solve for least squares
Vector<double> b = {1, 2, 3};
auto x = A.leastSquares(b);
x.printToStdOut();

// Eigenvalues and Eigenvectors
auto eigenvalues = A.eigenvaluesVector();
std::cout << "Eigenvalues: ";
for (auto val : eigenvalues) {
    std::cout << val << " ";
}
std::cout << std::endl;
```

## Contributing

Contributions to the Matrix Library are welcome! Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated. Please open an issue or pull request to get started.

## License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details.
