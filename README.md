# Matrix Library for C++

A versatile, template-based C++ library for matrix operations, designed to support both basic and advanced matrix manipulations. Whether you're working on mathematical problems, data analysis, or developing algorithms that require matrix operations, this library provides a robust set of features to facilitate your work.

The implementation is based on my linear algebra classes, brilliant.org linear algebra classes, and other youtube motivated videos. In the future i'm planning on adding `CUDA` and `OMP` parallelizaiton options. Currently only supporting `Open MP`. Currently basic operations, and algorithms are tested and are working, feel free to use them in your projects, if you encounter any bugs, errors, miscalculations feel free to contact me throughout this repository.

## Features

- **Basic Operations:** Addition, subtraction, multiplication, and scalar operations.
- **Advanced Manipulations:** Transpose, determinant calculation, inverse, LU decomposition, and more.
- **Specialized Functions:** Eigenvalues and eigenvectors calculation, matrix exponentiation, least squares solving, and QR decomposition.
- **Utility Functions:** Mean, max, min, and various norms (e.g., Frobenius, L1) calculations.
- **Parallel Processing Support:** Leverages OpenMP for CPU parallelization to accelerate computation on multicore processors.
- **Flexibility:** Template-based implementation supports various numerical types, including `int`, `float`, and `double`.

## Getting Started

### Prerequisites

Ensure your compiler supports C++17 (or later) and OpenMP for the best experience with this library.

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
