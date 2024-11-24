# Python Scientific Type

A new Python data type (`ScientificNumber`) that extends Python's numeric types with comprehensive scientific computing capabilities. This type seamlessly integrates mathematical, statistical, and matrix operations into a single, intuitive interface.

## Why a New Type?

Unlike regular Python numbers or NumPy arrays, `ScientificNumber` provides:
- Unified interface for scalar and matrix operations
- Built-in scientific constants and unit conversions
- Automatic type handling between different numerical representations
- Memory management for scientific calculations
- Comprehensive mathematical operations in a single type

## Features

- Basic arithmetic operations with type safety
- Matrix operations (determinant, rank, eigenvalues)
- Trigonometric functions
- Statistical functions
- Vector operations
- Unit conversions
- Physical constants
- Memory operations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from scientific_calculator import ScientificNumber
from scientific_functions import *

# Create scientific numbers
x = ScientificNumber(5)
y = ScientificNumber(3)

# Basic Operations with automatic type handling
result = BasicOperations.add(x, y)
print(result.value)  # Output: 8

# Matrix creation and operations
matrix = ScientificNumber(np.array([[1, 2], [3, 4]]))  # Create from numpy array
# Or use the utility function
matrix = MatrixOperations.create_matrix(2, 2, [1, 2, 3, 4])

# Matrix operations return ScientificNumber
determinant = MatrixOperations.determinant(matrix)
rank = MatrixOperations.rank(matrix)
eigenvalues = MatrixOperations.eigenvalues(matrix)

# Trigonometric operations with automatic unit handling
angle = ScientificNumber(45)  # degrees
rad = TrigonometricFunctions.deg_to_rad(angle)
sin_value = TrigonometricFunctions.sin(rad)

# Statistical operations on lists or arrays
data = [1, 2, 3, 4, 5]
mean = StatisticalFunctions.mean(data)
std_dev = StatisticalFunctions.std_dev(data)
```

## Type Features

The `ScientificNumber` type provides:

1. **Type Safety**: Automatically handles type conversions and validates operations
2. **Matrix Support**: Seamlessly works with both scalar and matrix values
3. **Operation History**: Tracks calculations for debugging and analysis
4. **Memory Management**: Built-in memory operations for scientific calculations
5. **Constants**: Access to physical and mathematical constants
6. **Unit Conversions**: Built-in unit conversion capabilities

## Classes

1. `ScientificNumber`: Core data type that encapsulates numerical values
2. `BasicOperations`: Arithmetic operations for the new type
3. `MatrixOperations`: Matrix manipulations preserving type safety
4. `TrigonometricFunctions`: Trigonometric calculations with unit handling
5. `StatisticalFunctions`: Statistical computations on collections
6. `SpecialFunctions`: Special mathematical functions
7. `VectorOperations`: Vector calculations
8. `UnitConversion`: Unit conversion utilities
9. `Constants`: Physical and mathematical constants
10. `MemoryOperations`: Memory management functions

## Requirements

- Python 3.6+
- NumPy
- SciPy
- SymPy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
