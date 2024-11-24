from scientific_calculator import ScientificNumber
import numpy as np

class BasicOperations:
    @staticmethod
    def add(a, b):
        """Add two numbers"""
        return ScientificNumber(a) + ScientificNumber(b)

    @staticmethod
    def subtract(a, b):
        """Subtract two numbers"""
        return ScientificNumber(a) - ScientificNumber(b)

    @staticmethod
    def multiply(a, b):
        """Multiply two numbers"""
        return ScientificNumber(a) * ScientificNumber(b)

    @staticmethod
    def divide(a, b):
        """Divide two numbers"""
        return ScientificNumber(a) / ScientificNumber(b)

    @staticmethod
    def floor_divide(a, b):
        """Floor division"""
        return ScientificNumber(a) // ScientificNumber(b)

    @staticmethod
    def modulo(a, b):
        """Modulo operation"""
        return ScientificNumber(a) % ScientificNumber(b)

    @staticmethod
    def power(a, b):
        """Raise a to power b"""
        return ScientificNumber(a) ** ScientificNumber(b)

class MatrixOperations:
    @staticmethod
    def create_matrix(rows, cols, values):
        """Create a matrix from values"""
        return ScientificNumber(np.array(values).reshape(rows, cols))

    @staticmethod
    def add_matrices(m1, m2):
        """Add two matrices"""
        return m1 + m2

    @staticmethod
    def subtract_matrices(m1, m2):
        """Subtract two matrices"""
        return m1 - m2

    @staticmethod
    def multiply_matrices(m1, m2):
        """Multiply two matrices"""
        return m1 * m2

    @staticmethod
    def transpose(matrix):
        """Get matrix transpose"""
        return matrix.matrix_transpose()

    @staticmethod
    def determinant(matrix):
        """Calculate matrix determinant"""
        return matrix.matrix_determinant()

    @staticmethod
    def inverse(matrix):
        """Calculate matrix inverse"""
        return matrix.matrix_inverse()

    @staticmethod
    def rank(matrix):
        """Calculate matrix rank"""
        return matrix.matrix_rank()

    @staticmethod
    def eigenvalues(matrix):
        """Calculate matrix eigenvalues"""
        return matrix.matrix_eigenvalues()

class TrigonometricFunctions:
    @staticmethod
    def sin(x):
        """Sine function"""
        if not isinstance(x, ScientificNumber):
            x = ScientificNumber(x)
        return ScientificNumber.sin(x)

    @staticmethod
    def cos(x):
        """Cosine function"""
        if not isinstance(x, ScientificNumber):
            x = ScientificNumber(x)
        return ScientificNumber.cos(x)

    @staticmethod
    def tan(x):
        """Tangent function"""
        if not isinstance(x, ScientificNumber):
            x = ScientificNumber(x)
        return ScientificNumber.tan(x)

    @staticmethod
    def arcsin(x):
        """Inverse sine"""
        return ScientificNumber.arcsin(ScientificNumber(x))

    @staticmethod
    def arccos(x):
        """Inverse cosine"""
        return ScientificNumber.arccos(ScientificNumber(x))

    @staticmethod
    def arctan(x):
        """Inverse tangent"""
        return ScientificNumber.arctan(ScientificNumber(x))

    @staticmethod
    def sinh(x):
        """Hyperbolic sine"""
        return ScientificNumber.sinh(ScientificNumber(x))

    @staticmethod
    def cosh(x):
        """Hyperbolic cosine"""
        return ScientificNumber.cosh(ScientificNumber(x))

    @staticmethod
    def tanh(x):
        """Hyperbolic tangent"""
        return ScientificNumber.tanh(ScientificNumber(x))

    @staticmethod
    def deg_to_rad(x):
        """Convert degrees to radians"""
        return ScientificNumber.deg_to_rad(ScientificNumber(x))

    @staticmethod
    def rad_to_deg(x):
        """Convert radians to degrees"""
        return ScientificNumber.rad_to_deg(ScientificNumber(x))

class StatisticalFunctions:
    @staticmethod
    def mean(data):
        """Calculate mean of data"""
        return ScientificNumber.mean(data)

    @staticmethod
    def median(data):
        """Calculate median of data"""
        return ScientificNumber.median(data)

    @staticmethod
    def variance(data):
        """Calculate variance of data"""
        return ScientificNumber.variance(data)

    @staticmethod
    def std_dev(data):
        """Calculate standard deviation"""
        return ScientificNumber.std_dev(data)

    @staticmethod
    def mode(data):
        """Calculate mode of data"""
        return ScientificNumber.mode(data)

    @staticmethod
    def skewness(data):
        """Calculate skewness"""
        return ScientificNumber.skewness(data)

    @staticmethod
    def kurtosis(data):
        """Calculate kurtosis"""
        return ScientificNumber.kurtosis(data)

    @staticmethod
    def normal_distribution(mu, sigma, size):
        """Generate normal distribution"""
        return ScientificNumber.normal_distribution(mu, sigma, size)

    @staticmethod
    def binomial_distribution(n, p, size):
        """Generate binomial distribution"""
        return ScientificNumber.binomial_distribution(n, p, size)

class SpecialFunctions:
    @staticmethod
    def sqrt(x):
        """Square root"""
        return ScientificNumber.sqrt(ScientificNumber(x))

    @staticmethod
    def exp(x):
        """Exponential function"""
        return ScientificNumber.exp(ScientificNumber(x))

    @staticmethod
    def log(x):
        """Natural logarithm"""
        return ScientificNumber.log(ScientificNumber(x))

    @staticmethod
    def factorial(n):
        """Factorial"""
        return ScientificNumber.factorial(ScientificNumber(n))

    @staticmethod
    def gcd(a, b):
        """Greatest common divisor"""
        return ScientificNumber.gcd(ScientificNumber(a), ScientificNumber(b))

    @staticmethod
    def lcm(a, b):
        """Least common multiple"""
        return ScientificNumber.lcm(ScientificNumber(a), ScientificNumber(b))

class VectorOperations:
    @staticmethod
    def dot_product(v1, v2):
        """Calculate dot product"""
        return ScientificNumber.dot_product(v1, v2)

    @staticmethod
    def cross_product(v1, v2):
        """Calculate cross product"""
        return ScientificNumber.cross_product(v1, v2)

    @staticmethod
    def magnitude(v):
        """Calculate vector magnitude"""
        return ScientificNumber.vector_magnitude(v)

class UnitConversion:
    @staticmethod
    def convert(value, from_unit, to_unit):
        """Convert between units"""
        return ScientificNumber.convert_units(value, from_unit, to_unit)

class Constants:
    @staticmethod
    def get(constant_name):
        """Get physical constant"""
        return ScientificNumber.get_constant(constant_name)

class MemoryOperations:
    @staticmethod
    def store(value):
        """Store value in memory"""
        ScientificNumber.memory_store(value)

    @staticmethod
    def recall():
        """Recall value from memory"""
        return ScientificNumber.memory_recall()

    @staticmethod
    def clear():
        """Clear memory"""
        ScientificNumber.memory_clear()

    @staticmethod
    def add(value):
        """Add to memory"""
        ScientificNumber.memory_add(value)

def demonstrate_all_functions():
    """Demonstrate usage of all functions"""
    print("\n=== Basic Operations ===")
    print("Addition: ", BasicOperations.add(5, 3).value)
    print("Subtraction: ", BasicOperations.subtract(5, 3).value)
    print("Multiplication: ", BasicOperations.multiply(5, 3).value)
    print("Division: ", BasicOperations.divide(6, 2).value)
    print("Floor Division: ", BasicOperations.floor_divide(7, 2).value)
    print("Modulo: ", BasicOperations.modulo(7, 3).value)
    print("Power: ", BasicOperations.power(2, 3).value)

    print("\n=== Matrix Operations ===")
    m1 = MatrixOperations.create_matrix(2, 2, [1, 2, 3, 4])
    m2 = MatrixOperations.create_matrix(2, 2, [5, 6, 7, 8])
    print("Matrix 1:")
    print(m1.value)
    print("Matrix 2:")
    print(m2.value)
    print("Matrix Addition:")
    print(MatrixOperations.add_matrices(m1, m2).value)
    print("Matrix Subtraction:")
    print(MatrixOperations.subtract_matrices(m2, m1).value)
    print("Matrix Multiplication:")
    print(MatrixOperations.multiply_matrices(m1, m2).value)
    print("Transpose:")
    print(MatrixOperations.transpose(m1).value)
    print("Determinant:")
    print(MatrixOperations.determinant(m1).value)
    print("Matrix Rank:")
    print(MatrixOperations.rank(m1).value)
    print("Eigenvalues:")
    print(MatrixOperations.eigenvalues(m1).value)

    print("\n=== Trigonometric Functions ===")
    angle = 45
    rad = TrigonometricFunctions.deg_to_rad(angle)
    print(f"Angle in radians: {rad.value}")
    print(f"Sin({angle}°): ", TrigonometricFunctions.sin(rad).value)
    print(f"Cos({angle}°): ", TrigonometricFunctions.cos(rad).value)
    print(f"Tan({angle}°): ", TrigonometricFunctions.tan(rad).value)
    print(f"Arcsin(0.5): ", TrigonometricFunctions.arcsin(0.5).value)
    print(f"Arccos(0.5): ", TrigonometricFunctions.arccos(0.5).value)
    print(f"Arctan(1): ", TrigonometricFunctions.arctan(1).value)
    print(f"Sinh(1): ", TrigonometricFunctions.sinh(1).value)
    print(f"Cosh(1): ", TrigonometricFunctions.cosh(1).value)
    print(f"Tanh(1): ", TrigonometricFunctions.tanh(1).value)

    print("\n=== Statistical Functions ===")
    data = [1, 2, 3, 4, 5]
    print("Data:", data)
    print("Mean: ", StatisticalFunctions.mean(data).value)
    print("Median: ", StatisticalFunctions.median(data).value)
    print("Variance: ", StatisticalFunctions.variance(data).value)
    print("Standard Deviation: ", StatisticalFunctions.std_dev(data).value)
    print("Mode: ", StatisticalFunctions.mode(data).value)
    print("Skewness: ", StatisticalFunctions.skewness(data).value)
    print("Kurtosis: ", StatisticalFunctions.kurtosis(data).value)
    print("\nDistributions:")
    print("Normal Distribution (first 5 values):", StatisticalFunctions.normal_distribution(0, 1, 5).value)
    print("Binomial Distribution (first 5 values):", StatisticalFunctions.binomial_distribution(10, 0.5, 5).value)

    print("\n=== Special Functions ===")
    print("Square root of 16: ", SpecialFunctions.sqrt(16).value)
    print("Exp(2): ", SpecialFunctions.exp(2).value)
    print("Log(10): ", SpecialFunctions.log(10).value)
    print("Factorial of 5: ", SpecialFunctions.factorial(5).value)
    print("GCD of 12 and 18: ", SpecialFunctions.gcd(12, 18).value)
    print("LCM of 12 and 18: ", SpecialFunctions.lcm(12, 18).value)

    print("\n=== Vector Operations ===")
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    print("Vector 1:", v1)
    print("Vector 2:", v2)
    print("Dot product: ", VectorOperations.dot_product(v1, v2).value)
    print("Cross product: ", VectorOperations.cross_product(v1, v2).value)
    print("Magnitude of v1: ", VectorOperations.magnitude(v1).value)

    print("\n=== Unit Conversion ===")
    print("1 meter to feet: ", UnitConversion.convert(1, "m", "ft").value)
    print("100 celsius to fahrenheit: ", UnitConversion.convert(100, "C", "F").value)
    print("1 kg to pounds: ", UnitConversion.convert(1, "kg", "lb").value)

    print("\n=== Constants ===")
    print("Pi: ", Constants.get("pi").value)
    print("Speed of light: ", Constants.get("speed_of_light").value)
    print("Planck constant: ", Constants.get("planck_constant").value)
    print("Boltzmann constant: ", Constants.get("boltzmann_constant").value)
    print("Electron mass: ", Constants.get("electron_mass").value)
    print("Avogadro number: ", Constants.get("avogadro_number").value)

    print("\n=== Memory Operations ===")
    MemoryOperations.store(10)
    print("Stored value: ", MemoryOperations.recall().value)
    MemoryOperations.add(5)
    print("After adding 5: ", MemoryOperations.recall().value)
    MemoryOperations.clear()
    print("After clearing: ", MemoryOperations.recall().value)

if __name__ == "__main__":
    demonstrate_all_functions()
