import math
import numpy as np
from scipy import stats
from sympy import symbols, Eq, solve
from sympy import init_printing
import requests

class ScientificNumber:
    history = []  # To store history of calculations
    _memory = 0

    def __init__(self, value):
        try:
            if isinstance(value, np.ndarray):
                self.value = value
                self.is_matrix = True
            elif isinstance(value, complex):
                self.value = value
                self.is_matrix = False
            else:
                self.value = float(value)
                self.is_matrix = False
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input: {str(e)}")

    def __repr__(self):
        if self.is_matrix:
            return f"ScientificNumber(matrix={self.value})"
        return f"ScientificNumber({self.value})"

    # --- Magic Methods for Arithmetic Operations ---
    def __add__(self, other):
        if isinstance(other, ScientificNumber):
            if self.is_matrix and other.is_matrix:
                return ScientificNumber(np.add(self.value, other.value))  # Matrix addition
            return ScientificNumber(self.value + other.value)
        return ScientificNumber(self.value + other)

    def __sub__(self, other):
        if isinstance(other, ScientificNumber):
            if self.is_matrix and other.is_matrix:
                return ScientificNumber(np.subtract(self.value, other.value))  # Matrix subtraction
            return ScientificNumber(self.value - other.value)
        return ScientificNumber(self.value - other)

    def __mul__(self, other):
        if isinstance(other, ScientificNumber):
            if self.is_matrix and other.is_matrix:
                return ScientificNumber(np.dot(self.value, other.value))  # Matrix multiplication
            return ScientificNumber(self.value * other.value)
        return ScientificNumber(self.value * other)

    def __truediv__(self, other):
        if isinstance(other, ScientificNumber):
            if self.is_matrix and other.is_matrix:
                raise ValueError("Matrix division is not defined in this context")
            return ScientificNumber(self.value / other.value)
        return ScientificNumber(self.value / other)

    def __floordiv__(self, other):
        if isinstance(other, ScientificNumber):
            return ScientificNumber(self.value // other.value)
        return ScientificNumber(self.value // other)

    def __mod__(self, other):
        if isinstance(other, ScientificNumber):
            return ScientificNumber(self.value % other.value)
        return ScientificNumber(self.value % other)

    def __pow__(self, other):
        if isinstance(other, ScientificNumber):
            return ScientificNumber(self.value ** other.value)
        return ScientificNumber(self.value ** other)

    def __neg__(self):
        return ScientificNumber(-self.value)

    # --- Bitwise Operations ---
    def __lshift__(self, other):
        if isinstance(other, ScientificNumber):
            return ScientificNumber(self.value << other.value)
        return ScientificNumber(self.value << other)

    def __rshift__(self, other):
        if isinstance(other, ScientificNumber):
            return ScientificNumber(self.value >> other.value)
        return ScientificNumber(self.value >> other)

    def __and__(self, other):
        if isinstance(other, ScientificNumber):
            return ScientificNumber(self.value & other.value)
        return ScientificNumber(self.value & other)

    def __or__(self, other):
        if isinstance(other, ScientificNumber):
            return ScientificNumber(self.value | other.value)
        return ScientificNumber(self.value | other)

    def __xor__(self, other):
        if isinstance(other, ScientificNumber):
            return ScientificNumber(self.value ^ other.value)
        return ScientificNumber(self.value ^ other)

    def __invert__(self):
        return ScientificNumber(~self.value)

    # --- Matrix Specific Operations ---
    def transpose(self):
        if not self.is_matrix:
            raise TypeError("Transpose is only available for matrices")
        return ScientificNumber(self.value.T)

    def inverse(self):
        if not self.is_matrix:
            raise TypeError("Inverse is only available for matrices")
        return ScientificNumber(np.linalg.inv(self.value))

    # --- Matrix Operations ---
    @staticmethod
    def create_matrix(rows, cols):
        """Create a matrix of given size with user input"""
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                value = float(input(f"Enter element at position ({i+1},{j+1}): "))
                row.append(value)
            matrix.append(row)
        return ScientificNumber(np.array(matrix))

    def matrix_transpose(self):
        """Return transpose of the matrix"""
        if not self.is_matrix:
            raise ValueError("Operation only valid for matrices")
        return ScientificNumber(np.transpose(self.value))

    def matrix_determinant(self):
        """Calculate determinant of the matrix"""
        if not self.is_matrix:
            raise ValueError("Operation only valid for matrices")
        if self.value.shape[0] != self.value.shape[1]:
            raise ValueError("Determinant only defined for square matrices")
        return ScientificNumber(np.linalg.det(self.value))

    def matrix_inverse(self):
        """Calculate inverse of the matrix if it exists"""
        if not self.is_matrix:
            raise ValueError("Operation only valid for matrices")
        if self.value.shape[0] != self.value.shape[1]:
            raise ValueError("Inverse only defined for square matrices")
        try:
            return ScientificNumber(np.linalg.inv(self.value))
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not invertible")

    def matrix_rank(self):
        """Calculate rank of the matrix"""
        if not self.is_matrix:
            raise ValueError("Operation only valid for matrices")
        return ScientificNumber(np.linalg.matrix_rank(self.value))

    def matrix_eigenvalues(self):
        """Calculate eigenvalues of the matrix"""
        if not self.is_matrix:
            raise ValueError("Operation only valid for matrices")
        if self.value.shape[0] != self.value.shape[1]:
            raise ValueError("Eigenvalues only defined for square matrices")
        return ScientificNumber(np.linalg.eigvals(self.value))

    # --- Scientific Functions ---
    @staticmethod
    def sin(number):
        if number.is_matrix:
            return ScientificNumber(np.sin(number.value))
        return ScientificNumber(math.sin(number.value))

    @staticmethod
    def cos(number):
        if number.is_matrix:
            return ScientificNumber(np.cos(number.value))
        return ScientificNumber(math.cos(number.value))

    @staticmethod
    def tan(number):
        if number.is_matrix:
            return ScientificNumber(np.tan(number.value))
        return ScientificNumber(math.tan(number.value))

    @staticmethod
    def log(number):
        if number.is_matrix:
            return ScientificNumber(np.log(number.value))
        return ScientificNumber(math.log(number.value))

    @staticmethod
    def exp(number):
        if number.is_matrix:
            return ScientificNumber(np.exp(number.value))
        return ScientificNumber(math.exp(number.value))

    @staticmethod
    def sqrt(number):
        if number.is_matrix:
            return ScientificNumber(np.sqrt(number.value))
        return ScientificNumber(math.sqrt(number.value))

    @staticmethod
    def deg_to_rad(number):
        if number.is_matrix:
            return ScientificNumber(np.radians(number.value))
        return ScientificNumber(math.radians(number.value))

    @staticmethod
    def rad_to_deg(number):
        if number.is_matrix:
            return ScientificNumber(np.degrees(number.value))
        return ScientificNumber(math.degrees(number.value))

    @staticmethod
    def sinh(number):
        if number.is_matrix:
            return ScientificNumber(np.sinh(number.value))
        return ScientificNumber(math.sinh(number.value))

    @staticmethod
    def cosh(number):
        if number.is_matrix:
            return ScientificNumber(np.cosh(number.value))
        return ScientificNumber(math.cosh(number.value))

    @staticmethod
    def tanh(number):
        if number.is_matrix:
            return ScientificNumber(np.tanh(number.value))
        return ScientificNumber(math.tanh(number.value))

    @staticmethod
    def arcsin(number):
        if number.is_matrix:
            return ScientificNumber(np.arcsin(number.value))
        return ScientificNumber(math.asin(number.value))

    @staticmethod
    def arccos(number):
        if number.is_matrix:
            return ScientificNumber(np.arccos(number.value))
        return ScientificNumber(math.acos(number.value))

    @staticmethod
    def arctan(number):
        if number.is_matrix:
            return ScientificNumber(np.arctan(number.value))
        return ScientificNumber(math.atan(number.value))

    @staticmethod
    def factorial(number):
        if number.is_matrix:
            raise TypeError("Factorial is not defined for matrices")
        if not float(number.value).is_integer() or number.value < 0:
            raise ValueError("Factorial is only defined for non-negative integers")
        return ScientificNumber(math.factorial(int(number.value)))

    @staticmethod
    def gcd(a, b):
        if a.is_matrix or b.is_matrix:
            raise TypeError("GCD is not defined for matrices")
        return ScientificNumber(math.gcd(int(a.value), int(b.value)))

    @staticmethod
    def lcm(a, b):
        if a.is_matrix or b.is_matrix:
            raise TypeError("LCM is not defined for matrices")
        return ScientificNumber(abs(a.value * b.value) // math.gcd(int(a.value), int(b.value)))

    # --- Statistical Methods ---
    @staticmethod
    def mean(data):
        return ScientificNumber(np.mean(data))

    @staticmethod
    def median(data):
        return ScientificNumber(np.median(data))

    @staticmethod
    def variance(data):
        return ScientificNumber(np.var(data))

    @staticmethod
    def std_dev(data):
        return ScientificNumber(np.std(data))

    @staticmethod
    def normal_distribution(mu, sigma, size):
        return ScientificNumber(np.random.normal(mu, sigma, size))

    @staticmethod
    def binomial_distribution(n, p, size):
        return ScientificNumber(np.random.binomial(n, p, size))

    @staticmethod
    def mode(data):
        return ScientificNumber(stats.mode(data)[0])

    @staticmethod
    def skewness(data):
        return ScientificNumber(stats.skew(data))

    @staticmethod
    def kurtosis(data):
        return ScientificNumber(stats.kurtosis(data))

    # --- Unit Conversion ---
    @staticmethod
    def convert_units(value, from_unit, to_unit):
        conversions = {
            ('m', 'km'): value / 1000,
            ('km', 'm'): value * 1000,
            ('kg', 'g'): value * 1000,
            ('g', 'kg'): value / 1000,
            ('C', 'F'): value * 9/5 + 32,
            ('F', 'C'): (value - 32) * 5/9
        }
        return ScientificNumber(conversions.get((from_unit, to_unit), value))

    # --- Physical Constants ---
    constants = {
        "speed_of_light": 3e8,  # m/s
        "gravitational_constant": 6.67430e-11,  # m^3 kg^-1 s^-2
        "planck_constant": 6.62607015e-34,  # J⋅s
        "boltzmann_constant": 1.380649e-23,  # J/K
        "electron_mass": 9.1093837015e-31,  # kg
        "proton_mass": 1.67262192369e-27,  # kg
        "elementary_charge": 1.602176634e-19,  # C
        "avogadro_number": 6.02214076e23,  # mol^-1
        "gas_constant": 8.31446261815324,  # J/(mol⋅K)
        "pi": math.pi,  # π
        "e": math.e,  # Euler's number
        "golden_ratio": (1 + math.sqrt(5)) / 2,  # φ
    }

    @staticmethod
    def get_constant(constant_name):
        return ScientificNumber(ScientificNumber.constants.get(constant_name, "Unknown Constant"))

    # --- Vector Calculations ---
    @staticmethod
    def dot_product(v1, v2):
        return ScientificNumber(np.dot(v1, v2))

    @staticmethod
    def cross_product(v1, v2):
        return ScientificNumber(np.cross(v1, v2))

    @staticmethod
    def vector_magnitude(v):
        return ScientificNumber(np.linalg.norm(v))

    # --- Equation Solving ---
    @staticmethod
    def solve_equation(equation):
        init_printing()  # Pretty printing
        x = symbols('x')
        eq = Eq(equation[0], equation[1])
        solution = solve(eq, x)
        return solution

    # --- History ---
    @staticmethod
    def add_to_history(calculation):
        ScientificNumber.history.append(calculation)

    @staticmethod
    def show_history():
        return ScientificNumber.history

    # --- Currency Conversion (Example) ---
    @staticmethod
    def convert_currency(amount, from_currency, to_currency):
        api_url = f'https://api.exchangerate-api.com/v4/latest/{from_currency}'
        response = requests.get(api_url).json()
        rate = response['rates'].get(to_currency)
        if rate:
            return amount * rate
        return "Currency conversion failed"

    # --- Polynomial Regression ---
    @staticmethod
    def polynomial_regression(x, y, degree):
        x = np.array(x)
        y = np.array(y)
        coefficients = np.polyfit(x, y, degree)
        return ScientificNumber(coefficients)

    # --- Memory Functions ---
    @classmethod
    def memory_store(cls, value):
        cls._memory = value.value if isinstance(value, ScientificNumber) else value

    @classmethod
    def memory_recall(cls):
        return ScientificNumber(cls._memory)

    @classmethod
    def memory_clear(cls):
        cls._memory = 0

    @classmethod
    def memory_add(cls, value):
        cls._memory += value.value if isinstance(value, ScientificNumber) else value

# --- Example Usage ---
if __name__ == "__main__":
    # Example of Polynomial Regression
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    print(ScientificNumber.polynomial_regression(x, y, 1))

    # Example of Vector Calculation
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    print("Dot product of vectors:", ScientificNumber.dot_product(v1, v2))
    print("Cross product of vectors:", ScientificNumber.cross_product(v1, v2))
    print("Magnitude of vector v1:", ScientificNumber.vector_magnitude(v1))

    # Example of Unit Conversion
    print(ScientificNumber.convert_units(1000, 'm', 'km'))

    # Example of Physical Constants
    print("Speed of light:", ScientificNumber.get_constant("speed_of_light"))

    # Example of Statistical Calculation
    data = [1, 2, 3, 4, 5]
    print("Mean of data:", ScientificNumber.mean(data))

    # Example of Currency Conversion
    print("Currency conversion (USD to EUR):", ScientificNumber.convert_currency(100, 'USD', 'EUR'))

    # Show history of operations
    ScientificNumber.add_to_history("Calculated polynomial regression.")
    ScientificNumber.add_to_history("Converted 1000 meters to kilometers.")
    print("History of calculations:", ScientificNumber.show_history())
