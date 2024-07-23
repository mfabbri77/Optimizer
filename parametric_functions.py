from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional

class ParametricFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray, *params: float) -> np.ndarray:
        pass

    @abstractmethod
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__

    def is3D(self) -> bool:
        return "3D" in self.__class__.__name__

class LinearFunction2D(ParametricFunction):
    def __call__(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        a = (y[-1] - y[0]) / (x[-1] - x[0])
        b = y[0] - a * x[0]
        return [a, b]

class LinearFunction3D(ParametricFunction):
    def __call__(self, x: np.ndarray, y: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * x + b * y + c

    def initial_guess(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[float]:
        A = np.column_stack((x, y, np.ones_like(x)))
        a, b, c = np.linalg.lstsq(A, z, rcond=None)[0]
        return [a, b, c]

class QuadraticFunction2D(ParametricFunction):
    def __call__(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * x**2 + b * x + c

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        a = (y[-1] - 2*y[len(y)//2] + y[0]) / ((x[-1] - x[0])**2)
        b = (y[-1] - y[0]) / (x[-1] - x[0]) - a * (x[-1] + x[0])
        c = y[0] - a * x[0]**2 - b * x[0]
        return [a, b, c]

class QuadraticFunction3D(ParametricFunction):
    def __call__(self, x: np.ndarray, y: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
        return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

    def initial_guess(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[float]:
        A = np.column_stack((x**2, y**2, x*y, x, y, np.ones_like(x)))
        params, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        return list(params)

import numpy as np

class ExponentialFunction2D(ParametricFunction):
    def __call__(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        # Clip the exponent to avoid overflow
        exponent = np.clip(b * x, -700, 700)
        exp_term = np.exp(exponent)
        # Clip the result to avoid overflow in multiplication
        return np.clip(a * exp_term, -1e300, 1e300) + c

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(x)
        x_max = np.max(x)
        
        a = (y_max - y_min) / 2
        b = np.log(y_max / max(y_min, 1e-10)) / (x_max - x_min)
        c = y_min
        
        # Clip initial guess to avoid extreme values
        a = np.clip(a, -1e10, 1e10)
        b = np.clip(b, -100, 100)
        c = np.clip(c, -1e10, 1e10)
        
        return [a, b, c]

class ExponentialFunction3D(ParametricFunction):
    def __call__(self, x: np.ndarray, y: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        exponent = np.clip(b * x + c * y, -700, 700)
        return a * np.exp(exponent) + d

    def initial_guess(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[float]:
        z_min, z_max = np.min(z), np.max(z)
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        
        a = (z_max - z_min) / 2
        b = 1 / max(x_range, 1e-10)
        c = 1 / max(y_range, 1e-10)
        d = z_min
        
        return [a, b, c, d]

class GaussianFunction2D(ParametricFunction):
    def __call__(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(-((x - b)**2) / (2 * c**2))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        a = np.max(y)
        b = x[np.argmax(y)]
        c = (x[-1] - x[0]) / 4
        return [a, b, c]

class GaussianFunction3D(ParametricFunction):
    def __call__(self, x: np.ndarray, y: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
        return a * np.exp(-((x - b)**2 / (2 * c**2) + (y - d)**2 / (2 * e**2))) + f

    def initial_guess(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[float]:
        a = np.max(z) - np.min(z)
        b = np.mean(x)
        c = np.std(x)
        d = np.mean(y)
        e = np.std(y)
        f = np.min(z)
        return [a, b, c, d, e, f]

class SineFunction2D(ParametricFunction):
    def __call__(self, x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        return a * np.sin(b * x + c) + d

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        a = (np.max(y) - np.min(y)) / 2
        b = 2 * np.pi / (x[-1] - x[0])
        c = 0
        d = np.mean(y)
        return [a, b, c, d]

class SineFunction3D(ParametricFunction):
    def __call__(self, x: np.ndarray, y: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
        return a * np.sin(b * x + c) * np.sin(d * y + e) + f

    def initial_guess(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[float]:
        a = (np.max(z) - np.min(z)) / 2
        b = 2 * np.pi / (np.max(x) - np.min(x))
        c = 0
        d = 2 * np.pi / (np.max(y) - np.min(y))
        e = 0
        f = np.mean(z)
        return [a, b, c, d, e, f]

class LogarithmicFunction2D(ParametricFunction):
    def __call__(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        epsilon = 1e-10
        return a * np.log(np.maximum(b * x, epsilon)) + c

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        positive_mask = x > 0
        x_positive = x[positive_mask]
        y_positive = y[positive_mask]

        if len(x_positive) < 2:
            return [1.0, 1.0, np.mean(y)]

        a = (y_positive[-1] - y_positive[0]) / (np.log(x_positive[-1]) - np.log(x_positive[0]))
        b = 1.0
        c = np.mean(y_positive) - a * np.mean(np.log(x_positive))

        return [a, b, c]

class LogarithmicFunction3D(ParametricFunction):
    def __call__(self, x: np.ndarray, y: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
        epsilon = 1e-10
        return a * np.log(np.maximum(b * x, epsilon)) + c * np.log(np.maximum(d * y, epsilon)) + e

    def initial_guess(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[float]:
        z_range = np.max(z) - np.min(z)
        x_log_range = np.log(np.maximum(np.max(x), 1e-10)) - np.log(np.maximum(np.min(x), 1e-10))
        y_log_range = np.log(np.maximum(np.max(y), 1e-10)) - np.log(np.maximum(np.min(y), 1e-10))
        
        a = z_range / max(x_log_range, 1e-10)
        b = 1.0
        c = z_range / max(y_log_range, 1e-10)
        d = 1.0
        e = np.min(z)
        
        return [a, b, c, d, e]

class PowerLawFunction2D(ParametricFunction):
    def __call__(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        epsilon = 1e-10
        return a * np.power(np.maximum(x, epsilon), b) + c

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        epsilon = 1e-10
        positive_mask = (x > 0) & (y > 0)
        x_positive = x[positive_mask]
        y_positive = y[positive_mask]

        if len(x_positive) < 2:
            return [1.0, 1.0, np.mean(y)]

        log_x = np.log(x_positive)
        log_y = np.log(y_positive - np.min(y_positive) + epsilon)

        try:
            slope, intercept = np.polyfit(log_x, log_y, 1)
            a = np.exp(intercept)
            b = slope
            c = np.min(y)
        except np.linalg.LinAlgError:
            a = 1.0
            b = 1.0
            c = np.mean(y)

        return [a, b, c]

class PowerLawFunction3D(ParametricFunction):
    def __call__(self, x: np.ndarray, y: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        epsilon = 1e-10
        # Use float64 for higher precision
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        # Clip the bases to avoid negative numbers in the power function
        safe_x = np.clip(np.maximum(x, epsilon), epsilon, 1e300)
        safe_y = np.clip(np.maximum(y, epsilon), epsilon, 1e300)
        # Clip the exponents to avoid overflow
        safe_b = np.clip(b, -100, 100)
        safe_c = np.clip(c, -100, 100)
        # Compute the result and clip to avoid overflow
        result = a * np.power(safe_x, safe_b) * np.power(safe_y, safe_c) + d
        return np.clip(result, -1e300, 1e300)

    def initial_guess(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[float]:
        a = np.max(z) - np.min(z)
        b = 1.0
        c = 1.0
        d = np.min(z)
        return [a, b, c, d]

class LogisticFunction2D(ParametricFunction):
    def __call__(self, x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        exponent = np.clip(-b * (x - c), -700, 700)
        return d + (a - d) / (1 + np.exp(exponent))

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        a = np.max(y)
        d = np.min(y)
        c = x[np.argmax(np.gradient(y))]
        b = 1
        return [a, b, c, d]

class LogisticFunction3D(ParametricFunction):
    def __call__(self, x: np.ndarray, y: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
        # Clip the exponent to avoid overflow
        exponent = np.clip(-b * (x - c) - d * (y - e), -700, 700)
        return f + (a - f) / (1 + np.exp(exponent))

    def initial_guess(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[float]:
        a = np.max(z)
        b = 1.0
        c = np.mean(x)
        d = 1.0
        e = np.mean(y)
        f = np.min(z)
        return [a, b, c, d, e, f]

class HyperbolicTangentFunction2D(ParametricFunction):
    def __call__(self, x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        return a * np.tanh(b * (x - c)) + d

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        a = (np.max(y) - np.min(y)) / 2
        b = 1
        c = np.mean(x)
        d = np.mean(y)
        return [a, b, c, d]

class HyperbolicTangentFunction3D(ParametricFunction):
    def __call__(self, x: np.ndarray, y: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
        return a * np.tanh(b * (x - c) + d * (y - e)) + f

    def initial_guess(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[float]:
        a = (np.max(z) - np.min(z)) / 2
        b = 1.0
        c = np.mean(x)
        d = 1.0
        e = np.mean(y)
        f = np.mean(z)
        return [a, b, c, d, e, f]

class PolynomialFunction2D(ParametricFunction):
    def __init__(self, degree: int):
        self.degree = degree

    def __call__(self, x: np.ndarray, *params: float) -> np.ndarray:
        return np.polyval(params, x)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        return list(np.polyfit(x, y, self.degree))

class PolynomialFunction3D(ParametricFunction):
    def __init__(self, degree: int):
        self.degree = degree

    def __call__(self, x: np.ndarray, y: np.ndarray, *params: float) -> np.ndarray:
        result = np.zeros_like(x)
        idx = 0
        for i in range(self.degree + 1):
            for j in range(i + 1):
                if idx < len(params):
                    result += params[idx] * (x ** (i - j)) * (y ** j)
                    idx += 1
        return result

    def initial_guess(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[float]:
        A = np.zeros((len(x), (self.degree + 1) * (self.degree + 2) // 2))
        idx = 0
        for i in range(self.degree + 1):
            for j in range(i + 1):
                A[:, idx] = (x ** (i - j)) * (y ** j)
                idx += 1
        params, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        return list(params)

def create_function(name: str, is_3d: bool) -> ParametricFunction:
    """Factory function to create parametric functions."""
    base_name = name.replace(' 3D', '')  # Remove '3D' suffix if present
    function_classes = {
        "Linear": LinearFunction2D if not is_3d else LinearFunction3D,
        "Quadratic": QuadraticFunction2D if not is_3d else QuadraticFunction3D,
        "Exponential": ExponentialFunction2D if not is_3d else ExponentialFunction3D,
        "Gaussian": GaussianFunction2D if not is_3d else GaussianFunction3D,
        "Sine": SineFunction2D if not is_3d else SineFunction3D,
        "Logarithmic": LogarithmicFunction2D if not is_3d else LogarithmicFunction3D,
        "Power Law": PowerLawFunction2D if not is_3d else PowerLawFunction3D,
        "Logistic": LogisticFunction2D if not is_3d else LogisticFunction3D,
        "Hyperbolic Tangent": HyperbolicTangentFunction2D if not is_3d else HyperbolicTangentFunction3D
    }
    if base_name.startswith("Polynomial"):
        degree = int(base_name.split("degree")[1].strip().split(")")[0])
        if degree > 3:
            raise ValueError(f"Polynomial functions of degree > 3 are not supported")
        return (PolynomialFunction2D if not is_3d else PolynomialFunction3D)(degree)
    
    if base_name not in function_classes:
        raise ValueError(f"Unknown function type: {name}")
    
    return function_classes[base_name]()
