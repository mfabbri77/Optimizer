from abc import ABC, abstractmethod
import numpy as np

class ParametricFunction(ABC):
    @abstractmethod
    def __call__(self, x, *params):
        pass

    @abstractmethod
    def initial_guess(self, x, y):
        pass

class LinearFunction(ParametricFunction):
    def __call__(self, x, a, b):
        return a * x + b

    def initial_guess(self, x, y):
        a = (y[-1] - y[0]) / (x[-1] - x[0])
        b = y[0] - a * x[0]
        return [a, b]

class QuadraticFunction(ParametricFunction):
    def __call__(self, x, a, b, c):
        return a * x**2 + b * x + c

    def initial_guess(self, x, y):
        a = (y[-1] - 2*y[len(y)//2] + y[0]) / ((x[-1] - x[0])**2)
        b = (y[-1] - y[0]) / (x[-1] - x[0]) - a * (x[-1] + x[0])
        c = y[0] - a * x[0]**2 - b * x[0]
        return [a, b, c]

class ExponentialFunction(ParametricFunction):
    def __call__(self, x, a, b, c):
        return a * np.exp(b * x) + c

    def initial_guess(self, x, y):
        a = (y[-1] - y[0]) / (np.exp(x[-1]) - np.exp(x[0]))
        b = np.log((y[-1] - y[0]) / (x[-1] - x[0]))
        c = y[0] - a * np.exp(b * x[0])
        return [a, b, c]

class GaussianFunction(ParametricFunction):
    def __call__(self, x, a, b, c):
        return a * np.exp(-((x - b)**2) / (2 * c**2))

    def initial_guess(self, x, y):
        a = np.max(y)
        b = x[np.argmax(y)]
        c = (x[-1] - x[0]) / 4
        return [a, b, c]