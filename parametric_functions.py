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

    def __str__(self):
        return "Linear"

class QuadraticFunction(ParametricFunction):
    def __call__(self, x, a, b, c):
        return a * x**2 + b * x + c

    def initial_guess(self, x, y):
        a = (y[-1] - 2*y[len(y)//2] + y[0]) / ((x[-1] - x[0])**2)
        b = (y[-1] - y[0]) / (x[-1] - x[0]) - a * (x[-1] + x[0])
        c = y[0] - a * x[0]**2 - b * x[0]
        return [a, b, c]

    def __str__(self):
        return "Quadratic"

class ExponentialFunction(ParametricFunction):
    def __call__(self, x, a, b, c):
        return a * np.exp(b * x) + c

    def initial_guess(self, x, y):
        a = (y[-1] - y[0]) / (np.exp(x[-1]) - np.exp(x[0]))
        b = np.log((y[-1] - y[0]) / (x[-1] - x[0]))
        c = y[0] - a * np.exp(b * x[0])
        return [a, b, c]

    def __str__(self):
        return "Exponential"

class GaussianFunction(ParametricFunction):
    def __call__(self, x, a, b, c):
        return a * np.exp(-((x - b)**2) / (2 * c**2))

    def initial_guess(self, x, y):
        a = np.max(y)
        b = x[np.argmax(y)]
        c = (x[-1] - x[0]) / 4
        return [a, b, c]

    def __str__(self):
        return "Gaussian"

class SineFunction(ParametricFunction):
    def __call__(self, x, a, b, c, d):
        return a * np.sin(b * x + c) + d

    def initial_guess(self, x, y):
        a = (np.max(y) - np.min(y)) / 2
        b = 2 * np.pi / (x[-1] - x[0])
        c = 0
        d = np.mean(y)
        return [a, b, c, d]

    def __str__(self):
        return "Sine"

class LogarithmicFunction(ParametricFunction):
    def __call__(self, x, a, b, c):
        # Aggiungiamo un piccolo valore epsilon per evitare log(0)
        epsilon = 1e-10
        return a * np.log(np.maximum(b * x, epsilon)) + c

    def initial_guess(self, x, y):
        # Filtriamo i valori non positivi
        positive_mask = x > 0
        x_positive = x[positive_mask]
        y_positive = y[positive_mask]

        if len(x_positive) < 2:
            # Se non abbiamo abbastanza punti positivi, usiamo una stima predefinita
            return [1.0, 1.0, np.mean(y)]

        # Calcoliamo la stima iniziale usando solo i valori positivi
        a = (y_positive[-1] - y_positive[0]) / (np.log(x_positive[-1]) - np.log(x_positive[0]))
        b = 1.0  # Possiamo iniziare con b = 1 come stima ragionevole
        c = np.mean(y_positive) - a * np.mean(np.log(x_positive))

        return [a, b, c]

    def __str__(self):
        return "Logarithmic"

class PowerLawFunction(ParametricFunction):
    def __call__(self, x, a, b, c):
        # Aggiungiamo un piccolo valore epsilon per evitare x^b quando x è negativo o zero
        epsilon = 1e-10
        return a * np.power(np.maximum(x, epsilon), b) + c

    def initial_guess(self, x, y):
        # Definiamo epsilon qui
        epsilon = 1e-10
        
        # Filtriamo i valori non positivi
        positive_mask = (x > 0) & (y > 0)
        x_positive = x[positive_mask]
        y_positive = y[positive_mask]

        if len(x_positive) < 2:
            # Se non abbiamo abbastanza punti positivi, usiamo una stima predefinita
            return [1.0, 1.0, np.mean(y)]

        # Calcoliamo la stima iniziale usando solo i valori positivi
        log_x = np.log(x_positive)
        log_y = np.log(y_positive - np.min(y_positive) + epsilon)

        try:
            slope, intercept = np.polyfit(log_x, log_y, 1)
            a = np.exp(intercept)
            b = slope
            c = np.min(y)  # Stima di c come il valore minimo di y
        except np.linalg.LinAlgError:
            # In caso di errore nel fitting, usiamo una stima predefinita
            a = 1.0
            b = 1.0
            c = np.mean(y)

        return [a, b, c]

    def __str__(self):
        return "Power Law"

class PolynomialFunction(ParametricFunction):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, x, *params):
        return np.polyval(params, x)

    def initial_guess(self, x, y):
        return np.polyfit(x, y, self.degree)

    def __str__(self):
        return f"Polynomial (degree {self.degree})"

class LogisticFunction(ParametricFunction):
    def __call__(self, x, a, b, c, d):
        return d + (a - d) / (1 + np.exp(-b * (x - c)))

    def initial_guess(self, x, y):
        a = np.max(y)
        d = np.min(y)
        c = x[np.argmax(np.gradient(y))]
        b = 1
        return [a, b, c, d]
 
    def __str__(self):
        return "Logistic"

class HyperbolicTangentFunction(ParametricFunction):
    def __call__(self, x, a, b, c, d):
        return a * np.tanh(b * (x - c)) + d

    def initial_guess(self, x, y):
        a = (np.max(y) - np.min(y)) / 2
        b = 1
        c = np.mean(x)
        d = np.mean(y)
        return [a, b, c, d]

    def __str__(self):
        return "Hyperbolic Tangent"

