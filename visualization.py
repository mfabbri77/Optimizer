import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def plot_results(x, y, func, params, ax=None):
    """
    Visualizes the optimization results.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(x, y, color='blue', label='Original data', s=10, marker='.', alpha=0.5)
    
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = func(x_fit, *params)
    ax.plot(x_fit, y_fit, color='red', label='Optimized function')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Optimization Results')
    ax.legend()
    ax.grid(True)
    
    if ax is None:
        plt.show()

def plot_residuals(x, y, func, params, ax=None):
    """
    Plots the residuals of the fit.
    """
    y_fit = func(x, *params)
    residuals = y - y_fit

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(x, residuals, color='blue', s=10, marker='.', alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('X')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    ax.grid(True)
    
    if ax is None:
        plt.show()

def generate_report(func, params, x, y):
    """
    Generates a report with optimized parameters and performance metrics.
    """
    y_fit = func(x, *params)
    r2 = r2_score(y, y_fit)
    rmse = np.sqrt(mean_squared_error(y, y_fit))

    report = f"""
    Optimization Report
    -------------------
    Function: {func.__class__.__name__}
    Optimized Parameters: {params}
    
    Performance Metrics:
    R-squared: {r2:.4f}
    RMSE: {rmse:.4f}
    """

    return report