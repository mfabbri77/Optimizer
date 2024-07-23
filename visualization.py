import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error

def plot_results(x, y, func, params, ax=None):
    """
    Visualizes the optimization results.
    """
    is_3d = func.is3D()
    
    if ax is None:
        if is_3d:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
        if is_3d and not isinstance(ax, Axes3D):
            raise ValueError("For 3D functions, ax must be a 3D Axes object")

    if is_3d:
        # For 3D functions, create a surface plot
        x_fit = np.linspace(min(x), max(x), 50)
        y_fit = np.linspace(min(y), max(y), 50)
        X, Y = np.meshgrid(x_fit, y_fit)
        Z = func(X.ravel(), Y.ravel(), *params).reshape(X.shape)
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        ax.scatter(x, y, func(x, y, *params), color='red', s=2, label='Data points', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Optimization Results')
        
        # Add a color bar to the correct figure
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    else:
        # For 2D functions, plot as before
        ax.scatter(x, y, color='blue', label='Original data', s=2, marker='.', alpha=0.5)
        
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = func(x_fit, *params)
        ax.plot(x_fit, y_fit, color='red', label='Optimized function')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Optimization Results')
    
    ax.legend()
    if not is_3d:
        ax.grid(True)
    
    plt.tight_layout()
    return fig, ax

def plot_residuals(x, y, z, func, params, ax=None, is_3d=False):
    """
    Plots the residuals of the fit.
    """
    if ax is None:
        if is_3d:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if is_3d:
        z_fit = func(x, y, *params)
        residuals = z - z_fit

        scatter = ax.scatter(x, y, residuals, c=residuals, cmap='coolwarm', s=2, alpha=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Residuals')
        ax.set_title('3D Residual Plot')

        # Add a color bar
        fig.colorbar(scatter, ax=ax, label='Residual Value')

        # Add a reference plane at z=0
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                           np.linspace(ylim[0], ylim[1], 10))
        Z = np.zeros_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.4, color='green')

    else:
        y_fit = func(x, *params)
        residuals = y - y_fit

        scatter = ax.scatter(x, residuals, c=residuals, cmap='coolwarm', s=2)
        ax.axhline(y=0, color='green')
        ax.set_xlabel('X')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')

        # Add a color bar
        fig.colorbar(scatter, ax=ax, label='Residual Value')

        ax.grid(True)

    fig.tight_layout()
    return ax, residuals

def generate_report(func, params, x, y):
    """
    Generates a report with optimized parameters and performance metrics.
    """
    is_3d = func.is3D()
    
    if is_3d:
        z = y  # For 3D functions, y is actually z
        y_fit = func(x, y, *params)
    else:
        y_fit = func(x, *params)

    r2 = r2_score(z if is_3d else y, y_fit)
    rmse = np.sqrt(mean_squared_error(z if is_3d else y, y_fit))

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
