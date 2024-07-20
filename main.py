import logging
from data_handling import read_data
from parametric_functions import LinearFunction, QuadraticFunction, ExponentialFunction, GaussianFunction
from optimization import select_best_function, optimize_parameters, advanced_optimization
from visualization import plot_results, plot_residuals, generate_report
from utils import setup_logging

logger = logging.getLogger(__name__)

def main():
    setup_logging()
    
    # Data reading
    filename = "dati_2d.txt"  # or "dati_3d.txt" for 3D data
    try:
        data = read_data(filename)
        if len(data) == 2:
            x, y = data
            logger.info("Processing 2D data")
        elif len(data) == 3:
            x, y, z = data
            logger.info("Processing 3D data")
        else:
            raise ValueError("Invalid data")
    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        return

    # List of functions to test
    functions = [LinearFunction(), QuadraticFunction(), ExponentialFunction(), GaussianFunction()]

    # Select the best function
    best_func = select_best_function(x, y, functions)
    logger.info(f"Best function selected: {best_func.__class__.__name__}")

    # Optimize parameters
    initial_guess = best_func.initial_guess(x, y)
    optimized_params = optimize_parameters(best_func, x, y, initial_guess)
    logger.info(f"Initial optimization complete. Parameters: {optimized_params}")

    # Advanced optimization
    final_params = advanced_optimization(best_func, x, y)
    logger.info(f"Advanced optimization complete. Final parameters: {final_params}")

    # Visualize results
    plot_results(x, y, best_func, final_params)
    plot_residuals(x, y, best_func, final_params)
    generate_report(best_func, final_params, x, y)

if __name__ == "__main__":
    main()