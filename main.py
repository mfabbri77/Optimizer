import logging
from data_handling import read_data, DataError
from parametric_functions import create_function
from optimization import select_best_function, advanced_optimization
from visualization import plot_results, plot_residuals, generate_report
from utils import setup_logging
from typing import List

logger = logging.getLogger(__name__)

def main():
    setup_logging()

if __name__ == "__main__":
    main()