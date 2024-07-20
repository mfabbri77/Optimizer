import logging
import concurrent.futures

def setup_logging():
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def parallel_execution(func, args_list):
    """
    Executes a function in parallel for multiple sets of arguments.
    
    :param func: Function to execute
    :param args_list: List of argument tuples for each function call
    :return: List of results
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda args: func(*args), args_list))
    return results

# Add more utility functions as needed