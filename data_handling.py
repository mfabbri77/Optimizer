import numpy as np
import logging

logger = logging.getLogger(__name__)

def read_data(filename):
    """
    Reads data from an external text file.
    Supports both 2D and 3D data.
    
    :param filename: Name of the file to read
    :return: Tuple containing the data (x, y) for 2D or (x, y, z) for 3D
    """
    data = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                key, value_str = line.strip().split('=')
                key = key.strip()
                values = [float(v) for v in value_str.strip()[1:-1].split(',')]
                data[key] = np.array(values)
        
        if len(data) == 2:
            logger.info(f"Read 2D data from file {filename}")
            return validate_data(data['x'], data['y'])
        elif len(data) == 3:
            logger.info(f"Read 3D data from file {filename}")
            return validate_data(data['x'], data['y'], data['z'])
        else:
            raise ValueError(f"File {filename} contains an invalid number of dimensions: {len(data)}")
    except Exception as e:
        logger.error(f"Error reading file {filename}: {str(e)}")
        raise

def validate_data(*args):
    """
    Validates the input data.
    
    :param args: Arrays of data to validate
    :return: Validated data arrays
    """
    if not all(len(arg) == len(args[0]) for arg in args):
        raise ValueError("All data arrays must have the same length")
    
    if any(np.isnan(arg).any() or np.isinf(arg).any() for arg in args):
        raise ValueError("Data contains NaN or Inf values")
    
    return args