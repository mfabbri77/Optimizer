import numpy as np
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class DataError(Exception):
    """Custom exception for data-related errors."""
    pass

def read_data(filename: str) -> Tuple[np.ndarray, ...]:
    """
    Reads data from an external text file.
    Supports both 2D and 3D data.
    
    :param filename: Name of the file to read
    :return: Tuple containing the data (x, y) for 2D or (x, y, z) for 3D
    :raises DataError: If there's an issue with reading or processing the data
    """
    data: dict = {}
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
            raise DataError(f"File {filename} contains an invalid number of dimensions: {len(data)}")
    except FileNotFoundError:
        raise DataError(f"File {filename} not found")
    except ValueError as e:
        raise DataError(f"Error parsing data in file {filename}: {str(e)}")
    except KeyError as e:
        raise DataError(f"Missing required data column in file {filename}: {str(e)}")
    except Exception as e:
        raise DataError(f"Unexpected error reading file {filename}: {str(e)}")

def save_residuals_to_file(filename: str, x: np.ndarray, y: np.ndarray, residuals: np.ndarray, z: Optional[np.ndarray] = None) -> None:
    """
    Saves the residuals data to a file in the format compatible with the program.

    :param filename: Name of the file to save the data to
    :param x: x-coordinate data
    :param y: y-coordinate data
    :param residuals: residual values
    :param z: z-coordinate data (optional, for 3D data)
    :raises DataError: If there's an issue with saving or processing the data
    """
    try:
        # Input validation
        if not all(isinstance(arr, np.ndarray) for arr in [x, y, residuals]):
            raise DataError("Input data must be numpy arrays")
        
        if not (len(x) == len(y) == len(residuals)):
            raise DataError("All input arrays must have the same length")

        if z is not None and len(z) != len(x):
            raise DataError("z array must have the same length as other arrays")

        # Prepare data for saving
        data_to_save = [('x', x)]
        if z is None:
            # 2D case: residuals replace y
            data_to_save.append(('y', residuals))
        else:
            # 3D case: y stays as is, residuals replace z
            data_to_save.extend([('y', y), ('z', residuals)])

        # Save data to file
        with open(filename, 'w') as f:
            for name, array in data_to_save:
                array_str = np.array2string(array, separator=',', threshold=np.inf, max_line_width=np.inf)
                f.write(f"{name}={array_str}\n")

        logger.info(f"Residuals data successfully saved to {filename}")

    except IOError as e:
        error_msg = f"IOError while saving file {filename}: {str(e)}"
        logger.error(error_msg)
        raise DataError(error_msg)
    except ValueError as e:
        error_msg = f"ValueError while processing data: {str(e)}"
        logger.error(error_msg)
        raise DataError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error while saving residuals to {filename}: {str(e)}"
        logger.error(error_msg)
        raise DataError(error_msg)

def validate_data(*args: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Validates the input data.
    
    :param args: Arrays of data to validate
    :return: Validated data arrays
    :raises DataError: If data validation fails
    """
    if not all(len(arg) == len(args[0]) for arg in args):
        raise DataError("All data arrays must have the same length")
    
    if any(np.isnan(arg).any() or np.isinf(arg).any() for arg in args):
        raise DataError("Data contains NaN or Inf values")
    
    return args

