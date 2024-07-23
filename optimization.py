import time
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Callable, Tuple, Optional
from parametric_functions import ParametricFunction
from multiprocessing import Pool, cpu_count, Value, Array, set_start_method
from functools import partial
import ctypes

# Set the start method to 'spawn' for better compatibility, especially on macOS
try:
    set_start_method('spawn')
except RuntimeError:
    pass  # The start method might already be set, which is fine

logger = logging.getLogger(__name__)

# Shared variables for progress tracking
progress = Value(ctypes.c_int, 0)
total_iterations = Value(ctypes.c_int, 0)

def objective_function(params, func, x, y, z=None):
    try:
        if z is None:
            pred = func(x, *params)
            residuals = y - pred
        else:
            pred = func(x, y, *params)
            residuals = z - pred
        
        # Clip residuals to avoid overflow in square
        clipped_residuals = np.clip(residuals, -1e150, 1e150)
        return np.sum(clipped_residuals**2)
    except (OverflowError, FloatingPointError, ValueError):
        return np.inf  # Return a large value if any numerical error occurs

def run_basin_hopping(args):
    func, x, y, z, initial_guess, niter = args
    objective = partial(objective_function, func=func, x=x, y=y, z=z)
    try:
        result = basinhopping(objective, initial_guess, niter=niter)
        return result.x, result.fun
    except Exception as e:
        logger.error(f"Error in basin hopping: {str(e)}")
        return initial_guess, np.inf

def run_differential_evolution(args):
    func, x, y, z, bounds, maxiter, popsize = args
    objective = partial(objective_function, func=func, x=x, y=y, z=z)
    try:
        result = differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize)
        return result.x, result.fun
    except Exception as e:
        logger.error(f"Error in differential evolution: {str(e)}")
        return [np.mean(b) for b in bounds], np.inf

def parallel_basin_hopping(func, x, y, z, initial_guess, niter=100, n_processes=None, progress_callback=None):
    if n_processes is None:
        n_processes = cpu_count()

    n_params = len(initial_guess)
    starting_points = [initial_guess + np.random.randn(n_params) for _ in range(n_processes)]

    args_list = [(func, x, y, z, start_point, niter // n_processes) for start_point in starting_points]

    with Pool(n_processes) as pool:
        try:
            results = []
            for i, result in enumerate(pool.imap_unordered(run_basin_hopping, args_list)):
                results.append(result)
                if progress_callback:
                    progress_callback(int((i + 1) / n_processes * 50))  # 50% for basin hopping
        except Exception as e:
            logger.error(f"Error in parallel basin hopping: {str(e)}")
            pool.terminate()
            raise

    best_params, best_score = min(results, key=lambda x: x[1])
    return best_params, best_score

def parallel_differential_evolution(func, x, y, z, bounds, maxiter=1000, popsize=15, n_processes=None, progress_callback=None):
    if n_processes is None:
        n_processes = cpu_count()

    args_list = [(func, x, y, z, bounds, maxiter // n_processes, popsize) for _ in range(n_processes)]

    with Pool(n_processes) as pool:
        try:
            results = []
            for i, result in enumerate(pool.imap_unordered(run_differential_evolution, args_list)):
                results.append(result)
                if progress_callback:
                    progress_callback(50 + int((i + 1) / n_processes * 50))  # 50-100% for differential evolution
        except Exception as e:
            logger.error(f"Error in parallel differential evolution: {str(e)}")
            pool.terminate()
            raise

    best_params, best_score = min(results, key=lambda x: x[1])
    return best_params, best_score

def advanced_optimization(func: ParametricFunction, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray], 
                          progress_callback: Callable = None, status_callback: Callable = None, 
                          time_callback: Callable = None, fast_mode: bool = False) -> List[float]:
    if func.is3D():
        initial_guess = func.initial_guess(x, y, z)
    else:
        initial_guess = func.initial_guess(x, y)
    
    if status_callback:
        status_callback("Starting parallel basin-hopping optimization...")
    
    niter_bh = 100 if fast_mode else 500
    best_params_bh, score_bh = parallel_basin_hopping(func, x, y, z, initial_guess, niter=niter_bh, progress_callback=progress_callback)

    if status_callback:
        status_callback("Starting parallel differential evolution...")
    
    bounds = [(-10, 10)] * len(initial_guess)  # Adjust bounds as needed
    maxiter_de = 1000 if fast_mode else 5000
    best_params_de, score_de = parallel_differential_evolution(func, x, y, z, bounds, maxiter=maxiter_de, progress_callback=progress_callback)

    # Calcola gli score usando evaluate_function
    score_bh = evaluate_function(func, x, y, z, best_params_bh)
    score_de = evaluate_function(func, x, y, z, best_params_de)

    if score_bh > score_de:
        if status_callback:
            status_callback(f"Basin-hopping solution selected as best with score={score_bh}")
        return best_params_bh
    else:
        if status_callback:
            status_callback(f"Differential evolution solution selected as best with score={score_de}")
        return best_params_de

def select_best_function(x: np.ndarray, y: np.ndarray, functions: List[ParametricFunction], 
                         z: Optional[np.ndarray] = None, 
                         progress_callback: Callable = None, 
                         status_callback: Callable = None, 
                         time_callback: Callable = None) -> Tuple[ParametricFunction, List[float]]:
    best_score = float('-inf')
    best_func = None
    best_params = None

    for i, func in enumerate(functions):
        try:
            if func.is3D():
                initial_guess = func.initial_guess(x, y, z)
            else:
                initial_guess = func.initial_guess(x, y)
            
            params = optimize_parameters(func, x, y, z, initial_guess)
            score = evaluate_function(func, x, y, z, params)

            if score > best_score:
                best_score = score
                best_func = func
                best_params = params

            if status_callback:
                status_callback(f"Tested {str(func)}: score = {score}")
        
        except Exception as e:
            logger.error(f"Error with {str(func)}: {str(e)}")

        if progress_callback:
            progress_callback(int((i + 1) / len(functions) * 40))

    if status_callback:
        if best_func:
            status_callback(f"Best function selected: {str(best_func)}")
        else:
            status_callback("No suitable function found")
    
    return best_func, best_params

def optimize_parameters(func: ParametricFunction, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray], initial_guess: List[float], method: str = 'Nelder-Mead') -> List[float]:
    objective = partial(objective_function, func=func, x=x, y=y, z=z)
    result = minimize(objective, initial_guess, method=method)
    return result.x.tolist()

def evaluate_function(func: ParametricFunction, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray], params: List[float]) -> float:
    if z is None:
        z_pred = func(x, *params)
        z_true = y
    else:
        z_pred = func(x, y, *params)
        z_true = z
    
    valid_mask = np.isfinite(z_pred)
    z_valid = z_true[valid_mask]
    z_pred_valid = z_pred[valid_mask]

    if len(z_valid) == 0:
        return 0  # Worst score

    # R-squared score
    r2 = r2_score(z_valid, z_pred_valid)
    
    # Normalized RMSE
    rmse = np.sqrt(mean_squared_error(z_valid, z_pred_valid))
    z_range = np.max(z_valid) - np.min(z_valid)
    normalized_rmse = rmse / z_range if z_range != 0 else 1
    
    # Cross-validation
    kf = KFold(n_splits=5)
    cv_scores = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if z is not None:
            z_train, z_test = z[train_index], z[test_index]
        else:
            z_train, z_test = y_train, y_test
        
        train_params = optimize_parameters(func, x_train, y_train, z_train if z is not None else None, params)
        
        if z is None:
            z_test_pred = func(x_test, *train_params)
        else:
            z_test_pred = func(x_test, y_test, *train_params)
        
        valid_mask = np.isfinite(z_test_pred)
        z_test_valid = z_test[valid_mask]
        z_test_pred_valid = z_test_pred[valid_mask]
        
        if len(z_test_valid) > 0:
            cv_scores.append(r2_score(z_test_valid, z_test_pred_valid))
    
    mean_cv_r2 = np.mean(cv_scores) if cv_scores else 0

    # Combine metrics
    raw_score = 0.4 * r2 + 0.4 * (1 - normalized_rmse) + 0.2 * mean_cv_r2
    
    # Normalize the score to [0, 1]
    #normalized_score = (raw_score + 1) / 2  # Assuming raw_score is in [-1, 1]
    normalized_score = raw_score 
    normalized_score = np.clip(normalized_score, 0, 1)  # Ensure it's in [0, 1]

    return normalized_score

def update_progress():
    global progress, total_iterations
    while True:
        with progress.get_lock():
            current = progress.value
            total = total_iterations.value
        if total > 0:
            percent = (current / total) * 100
            print(f"Progress: {percent:.2f}%")
        elif current > 0:
            print(f"Progress: {current} iterations completed")
        if current >= total and total > 0:
            break
        time.sleep(1)  # Update every second

