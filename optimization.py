import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import logging

logger = logging.getLogger(__name__)


def select_best_function(x, y, functions, progress_callback=None, status_callback=None, time_callback=None):
    """
    Selects the best function for the given data.
    """
    best_score = float('-inf')
    best_func = None

    for i, func in enumerate(functions):
        if status_callback:
            status_callback(f"Testing {str(func)} function...")
        
        try:
            initial_guess = func.initial_guess(x, y)
            params = optimize_parameters(func, x, y, initial_guess)
            score = evaluate_function(func, x, y, params)
        
            if score > best_score:
                best_score = score
                best_func = func
        except Exception as e:
            print(f"Error with {str(func)}: {str(e)}")
            continue  # Skip this function and move to the next one

        if progress_callback:
            progress_callback(int((i + 1) / len(functions) * 40))  # Use only 40% for function selection

    if status_callback:
        if best_func:
            status_callback(f"Best function selected: {str(best_func)}")
        else:
            status_callback("No suitable function found")
    
    return best_func

def optimize_parameters(func, x, y, initial_guess, method='Nelder-Mead'):
    """
    Optimizes parameters for a given function.
    """
    def objective(params):
        return np.sum((y - func(x, *params))**2)

    result = minimize(objective, initial_guess, method=method)
    return result.x

def advanced_optimization(func, x, y, progress_callback=None, status_callback=None, time_callback=None, fast_mode=False):
    """
    Applies advanced optimization strategies.
    """
    initial_guess = func.initial_guess(x, y)
    
    if status_callback:
        status_callback("Starting basin-hopping optimization...")
    
    # Strategy 1: Basin-hopping
    niter_bh = 10 if fast_mode else 100
    def basin_hopping_callback(x, f, accept):
        nonlocal bh_iter
        progress = 40 + int((bh_iter / niter_bh) * 30)
        if progress_callback:
            progress_callback(progress)
        if time_callback:
            time_callback(progress)
        if status_callback and bh_iter:
            status_callback(f"Basin-hopping iteration {bh_iter}/{niter_bh}")
        bh_iter += 1
        return False

    bh_iter = 0
    result_bh = basinhopping(lambda params: -evaluate_function(func, x, y, params),
                             initial_guess, niter=niter_bh, callback=basin_hopping_callback)
    
    if progress_callback:
        progress_callback(70)
    
    if status_callback:
        status_callback("Starting differential evolution...")
    
    # Strategy 2: Differential Evolution
    de_maxiter = 250 if fast_mode else 1000
    def differential_evolution_callback(xk, convergence=None):
        nonlocal de_iter
        progress = 70 + int((de_iter / de_maxiter) * 30)
        if progress_callback:
            progress_callback(progress)
        if time_callback:
            time_callback(progress)
        if status_callback and de_iter:
            status_callback(f"Differential evolution iteration {de_iter}/{de_maxiter}")
        de_iter += 1
        return False

    de_iter = 0
    bounds = [(-10, 10)] * len(initial_guess)
    result_de = differential_evolution(lambda params: -evaluate_function(func, x, y, params),
                                       bounds, callback=differential_evolution_callback,
                                       maxiter=de_maxiter, polish=True, popsize=4 if fast_mode else 16)
    
    if progress_callback:
        progress_callback(100)
    
    # Compare results and choose the best
    if evaluate_function(func, x, y, result_bh.x) > evaluate_function(func, x, y, result_de.x):
        if status_callback:
            status_callback("Basin-hopping solution selected as best")
        return result_bh.x
    else:
        if status_callback:
            status_callback("Differential evolution solution selected as best")
        return result_de.x


def evaluate_function(func, x, y, params):
    """
    Evaluates the function using multiple metrics.
    """
    y_pred = func(x, *params)
    
    # Rimuovi eventuali NaN o infiniti da y_pred
    valid_mask = np.isfinite(y_pred)
    y_valid = y[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if len(y_valid) == 0:
        return float('-inf')  # Restituisci un punteggio molto basso se non ci sono predizioni valide

    r2 = r2_score(y_valid, y_pred_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
    
    # Cross-validation
    kf = KFold(n_splits=5)
    cv_scores = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Optimize parameters on training data
        train_params = optimize_parameters(func, x_train, y_train, params)
        
        # Evaluate on test data
        y_test_pred = func(x_test, *train_params)
        
        # Rimuovi eventuali NaN o infiniti
        valid_mask = np.isfinite(y_test_pred)
        y_test_valid = y_test[valid_mask]
        y_test_pred_valid = y_test_pred[valid_mask]
        
        if len(y_test_valid) > 0:
            cv_scores.append(r2_score(y_test_valid, y_test_pred_valid))
    
    # Combine metrics (you can adjust the weights)
    score = 0.5 * r2 + 0.3 * (1 / (rmse + 1e-10)) + 0.2 * np.mean(cv_scores) if cv_scores else 0
    
    return score
    