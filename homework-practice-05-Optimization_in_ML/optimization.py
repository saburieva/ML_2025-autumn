import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.optimize import line_search
from scipy.linalg import cho_factor, cho_solve
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c

        alpha_0 = self.alpha_0 if previous_alpha is None else previous_alpha

        if self._method == 'Armijo':
            return self._armijo_search(oracle, x_k, d_k, alpha_0)
        
        elif self._method == 'Wolfe':
            # Use scipy's line_search with Wolfe conditions
            result = line_search(
                f=lambda x: oracle.func(x),
                myfprime=lambda x: oracle.grad(x),
                xk=x_k,
                pk=d_k,
                gfk=oracle.grad(x_k),
                old_fval=oracle.func(x_k),
                c1=self.c1,
                c2=self.c2
            )
            
            alpha = result[0]
            if alpha is not None:
                return alpha
            else:
                # Fallback to Armijo if Wolfe fails
                return self._armijo_search(oracle, x_k, d_k, alpha_0)
        
        else:
            return None

    def _armijo_search(self, oracle, x_k, d_k, alpha_0):
        """
        Backtracking line search satisfying Armijo condition.
        """
        phi_0 = oracle.func_directional(x_k, d_k, 0.0)
        derphi_0 = oracle.grad_directional(x_k, d_k, 0.0)
        
        alpha = alpha_0
        c1 = self.c1
        
        while alpha > 1e-12:  # Prevent infinite loop with very small alpha
            phi_alpha = oracle.func_directional(x_k, d_k, alpha)
            
            # Armijo condition: phi(alpha) <= phi(0) + c1 * alpha * derphi(0)
            if phi_alpha <= phi_0 + c1 * alpha * derphi_0:
                return alpha
            
            alpha *= 0.5
        
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    start_time = datetime.now()
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        try:
            # Compute function value and gradient
            f_k = oracle.func(x_k)
            g_k = oracle.grad(x_k)
            grad_norm = np.linalg.norm(g_k)
            
            # Check for computational errors
            if (np.isinf(f_k) or np.isnan(f_k) or 
                np.any(np.isinf(g_k)) or np.any(np.isnan(g_k))):
                return x_k, 'computational_error', history
            
            # Store trace information
            if trace:
                current_time = (datetime.now() - start_time).total_seconds()
                history['time'].append(current_time)
                history['func'].append(f_k)
                history['grad_norm'].append(grad_norm)
                if x_k.size <= 2:
                    history['x'].append(x_k.copy())
            
            # Display debug information
            if display:
                print(f"Iteration {iteration}: f(x) = {f_k:.6f}, ||grad|| = {grad_norm:.6f}")
            
            # Check stopping criterion
            if grad_norm < tolerance:
                return x_k, 'success', history
            
            # Compute search direction (negative gradient for gradient descent)
            d_k = -g_k
            
            # Line search
            alpha = line_search_tool.line_search(oracle, x_k, d_k)
            
            if alpha is None:
                return x_k, 'computational_error', history
            
            # Update x
            x_k = x_k + alpha * d_k
            
        except (ValueError, LinAlgError) as e:
            if display:
                print(f"Error at iteration {iteration}: {e}")
            return x_k, 'computational_error', history
    
    # Check final gradient norm
    final_grad_norm = np.linalg.norm(oracle.grad(x_k))
    if final_grad_norm < tolerance:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    start_time = datetime.now()
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        try:
            # Compute function value, gradient and Hessian
            f_k = oracle.func(x_k)
            g_k = oracle.grad(x_k)
            H_k = oracle.hess(x_k)
            grad_norm = np.linalg.norm(g_k)
            
            # Check for computational errors
            if (np.isinf(f_k) or np.isnan(f_k) or 
                np.any(np.isinf(g_k)) or np.any(np.isnan(g_k)) or
                np.any(np.isinf(H_k)) or np.any(np.isnan(H_k))):
                return x_k, 'computational_error', history
            
            # Store trace information
            if trace:
                current_time = (datetime.now() - start_time).total_seconds()
                history['time'].append(current_time)
                history['func'].append(f_k)
                history['grad_norm'].append(grad_norm)
                if x_k.size <= 2:
                    history['x'].append(x_k.copy())
            
            # Display debug information
            if display:
                print(f"Iteration {iteration}: f(x) = {f_k:.6f}, ||grad|| = {grad_norm:.6f}")
            
            # Check stopping criterion
            if grad_norm < tolerance:
                return x_k, 'success', history
            
            # Solve Newton system: H_k * d_k = -g_k using Cholesky decomposition
            try:
                c, lower = cho_factor(H_k)
                d_k = -cho_solve((c, lower), g_k)
            except LinAlgError:
                return x_k, 'newton_direction_error', history
            
            # Line search
            alpha = line_search_tool.line_search(oracle, x_k, d_k)
            
            if alpha is None:
                return x_k, 'computational_error', history
            
            # Update x
            x_k = x_k + alpha * d_k
            
        except (ValueError, LinAlgError) as e:
            if display:
                print(f"Error at iteration {iteration}: {e}")
            return x_k, 'computational_error', history
    
    # Check final gradient norm
    final_grad_norm = np.linalg.norm(oracle.grad(x_k))
    if final_grad_norm < tolerance:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


class LogRegL2OptimizedOracle:
    """
    Optimized oracle for L2-regularized logistic regression.
    """
    def __init__(self, A, b, regcoef):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        
        # Cache for matrix-vector products
        self._last_x = None
        self._last_Ax = None
        self._last_test_point = None
        self._last_test_Ax = None
        
    def _compute_Ax(self, x):
        """Compute A*x with caching."""
        if self._last_x is not None and np.array_equal(x, self._last_x):
            return self._last_Ax
        
        if self._last_test_point is not None and np.array_equal(x, self._last_test_point):
            return self._last_test_Ax
        
        Ax = self.A.dot(x)
        self._last_x = x.copy()
        self._last_Ax = Ax
        return Ax
    
    def func(self, x):
        """Function value."""
        Ax = self._compute_Ax(x)
        margins = self.b * Ax
        loss = np.mean(np.logaddexp(0, -margins))
        reg = 0.5 * self.regcoef * np.sum(x**2)
        return loss + reg
    
    def grad(self, x):
        """Gradient."""
        Ax = self._compute_Ax(x)
        margins = self.b * Ax
        sigmoid = 1.0 / (1.0 + np.exp(-margins))
        grad_loss = -self.A.T.dot(self.b * (1 - sigmoid)) / len(self.b)
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg
    
    def hess(self, x):
        """Hessian."""
        Ax = self._compute_Ax(x)
        margins = self.b * Ax
        sigmoid = 1.0 / (1.0 + np.exp(-margins))
        diag = sigmoid * (1 - sigmoid)
        
        # Hessian = A^T * D * A / n + regcoef * I
        n = len(self.b)
        hess_loss = self.A.T.dot((self.A * diag[:, np.newaxis])) / n
        hess_reg = self.regcoef * np.eye(len(x))
        return hess_loss + hess_reg
    
    def func_directional(self, x, d, alpha):
        """Function value in direction."""
        # Compute Ax and Ad if not cached
        if (self._last_x is None or not np.array_equal(x, self._last_x) or
            self._last_test_point is not None and np.array_equal(d, self._last_test_point - x)):
            Ax = self._compute_Ax(x)
            Ad = self.A.dot(d)
        else:
            Ax = self._last_Ax
            Ad = self.A.dot(d)
        
        # Cache the test point
        test_point = x + alpha * d
        test_Ax = Ax + alpha * Ad
        self._last_test_point = test_point.copy()
        self._last_test_Ax = test_Ax
        
        margins = self.b * test_Ax
        loss = np.mean(np.logaddexp(0, -margins))
        reg = 0.5 * self.regcoef * np.sum(test_point**2)
        return loss + reg
    
    def grad_directional(self, x, d, alpha):
        """Directional derivative."""
        # Compute Ax and Ad if not cached
        if (self._last_x is None or not np.array_equal(x, self._last_x) or
            self._last_test_point is not None and np.array_equal(d, self._last_test_point - x)):
            Ax = self._compute_Ax(x)
            Ad = self.A.dot(d)
        else:
            Ax = self._last_Ax
            Ad = self.A.dot(d)
        
        # Cache the test point
        test_point = x + alpha * d
        test_Ax = Ax + alpha * Ad
        self._last_test_point = test_point.copy()
        self._last_test_Ax = test_Ax
        
        margins = self.b * test_Ax
        sigmoid = 1.0 / (1.0 + np.exp(-margins))
        grad_loss = -self.b * (1 - sigmoid)
        grad_dir_loss = np.dot(Ad, grad_loss) / len(self.b)
        grad_dir_reg = self.regcoef * np.dot(test_point, d)
        return grad_dir_loss + grad_dir_reg
