import numpy as np
import scipy
from scipy.special import expit
import scipy.sparse


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)

    def func(self, x):
        # Compute Ax
        Ax = self.matvec_Ax(x)
        
        # Compute margins = b_i * a_i^T x
        margins = self.b * Ax
        
        # Compute logistic loss using logaddexp to avoid overflow
        # log(1 + exp(-z)) = logaddexp(0, -z)
        logistic_loss = np.mean(np.logaddexp(0, -margins))
        
        # Compute regularization term
        reg_term = 0.5 * self.regcoef * np.dot(x, x)
        
        return logistic_loss + reg_term

    def grad(self, x):
        # Compute Ax
        Ax = self.matvec_Ax(x)
        
        # Compute margins = b_i * a_i^T x
        margins = self.b * Ax
        
        # Compute sigmoid probabilities using expit to avoid overflow
        # sigmoid(-z) = 1 / (1 + exp(z)) = expit(-z)
        probabilities = expit(-margins)
        
        # Compute gradient of logistic loss: -A^T(b * (1 - sigmoid)) / m
        grad_loss = -self.matvec_ATx(self.b * (1 - probabilities)) / self.m
        
        # Add regularization gradient
        grad_reg = self.regcoef * x
        
        return grad_loss + grad_reg

    def hess(self, x):
        # Compute Ax
        Ax = self.matvec_Ax(x)
        
        # Compute margins = b_i * a_i^T x
        margins = self.b * Ax
        
        # Compute sigmoid probabilities
        probabilities = expit(-margins)
        
        # Compute diagonal matrix s = sigmoid * (1 - sigmoid) * b^2 / m
        # Note: b_i^2 = 1 since b_i ∈ {-1, 1}
        s = probabilities * (1 - probabilities) / self.m
        
        # Compute Hessian of logistic loss: A^T * diag(s) * A
        hess_loss = self.matmat_ATsA(s)
        
        # Add regularization Hessian
        hess_reg = self.regcoef * np.eye(len(x))
        
        return hess_loss + hess_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        # Cache for precomputed values
        self._last_x = None
        self._last_Ax = None
        self._last_d = None
        self._last_Ad = None

    def func_directional(self, x, d, alpha):
        # Check if we can reuse precomputed values
        if (self._last_x is not None and np.array_equal(x, self._last_x) and 
            self._last_d is not None and np.array_equal(d, self._last_d)):
            Ax = self._last_Ax
            Ad = self._last_Ad
        else:
            # Compute and cache Ax and Ad
            Ax = self.matvec_Ax(x)
            Ad = self.matvec_Ax(d)
            self._last_x = x.copy()
            self._last_Ax = Ax.copy()
            self._last_d = d.copy()
            self._last_Ad = Ad.copy()
        
        # Compute A(x + alpha*d) = Ax + alpha*Ad
        Ax_alpha = Ax + alpha * Ad
        
        # Compute margins
        margins = self.b * Ax_alpha
        
        # Compute logistic loss
        logistic_loss = np.mean(np.logaddexp(0, -margins))
        
        # Compute regularization term for x + alpha*d
        x_alpha = x + alpha * d
        reg_term = 0.5 * self.regcoef * np.dot(x_alpha, x_alpha)
        
        return logistic_loss + reg_term

    def grad_directional(self, x, d, alpha):
        # Check if we can reuse precomputed values
        if (self._last_x is not None and np.array_equal(x, self._last_x) and 
            self._last_d is not None and np.array_equal(d, self._last_d)):
            Ax = self._last_Ax
            Ad = self._last_Ad
        else:
            # Compute and cache Ax and Ad
            Ax = self.matvec_Ax(x)
            Ad = self.matvec_Ax(d)
            self._last_x = x.copy()
            self._last_Ax = Ax.copy()
            self._last_d = d.copy()
            self._last_Ad = Ad.copy()
        
        # Compute A(x + alpha*d) = Ax + alpha*Ad
        Ax_alpha = Ax + alpha * Ad
        
        # Compute margins
        margins = self.b * Ax_alpha
        
        # Compute sigmoid probabilities
        probabilities = expit(-margins)
        
        # Compute directional derivative of logistic loss
        # d/d_alpha [logistic_loss] = - (b * (1 - probabilities))^T Ad / m
        grad_dir_loss = -np.dot(self.b * (1 - probabilities), Ad) / self.m
        
        # Compute directional derivative of regularization
        x_alpha = x + alpha * d
        grad_dir_reg = self.regcoef * np.dot(x_alpha, d)
        
        return grad_dir_loss + grad_dir_reg


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    # Define matrix-vector multiplication functions that work with both dense and sparse matrices
    def matvec_Ax(x):
        return A.dot(x)
    
    def matvec_ATx(x):
        return A.T.dot(x)
    
    def matmat_ATsA(s):
        # s is a vector of length m (number of samples)
        # We need to compute A^T * diag(s) * A
        
        if scipy.sparse.issparse(A):
            # For sparse matrices, use efficient computation
            S = scipy.sparse.diags(s)
            return (A.T @ S @ A).tocsc()
        else:
            # For dense matrices
            return A.T @ (s[:, np.newaxis] * A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    grad = np.zeros(n)
    f_x = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_x_eps = func(x + eps * e_i)
        grad[i] = (f_x_eps - f_x) / eps
    
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    hess = np.zeros((n, n))
    f_x = func(x)
    
    # Precompute f(x + eps * e_i) for all i
    f_x_eps_i = np.zeros(n)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_x_eps_i[i] = func(x + eps * e_i)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            
            # Compute f(x + eps * e_i + eps * e_j)
            f_x_eps_ij = func(x + eps * e_i + eps * e_j)
            
            # Apply finite difference formula
            hess[i, j] = (f_x_eps_ij - f_x_eps_i[i] - f_x_eps_i[j] + f_x) / (eps * eps)
    
    return hess


# Test functions to verify the implementation
def test_oracle_implementation():
    """Test function to verify the oracle implementation"""
    print("Testing oracle implementation...")
    
    # Generate small test data
    np.random.seed(42)
    m, n = 10, 3  # small dimensions for testing
    A = np.random.randn(m, n)
    b = np.random.choice([-1, 1], size=m)
    regcoef = 0.1
    
    # Create oracle
    oracle = create_log_reg_oracle(A, b, regcoef, 'usual')
    
    # Test point
    x_test = np.random.randn(n)
    
    print(f"Test point x: {x_test}")
    
    # Test function value
    f_val = oracle.func(x_test)
    print(f"Function value: {f_val:.6f}")
    
    # Test gradient
    grad_analytic = oracle.grad(x_test)
    grad_numeric = grad_finite_diff(oracle.func, x_test)
    
    print(f"Analytic gradient: {grad_analytic}")
    print(f"Numeric gradient:  {grad_numeric}")
    print(f"Gradient difference norm: {np.linalg.norm(grad_analytic - grad_numeric):.2e}")
    
    # Test Hessian
    hess_analytic = oracle.hess(x_test)
    hess_numeric = hess_finite_diff(oracle.func, x_test)
    
    print(f"Analytic Hessian shape: {hess_analytic.shape}")
    print(f"Numeric Hessian shape:  {hess_numeric.shape}")
    print(f"Hessian difference norm: {np.linalg.norm(hess_analytic - hess_numeric):.2e}")
    
    # Test with sparse matrix
    print("\nTesting with sparse matrix...")
    A_sparse = scipy.sparse.csr_matrix(A)
    oracle_sparse = create_log_reg_oracle(A_sparse, b, regcoef, 'usual')
    
    f_val_sparse = oracle_sparse.func(x_test)
    grad_sparse = oracle_sparse.grad(x_test)
    
    print(f"Function value difference (dense vs sparse): {abs(f_val - f_val_sparse):.2e}")
    print(f"Gradient difference norm (dense vs sparse): {np.linalg.norm(grad_analytic - grad_sparse):.2e}")

if __name__ == "__main__":
    test_oracle_implementation()
