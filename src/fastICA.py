import numpy as np
import scipy.linalg as lin

class fastICA:
    """
        Class for the fast Independent Component Analysis algorithm.
        Inspired by:
        * https://www.cs.helsinki.fi/u/ahyvarin/papers/TNN99new.pdf
        * https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf

        Args:
            general (bool): true if algorithm needs to be general, false if the algorithm needs to be robust. Regulates
                which functions are being used for determining non gaussianity. Described in:
                https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf

        Attributes:
            _f (anonymous function): Non quadratic non linear function used for calculating non gaussianity.
            _g (anonymous function): The derivative of f. Used for calculating non gaussianity.
            _dg (anonymous function): The second derivative of f. Used for calculating non gaussianity.

    """

    def __init__(self, general=True):
        self._f = lambda x: np.log(np.cosh(x))
        self._g = lambda x: np.tanh(x)
        self._dg = lambda x: 1 - np.tanh(x)
        if not general:
            self._f = lambda x: -np.power(np.e, -np.divide(np.power(x, 2), 2))
            self._g = lambda x: x*np.power(np.e, -np.divide(np.power(x, 2), 2))
            self._dg = lambda x: (1-x*x)*np.power(np.e, -np.divide(np.power(x, 2), 2))

    def _whitening(self, X):
        """
        Apply whitening on input data. The data needs to be whitened before components can be extracted.

        Keyword arguments:
        X -- ndarray of shape (N, M)
        """
        X -= X.mean(axis=1, keepdims=True)  # Centre the data
        w, v = np.linalg.eig(np.cov(X))  # Calculate eigenvalues and vectors
        return np.matmul(np.matmul(np.matmul(v, lin.sqrtm(np.diag(w))), v.T), X)  # Return the whitened matrix

    def single_component_extraction(self, X):
        """
        Calculate a single component from input data

        Keyword arguments:
        X -- ndarray of shape (N, M)
        """
        raise NotImplementedError

    def multiple_component_extraction(self, X, c):
        """
        Calculate a single component from input data

        Keyword arguments:
        X -- ndarray of shape (N, M)
        c -- number of desired components, c <= N.
        """
        if c > X.shape[0]:
            raise AssertionError('C needs to be smaller or equal to N.')

        raise NotImplementedError
