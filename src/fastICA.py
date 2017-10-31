import numpy as np

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


    def _whitening(X):
        """
        Apply whitening on input data.

        Keyword arguments:
        X -- ndarray of shape (N, M)
        """
        raise NotImplementedError

    def single_component_extraction(X):
        """
        Calculate a single component from input data

        Keyword arguments:
        X -- ndarray of shape (N, M)
        """
        raise NotImplementedError