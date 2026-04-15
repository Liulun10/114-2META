import numpy as np

def F2(x):
    """
    Schwefel's Problem 2.22
    """
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F9(x):
    """
    Rastrigin Function (具有大量局部最佳解)
    """
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)