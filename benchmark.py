import numpy as np

def F2(x):
    """
    Schwefel's Problem 2.22
    """
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F6(x):
    """Step Function (Discontinuous)"""
    return np.sum(np.floor(x + 0.5)**2)

def F9(x):
    """
    Rastrigin Function (具有大量局部最佳解)
    """
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def F11(x):
    part1 = np.sum(x**2) / 4000
    part2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return part1 - part2 + 1

def F13(x):
    """Ackley Function (Multimodal)"""
    dim = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / dim)
    return term1 + term2 + 20 + np.exp(1)

def F15(x):
    return np.sum(x**2)

def F17(x):
    """Penalized Function (Complex Multimodal)"""
    def y(xi):
        return 1 + (xi + 1) / 4
    
    dim = len(x)
    term1 = 10 * (np.sin(np.pi * y(x[0])))**2
    term2 = np.sum((y(x[:-1]) - 1)**2 * (1 + 10 * (np.sin(np.pi * y(x[1:])))**2))
    term3 = (y(x[-1]) - 1)**2
    
    # 懲罰項
    u = np.where(x > 10, 100 * (x - 10)**4, 
                 np.where(x < -10, 100 * (-x - 10)**4, 0))
    
    return (np.pi / dim) * (term1 + term2 + term3) + np.sum(u)

functions = {
    "F2": F2,
    "F6": F6,
    "F9": F9,
    "F11": F11,
    "F13": F13,
    "F15": F15,
    "F17": F17
}
