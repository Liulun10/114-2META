from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    def __init__(self, obj_func, dim, bounds, max_iter, pop_size):
        self.obj_func = obj_func
        self.dim = dim
        self.bounds = bounds  # 例如: (-100, 100)
        self.max_iter = max_iter
        self.pop_size = pop_size

    @abstractmethod
    def run(self):
        pass