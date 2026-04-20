from abc import ABC, abstractmethod
import numpy as np

class BaseAlgorithm(ABC):
    def __init__(self, obj_func, dim, search_range, max_iter, pop_size):
        self.obj_func = obj_func
        self.dim = dim
        # 如果傳進來的是單一範圍如 [-10, 10]，轉成列表格式
        if isinstance(search_range[0], (int, float)):
            self.lb = np.array([search_range[0]] * dim)
            self.ub = np.array([search_range[1]] * dim)
        else:
            # 如果是 [[-5, 10], [0, 15]] 這種格式
            self.lb = np.array([r[0] for r in search_range])
            self.ub = np.array([r[1] for r in search_range])
        self.max_iter = max_iter
        self.pop_size = pop_size

    @abstractmethod
    def run(self):
        pass