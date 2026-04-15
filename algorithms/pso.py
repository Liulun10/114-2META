import numpy as np
from base_algo import BaseAlgorithm

class PSO(BaseAlgorithm):
    def __init__(self, obj_func, dim, bounds, max_iter, pop_size, w=0.8, c1=1.5, c2=1.5):
        super().__init__(obj_func, dim, bounds, max_iter, pop_size)
        self.w = w    # 慣性權重
        self.c1 = c1  # 個體學習因子
        self.c2 = c2  # 群體學習因子
        
        # 初始化粒子位置與速度
        self.X = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        self.V = np.random.uniform(-1, 1, (pop_size, dim))
        
        # 初始化個體最佳與全域最佳
        self.pBest_X = self.X.copy()
        self.pBest_score = np.array([float('inf')] * pop_size)
        self.gBest_X = None
        self.gBest_score = float('inf')

    def run(self):
        convergence_curve = []

        for t in range(self.max_iter):
            # 1. 計算適應值 (Fitness)
            for i in range(self.pop_size):
                score = self.obj_func(self.X[i])
                
                # 更新個體最佳 pBest
                if score < self.pBest_score[i]:
                    self.pBest_score[i] = score
                    self.pBest_X[i] = self.X[i].copy()
                
                # 更新全域最佳 gBest
                if score < self.gBest_score:
                    self.gBest_score = score
                    self.gBest_X = self.X[i].copy()

            # 2. 更新速度與位置
            r1, r2 = np.random.rand(2)
            self.V = (self.w * self.V + 
                      self.c1 * r1 * (self.pBest_X - self.X) + 
                      self.c2 * r2 * (self.gBest_X - self.X))
            
            self.X = self.X + self.V

            # 3. 邊界處理 (防止噴出定義域)
            self.X = np.clip(self.X, self.bounds[0], self.bounds[1])
            
            convergence_curve.append(self.gBest_score)
            
        return self.gBest_X, self.gBest_score, convergence_curve