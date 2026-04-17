import numpy as np
from base_algo import BaseAlgorithm

class PSO(BaseAlgorithm):
    def __init__(self, obj_func, dim, bounds, max_iter, pop_size, w=0.8, c1=1.5, c2=1.5):
        super().__init__(obj_func, dim, bounds, max_iter, pop_size)
        self.w = w    # 慣性權重
        self.c1 = c1  # 個體學習因子
        self.c2 = c2  # 群體學習因子

        # 支援對稱邊界 [-a, a] 或非對稱邊界 [[-a, b], [-c, d], ...]
        bounds_arr = np.array(bounds)
        if bounds_arr.ndim == 1:
            # 對稱邊界：每個維度相同
            self.lb = np.full(dim, bounds_arr[0], dtype=float)
            self.ub = np.full(dim, bounds_arr[1], dtype=float)
        else:
            # 非對稱邊界：每個維度獨立 (shape: [dim, 2])
            self.lb = bounds_arr[:, 0].astype(float)
            self.ub = bounds_arr[:, 1].astype(float)
        
        # 初始化粒子位置與速度
        self.X = np.random.uniform(self.lb, self.ub, (pop_size, dim))
        self.V = np.random.uniform(-1, 1, (pop_size, dim))
        
        # 初始化個體最佳與全域最佳
        self.pBest_X = self.X.copy()
        self.pBest_score = np.array([float('inf')] * pop_size)
        self.gBest_X = None
        self.gBest_score = float('inf')

        # 速度上限：搜尋範圍的 20%，防止粒子爆炸
        self.Vmax = (self.ub - self.lb) * 0.2

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

            # 2. 更新速度與位置 每個粒子、每個維度都有獨立的隨機擾動
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.V = (self.w * self.V + 
                      self.c1 * r1 * (self.pBest_X - self.X) + 
                      self.c2 * r2 * (self.gBest_X - self.X))
            
            # 限制速度在 [-Vmax, Vmax]，防止粒子飛出
            self.V = np.clip(self.V, -self.Vmax, self.Vmax)
            
            self.X = self.X + self.V

            # 3. 邊界處理 (防止噴出定義域)
            self.X = np.clip(self.X, self.lb, self.ub)
            
            convergence_curve.append(self.gBest_score)
            
        return self.gBest_X, self.gBest_score, convergence_curve