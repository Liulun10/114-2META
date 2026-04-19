import numpy as np
from base_algo import BaseAlgorithm

class ABC(BaseAlgorithm):
    def run(self):
        # ABC 參數設定
        n_employed = self.pop_size // 2
        n_onlooker = self.pop_size // 2
        limit = self.dim * self.pop_size # 蜜源被放棄的閾值
        
        # 1. 初始化蜜源
        foods = np.random.uniform(self.lb, self.ub, (n_employed, self.dim))
        fitness = np.zeros(n_employed)
        trial = np.zeros(n_employed) # 記錄蜜源多久沒被改進
        
        def get_fitness(sol):
            f = self.obj_func(sol)
            # 將目標函數值轉換為 ABC 的適應度 (Fitness) 概念
            return 1 / (1 + f) if f >= 0 else 1 + abs(f)

        # 初始化適應度
        for i in range(n_employed):
            fitness[i] = get_fitness(foods[i])
            
        gbest_score = float('inf')
        gbest_pos = None
        convergence_curve = []
        mean_fitness_curve = []

        for t in range(self.max_iter):
            # --- A. 雇傭蜂階段 (Employed Bee Phase) ---
            for i in range(n_employed):
                k = np.random.choice([j for j in range(n_employed) if j != i])
                phi = np.random.uniform(-1, 1, self.dim)
                
                new_sol = foods[i] + phi * (foods[i] - foods[k])
                new_sol = np.clip(new_sol, self.lb, self.ub)
                
                new_fit = get_fitness(new_sol)
                if new_fit > fitness[i]:
                    foods[i] = new_sol
                    fitness[i] = new_fit
                    trial[i] = 0
                else:
                    trial[i] += 1

            # --- B. 觀察蜂階段 (Onlooker Bee Phase) ---
            probs = fitness / np.sum(fitness)
            for _ in range(n_onlooker):
                i = np.random.choice(range(n_employed), p=probs)
                k = np.random.choice([j for j in range(n_employed) if j != i])
                phi = np.random.uniform(-1, 1, self.dim)
                
                new_sol = foods[i] + phi * (foods[i] - foods[k])
                new_sol = np.clip(new_sol, self.lb, self.ub)
                
                new_fit = get_fitness(new_sol)
                if new_fit > fitness[i]:
                    foods[i] = new_sol
                    fitness[i] = new_fit
                    trial[i] = 0
                else:
                    trial[i] += 1

            # 更新當前全域最佳解，並統一計算本代所有蜜源的目標函數值
            current_scores = np.array([self.obj_func(foods[i]) for i in range(n_employed)])
            for i in range(n_employed):
                if current_scores[i] < gbest_score:
                    gbest_score = current_scores[i]
                    gbest_pos = foods[i].copy()
            
            # --- C. 偵查蜂階段 (Scout Bee Phase) ---
            max_trial_idx = np.argmax(trial)
            if trial[max_trial_idx] > limit:
                foods[max_trial_idx] = np.random.uniform(self.lb, self.ub, self.dim)
                fitness[max_trial_idx] = get_fitness(foods[max_trial_idx])
                trial[max_trial_idx] = 0
                
            convergence_curve.append(gbest_score)
            mean_fitness_curve.append(float(np.mean(current_scores)))
            
        return gbest_pos, gbest_score, convergence_curve, mean_fitness_curve