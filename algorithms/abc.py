import numpy as np
#from base_algorithm import MetaHeuristic
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
        self.history = []

        for t in range(self.max_iter):
            # --- A. 雇傭蜂階段 (Employed Bee Phase) ---
            for i in range(n_employed):
                # 隨機選一個不同的蜜源來進行擾動
                k = np.random.choice([j for j in range(n_employed) if j != i])
                phi = np.random.uniform(-1, 1, self.dim)
                
                # 產生新解
                new_sol = foods[i] + phi * (foods[i] - foods[k])
                new_sol = np.clip(new_sol, self.lb, self.ub)
                
                new_fit = get_fitness(new_sol)
                if new_fit > fitness[i]: # 貪婪選擇
                    foods[i] = new_sol
                    fitness[i] = new_fit
                    trial[i] = 0
                else:
                    trial[i] += 1

            # --- B. 觀察蜂階段 (Onlooker Bee Phase) ---
            # 根據適應度計算被選中的機率 (輪盤法)
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

            # 更新當前全域最佳解
            for i in range(n_employed):
                current_score = self.obj_func(foods[i])
                if current_score < gbest_score:
                    gbest_score = current_score
                    gbest_pos = foods[i].copy()
            
            # --- C. 偵查蜂階段 (Scout Bee Phase) ---
            # 如果某蜜源超過 limit 次沒改進，則放棄該蜜源，隨機重新產生
            max_trial_idx = np.argmax(trial)
            if trial[max_trial_idx] > limit:
                foods[max_trial_idx] = np.random.uniform(self.lb, self.ub, self.dim)
                fitness[max_trial_idx] = get_fitness(foods[max_trial_idx])
                trial[max_trial_idx] = 0
                
            self.history.append(gbest_score)
            
        return gbest_score, gbest_pos, self.history
