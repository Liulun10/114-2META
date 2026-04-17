import numpy as np
from base_algo import BaseAlgorithm


class GA(BaseAlgorithm):
    """
    Real-valued Genetic Algorithm (實數編碼遺傳演算法)

    Parameters
    ----------
    obj_func     : callable  目標函數 (越小越好)
    dim          : int       搜尋維度
    bounds       : list      邊界，格式同 PSO：
                             對稱 [-a, a] 或非對稱 [[-a,b],[c,d],...]
    max_iter     : int       最大代數
    pop_size     : int       族群大小 (建議為偶數)
    pc           : float     交配率 (crossover probability)，預設 0.9
    pm           : float     每個基因的突變率，預設 1/dim
    eta_c        : float     SBX 分佈指數，越大子代越靠近父代，預設 20
    eta_m        : float     Polynomial Mutation 分佈指數，預設 20
    tournament_k : int       錦標賽選擇人數，預設 3
    elitism      : int       精英保留數量，預設 1
    """

    def __init__(self, obj_func, dim, bounds, max_iter, pop_size,
                 pc=0.9, pm=None, eta_c=20, eta_m=20,
                 tournament_k=3, elitism=1):
        super().__init__(obj_func, dim, bounds, max_iter, pop_size)

        # ── 邊界處理（與 PSO 一致，支援非對稱邊界）──────────────────────
        bounds_arr = np.array(bounds)
        if bounds_arr.ndim == 1:
            self.lb = np.full(dim, bounds_arr[0], dtype=float)
            self.ub = np.full(dim, bounds_arr[1], dtype=float)
        else:
            self.lb = bounds_arr[:, 0].astype(float)
            self.ub = bounds_arr[:, 1].astype(float)

        # ── GA 參數 ───────────────────────────────────────────────────────
        self.pc           = pc
        self.pm           = pm if pm is not None else 1.0 / dim
        self.eta_c        = eta_c
        self.eta_m        = eta_m
        self.tournament_k = tournament_k
        self.elitism      = elitism

    # ═══════════════════════════════════════════════════════════════════════
    #  Selection：錦標賽選擇 (Tournament Selection)
    # ═══════════════════════════════════════════════════════════════════════
    def _tournament_select(self, pop, fitness):
        """隨機抽 k 個個體，回傳適應值最小者的複製"""
        candidates = np.random.choice(len(pop), self.tournament_k, replace=False)
        best = candidates[np.argmin(fitness[candidates])]
        return pop[best].copy()

    # ═══════════════════════════════════════════════════════════════════════
    #  Crossover：Simulated Binary Crossover (SBX)
    # ═══════════════════════════════════════════════════════════════════════
    def _sbx_crossover(self, p1, p2):
        """
        SBX 模擬二進位交配的分佈特性，適合連續空間最佳化。
        eta_c 越大，子代越集中於父代附近（開發）；越小則探索範圍越廣。
        """
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() > self.pc:
            return c1, c2  # 不觸發交配，直接複製

        for i in range(self.dim):
            if np.random.rand() <= 0.5 and abs(p1[i] - p2[i]) > 1e-10:
                u = np.random.rand()
                eta = self.eta_c
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))

                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])

        c1 = np.clip(c1, self.lb, self.ub)
        c2 = np.clip(c2, self.lb, self.ub)
        return c1, c2

    # ═══════════════════════════════════════════════════════════════════════
    #  Mutation：Polynomial Mutation
    # ═══════════════════════════════════════════════════════════════════════
    def _polynomial_mutation(self, individual):
        """
        每個基因以機率 pm 進行突變。
        Polynomial Mutation 根據當前值到邊界的距離自適應縮放擾動量，
        避免突變後飛出邊界。
        """
        child = individual.copy()
        eta = self.eta_m
        span = self.ub - self.lb  # 各維度的搜尋寬度

        for i in range(self.dim):
            if np.random.rand() < self.pm:
                u = np.random.rand()
                delta_l = (child[i] - self.lb[i]) / span[i]
                delta_r = (self.ub[i] - child[i]) / span[i]

                if u <= 0.5:
                    delta = (2.0 * u + (1 - 2.0 * u) * (1 - delta_l) ** (eta + 1)) \
                            ** (1.0 / (eta + 1)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1 - u) + 2.0 * (u - 0.5) * (1 - delta_r) ** (eta + 1)) \
                            ** (1.0 / (eta + 1))

                child[i] += delta * span[i]

        return np.clip(child, self.lb, self.ub)

    # ═══════════════════════════════════════════════════════════════════════
    #  Main Loop
    # ═══════════════════════════════════════════════════════════════════════
    def run(self):
        # ── 初始化族群，均勻分佈在搜尋空間 ─────────────────────────────
        pop     = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self.obj_func(ind) for ind in pop])

        best_idx   = np.argmin(fitness)
        best_pos   = pop[best_idx].copy()
        best_score = fitness[best_idx]

        convergence_curve = []

        for _ in range(self.max_iter):
            new_pop = []

            # ── 精英保留：最優的 elitism 個直接進入下一代 ───────────────
            elite_idxs = np.argsort(fitness)[:self.elitism]
            for idx in elite_idxs:
                new_pop.append(pop[idx].copy())

            # ── 產生子代，填滿族群至 pop_size ───────────────────────────
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitness)
                p2 = self._tournament_select(pop, fitness)

                c1, c2 = self._sbx_crossover(p1, p2)

                c1 = self._polynomial_mutation(c1)
                c2 = self._polynomial_mutation(c2)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            # ── 更新族群與適應值 ─────────────────────────────────────────
            pop     = np.array(new_pop)
            fitness = np.array([self.obj_func(ind) for ind in pop])

            # ── 更新全域最佳 ─────────────────────────────────────────────
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_score:
                best_score = fitness[gen_best_idx]
                best_pos   = pop[gen_best_idx].copy()

            convergence_curve.append(best_score)

        return best_pos, best_score, convergence_curve