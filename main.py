from benchmark import F2  # 假設你定義了 F2 函數
from algorithms.pso import PSO
import matplotlib.pyplot as plt

def main():
    # 設定參數
    dim = 30
    bounds = (-100, 100)
    max_iter = 500
    pop_size = 50
    runs = 50
    
    all_results = []

    for r in range(runs):
        pso = PSO(F2, dim, bounds, max_iter, pop_size)
        best_pos, best_score, curve = pso.run()
        all_results.append(best_score)
        print(f"Run {r+1}: Best Score = {best_score}")

    # 之後可以在這裡加入統計分析與繪圖邏輯
    print(f"\nAverage Best Score over 50 runs: {sum(all_results)/runs}")

if __name__ == "__main__":
    main()