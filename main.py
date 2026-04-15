from benchmark import F2  # 確保 benchmarks.py 已有 F2 定義
from algorithms.pso import PSO
from utils.visualizer import plot_convergence_curve, plot_box_result
import numpy as np

def main():
    # 實驗參數
    dim = 30
    bounds = (-100, 100)
    max_iter = 500
    pop_size = 50
    runs = 50  # 執行 50 次
    test_func_name = "F2" # 自行設定
    
    all_runs_best_scores = []    # 紀錄每次 run 的最終結果
    all_runs_curves = []         # 紀錄每次 run 的完整收斂過程

    print(f"Starting 50 runs of PSO on F2...")

    for r in range(runs):
        pso = PSO(test_func_name, dim, bounds, max_iter, pop_size)
        best_pos, best_score, curve = pso.run()
        
        all_runs_best_scores.append(best_score)
        all_runs_curves.append(curve)
        
        if (r + 1) % 10 == 0:
            print(f"Completed {r + 1}/{runs} runs. Current Avg: {np.mean(all_runs_best_scores):.4e}")

    # --- 統計分析 ---
    print("\n" + "="*30)
    print(f"Final Statistics for PSO on F2:")
    print(f"Best: {np.min(all_runs_best_scores):.4e}")
    print(f"Worst: {np.max(all_runs_best_scores):.4e}")
    print(f"Mean: {np.mean(all_runs_best_scores):.4e}")
    print(f"Std: {np.std(all_runs_best_scores):.4e}")
    print("="*30)

    # --- 呼叫視覺化 ---
    print("Generating charts...")
    plot_convergence_curve(all_runs_curves, title=f"PSO Convergence (F2, Dim={dim})")
    plot_box_result([all_runs_best_scores], algorithm_names=["PSO"])

if __name__ == "__main__":
    main()

