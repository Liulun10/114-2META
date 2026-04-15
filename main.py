import benchmark
from algorithms.pso import PSO
from utils.visualizer import plot_convergence_curve, plot_box_result
import numpy as np

def run_experiment(func_name, dim=30, max_iter=500, pop_size=50, runs=50):
    """
    針對單一函數執行完整的 50 Runs 實驗並產出圖表
    """
    obj_func = benchmark.functions[func_name]
    bounds = (-100, 100) # 大部分基準函數的預設範圍
    
    all_runs_best_scores = []
    all_runs_curves = []

    print(f"\n" + "═"*50)
    print(f"啟動實驗: {func_name} (Dim: {dim}, Runs: {runs})")
    
    for r in range(runs):
        # ps0, ga, abc 都在這裡改，不需要的直接註解
        pso = PSO(obj_func, dim, bounds, max_iter, pop_size)
        best_pos, best_score, curve = pso.run()
        
        all_runs_best_scores.append(best_score)
        all_runs_curves.append(curve)
        
        # 每 10 次顯示一次進度，避免畫面太亂
        if (r + 1) % 10 == 0:
            print(f"  > Progress: {r + 1}/{runs} runs...")

    # 計算統計數據
    avg_score = np.mean(all_runs_best_scores)
    std_score = np.std(all_runs_best_scores)
    print(f"完成實驗: {func_name} 平均值: {avg_score:.2e}, 標準差: {std_score:.2e}, 最佳值: {np.min(all_runs_best_scores):.2e}")

    # --- 關鍵：產出該函數專屬的圖表 ---
    plot_convergence_curve(
        all_runs_curves, 
        title=f"PSO Convergence: {func_name}", 
        function_name=func_name
    )
    
    plot_box_result(
        [all_runs_best_scores], 
        algorithm_names=["PSO"], 
        function_name=func_name
    )

def main():
    # 這裡定義你想要跑的所有函數清單
    # 你可以從 benchmarks.functions.keys() 抓全部，也可以手動指定
    target_functions = ["F2", "F6", "F9", "F11", "F13", "F15", "F17"]
    
    for func_name in target_functions:
        try:
            run_experiment(func_name)
        except Exception as e:
            print(f"執行 {func_name} 時發生錯誤: {e}")

    print("\n" + "═"*50)
    print("所有實驗已完成，請至 results 資料夾查看結果。")

if __name__ == "__main__":
    main()

