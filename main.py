import benchmarks
from algorithms.pso import PSO
from algorithms.abc import ABC as BeeAlgo
from algorithms.ga import GA
from utils.visualizer import plot_convergence_curve, plot_box_result, plot_3d_benchmarks
import numpy as np
import time
import os
import csv

def run_experiment(func_name, max_iter=100, pop_size=50, runs=50):#, dim=30
    # 1. 從 benchmark 獲取函數細節
    details = benchmarks.get_function_details(func_name)
    if details is None:
        print(f"找不到函數 {func_name} 的定義")
        return

    obj_func = getattr(benchmarks, func_name)
    dim = details['dim']
    bounds = details['range'] # 這裡會拿到 [-10, 10] 或 [[-5, 10], [0, 15]]
    
    all_runs_best_scores = []
    all_runs_curves = []

    print(f"\n" + "═"*50)
    print(f"啟動實驗: {func_name} (Dim: {dim}, Runs: {runs})")

    exp_start_time = time.time()
    
    for r in range(runs):
        # ps0, ga, abc 都在這裡改，不需要的直接註解
        model = PSO(obj_func, dim, bounds, max_iter, pop_size)
        # model = GA(obj_func, dim, bounds, max_iter, pop_size)

        best_pos,best_score, curve = model.run()
        # model = BeeAlgo(
        #     obj_func=obj_func, 
        #     dim=dim, 
        #     search_range=bounds, 
        #     pop_size=pop_size, 
        #     max_iter=max_iter
        # )
        # best_score,best_pos, curve = model.run()  # 注意 ABC 的回傳順序與 PSO 不同，請根據實際情況調整

        all_runs_best_scores.append(best_score)
        all_runs_curves.append(curve)
        
        # 每 10 次顯示一次進度，避免畫面太亂
        if (r + 1) % 10 == 0:
            print(f"  > Progress: {r + 1}/{runs} runs...")

    exp_duration = time.time() - exp_start_time

    # 計算統計數據
    avg_score = np.mean(all_runs_best_scores)
    std_score = np.std(all_runs_best_scores)
    print(f"完成實驗: {func_name} 耗時: {exp_duration:.2f}s, 平均值: {avg_score:.8f}, 標準差: {std_score:.8f}, 最佳值: {np.min(all_runs_best_scores):.8f}")

    # --- 關鍵：產出該函數專屬的圖表 ---
    plot_convergence_curve(
        all_runs_curves, 
        title=f"PSO Convergence: {func_name}", 
        function_name=func_name,
        algorithm_used = "PSO"
    )
    # plot_convergence_curve(
    #     all_runs_curves, 
    #     title=f"ABC Convergence: {func_name}", 
    #     function_name=func_name
    # )
    
    plot_box_result(
        [all_runs_best_scores], 
        algorithm_names=["PSO"], 
        function_name=func_name,
        algorithm_used = "PSO"
    )
    # plot_box_result(
    #     [all_runs_best_scores], 
    #     algorithm_names=["ABC"], 
    #     function_name=func_name
    # )

    return {
        "Function":  func_name,
        "Runs":      runs,
        "Mean":      f"{avg_score:.8f}",
        "Std":       f"{std_score:.8f}",
        "Best":      f"{best_score:.8f}",
        "Time(s)":   f"{exp_duration:.2f}"
    }

def save_results_csv(results, filename="results_PSO_100iters.csv"):
    # filename手動改成實驗的model name
    """將所有實驗結果存成 CSV 至 results 資料夾"""
    folder = "results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, filename)
    fieldnames = ["Function", "Runs", "Mean", "Std", "Best", "Time(s)"]
    # fieldnames = ["Function", "Time(s)"]
 
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
 
    print(f"實驗結果已儲存至: {path}")

def main():
    # 這裡定義你想要跑的所有函數清單
    # 你可以從 benchmarks.functions.keys() 抓全部，也可以手動指定
    
    target_functions = ["F2", "F6", "F9", "F11", "F13", "F15", "F17"]
    all_results = []

    # print("正在產生 benchmark 函數 3D 地形圖...")
    # plot_3d_benchmarks(benchmarks)
    
    for func_name in target_functions:
        try:
            result = run_experiment(func_name, max_iter=1000, pop_size=50, runs=50)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"執行 {func_name} 時發生錯誤: {e}")
    

    print("\n" + "═"*50)
    print(f"所有函數實驗完成！請至 results 資料夾查看結果")

    # 儲存彙整 CSV
    if all_results:
        save_results_csv(all_results)

if __name__ == "__main__":
    main()

