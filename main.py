import benchmarks
from algorithms.pso import PSO
from algorithms.abc import ABC as BeeAlgo
from algorithms.ga import GA
from utils.visualizer import plot_convergence_curve, plot_box_result, plot_3d_benchmarks
import numpy as np
import time
import os
import csv


"""
實驗執行前須手動修改以下幾個地方：
1. main.py 的 run_experiment 裡的 model = PSO(...) 這行，改成你要跑的演算法（PSO、GA、ABC）
2. main.py 的 save_results_csv 裡的 filename 參數及iteration數量，改成對應的檔名（例如 results_GA_1000iters.csv）
3. plot_convergence_curve, plot_box_result 裡的 algorithm_used 參數，改成對應的演算法名稱（例如 "GA"）
"""

def run_experiment(func_name, max_iter=1000, pop_size=50, runs=50):#, dim=30
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
    all_runs_mean_curves = []  
    all_runs_best_pos = []

    print(f"\n" + "═"*50)
    print(f"啟動實驗: {func_name} (Dim: {dim}, Runs: {runs})")

    exp_start_time = time.time()
    
    for r in range(runs):
        # ps0, ga, abc 都在這裡改，不需要的直接註解
        # model = PSO(obj_func, dim, bounds, max_iter, pop_size)
        model = GA(obj_func, dim, bounds, max_iter, pop_size)

        
        # model = BeeAlgo(
        #     obj_func=obj_func, 
        #     dim=dim, 
        #     search_range=bounds, 
        #     pop_size=pop_size, 
        #     max_iter=max_iter
        # )
        best_pos,best_score, curve, mean_curve = model.run()

        all_runs_best_scores.append(best_score)
        all_runs_curves.append(curve)
        all_runs_mean_curves.append(mean_curve) 
        all_runs_best_pos.append(best_pos)
        
        # 每 10 次顯示一次進度，避免畫面太亂
        if (r + 1) % 10 == 0:
            print(f"  > Progress: {r + 1}/{runs} runs...")

    exp_duration = time.time() - exp_start_time

    # 計算統計數據
    avg_score = np.mean(all_runs_best_scores)
    std_score = np.std(all_runs_best_scores)
    print(f"完成實驗: {func_name} 耗時: {exp_duration:.2f}s, 平均值: {avg_score:.8f}, 標準差: {std_score:.8f}, 最佳值: {np.min(all_runs_best_scores):.8f}")

    # 問題 4 的 avg mean fitness（末代）
    avg_mean_fitness_last = np.mean([c[-1] for c in all_runs_mean_curves])

    # 問題 6 的距離誤差
    target_x = details['target_x']  # 從 benchmarks.get_function_details 取得
    if target_x is not None:
        dists = [np.linalg.norm(pos - np.array(target_x)) for pos in all_runs_best_pos]
        dist_mean, dist_std = np.mean(dists), np.std(dists)
        dist_mean_str = f"{dist_mean:.8f}"
        dist_std_str = f"{dist_std:.8f}"    
    else:
        dist_mean_str, dist_std_str = "N/A", "N/A"

    # --- 關鍵：產出該函數專屬的圖表 ---
    plot_convergence_curve(
        all_runs_curves, 
        title=f"GA Convergence: {func_name}", 
        function_name=func_name,
        algorithm_used = "GA",
        iters = max_iter
    )

    
    plot_box_result(
        [all_runs_best_scores], 
        algorithm_names=["GA"], 
        function_name=func_name,
        algorithm_used = "GA",
        iters = max_iter        
    )


    return {
        "Function":  func_name,
        "Runs":      runs,
        "Mean":      f"{avg_score:.8f}",
        "Std":       f"{std_score:.8f}",
        "Best":      f"{best_score:.8f}",
        "DistMean":  dist_mean_str,
        "DistStd":   dist_std_str,
        "AvgMean_Fitness":   f"{avg_mean_fitness_last:.8f}",
        "Time(s)":   f"{exp_duration:.2f}"
    }

def save_results_csv(results, filename="results_GA_1000iters.csv"):
    # filename手動改成實驗的model name
    """將所有實驗結果存成 CSV 至 results 資料夾"""
    folder = "results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, filename)
    fieldnames = ["Function", "Runs", "Mean", "Std", "Best", "DistMean", "DistStd", "AvgMean_Fitness", "Time(s)"]
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

