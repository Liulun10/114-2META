import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def save_plot(filename):
    """
    輔助函式：確保 results 資料夾存在並儲存圖片
    """
    folder = "results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    path = os.path.join(folder, filename)
    plt.savefig(path)
    print(f"圖表已儲存至: {path}")

def plot_convergence_curve(all_runs_curves, title="PSO Convergence Curve", function_name="UNKNOWN", algorithm_used="UNKOWN"):
    # 1. 增加圖表總寬度
    fig = plt.figure(figsize=(14, 6))
    
    # 2. 為右側文字預留空間 (left, bottom, right, top 範圍為 0~1)
    # 將 right 設為 0.8，表示圖表主體只畫到 80% 的位置，右邊留 20% 空白
    plt.subplots_adjust(right=0.8)
    
    curves = np.array(all_runs_curves)
    avg_curve = np.mean(curves, axis=0)
    
    # 計算統計值
    final_scores = curves[:, -1]
    mean_val = np.mean(final_scores)
    std_val = np.std(final_scores)
    best_final = np.min(final_scores)

    # 畫線邏輯
    for i in range(len(curves)):
        plt.plot(curves[i], color='gray', alpha=0.1)
    plt.plot(avg_curve, color='blue', linewidth=2, label='Average Convergence')
    
    # 3. 調整文字框樣式與位置
    stats_text = (
        f"Statistics (50 runs)\n"
        f"{'─'*22}\n"
        f"Function: {function_name}\n\n"
        f"Mean:\n {mean_val:.8f}\n\n"
        f"Std Dev:\n {std_val:.8f}\n\n"
        f"Final Best:\n {best_final:.8f}\n"
        f"{'─'*22}"
    )
    
    # x=1.05 代表放在圖表框線外一點點的位置
    plt.text(1.05, 0.5, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='center', linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='navy', alpha=0.8))

    plt.title(title, fontsize=14)
    plt.xlabel('Iteration')
    plt.ylabel('Best Score (Log Scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='upper right')
    
    save_plot(f"convergence_{function_name}_{algorithm_used}_100iters.png")
    plt.show()
    plt.close()

def plot_box_result(all_final_scores, algorithm_names=["PSO"], function_name="UNKNOWN", algorithm_used="UNKOWN"):
    """
    畫出 50 runs 最後結果的盒鬚圖，用於比較穩定性
    """
    plt.figure(figsize=(8, 6))
    plt.boxplot(all_final_scores, labels=algorithm_names)
    plt.title('Final Best Score Distribution (50 Runs)')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(f"boxplot_{function_name}_{algorithm_used}_100iters.png")
    plt.show()
    plt.close()

def plot_3d_benchmarks(benchmarks_module, resolution=150):
    """
    畫出所有 benchmark 函數的 3D 曲面圖，存至 results/3d_benchmarks.png。
 
    Parameters
    ----------
    benchmarks_module : module
        直接傳入 benchmarks 模組（import benchmarks 後傳進來即可）。
    resolution : int
        每個軸的格點數，越高越細緻但越慢，預設 150。
 
    Usage
    -----
        import benchmarks
        from utils.visualizer import plot_3d_benchmarks
        plot_3d_benchmarks(benchmarks)
    """
 
    # ── 函數清單：(函數名, 子圖標題, x1 範圍, x2 範圍, 維度) ────────────
    # F15 是 4 維函數，這裡固定 x3=x4=0，掃 x1/x2 的截面來呈現地形
    FUNC_CONFIG = [
        ("F2",  "F2: Schwefel 2.22",  (-10, 10),       (-10, 10),     2),
        ("F6",  "F6: Step",           (-100, 100),     (-100, 100),   2),
        ("F9",  "F9: Rastrigin",      (-5.12, 5.12),   (-5.12, 5.12), 2),
        ("F11", "F11: Griewank",      (-600, 600),     (-600, 600),   2),
        ("F13", "F13: Penalized",     (-50, 50),       (-50, 50),     2),
        ("F15", "F15: Kowalik\n(D=4, x3=x4=0 slice)", (-5, 5), (-5, 5), 4),
        ("F17", "F17: Branin",        (-5, 10),        (0, 15),       2),
    ]
 
    n_funcs = len(FUNC_CONFIG)
    ncols = 2
    nrows = (n_funcs + 1) // ncols  # 4 列
 
    fig = plt.figure(figsize=(14, nrows * 6))
    fig.suptitle("3D Surface Plots of Benchmark Functions (D=2)",
                 fontsize=16, fontweight='bold', y=0.95)
 
    for idx, (fname, title, x1_rng, x2_rng, dim) in enumerate(FUNC_CONFIG):
        func = getattr(benchmarks_module, fname)
 
        # 建立格點
        x1 = np.linspace(x1_rng[0], x1_rng[1], resolution)
        x2 = np.linspace(x2_rng[0], x2_rng[1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
 
        # 計算函數值
        Z = np.zeros_like(X1)
        for i in range(resolution):
            for j in range(resolution):
                if dim == 4:
                    pt = np.array([X1[i, j], X2[i, j], 0.0, 0.0])
                else:
                    pt = np.array([X1[i, j], X2[i, j]])
                Z[i, j] = func(pt)
 
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        surf = ax.plot_surface(X1, X2, Z, cmap=cm.viridis,
                               alpha=0.85, linewidth=0, antialiased=True)
 
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel("x₁", fontsize=9, labelpad=4)
        ax.set_ylabel("x₂", fontsize=9, labelpad=4)
        ax.set_zlabel("f(x)", fontsize=9, labelpad=4)
        ax.tick_params(labelsize=7)
        fig.colorbar(surf, ax=ax, shrink=0.45, aspect=8, pad=0.12)
 
    # 最後一格空白（7 個函數 / 2 列 → 第 8 格留空）
    if n_funcs % ncols != 0:
        fig.add_subplot(nrows, ncols, n_funcs + 1).axis('off')
 
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, hspace=0.4, wspace=0.3)
    save_plot("3d_benchmarks.png")
    plt.show()
    plt.close()