import matplotlib.pyplot as plt
import numpy as np
import os

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

def plot_convergence_curve(all_runs_curves, title="PSO Convergence Curve", function_name="UNKNOWN"):
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
        f"Mean:\n {mean_val:.4e}\n\n"
        f"Std Dev:\n {std_val:.4e}\n\n"
        f"Final Best:\n {best_final:.4e}\n"
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
    
    save_plot(f"convergence_{function_name}.png")
    plt.show()
    plt.close()

def plot_box_result(all_final_scores, algorithm_names=["PSO"], function_name="UNKNOWN"):
    """
    畫出 50 runs 最後結果的盒鬚圖，用於比較穩定性
    """
    plt.figure(figsize=(8, 6))
    plt.boxplot(all_final_scores, labels=algorithm_names)
    plt.title('Final Best Score Distribution (50 Runs)')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(f"boxplot_{function_name}.png")
    plt.show()