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

def plot_convergence_curve(all_runs_curves, title="PSO Convergence Curve", func_name="UNKNOWN"):
    plt.figure(figsize=(12, 6)) # 稍微加寬一點，給右邊文字留空間
    
    curves = np.array(all_runs_curves)
    avg_curve = np.mean(curves, axis=0)
    
    # 計算最終統計值
    final_scores = curves[:, -1]
    mean_val = np.mean(final_scores)
    std_val = np.std(final_scores)
    best_final = np.min(final_scores)

    # 1. 畫出所有 run 的淡色線
    for i in range(len(curves)):
        plt.plot(curves[i], color='gray', alpha=0.1)
    
    # 2. 畫出平均收斂線
    plt.plot(avg_curve, color='blue', linewidth=2, label='Average Convergence')
    
    # 3. 製作資訊文字框 (統計數值)
    stats_text = (
        f"Statistics (50 runs):\n"
        f"{'─'*20}\n"
        f"Mean: {mean_val:.2e}\n"
        f"Std Dev: {std_val:.2e}\n"
        f"Final Best: {best_final:.2e}"
    )
    
    # 將文字標註在圖表右側 (使用座標系變換，放在圖表框外或右緣)
    # transform=plt.gca().transAxes 代表使用 0~1 的相對座標
    plt.text(1.02, 0.5, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Best Score (Log Scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='upper right')
    
    save_plot(f"convergence_{func_name}.png")
    plt.show()
    plt.close() # 關閉畫布避免記憶體累積

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