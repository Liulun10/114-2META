import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_curve(all_runs_curves, title="PSO Convergence Curve"):
    """
    all_runs_curves: list of lists (50 runs * max_iter)
    """
    plt.figure(figsize=(10, 6))
    
    # 轉換成 numpy array 方便計算
    curves = np.array(all_runs_curves)
    avg_curve = np.mean(curves, axis=0)
    
    # 畫出每一條 run 的淡色線（選配，可以看到穩定度）
    for i in range(len(curves)):
        plt.plot(curves[i], color='gray', alpha=0.1)
    
    # 畫出平均收斂線
    plt.plot(avg_curve, color='blue', linewidth=2, label='Average Convergence')
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Best Score (Log Scale recommended)')
    plt.yscale('log')  # 最佳解通常差距很大，建議用 log 尺度
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # 存檔或顯示
    plt.savefig('pso_convergence.png')
    plt.show()

def plot_box_result(all_final_scores, algorithm_names=["PSO"]):
    """
    畫出 50 runs 最後結果的盒鬚圖，用於比較穩定性
    """
    plt.figure(figsize=(8, 6))
    plt.boxplot(all_final_scores, labels=algorithm_names)
    plt.title('Final Best Score Distribution (50 Runs)')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('result_boxplot.png')
    plt.show()