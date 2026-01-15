import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置路径 (根据你的实际路径修改)
SAVE_DIR = Path(r"E:\2024liushaoxuan\learn\shiyan1\data\output\m10_checkpoints_eedscy_2.0")
PRED_FILE = SAVE_DIR / 'predictions.npz'


def visualize_results():
    if not PRED_FILE.exists():
        print(f"❌ 文件不存在: {PRED_FILE}")
        return

    print(f"📂 加载预测结果: {PRED_FILE}")
    data = np.load(PRED_FILE)
    preds = data['predictions']
    targets = data['targets']

    # 展平
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    # 1. 绘制整体趋势图
    plt.figure(figsize=(15, 6))
    plt.plot(targets, label='True Values', alpha=0.7, color='blue')
    plt.plot(preds, label='Predictions', alpha=0.7, color='red', linestyle='--')
    plt.title(f'Prediction vs True (RMSE: {np.sqrt(np.mean((preds - targets) ** 2)):.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / 'vis_trend.png')
    plt.show()

    # 2. 绘制散点图 (查看相关性)
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.3, s=10)

    # 画对角线 (完美预测线)
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Scatter Plot: True vs Pred')
    plt.grid(True, alpha=0.3)
    plt.savefig(SAVE_DIR / 'vis_scatter.png')
    plt.show()

    # 3. 局部放大 (查看前200个点，看细节)
    plt.figure(figsize=(15, 6))
    plt.plot(targets[:200], label='True', marker='o', markersize=4)
    plt.plot(preds[:200], label='Pred', marker='x', markersize=4)
    plt.title('Zoom in: First 200 hours')
    plt.legend()
    plt.grid(True)
    plt.savefig(SAVE_DIR / 'vis_zoom.png')
    plt.show()


if __name__ == "__main__":
    visualize_results()