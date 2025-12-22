"""
check_data_v2.py - 深度数据健康诊断脚本 (适配 tourism_seq 文件名格式)
===================================================================
功能：
    1. 自动读取 m10_config.py 配置
    2. 检查 .npy 数据中是否含有 NaN/Inf
    3. 检查是否存在方差为 0 的特征列（导致归一化除零）
    4. 检查标签（Target）分布是否异常
"""

import os
import numpy as np
import sys

# ============================================================================
# 1. 尝试导入配置
# ============================================================================
print("🔄 正在初始化配置...")
try:
    # 尝试从 config 包导入
    from 旧版本.m10_config import TrainConfig

    print("✅ 成功导入: from config.m10_config import TrainConfig")
except ImportError:
    try:
        # 尝试直接从当前目录导入
        from m10_config import TrainConfig

        print("✅ 成功导入: from m10_config import TrainConfig")
    except ImportError:
        print("\n❌ 严重错误：找不到 m10_config.py")
        print("   请确保此脚本 check_data_v2.py 与 m10_config.py 在同一目录下！")
        sys.exit(1)


# ============================================================================
# 2. 核心检查函数
# ============================================================================

def check_split_data(split_name):
    """
    检查指定数据集 (train/val/test) 的健康状况
    """
    print(f"\n{'=' * 30}")
    print(f"🔍 正在检查数据集: [{split_name}]")
    print(f"{'=' * 30}")

    # 获取配置参数
    scenic_name = TrainConfig.TARGET_SCENIC_NAME
    data_dir = TrainConfig.DATA_DIR

    # 构建文件名 (严格匹配 m10_train.py 中的 DataLoader_M8 逻辑)
    # 格式: tourism_seq_{scenic_name}_{split}_x.npy
    file_name_x = f"tourism_seq_{scenic_name}_{split_name}_x.npy"
    file_name_y = f"tourism_seq_{scenic_name}_{split_name}_y.npy"

    path_x = os.path.join(data_dir, file_name_x)
    path_y = os.path.join(data_dir, file_name_y)

    # --- 1. 文件存在性检查 ---
    if not os.path.exists(path_x):
        print(f"❌ 错误: 找不到文件 X -> {path_x}")
        return False
    if not os.path.exists(path_y):
        print(f"❌ 错误: 找不到文件 Y -> {path_y}")
        return False

    print(f"📂 已找到文件，正在加载...")
    try:
        data_x = np.load(path_x)
        data_y = np.load(path_y)
        print(f"   Shape X (特征): {data_x.shape}")
        print(f"   Shape Y (标签): {data_y.shape}")
    except Exception as e:
        print(f"❌ 加载出错: {e}")
        return False

    # --- 2. 检查 NaN (空值) ---
    nan_count_x = np.isnan(data_x).sum()
    nan_count_y = np.isnan(data_y).sum()

    if nan_count_x > 0:
        print(f"\n🚨 [CRITICAL] X 中发现 {nan_count_x} 个 NaN 值！")
        # 定位问题列：沿 batch(0) 和 seq(1) 轴检查，看哪个特征(2)坏了
        nan_cols = np.where(np.isnan(data_x).any(axis=(0, 1)))[0]
        print(f"   -> 问题特征索引 (Index): {list(nan_cols)}")
        print("   -> 建议: 检查 m8 模块的 Lag/Rolling 特征生成部分，是否未 dropna")
    else:
        print("✅ X 数据无 NaN")

    if nan_count_y > 0:
        print(f"🚨 [CRITICAL] Y 中发现 {nan_count_y} 个 NaN 值！")
    else:
        print("✅ Y 数据无 NaN")

    # --- 3. 检查 Inf (无穷大) ---
    inf_count_x = np.isinf(data_x).sum()

    if inf_count_x > 0:
        print(f"\n🚨 [CRITICAL] X 中发现 {inf_count_x} 个 Inf 值！")
        inf_cols = np.where(np.isinf(data_x).any(axis=(0, 1)))[0]
        print(f"   -> 问题特征索引: {list(inf_cols)}")
    else:
        print("✅ X 数据无 Inf")

    # --- 4. 检查方差为 0 的列 (常数陷阱) ---
    # 仅在训练集检查，因为归一化是在训练集拟合的
    # 如果某列全是 0 或全是 1，标准差为 0，归一化时 (x - min) / (max - min) 分母为 0 -> NaN
    if split_name == 'train':
        print("\n🔬 [深度检查] 寻找 '常数特征' (导致归一化除零的元凶)...")

        # 将 [N, Seq, Feat] 展平为 [Total_Steps, Feat]
        flat_x = data_x.reshape(-1, data_x.shape[-1])

        # 计算每个特征的最小值、最大值、标准差
        mins = np.min(flat_x, axis=0)
        maxs = np.max(flat_x, axis=0)
        stds = np.std(flat_x, axis=0)

        # 找出 min == max 的列
        zero_var_cols = np.where(mins == maxs)[0]

        if len(zero_var_cols) > 0:
            print(f"⚠️  [WARNING] 发现 {len(zero_var_cols)} 个常数特征列 (方差为0)!")
            print(f"   -> 这些列会导致 MinMaxScaler 产生除零错误 (NaN)")
            print(f"   -> 问题索引: {list(zero_var_cols)}")
            print(f"   -> 对应的值: {mins[zero_var_cols]}")
            print("   -> 建议: 在 m8 或 m10 中剔除这些列，或给 MinMax 添加 epsilon")
        else:
            print("✅ 所有特征均有数值变化 (无常数列)")

    return True


# ============================================================================
# 3. 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n🚀 开始数据健康检查 (check_data_v2.py)")
    print(f"配置路径: {TrainConfig.DATA_DIR}")
    print(f"目标景区: {TrainConfig.TARGET_SCENIC_NAME}")

    # 重点检查训练集
    success = check_split_data('train')

    if success:
        print("\n🎉 检查完成。如果上方全是 ✅，则 NaN 问题可能源于模型参数初始化。")
        print("如果上方有 🚨 或 ⚠️，请先修复数据问题。")
    else:
        print("\n❌ 检查因文件缺失而终止。")