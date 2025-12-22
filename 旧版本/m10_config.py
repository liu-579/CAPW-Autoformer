"""
m10_config.py - 三阶段渐进式训练配置文件
===================================================================
包含所有训练超参数、三阶段训练策略参数和正则化配置

作者：[您的姓名]
日期：2025-12-18
版本：v1.0 - 硕士论文核心训练脚本
"""

import os
import torch


class TrainConfig:
    """
    三阶段渐进式训练配置

    训练策略：
        Phase 1: 预热骨干网络（冻结权重网络）
        Phase 2: 学习动态权重（冻结骨干网络）
        Phase 3: 端到端联合微调（解冻所有参数）
    """

    # ========== 数据配置 ==========
    TARGET_SCENIC_NAME = "鄂尔多斯草原"  # 目标景区名称
    DATA_DIR = r"/data/output/m8_eedscy/datasets"  # 数据目录

    # ========== 基础训练参数 ==========
    BATCH_SIZE = 64  # 批次大小
    LEARNING_RATE = 1e-5  # 初始学习率
    NUM_EPOCHS = 100  # 总训练轮数

    # ========== 三阶段训练策略参数 (核心) ==========
    PHASE_1_EPOCHS = 20  # Phase 1: 预热骨干网络
    PHASE_2_EPOCHS = 20  # Phase 2: 学习动态权重
    # Phase 3 = NUM_EPOCHS - PHASE_1_EPOCHS - PHASE_2_EPOCHS (联合微调)

    # ========== 正则化参数 ==========
    L1_LAMBDA = 1e-5  # L1 正则化系数（权重稀疏化）
    WEIGHT_DECAY = 1e-5  # AdamW 权重衰减
    GRADIENT_CLIP = 1.0  # 梯度裁剪阈值

    # ========== 温度退火参数 ==========
    TEMP_START = 5.0  # 初始温度（平滑权重分布）
    TEMP_END = 0.1  # 最终温度（尖锐权重分布）
    # 温度退火策略：使权重从平滑探索过渡到精确选择

    # ========== 优化器配置 ==========
    OPTIMIZER = 'adamw'  # 优化器类型
    BETAS = (0.9, 0.999)  # AdamW 的 beta 参数
    EPS = 1e-8  # AdamW 的 epsilon

    # ========== 学习率调度器 ==========
    USE_LR_SCHEDULER = True  # 是否使用学习率调度
    LR_SCHEDULER_TYPE = 'cosine'  # 调度器类型
    LR_WARMUP_EPOCHS = 5  # 预热轮数
    LR_MIN = 1e-6  # 最小学习率

    # ========== 早停策略 ==========
    EARLY_STOPPING = True  # 是否启用早停
    PATIENCE = 15  # 早停容忍轮数
    MIN_DELTA = 1e-5  # 最小改进阈值

    # ========== 验证与保存 ==========
    VAL_INTERVAL = 1  # 验证间隔（每 N 轮验证一次）
    SAVE_BEST_ONLY = True  # 是否只保存最优模型
    SAVE_INTERVAL = 10  # 常规保存间隔

    # ========== 设备配置 ==========
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4  # DataLoader 工作进程数
    PIN_MEMORY = True  # 是否使用锁页内存

    # ========== 输出路径 ==========
    CHECKPOINT_DIR = DATA_DIR+"data/output/m10"+"./checkpoints"  # 模型保存目录
    LOG_DIR = DATA_DIR+"data/output/m10"+"./logs"  # 日志保存目录
    RESULT_DIR = DATA_DIR+"data/output/m10"+"./results"  # 结果保存目录

    # ========== 日志与可视化 ==========
    LOG_INTERVAL = 10  # 训练日志打印间隔（每 N 个 batch）
    SAVE_WEIGHT_HISTORY = True  # 是否保存权重历史
    PLOT_TRAINING_CURVES = True  # 是否绘制训练曲线

    # ========== 随机种子 ==========
    SEED = 42  # 随机种子（保证可复现性）

    @classmethod
    def get_phase_3_epochs(cls) -> int:
        """计算 Phase 3 的训练轮数"""
        return cls.NUM_EPOCHS - cls.PHASE_1_EPOCHS - cls.PHASE_2_EPOCHS

    @classmethod
    def validate_config(cls):
        """验证配置的合理性"""
        assert cls.PHASE_1_EPOCHS + cls.PHASE_2_EPOCHS < cls.NUM_EPOCHS, \
            f"Phase 1 + Phase 2 轮数 ({cls.PHASE_1_EPOCHS + cls.PHASE_2_EPOCHS}) 必须小于总轮数 ({cls.NUM_EPOCHS})"

        assert cls.TEMP_START > cls.TEMP_END, \
            f"初始温度 ({cls.TEMP_START}) 必须大于最终温度 ({cls.TEMP_END})"

        assert cls.L1_LAMBDA >= 0, "L1 正则化系数必须非负"

        print("✅ 配置验证通过")

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 80)
        print("训练配置详情")
        print("=" * 80)

        print("\n【数据配置】")
        print(f"  目标景区: {cls.TARGET_SCENIC_NAME}")
        print(f"  数据目录: {cls.DATA_DIR}")
        print(f"  批次大小: {cls.BATCH_SIZE}")

        print("\n【三阶段训练策略】")
        print(f"  总训练轮数: {cls.NUM_EPOCHS}")
        print(f"  Phase 1 (预热骨干): {cls.PHASE_1_EPOCHS} 轮")
        print(f"  Phase 2 (学习权重): {cls.PHASE_2_EPOCHS} 轮")
        print(f"  Phase 3 (联合微调): {cls.get_phase_3_epochs()} 轮")

        print("\n【正则化与退火】")
        print(f"  L1 正则化系数: {cls.L1_LAMBDA}")
        print(f"  温度退火: {cls.TEMP_START} → {cls.TEMP_END}")

        print("\n【优化器配置】")
        print(f"  学习率: {cls.LEARNING_RATE}")
        print(f"  权重衰减: {cls.WEIGHT_DECAY}")
        print(f"  梯度裁剪: {cls.GRADIENT_CLIP}")

        print("\n【早停策略】")
        print(f"  启用早停: {cls.EARLY_STOPPING}")
        print(f"  容忍轮数: {cls.PATIENCE}")

        print("\n【设备配置】")
        print(f"  训练设备: {cls.DEVICE}")
        print(f"  工作进程: {cls.NUM_WORKERS}")

        print("=" * 80 + "\n")

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_DIR, exist_ok=True)
        print(f"✅ 目录已创建/确认存在")


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("训练配置测试")
    print("=" * 80)

    # 验证配置
    TrainConfig.validate_config()

    # 打印配置
    TrainConfig.print_config()

    # 创建目录
    TrainConfig.create_directories()

    print("\n" + "=" * 80)
    print("配置测试完成！")
    print("=" * 80)