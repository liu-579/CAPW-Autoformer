# -*- coding: utf-8 -*-
"""
M10 三阶段渐进式训练配置 (Final Thesis Version)
Three-Phase Progressive Training Configuration
用途: 定义训练超参数、策略参数、路径配置
"""

from pathlib import Path


class TrainingConfig:
    """三阶段渐进式训练配置类"""

    # ==================== 数据路径配置 ====================
    # 项目根目录
    BASE_DIR = Path(__file__).parent.parent

    scenic_name = "xsw"

    # m8b 输出目录 (数据来源)
    DATA_DIR = BASE_DIR / f"data/output/m8b_{scenic_name}_2.0"  # 输入目录

    # m10 输出目录 (模型保存)
    SAVE_DIR = Path(BASE_DIR / f'data/output/m10_checkpoints_{scenic_name}_4.2_f')

    # 数据文件路径
    TRAIN_X = DATA_DIR / 'train_x.npy'
    TRAIN_Y = DATA_DIR / 'train_y.npy'
    VAL_X = DATA_DIR / 'val_x.npy'
    VAL_Y = DATA_DIR / 'val_y.npy'
    TEST_X = DATA_DIR / 'test_x.npy'
    TEST_Y = DATA_DIR / 'test_y.npy'

    # 配置文件路径
    FEATURE_MAP = DATA_DIR / 'feature_map.json'
    SCALER_PATH = DATA_DIR / 'scalers.pkl'

    # ==================== 训练基础参数 ====================

    BATCH_SIZE = 16  # 批次大小
    NUM_EPOCHS = 1000  # 总训练轮数
    LEARNING_RATE = 1e-3  # 初始学习率

    # 【修改】暂时设为 0，解除约束，让模型先学会拟合波动
    WEIGHT_DECAY = 1e-3  # 权重衰减 (L2 正则化)

    # ==================== 三阶段训练策略 ====================

    # Phase 1: 预热骨干网络 (Warm-up Backbone)
    PHASE_1_EPOCHS = 1

    # Phase 2: 权重网络学习 (Weight Network Learning)
    PHASE_2_EPOCHS = 1

    # Phase 3: 联合微调 (Joint Fine-tuning)
    # 轮数: NUM_EPOCHS - PHASE_1_EPOCHS - PHASE_2_EPOCHS

    # ==================== 温度退火策略 ====================

    TEMP_START = 5.0  # 初始温度
    # 【修改】降低最终温度，使权重更尖锐 (One-hot like)，避免平均化
    TEMP_END = 0.01
    TEMP_SCHEDULE = 'linear'  # 退火策略

    # ==================== 组合损失函数权重 ====================
    # 组合 Loss = weight_mse * MSE + weight_pearson * (1 - Pearson)

    # MSE 负责绝对数值准确性
    LOSS_WEIGHT_MSE = 1.0

    # Pearson 负责趋势形状 (范围 -1到1，Loss范围 0到2)
    # 建议从 0.1 或 0.01 开始，因为 (1-r) 的数值通常比 MSE(0.01左右) 大得多
    # 如果设为 1.0，模型可能会为了刷相关性而忽略绝对数值
    LOSS_WEIGHT_PEARSON = 1

    # 【新增】自适应峰值加权配置 (Adaptive Weighted MSE)
    # 目的：解决削峰问题，强迫模型关注高数值样本
    USE_ADAPTIVE_WEIGHT = True  # 是否启用自适应加权

    PEAK_SIGMA = 1.5  # 阈值界定: Mean + Sigma * Std (建议 1.5 或 2.0)
    PEAK_PENALTY_WEIGHT = 25  # 超过阈值的样本，Loss 权重放大倍数 (建议 5.0 - 10.0)

    # 加权 MSE 在总 Loss 中的系数
    # Total Loss = (1 - CCC) + LOSS_WEIGHT_PEAK_MSE * WeightedMSE
    LOSS_WEIGHT_PEAK_MSE = 0.5
    # ==================== 正则化参数 ====================

    # L1 正则化: 稀疏化权重
    # 【修改】设为 0.0，彻底解除稀疏约束，防止模型输出死直线
    L1_LAMBDA = 0.0

    # 组间多样性正则化
    DIVERSITY_LAMBDA = 1e-5  # 多样性损失系数
    USE_DIVERSITY_LOSS = False  # 是否启用多样性损失

    # ==================== 优化器配置 ====================

    OPTIMIZER = 'adamw'  # 优化器类型
    BETAS = (0.9, 0.999)
    EPS = 1e-8

    # ==================== 学习率调度器配置 ====================

    SCHEDULER = 'cosine'  # 'cosine', 'step', 'plateau', 'none'

    # Cosine Annealing 参数
    LR_MIN = 1e-7  # 最小学习率

    # Step LR 参数
    LR_STEP_SIZE = 20
    LR_GAMMA = 0.5

    # Reduce on Plateau 参数
    LR_PATIENCE = 5
    LR_FACTOR = 0.5

    # ==================== 早停策略 ====================
    # 注意：在三阶段训练中，早停主要控制 Phase 3，
    # 但我们现在的逻辑是跑完所有 Epoch 以收集各阶段最佳模型
    EARLY_STOP = True
    PATIENCE = 70
    MIN_DELTA = 1e-4

    # ==================== 梯度裁剪 ====================

    GRAD_CLIP = True
    MAX_GRAD_NORM = 1.0

    # ==================== 设备配置 ====================

    DEVICE = 'cuda'
    NUM_WORKERS = 0  # Windows下设为0更稳定，Linux可设为4
    PIN_MEMORY = True

    # ==================== 日志与保存配置 ====================

    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5

    # 通用保存路径
    SAVE_LOG = SAVE_DIR / 'training_log.csv'
    SAVE_LATEST_MODEL = SAVE_DIR / 'latest_model.pth'

    # 【新增】分阶段最佳模型保存路径
    SAVE_BEST_MODEL_PHASE1 = SAVE_DIR / 'best_model_phase1.pth'
    SAVE_BEST_MODEL_PHASE2 = SAVE_DIR / 'best_model_phase2.pth'
    SAVE_BEST_MODEL_PHASE3 = SAVE_DIR / 'best_model_phase3.pth'

    # ==================== 可视化配置 ====================

    SAVE_ATTENTION_WEIGHTS = True
    SAVE_PRED_VS_TRUE = True

    # ==================== 随机种子 ====================

    SEED = 42

    # ==================== 辅助方法 ====================

    @classmethod
    def setup_dirs(cls):
        """创建必要的目录"""
        cls.SAVE_DIR.mkdir(exist_ok=True, parents=True)
        print(f"✓ 创建保存目录: {cls.SAVE_DIR}")

    @classmethod
    def validate(cls):
        """验证配置参数的合法性"""
        assert cls.DATA_DIR.exists(), f"数据目录不存在: {cls.DATA_DIR}"

        total_phase_epochs = cls.PHASE_1_EPOCHS + cls.PHASE_2_EPOCHS
        assert total_phase_epochs < cls.NUM_EPOCHS, \
            f"阶段1和阶段2的总轮数 ({total_phase_epochs}) 必须小于总轮数 ({cls.NUM_EPOCHS})"

        print("✓ 配置参数验证通过")
        return True

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 70)
        print("M10 三阶段训练配置 (Final Thesis)")
        print("=" * 70)
        print(f"  - Phase 1 (预热):  {cls.PHASE_1_EPOCHS} epochs")
        print(f"  - Phase 2 (权重):  {cls.PHASE_2_EPOCHS} epochs")
        print(f"  - Phase 3 (微调):  {cls.NUM_EPOCHS - cls.PHASE_1_EPOCHS - cls.PHASE_2_EPOCHS} epochs")
        print(f"  - Loss Function:   Huber Loss")
        print(f"  - L1 Lambda:       {cls.L1_LAMBDA} (Disabled)")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    TrainingConfig.validate()
    TrainingConfig.print_config()