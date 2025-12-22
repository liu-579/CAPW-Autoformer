# -*- coding: utf-8 -*-
"""
M10 三阶段渐进式训练配置
Three-Phase Progressive Training Configuration
用途: 定义训练超参数、策略参数、路径配置
"""

from pathlib import Path


class TrainingConfig:
    """三阶段渐进式训练配置类"""

    # ==================== 数据路径配置 ====================
    from pathlib import Path
    # 项目根目录
    BASE_DIR = Path(__file__).parent.parent

    # m8b 输出目录 (数据来源)
    DATA_DIR =BASE_DIR/ "data/output/m8b_eedscy"   # 输出目录

    # m10 输出目录 (模型保存)
    SAVE_DIR = Path(BASE_DIR/'data/output/m10_checkpoints_eedscy')

    # 数据文件路径
    TRAIN_X = DATA_DIR / 'train_x.npy'
    TRAIN_Y = DATA_DIR / 'train_y.npy'
    VAL_X = DATA_DIR / 'val_x.npy'
    VAL_Y = DATA_DIR / 'val_y.npy'
    TEST_X = DATA_DIR / 'test_x.npy'
    TEST_Y = DATA_DIR / 'test_y.npy'

    # 配置文件路径
    FEATURE_MAP = DATA_DIR / 'feature_map.json'
    SCALER_PATH = DATA_DIR / 'scaler.pkl'

    # ==================== 训练基础参数 ====================

    BATCH_SIZE = 64  # 批次大小
    NUM_EPOCHS = 100  # 总训练轮数
    LEARNING_RATE = 1e-4  # 初始学习率
    WEIGHT_DECAY = 1e-5  # 权重衰减 (L2 正则化)

    # ==================== 三阶段训练策略 ====================

    # Phase 1: 预热骨干网络 (Warm-up Backbone)
    # 目标: 让 Transformer 先学会基本的时序模式
    # 策略: 冻结权重网络，只训练骨干
    PHASE_1_EPOCHS = 20

    # Phase 2: 权重网络学习 (Weight Network Learning)
    # 目标: 让模型学会"看环境脸色"分配权重
    # 策略: 冻结骨干网络，只训练权重网络
    PHASE_2_EPOCHS = 20

    # Phase 3: 联合微调 (Joint Fine-tuning)
    # 目标: 端到端优化，融会贯通
    # 策略: 解冻所有参数
    # 轮数: NUM_EPOCHS - PHASE_1_EPOCHS - PHASE_2_EPOCHS = 60

    # ==================== 温度退火策略 ====================

    # Temperature Annealing: 从平滑到尖锐
    # 高温 -> 权重分布平滑 (探索阶段)
    # 低温 -> 权重分布尖锐 (利用阶段)

    TEMP_START = 5.0  # 初始温度 (平滑权重分布)
    TEMP_END = 0.1  # 最终温度 (尖锐权重分布)
    TEMP_SCHEDULE = 'linear'  # 退火策略: 'linear' 或 'cosine'

    # ==================== 正则化参数 ====================

    # L1 正则化: 稀疏化权重 (提高可解释性)
    L1_LAMBDA = 1e-4  # 权重稀疏正则化系数

    # 组间多样性正则化: 鼓励不同情感组的权重有差异
    DIVERSITY_LAMBDA = 1e-5  # 多样性损失系数
    USE_DIVERSITY_LOSS = False  # 是否启用多样性损失

    # ==================== 优化器配置 ====================

    OPTIMIZER = 'adamw'  # 优化器类型: 'adam', 'adamw', 'sgd'
    BETAS = (0.9, 0.999)  # Adam/AdamW 的 beta 参数
    EPS = 1e-8  # Adam/AdamW 的 epsilon

    # ==================== 学习率调度器配置 ====================

    SCHEDULER = 'cosine'  # 'cosine', 'step', 'plateau', 'none'

    # Cosine Annealing 参数
    LR_MIN = 1e-6  # 最小学习率

    # Step LR 参数
    LR_STEP_SIZE = 20  # 学习率衰减步长
    LR_GAMMA = 0.5  # 学习率衰减因子

    # Reduce on Plateau 参数
    LR_PATIENCE = 5  # 学习率衰减耐心值
    LR_FACTOR = 0.5  # 学习率衰减因子

    # ==================== 早停策略 ====================

    EARLY_STOP = True  # 是否启用早停
    PATIENCE = 15  # 验证集无改善的最大轮数
    MIN_DELTA = 1e-4  # 最小改善阈值

    # ==================== 梯度裁剪 ====================

    GRAD_CLIP = True  # 是否启用梯度裁剪
    MAX_GRAD_NORM = 1.0  # 梯度裁剪的最大范数

    # ==================== 设备配置 ====================

    DEVICE = 'cuda'  # 'cuda' 或 'cpu'
    NUM_WORKERS = 4  # DataLoader 工作进程数
    PIN_MEMORY = True  # 是否使用 pin_memory 加速

    # ==================== 日志与保存配置 ====================

    LOG_INTERVAL = 10  # 每隔多少个 batch 打印一次日志
    SAVE_INTERVAL = 5  # 每隔多少个 epoch 保存一次检查点

    # 保存文件路径
    SAVE_LOG = SAVE_DIR / 'training_log.csv'
    SAVE_BEST_MODEL = SAVE_DIR / 'best_model.pth'
    SAVE_LATEST_MODEL = SAVE_DIR / 'latest_model.pth'
    SAVE_TEST_RESULT = SAVE_DIR / 'test_result.json'
    SAVE_WEIGHT_ANALYSIS = SAVE_DIR / 'weight_analysis.npz'

    # ==================== 可视化配置 ====================

    SAVE_ATTENTION_WEIGHTS = True  # 是否保存注意力权重
    SAVE_PRED_VS_TRUE = True  # 是否保存预测值与真实值对比

    # ==================== 随机种子 ====================

    SEED = 42  # 随机种子 (保证可复现)

    # ==================== 辅助方法 ====================

    @classmethod
    def setup_dirs(cls):
        """创建必要的目录"""
        cls.SAVE_DIR.mkdir(exist_ok=True, parents=True)
        print(f"✓ 创建保存目录: {cls.SAVE_DIR}")

    @classmethod
    def validate(cls):
        """验证配置参数的合法性"""
        # 检查数据路径
        assert cls.DATA_DIR.exists(), f"数据目录不存在: {cls.DATA_DIR}"
        assert cls.FEATURE_MAP.exists(), f"特征映射文件不存在: {cls.FEATURE_MAP}"
        assert cls.SCALER_PATH.exists(), f"归一化器文件不存在: {cls.SCALER_PATH}"

        # 检查阶段轮数
        total_phase_epochs = cls.PHASE_1_EPOCHS + cls.PHASE_2_EPOCHS
        assert total_phase_epochs < cls.NUM_EPOCHS, \
            f"阶段1和阶段2的总轮数 ({total_phase_epochs}) 必须小于总轮数 ({cls.NUM_EPOCHS})"

        # 检查温度参数
        assert cls.TEMP_START > cls.TEMP_END > 0, \
            f"温度参数必须满足: TEMP_START > TEMP_END > 0"

        # 检查正则化参数
        assert cls.L1_LAMBDA >= 0, f"L1_LAMBDA 必须非负"

        # 检查学习率参数
        assert cls.LEARNING_RATE > cls.LR_MIN > 0, \
            f"学习率参数必须满足: LEARNING_RATE > LR_MIN > 0"

        print("✓ 配置参数验证通过")
        return True

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 70)
        print("M10 三阶段训练配置")
        print("=" * 70)

        print("\n【数据路径】")
        print(f"  - 数据目录:        {cls.DATA_DIR}")
        print(f"  - 保存目录:        {cls.SAVE_DIR}")

        print("\n【训练参数】")
        print(f"  - 批次大小:        {cls.BATCH_SIZE}")
        print(f"  - 总轮数:          {cls.NUM_EPOCHS}")
        print(f"  - 学习率:          {cls.LEARNING_RATE}")
        print(f"  - 权重衰减:        {cls.WEIGHT_DECAY}")

        print("\n【三阶段策略】")
        print(f"  - Phase 1 (预热):  {cls.PHASE_1_EPOCHS} epochs")
        print(f"  - Phase 2 (权重):  {cls.PHASE_2_EPOCHS} epochs")
        print(f"  - Phase 3 (微调):  {cls.NUM_EPOCHS - cls.PHASE_1_EPOCHS - cls.PHASE_2_EPOCHS} epochs")

        print("\n【温度退火】")
        print(f"  - 初始温度:        {cls.TEMP_START}")
        print(f"  - 最终温度:        {cls.TEMP_END}")
        print(f"  - 退火策略:        {cls.TEMP_SCHEDULE}")

        print("\n【正则化】")
        print(f"  - L1 系数:         {cls.L1_LAMBDA}")
        print(f"  - 梯度裁剪:        {cls.GRAD_CLIP} (max_norm={cls.MAX_GRAD_NORM})")

        print("\n【早停策略】")
        print(f"  - 启用早停:        {cls.EARLY_STOP}")
        print(f"  - 耐心值:          {cls.PATIENCE}")

        print("=" * 70 + "\n")


# 配置验证
if __name__ == "__main__":
    TrainingConfig.validate()
    TrainingConfig.print_config()