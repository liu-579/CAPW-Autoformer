# config/m4_baseline_config.py
"""
模块 4：基线模型微调配置文件
所有超参数和路径配置
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

class BaselineConfig:
    """基线模型训练配置"""
    real_path = r"E:\2024liushaoxuan\learn\shiyan1"
    # ==================== 路径配置 ====================
    # 数据路径
    DATA_DIR = Path(r"E:\2024liushaoxuan\learn\shiyan1/data\输出\dataset_v1")
    TRAIN_FILE = DATA_DIR / "train.csv"
    VAL_FILE = DATA_DIR / "val.csv"
    TEST_FILE = DATA_DIR / "test.csv"

    # 输出路径
    OUTPUT_DIR = Path(r"E:\2024liushaoxuan\learn\shiyan1/data\输出\m4_macbert")    # macbert的输出目录
    OUTPUT_DIR = Path(r"E:\2024liushaoxuan\learn\shiyan1/data\输出\m4_Robert")    # roberta的输出目录
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / "best_model"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    METRICS_FILE = OUTPUT_DIR / "metrics.json"
    LOG_FILE = OUTPUT_DIR / "training.log"

    # ==================== 模型配置 ====================
    # 预训练模型（可替换为其他中文模型）
    # MODEL_NAME = "hfl/chinese-macbert-base"
    # 备选模型（取消注释即可使用）:
    # MODEL_NAME = "bert-base-chinese"
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
    # MODEL_NAME = "hfl/chinese-electra-180g-base-discriminator"

    # 模型参数
    MAX_LENGTH = 512  # 最大序列长度
    NUM_LABELS = 5  # 输出维度（5个情感维度）
    HIDDEN_DROPOUT = 0.1  # Dropout比率

    # ==================== 训练配置 ====================
    # 基础参数
    BATCH_SIZE = 16  # 批次大小
    NUM_EPOCHS = 100  # 最大训练轮数
    LEARNING_RATE = 2e-5  # 学习率
    WEIGHT_DECAY = 0.01  # 权重衰减
    WARMUP_RATIO = 0.1  # 预热比例

    # 优化器配置
    ADAM_EPSILON = 1e-8
    MAX_GRAD_NORM = 1.0  # 梯度裁剪

    # 早停配置
    EARLY_STOPPING_PATIENCE = 10  # 早停耐心值
    EARLY_STOPPING_METRIC = "avg_pearsonr"  # 监控指标
    EARLY_STOPPING_MODE = "max"  # 最大化还是最小化

    # ==================== 断点续传配置 ====================
    # 是否启用断点续传
    RESUME_TRAINING = True

    # 检查点保存策略
    SAVE_CHECKPOINT_EVERY_EPOCH = True  # 每个epoch都保存检查点
    SAVE_BEST_ONLY = False  # False: 保存所有检查点, True: 只保存最佳模型

    # 检查点加载选项
    CHECKPOINT_TYPE = "latest"  # 'latest': 加载最新检查点, 'best': 加载最佳检查点, 'specific': 加载指定epoch
    RESUME_FROM_EPOCH = None  # 当CHECKPOINT_TYPE='specific'时，指定要恢复的epoch数

    # 检查点清理
    KEEP_CHECKPOINT_MAX = 5  # 最多保留几个历史检查点，None表示保留所有

    # 强制重新训练（忽略已有检查点）
    FORCE_RESTART = False  # True: 删除所有检查点从头开始, False: 尝试恢复

    # ==================== 随机种子 ====================
    RANDOM_SEED = 42

    # ==================== 评估配置 ====================
    # 情感维度
    DIMENSIONS = ["景色", "交通", "美食", "门票", "服务"]

    # 合格阈值
    QUALIFIED_THRESHOLD = 0.95  # 平均Pearsonr阈值

    # 评估指标
    METRICS = ["pearsonr", "rmse", "mae"]

    # ==================== 日志配置 ====================
    LOG_INTERVAL = 10  # 日志打印间隔（步数）
    EVAL_STEPS = None  # 评估间隔（None表示每个epoch评估）
    SAVE_TOTAL_LIMIT = 2  # 最多保存的checkpoint数量

    # ==================== 设备配置 ====================
    DEVICE = "cuda"  # cuda 或 cpu（自动检测）
    NUM_WORKERS = 4  # 数据加载线程数

    # ==================== 可视化配置 ====================
    FIGURE_DPI = 300
    FIGURE_SIZE = (8, 6)
    SCATTER_ALPHA = 0.6
    SCATTER_SIZE = 30

    @classmethod
    def create_dirs(cls):
        """创建所有必要的目录"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def to_dict(cls):
        """转换为字典（用于保存配置）"""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in cls.__dict__.items()
            if not k.startswith('_') and k.isupper()
        }

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 80)
        print("训练配置")
        print("=" * 80)

        print("\n[路径配置]")
        print(f"训练集: {cls.TRAIN_FILE}")
        print(f"验证集: {cls.VAL_FILE}")
        print(f"测试集: {cls.TEST_FILE}")
        print(f"输出目录: {cls.OUTPUT_DIR}")

        print("\n[模型配置]")
        print(f"预训练模型: {cls.MODEL_NAME}")
        print(f"最大长度: {cls.MAX_LENGTH}")
        print(f"输出维度: {cls.NUM_LABELS}")
        print(f"Dropout: {cls.HIDDEN_DROPOUT}")

        print("\n[训练配置]")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"权重衰减: {cls.WEIGHT_DECAY}")
        print(f"预热比例: {cls.WARMUP_RATIO}")

        print("\n[早停配置]")
        print(f"耐心值: {cls.EARLY_STOPPING_PATIENCE}")
        print(f"模式: {cls.EARLY_STOPPING_MODE}")
        print(f"合格阈值: {cls.QUALIFIED_THRESHOLD}")

        print("\n[断点续传配置]")
        print(f"启用续传: {cls.RESUME_TRAINING}")
        print(f"强制重启: {cls.FORCE_RESTART}")
        print(f"检查点类型: {cls.CHECKPOINT_TYPE}")
        print(f"每轮保存: {cls.SAVE_CHECKPOINT_EVERY_EPOCH}")
        print(f"只保存最佳: {cls.SAVE_BEST_ONLY}")
        print(f"保留数量: {cls.KEEP_CHECKPOINT_MAX}")

        print("\n[设备配置]")
        print(f"设备: {cls.DEVICE}")
        print(f"随机种子: {cls.RANDOM_SEED}")

        print("\n" + "=" * 80)