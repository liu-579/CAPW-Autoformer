"""
M4-b 全量推理配置文件
用于对清洗后的评论进行批量情感分析
"""

from pathlib import Path
from datetime import datetime


class M4bConfig:
    """M4-b 全量推理配置"""

    # ==================== 路径配置 ====================
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent

    # 输入文件：M1清洗后的评论数据
    INPUT_FILE = PROJECT_ROOT / "data/输出" / "m1_cleaned_comments.jsonl"

    # 输出目录
    OUTPUT_DIR = PROJECT_ROOT / "data/output" / "m4_b_inference"
    OUTPUT_FILE = OUTPUT_DIR / "m4_b_sentiment.jsonl"
    LOG_FILE = OUTPUT_DIR / "inference_log.txt"

    # M4训练好的模型路径
    MODEL_CHECKPOINT_DIR = PROJECT_ROOT / "data/输出/m4_macbert/checkpoints/best_model"
    # 使用最佳模型
    BEST_MODEL_PATH = MODEL_CHECKPOINT_DIR / "best_checkpoint.pt"

    # ==================== 模型配置 ====================
    # 预训练模型名称（需要与M4训练时一致）
    MODEL_NAME = "hfl/chinese-macbert-base"

    # 最大序列长度
    MAX_LENGTH = 128

    # 输出维度数（5个情感维度）
    NUM_LABELS = 5

    # Dropout比率（推理时不起作用，但加载模型需要）
    HIDDEN_DROPOUT = 0.1

    # 情感维度名称（按训练时的顺序）
    DIMENSIONS = [
        "景色",
        "交通",
        "美食",
        "门票",
        "服务"
    ]

    # ==================== 推理配置 ====================
    # 批处理大小（根据显存调整）
    BATCH_SIZE = 32

    # DataLoader工作线程数
    NUM_WORKERS = 4

    # 是否使用GPU
    USE_GPU = True

    # 推理温度（0表示贪心解码，实际上回归任务不使用温度）
    TEMPERATURE = 0.0

    # 是否显示进度条
    SHOW_PROGRESS = True

    # ==================== 分数映射配置 ====================
    # 归一化映射（训练时使用）
    # -2 → 0.0, -1 → 0.25, 0 → 0.5, 1 → 0.75, 2 → 1.0
    NORMALIZE_MAPPING = {-2: 0.0, -1: 0.25, 0: 0.5, 1: 0.75, 2: 1.0}

    # 反归一化：将[0,1]映射回[-2,2]的整数
    # score_int = round(score * 4 - 2)
    # 然后裁剪到[-2, -1, 0, 1, 2]
    MIN_SCORE = -2
    MAX_SCORE = 2

    # ==================== 输出配置 ====================
    # 随机抽样显示的评论数量
    SAMPLE_DISPLAY_COUNT = 5

    # 是否保存推理时的原始连续分数（0-1）
    SAVE_RAW_SCORES = False

    # 日期字段格式
    DATE_FORMAT = "%Y-%m-%d"

    # ==================== 日志配置 ====================
    # 日志级别
    LOG_LEVEL = "INFO"

    # 是否将日志同时输出到控制台
    LOG_TO_CONSOLE = True

    # 统计显存的时间间隔（秒）
    MEMORY_LOG_INTERVAL = 10

    # ==================== 其他配置 ====================
    # 随机种子（保证可复现）
    RANDOM_SEED = 42

    # 遇到错误时是否继续处理
    CONTINUE_ON_ERROR = True

    # 错误记录文件
    ERROR_LOG_FILE = OUTPUT_DIR / "error_log.txt"

    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("\n" + "=" * 80)
        print("M4-b 全量推理配置")
        print("=" * 80)
        print(f"输入文件: {cls.INPUT_FILE}")
        print(f"输出文件: {cls.OUTPUT_FILE}")
        print(f"模型路径: {cls.BEST_MODEL_PATH}")
        print(f"预训练模型: {cls.MODEL_NAME}")
        print(f"最大长度: {cls.MAX_LENGTH}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"工作线程: {cls.NUM_WORKERS}")
        print(f"使用GPU: {cls.USE_GPU}")
        print(f"情感维度: {', '.join(cls.DIMENSIONS)}")
        print(f"分数范围: [{cls.MIN_SCORE}, {cls.MAX_SCORE}]")
        print("=" * 80 + "\n")