# m4_b_config.py
"""
M4-b 全量推理配置文件
用于对清洗后的评论进行批量情感分析 (数据库输入/输出版)
"""

from pathlib import Path
from datetime import datetime


class M4bConfig:
    """M4-b 全量推理配置"""

    # ==================== 路径配置 ====================
    PROJECT_ROOT = Path(__file__).parent.parent

    # 输出目录 (仅用于日志和临时文件)
    OUTPUT_DIR = PROJECT_ROOT / "data/output" / "m4_sql_b_inference"
    LOG_FILE = OUTPUT_DIR / "inference_log.txt"

    # M4训练好的模型路径
    # 注意：根据您的输入，模型目录修正为 m4_Robert_new_2.0
    MODEL_CHECKPOINT_DIR = PROJECT_ROOT / "data/输出/m4_Robert_new_2.0/checkpoints/best_model"
    BEST_MODEL_PATH = MODEL_CHECKPOINT_DIR / "best_checkpoint.pt"

    # ==================== 数据库配置 ====================
    DATABASE_CONFIG = {
        'host': 'localhost',  # 数据库地址
        'port': 3306,  # 端口
        'user': 'root',  # 用户名
        'password': 'password',  # 密码
        'db': 'yq_data',  # 数据库名
        'charset': 'utf8mb4'
    }

    # ==================== 数据源配置 (MySQL 输入) ====================
    DATA_SOURCE_CONFIG = {
        'input_table': 'yq_clean_all',  # 原评论数据表
        'content_field': 'format_content',  # 评论正文的字段名，用于模型推理

        # 需要读取的原始字段列表
        'all_fields': [
            'id', 'original_id', 'scenic_name', 'source_type', 'author', 'title',
            'release_date', 'isvalid', 'highly_relevant', 'keywords',
            'feature_words', 'format_content', 'negative_probability',
            'sentiment', 'scores_comment', 'source_table'
        ],

        # SQL WHERE条件
        # 您在输入中注释掉了此项，这里设置为 None，表示不进行筛选（全量处理）
        'filter_condition': None,
    }

    # ==================== 结果输出配置 (MySQL 输出) ====================
    DATA_OUTPUT_CONFIG = {
        'output_table': 'yq_inference_all',  # 推理结果存放的新表名
        'create_table_if_not_exists': True,  # 如果表不存在则创建
        'batch_size': 5000,  # 数据库批量插入大小
    }

    # ==================== 模型配置 ====================
    # 使用 RoBERTa 模型
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
    MAX_LENGTH = 128
    NUM_LABELS = 5

    # 5个情感维度
    DIMENSIONS = ['景色', '交通', '美食', '消费', '服务']

    # 数据库中新增的5个评分字段名
    SENTIMENT_SCORE_FIELDS = [
        'scenery_score',
        'transportation_score',
        'food_score',
        'consumption_score',
        'service_score'
    ]

    # ==================== 批处理配置 ====================
    BATCH_SIZE = 64

    # ==================== 辅助方法 ====================
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 60)
        print(f"M4-b 全量推理配置 (Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print("-" * 60)

        # 打印路径
        print(f"项目根目录: {cls.PROJECT_ROOT}")
        print(f"日志文件:   {cls.LOG_FILE}")
        print(f"模型路径:   {cls.BEST_MODEL_PATH}")

        # 打印数据库配置 (隐藏密码)
        db_safe = cls.DATABASE_CONFIG.copy()
        if 'password' in db_safe:
            db_safe['password'] = '******'
        print(f"\n[数据库连接]")
        print(f"配置: {db_safe}")

        # 打印输入配置
        print(f"\n[数据源输入]")
        print(f"源表名: {cls.DATA_SOURCE_CONFIG['input_table']}")
        print(f"内容字段: {cls.DATA_SOURCE_CONFIG['content_field']}")
        print(f"读取字段数: {len(cls.DATA_SOURCE_CONFIG['all_fields'])}")
        print(
            f"筛选条件: {cls.DATA_SOURCE_CONFIG['filter_condition'] if cls.DATA_SOURCE_CONFIG['filter_condition'] else '无 (全量)'}")

        # 打印输出配置
        print(f"\n[结果输出]")
        print(f"目标表名: {cls.DATA_OUTPUT_CONFIG['output_table']}")
        print(f"批量大小: {cls.DATA_OUTPUT_CONFIG['batch_size']}")

        # 打印模型配置
        print(f"\n[模型参数]")
        print(f"基础模型: {cls.MODEL_NAME}")
        print(f"最大长度: {cls.MAX_LENGTH}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"情感维度: {cls.DIMENSIONS}")

        print("=" * 60 + "\n")