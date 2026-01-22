# -*- coding: utf-8 -*-
"""
数据清洗配置文件（小时尺度版本）
ETL Engineer: Data Warehouse Configuration
用途: 定义数据库连接、目标景区、列名映射等配置
版本: m8a_h - 适配小时尺度数据
"""

# ==================== 数据库连接配置 ====================
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'password',  # 请修改为实际密码
    'database': 'yq_data',  # 请修改为实际数据库名
    'charset': 'utf8mb4'
}

# ==================== 目标景区配置 ====================
TARGET_SCENIC_NAME = "鄂尔多斯草原"

# ==================== 表名配置 ====================
SOURCE_TABLE = "fused_training_data_hourly"    # 原表（小时尺度）
OUTPUT_TABLE_PREFIX = "cleaned_feature_store_hourly"   # 输出表前缀

# ==================== 时间字段配置 ====================
DATETIME_FIELD = 'datetime'  # 小时尺度的完整时间戳字段
DATE_FIELD = 'date'  # 日期字段（如果存在）

# ==================== 中文文本列定义 ====================
# 这些列是原始中文文本，我们会利用它们（或日期）生成新特征
# 之后这些原始列会在 strong_cleaning 阶段被 drop 掉
TEXT_COLS = {
    'day_of_week': 'str',      # 原始星期文本 (将被周期性特征替代)
    'holiday': 'str',          # 原始节假日文本 (将被 chinese_calendar 逻辑替代)
    'weather_cond': 'str',     # 天气状况 (保持 LabelEncoder)
    'wind_dir': 'str'          # 风向 (保持 LabelEncoder)
}

# 需要转换为数值的列
NUMERIC_CONVERSION_COLS = ['passenger_count']

# ==================== 输出文件配置 ====================
CSV_ENCODING = 'utf-8-sig'
CSV_OUTPUT_PREFIX = "debug_cleaned_hourly"

# ==================== 数据清洗参数 ====================
DROP_EMPTY_ROWS = True
DROP_EMPTY_COLS = True
DROP_CONSTANT_COLS = True
VARIANCE_THRESHOLD = 0.0

# ==================== 保留列配置 ====================
# 无论如何都保留的列，建议保留 datetime 和 date 方便后续验证
PRESERVE_COLS = ['datetime', 'date', 'scenic_name', 'is_day_off', 'hour']

# ==================== 日志配置 ====================
LOG_LEVEL = 'INFO'
SHOW_ENCODING_MAPPING = True