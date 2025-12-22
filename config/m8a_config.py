# -*- coding: utf-8 -*-
"""
数据清洗配置文件
ETL Engineer: Data Warehouse Configuration
用途: 定义数据库连接、目标景区、列名映射等配置
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
TARGET_SCENIC_NAME = "响沙湾"

# ==================== 表名配置 ====================
SOURCE_TABLE = "fused_training_data"
OUTPUT_TABLE_PREFIX = "cleaned_feature_store"  # 实际表名会加上景区名后缀

# ==================== 中文文本列定义 ====================
# 需要进行编码处理的中文列
TEXT_COLS = {
    'day_of_week': 'str',      # 星期 (使用固定字典映射)
    'holiday': 'str',          # 节假日 (二值化处理)
    'weather_cond': 'str',     # 天气状况 (LabelEncoder)
    'wind_dir': 'str'          # 风向 (LabelEncoder)
}

# 需要转换为数值的列
NUMERIC_CONVERSION_COLS = ['passenger_count']

# ==================== 星期映射字典 (固定映射,防止排序混乱) ====================
DAY_OF_WEEK_MAPPING = {
    '周一': 1,
    '周二': 2,
    '周三': 3,
    '周四': 4,
    '周五': 5,
    '周六': 6,
    '周日': 7
}

# ==================== 节假日二值化规则 ====================
HOLIDAY_NORMAL_VALUE = "非节假日"  # 正常值映射为 0
# 其他所有值(春节/国庆等)映射为 1

# ==================== 输出文件配置 ====================
CSV_ENCODING = 'utf-8-sig'  # 带BOM头,防止Excel中文乱码
CSV_OUTPUT_PREFIX = "debug_cleaned"

# ==================== 数据清洗参数 ====================
DROP_EMPTY_ROWS = True          # 是否删除空行
DROP_EMPTY_COLS = True          # 是否删除全空列
DROP_CONSTANT_COLS = True       # 是否删除常数列
VARIANCE_THRESHOLD = 0.0        # 方差阈值,0表示完全常数

# ==================== 保留列配置 ====================
PRESERVE_COLS = ['date', 'scenic_name']  # 无论如何都保留的列

# ==================== 日志配置 ====================
LOG_LEVEL = 'INFO'
SHOW_ENCODING_MAPPING = True    # 是否打印编码映射关系