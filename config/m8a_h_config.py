# -*- coding: utf-8 -*-
"""
数据清洗配置文件（小时尺度版本）
ETL Engineer: Data Warehouse Configuration
用途: 定义数据库连接、目标景区、列名映射等配置
版本: m8a_h - 适配小时尺度数据
修改: 添加淡旺季特征支持（从m7c季节分析结果读取）
"""
from pathlib import Path

# ==================== 基础路径配置 ====================
BASE_DIR = Path(__file__).parent.parent

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
SOURCE_TABLE = "m7b_fused_training_data_hourly_2"    # 原表（小时尺度）
OUTPUT_TABLE_PREFIX = "m8a_cleaned_feature_store_hourly_2"   # 输出表前缀

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

# ==================== 淡旺季特征配置 ====================
# 是否启用淡旺季特征
ENABLE_SEASON_FEATURE = True

# 淡旺季配置文件路径（由 m7c_season_analysis.py 生成）
SEASON_CONFIG_PATH = BASE_DIR / "data/output/m7b_season_analysis_xsw/season_config_52weeks.json"

# 财年起始日期（与 m7c 配置保持一致）
FISCAL_YEAR_START_MONTH = 10  # 10月
FISCAL_YEAR_START_DAY = 1     # 1日

# 淡旺季特征列名
SEASON_CODE_FIELD = 'season_code'      # 季节代码 (0=淡季, 1=平季, 2=旺季)
SEASON_NAME_FIELD = 'season_name'      # 季节英文名称 (low, normal, peak)
FISCAL_WEEK_FIELD = 'fiscal_week'      # 财年周序号 (1-52)

# 是否生成 One-Hot 编码的季节特征
ENABLE_SEASON_ONEHOT = True
# One-Hot 编码列名
SEASON_ONEHOT_COLS = {
    0: 'is_low_season',     # 淡季标志
    1: 'is_normal_season',  # 平季标志
    2: 'is_peak_season'     # 旺季标志
}

# ==================== 天气等级分类映射配置 ====================
# 是否启用天气等级分类特征（将 weather_cond 中文文本映射为好/中/坏天气等级）
ENABLE_WEATHER_GRADE = True

# 天气等级映射: 好天气=2, 中等天气=1, 坏天气=0
WEATHER_GRADE_FIELD = 'weather_grade'  # 生成的天气等级列名

WEATHER_GRADE_MAPPING = {
    # --- 好天气 (2) ---
    '晴': 2,
    '多云': 2,
    '阴': 2,
    # --- 中等天气 (1) ---
    '小雨': 1,
    '雨': 1,
    '小雪': 1,
    '雾': 1,
    '霾': 1,
    '浮尘': 1,
    # --- 坏天气 (0) ---
    '扬沙': 0,
    '中雨': 0,
    '大雨': 0,
    '暴雨': 0,
    '大暴雨': 0,
    '特大暴雨': 0,
    '雪': 0,
    '中雪': 0,
    '雨夹雪': 0,
}

# 未匹配到映射表的天气默认归入中等天气
WEATHER_GRADE_DEFAULT = 1

# 天气等级标签（用于日志打印）
WEATHER_GRADE_LABELS = {2: '好天气', 1: '中等天气', 0: '坏天气'}
