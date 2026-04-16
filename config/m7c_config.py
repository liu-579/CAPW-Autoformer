# -*- coding: utf-8 -*-
"""
M7C 淡旺季分析配置文件
Season Analysis Configuration using Jenks Natural Breaks
用途: 定义数据源、财年设置、分类参数等
"""
from pathlib import Path
# ==================== 数据库连接配置 ====================
DB_CONFIG = {
    'host': '10.35.16.203',
    'port': 3306,
    'user': 'root',
    'password': 'password',
    'database': 'yq_data',
    'charset': 'utf8mb4'
}
# ==================== 数据源类型配置 ====================
# 数据源类型: 'hourly' (小时级数据) 或 'daily' (日度数据)
# 'hourly': 小时级数据，聚合时计算每周平均每小时客流
# 'daily': 日度数据，聚合时计算每周平均每日客流
DATA_SOURCE_TYPE = 'daily'  # 可选: 'hourly', 'daily'

# ==================== 输入表配置 ====================
# 根据数据源类型配置对应的表名
# 小时级数据表
INPUT_TABLE_HOURLY = "cleaned_feature_store_hourly_鄂尔多斯草原"
# 日度数据表
INPUT_TABLE_DAILY = "cleaned_feature_store_2_鄂尔多斯草原"

# 根据数据源类型自动选择输入表
INPUT_TABLE = INPUT_TABLE_HOURLY if DATA_SOURCE_TYPE == 'hourly' else INPUT_TABLE_DAILY

DATETIME_COLUMN = "datetime"
TARGET_COLUMN = "passenger_count"
# ==================== 财年配置 ====================
# 数据范围: 2023.10.01 - 2025.09.30
# 财年定义: 从10月1日开始，到次年9月30日结束
FISCAL_YEAR_START_MONTH = 10  # 财年起始月份（10月）
FISCAL_YEAR_START_DAY = 1     # 财年起始日期（1日）
# 两个完整财年
FISCAL_YEARS = [
    {'year': 2023, 'start': '2023-10-01', 'end': '2024-10-01', 'label': 'FY2023-2024'},
    {'year': 2024, 'start': '2024-10-01', 'end': '2025-10-01', 'label': 'FY2024-2025'},
]
# ==================== 聚合配置 ====================
# 周聚合方式
AGGREGATION_METHOD = 'mean'  # 'mean', 'sum', 'median'
WEEKS_PER_YEAR = 52  # 每年52周
# ==================== Jenks 分类配置 ====================
N_CLASSES = 3  # 分类数量: 淡季(0), 平季(1), 旺季(2)
# 季节标签映射
SEASON_LABELS = {
    0: {'name': 'low', 'cn': '淡季', 'color': '#3498db'},
    1: {'name': 'normal', 'cn': '平季', 'color': '#f39c12'},
    2: {'name': 'peak', 'cn': '旺季', 'color': '#e74c3c'},
}
# GVF 阈值（Goodness of Variance Fit）
# GVF > 0.8 表示分类效果良好
GVF_THRESHOLD = 0.8
# ==================== 输出配置 ====================
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data/output/m7b_season_analysis_eedscy"
# 输出文件
OUTPUT_FILES = {
    'weekly_data': 'weekly_aggregated_data.csv',      # 周聚合数据
    'season_result': 'season_classification.json',    # 分类结果
    'comparison': 'two_year_comparison.csv',          # 两年比对
    'review_list': 'review_needed.csv',               # 待审核列表
    'final_config': 'season_config_52weeks.json',     # 最终52周配置
    'visualization': 'season_visualization.png',      # 可视化图表
}
# ==================== 可视化配置 ====================
FIGURE_SIZE = (16, 10)
DPI = 150
# ==================== 日志配置 ====================
LOG_LEVEL = 'INFO'
SHOW_DETAILS = True