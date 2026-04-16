"""
数据融合与特征工程配置文件（小时尺度版本）
功能：定义情感数据（日度）与物理特征数据（小时）融合的所有参数
版本：m7_b - 将日度情感数据广播到小时尺度
"""
# ==================== 数据库连接配置 ====================
DB_CONFIG = {
    'host': 'localhost',  # MySQL 服务器地址
    'port': 3306,  # MySQL 端口号
    'user': 'root',  # 数据库用户名
    'password': 'password',  # 数据库密码（请根据实际情况修改）
    'database': 'yq_data'  # 数据库名称
}
# ==================== 表名配置 ====================
# 情感数据源表名（存储每日各景区的情感分析结果 - 日度数据）
SENTIMENT_TABLE = 'yq_sentiment_daily_2'
# 最终融合数据的输出表名（用于机器学习训练的宽表 - 小时尺度）
OUTPUT_TABLE = 'm7b_fused_training_data_hourly_2'
# ==================== 景区映射配置 ====================
# 关键映射字典：将情感表中的景区名称映射到其对应的物理特征数据表（小时尺度）
# 键 (Key): 情感表中 scenic_name 字段的中文景区名称
# 值 (Value): 该景区的小时尺度物理特征数据表名（包含客流、天气等）
SCENIC_MAPPING = {
    '响沙湾': 'history_hourly_xsw',
    '鄂尔多斯草原': 'history_hourly_eedscy',
    # 请根据实际景区名称和表名进行扩展
    # '张家界景区': 'scenic_spot_12345678_hourly',
    # '西湖景区': 'scenic_spot_87654321_hourly',
}
# ==================== 特征工程参数 ====================

# ---------- 情感特征配置 ----------
# 情感维度字段列表：需要进行特征衍生的5个情感维度字段名
SENTIMENT_COLS = [
    'scenery_score',  # 景色得分
    'transportation_score',  # 交通便利度得分
    'food_score',  # 餐饮得分
    'consumption_score',  # 消费水平得分
    'service_score'  # 服务得分
]

# 情感特征：移动平均窗口列表（单位：小时）
# 对每个情感维度分别计算多个窗口的移动平均特征
# 24h=1天, 48h=2天, 168h=7天, 336h=14天
SENTIMENT_ROLLING_WINDOWS = [168, 336]

# 注意：情感特征不再构建滞后项(Lag)，仅构建移动平均项(Rolling)

# ---------- 历史客流特征配置 ----------
# 客流目标列名
PASSENGER_COL = 'passenger_count'

# 客流滞后特征：定义需要生成的滞后小时数
# 1h=上一小时, 2h=前2小时, 3h=前3小时, 24h=昨日同时, 48h=前天同时, 168h=上周同时
PASSENGER_LAG_HOURS = [1, 2, 3, 24, 48, 168]

# 客流移动平均窗口列表（单位：小时）
# 24h=日均, 48h=2日均, 168h=周均, 336h=2周均
PASSENGER_ROLLING_WINDOWS = [24, 168, 336]
# ==================== 数据处理配置 ====================
# 数据库写入模式
# 'append': 追加模式，适用于循环写入多个景区数据
# 'replace': 替换模式，会清空原表（首次写入时使用）
DB_WRITE_MODE = 'replace'
# 日期字段名称（用于数据融合的关键字段）
DATE_FIELD = 'date'
# 小时字段名称（物理特征表中的小时字段）
HOUR_FIELD = 'hour'
# 日期时间字段名称（融合后的完整时间戳字段）
DATETIME_FIELD = 'datetime'
# 景区名称字段（用于筛选和标识）
SCENIC_NAME_FIELD = 'scenic_name'
# 是否在输出表中保留景区名称列（用于区分数据来源）
KEEP_SCENIC_NAME = True
# 是否在输出表中添加源表标识列
ADD_SOURCE_TABLE_COLUMN = True
# ==================== 日志配置 ====================
# 是否启用详细日志输出
VERBOSE_LOGGING = True
# 是否在特征工程后显示数据统计信息
SHOW_DATA_STATS = True
# ==================== 辅助函数 ====================
def get_database_url():
    """
    生成 SQLAlchemy 数据库连接字符串
    返回:
        str: 格式为 mysql+pymysql://user:password@host:port/database
    """
    return (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
def get_lag_feature_name(base_col, lag_hours):
    """
    生成滞后特征的列名
    参数:
        base_col (str): 原始列名（如 'scenery_score'）
        lag_hours (int): 滞后小时数（如 1, 24）
    返回:
        str: 滞后特征列名（如 'scenery_score_lag1h'）
    """
    return f"{base_col}_lag{lag_hours}h"
def get_rolling_feature_name(base_col, window):
    """
    生成移动平均特征的列名
    参数:
        base_col (str): 原始列名（如 'scenery_score'）
        window (int): 移动窗口大小（如 24）
    返回:
        str: 移动平均特征列名（如 'scenery_score_roll24h'）
    """
    return f"{base_col}_roll{window}h"
