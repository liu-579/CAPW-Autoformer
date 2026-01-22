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
OUTPUT_TABLE = 'fused_training_data_hourly'

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
# 滞后特征配置：定义需要生成的滞后小时数
# 例如 [1, 2, 3, 24, 48, 168] 表示生成 T-1小时、T-2小时、T-3小时、T-24小时（1天前）、T-48小时（2天前）、T-168小时（7天前）的特征值
LAG_HOURS = [1, 2, 3, 24, 48, 168]

# 移动平均窗口大小：定义计算移动平均的时间窗口（单位：小时）
# 例如 24 表示计算过去24小时的平均值（不包括当前小时，避免数据泄露）
ROLLING_WINDOW = 24

# 情感维度字段列表：需要进行特征衍生的5个情感维度字段名
# 这些字段将分别生成滞后特征和移动平均特征
SENTIMENT_COLS = [
    'scenery_score',  # 景色得分
    'transportation_score',  # 交通便利度得分
    'food_score',  # 餐饮得分
    'consumption_score',  # 消费水平得分
    'service_score'  # 服务得分
]

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