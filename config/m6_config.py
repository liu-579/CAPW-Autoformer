"""
配置文件 - 数据仓库ETL任务配置
用于存储所有可配置的参数、数据库连接信息和输出设置
"""

# ==================== 数据库连接配置 ====================
DATABASE_CONFIG = {
    'host': 'localhost',          # 数据库主机地址
    'port': 3306,                 # 数据库端口
    'user': 'root',      # 数据库用户名
    'password': 'password',  # 数据库密码
    'database': 'yq_data',  # 数据库名称
    'charset': 'utf8mb4'          # 字符集
}

# SQLAlchemy 连接字符串模板
def get_database_url():
    """生成 SQLAlchemy 数据库连接 URL"""
    return (f"mysql+pymysql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
            f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}"
            f"/{DATABASE_CONFIG['database']}?charset={DATABASE_CONFIG['charset']}")


# ==================== 数据源配置 ====================
SOURCE_TABLE = 'history_passenger_flow_data'  # 源数据表名

# 需要处理的景区ID列表
SCENIC_SPOT_IDS = ['9019043', '19012822']


# ==================== 字段映射配置 ====================
# 源表字段名
SOURCE_FIELDS = {
    'fence_id': 'fenceId',           # 景区ID字段
    'particle_size': 'particleSiz',  # 聚合粒度字段
    'date': 'date',                  # 日期字段
    'passenger_count': 'all',        # 客流总人数字段
    'day_of_week': 'dayOfWeek',      # 星期字段
    'holiday': 'holiday',            # 节假日字段
    'weather': 'weather'             # 天气字段（JSON格式）
}

# 粒度标识符
GRANULARITY = {
    'daily': 'd',    # 天粒度标识
    'hourly': 'h'    # 小时粒度标识
}


# ==================== 业务逻辑配置 ====================
# 节假日字段默认填充值
DEFAULT_HOLIDAY_VALUE = '非节假日'

# 日期格式
DATE_FORMAT = '%Y%m%d'  # 输入日期格式 "20231005"


# ==================== 天气字段聚合规则配置 ====================
# 定义每个天气字段的聚合方法
WEATHER_AGGREGATION_RULES = {
    'weather_cond': 'mode',      # 天气状况：众数
    'temp_high': 'max',          # 最高温度：最大值
    'temp_low': 'min',           # 最低温度：最小值
    'wind_dir': 'mode',          # 风向：众数
    'wind_power': 'mean',        # 风力：平均值
    'humidity': 'mean'           # 湿度：平均值
}

# MongoDB JSON 字段映射（从 weather JSON 中提取的字段）
MONGO_JSON_FIELDS = {
    'weather': 'weather_cond',           # 天气状况
    'temperature': 'temperature',        # 温度（需特殊处理 $numberInt）
    'winddirection': 'wind_dir',         # 风向
    'windpower': 'wind_power',           # 风力
    'humidity': 'humidity'               # 湿度（需特殊处理 $numberInt）
}


# ==================== 输出配置 ====================
# 输出表名模板
OUTPUT_TABLE_TEMPLATE = 'scenic_spot_{fence_id}_daily'

# 是否在插入前删除已存在的表
DROP_TABLE_IF_EXISTS = True

# 数据库写入模式 ('replace': 替换, 'append': 追加, 'fail': 如果表存在则失败)
DB_WRITE_MODE = 'replace'


# ==================== 日志配置 ====================
# 是否启用详细日志
VERBOSE_LOGGING = True

# 日志级别
LOG_LEVEL = 'INFO'  # 可选: DEBUG, INFO, WARNING, ERROR

# 是否在发现数据异常时打印警告
WARN_ON_MISSING_WEATHER = True  # 当某天有客流但无天气数据时警告


# ==================== 数据质量检查配置 ====================
# 是否执行数据质量检查
ENABLE_DATA_QUALITY_CHECK = True

# 最小记录数阈值（如果处理的记录数少于此值，发出警告）
MIN_RECORDS_THRESHOLD = 10

# 是否检查日期连续性
CHECK_DATE_CONTINUITY = False