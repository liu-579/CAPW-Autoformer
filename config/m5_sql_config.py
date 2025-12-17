"""
M5 日情感聚合配置文件（MySQL版）
功能：配置M5日聚合任务的所有参数
日期：2025-12-15
"""

from pathlib import Path


class M5Config:
    """M5 日情感聚合任务配置"""

    # ==================== 路径配置 ====================
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = DATA_DIR / "output" / "m5"

    # 日志文件
    LOG_FILE = OUTPUT_DIR / "aggregation_log.txt"

    # ==================== 数据库配置 ====================
    DATABASE_CONFIG = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': 'password',
        'db': 'yq_data',
        'charset': 'utf8mb4'
    }

    # ==================== 数据源配置 ====================
    # 输入表：M4-b的情感分析结果表
    INPUT_TABLE = 'yq_inference_all'

    # 输出表：日度聚合结果表
    OUTPUT_TABLE = 'yq_sentiment_daily'

    # 景区列表
    SCENIC_NAMES = ['响沙湾', '鄂尔多斯草原']

    # 日期字段名
    DATE_FIELD = 'release_date'

    # 景区字段名
    SCENIC_FIELD = 'scenic_name'

    # 需要读取的字段
    SOURCE_FIELDS = [
        'id',
        'scenic_name',
        'release_date',
        'scenery_score',
        'transportation_score',
        'food_score',
        'consumption_score',
        'service_score'
    ]

    # ==================== 情感维度配置 ====================
    # 情感维度列表（与M4输出的字段对应）
    DIMENSIONS = ['景色', '交通', '美食', '消费', '服务']

    # 数据库字段与维度的映射
    SCORE_FIELD_MAPPING = {
        '景色': 'scenery_score',
        '交通': 'transportation_score',
        '美食': 'food_score',
        '消费': 'consumption_score',
        '服务': 'service_score'
    }

    # ==================== 缺失值填充配置 ====================
    # 移动平均窗口大小（天数）
    MOVING_AVERAGE_WINDOW = 7

    # 是否使用全局均值作为备选填充方案
    USE_GLOBAL_MEAN_FALLBACK = True

    # ==================== 输出格式配置 ====================
    # 保留的小数位数
    DECIMAL_PLACES = 3

    # ==================== 日志配置 ====================
    LOG_LEVEL = "INFO"
    LOG_TO_CONSOLE = True

    # ==================== 辅助方法 ====================
    @classmethod
    def create_dirs(cls):
        """创建必要的输出目录"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        """打印当前配置信息"""
        print("\n" + "=" * 80)
        print("M5 日情感聚合配置（MySQL版）")
        print("=" * 80)

        # 数据库配置（隐藏密码）
        db_safe = cls.DATABASE_CONFIG.copy()
        db_safe['password'] = '******'
        print(f"【数据库连接】")
        print(f"  配置: {db_safe}")

        print(f"\n【数据源】")
        print(f"  输入表: {cls.INPUT_TABLE}")
        print(f"  输出表: {cls.OUTPUT_TABLE}")
        print(f"  景区列表: {cls.SCENIC_NAMES}")
        print(f"  日期字段: {cls.DATE_FIELD}")

        print(f"\n【情感维度】")
        print(f"  维度列表: {cls.DIMENSIONS}")
        print(f"  维度数量: {len(cls.DIMENSIONS)}")

        print(f"\n【填充策略】")
        print(f"  移动平均窗口: {cls.MOVING_AVERAGE_WINDOW} 天")
        print(f"  全局均值填充: {'启用' if cls.USE_GLOBAL_MEAN_FALLBACK else '禁用'}")

        print(f"\n【输出格式】")
        print(f"  保留小数位: {cls.DECIMAL_PLACES}")

        print(f"\n【日志配置】")
        print(f"  日志文件: {cls.LOG_FILE}")
        print(f"  日志级别: {cls.LOG_LEVEL}")
        print(f"  控制台输出: {'启用' if cls.LOG_TO_CONSOLE else '禁用'}")
        print("=" * 80 + "\n")

    @classmethod
    def validate(cls):
        """验证配置的合法性"""
        if not cls.DIMENSIONS:
            raise ValueError("情感维度列表不能为空")

        if not cls.SCENIC_NAMES:
            raise ValueError("景区列表不能为空")

        if cls.MOVING_AVERAGE_WINDOW < 1:
            raise ValueError(f"移动平均窗口必须 >= 1，当前值: {cls.MOVING_AVERAGE_WINDOW}")

        if cls.DECIMAL_PLACES < 0:
            raise ValueError(f"小数位数不能为负，当前值: {cls.DECIMAL_PLACES}")

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if cls.LOG_LEVEL not in valid_log_levels:
            raise ValueError(f"无效的日志级别: {cls.LOG_LEVEL}")

        print("✓ 配置验证通过")


if __name__ == "__main__":
    M5Config.print_config()
    try:
        M5Config.validate()
        print("\n✓ 配置文件正常")
    except Exception as e:
        print(f"\n✗ 配置错误: {e}")