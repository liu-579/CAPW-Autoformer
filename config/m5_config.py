"""
M5 日情感聚合配置文件
功能：配置M5日聚合任务的所有参数
作者：AI Assistant
日期：2025-12-01
"""

from pathlib import Path


class M5Config:
    """M5 日情感聚合任务配置"""

    # ==================== 路径配置 ====================
    # 项目根目录
    BASE_DIR = Path(__file__).parent.parent

    # 数据目录
    DATA_DIR = BASE_DIR / "data"
    # 输出目录
    OUTPUT_DIR = DATA_DIR / "output" / "m5"

    # 输入文件：M4-b的输出结果（按条的情感分析结果）
    # 格式：{"id": "...", "text": "...", "date": "2021-06-15", "year": 2021,
    #        "month": 6, "day": 15, "景色": 1, "交通": 0, ...}
    INPUT_FILE = BASE_DIR / r"data/output/m4_b_inference/m4_b_sentiment.jsonl"

    # 输出文件：按天聚合的情感均值
    # 格式：{"date": "2021-06-15", "景色": 1.230, "交通": 0.750, ..., "n_comments": 42}
    OUTPUT_FILE = OUTPUT_DIR / "m5_sentiment_daily_int.jsonl"

    # 日志文件：记录聚合过程的详细信息
    LOG_FILE = OUTPUT_DIR / "inference_log.txt"

    # ==================== 情感维度配置 ====================
    # 情感维度列表（必须与M4输出的维度完全一致）
    # 这5个维度是景区评论的核心情感类别
    DIMENSIONS = [
        "景色",  # 景色美观度
        "交通",  # 交通便利性
        "美食",  # 美食满意度
        "门票",  # 门票价格合理性
        "服务"  # 服务质量
    ]

    # ==================== 缺失值填充配置 ====================
    # 移动平均窗口大小（天数）
    # 说明：当某天无评论时，优先使用前N天的平均值填充
    # 示例：MOVING_AVERAGE_WINDOW=7 表示使用前7天的均值
    # 注意：只使用有实际评论的天数计算均值，空白天不参与
    MOVING_AVERAGE_WINDOW = 7

    # 是否使用全局均值作为备选填充方案
    # 说明：当移动平均也无法填充时（前N天都没有评论），使用全局均值
    # True: 启用全局均值填充，确保所有日期都有数据
    # False: 不填充，保留空白
    USE_GLOBAL_MEAN_FALLBACK = True

    # ==================== 输出格式配置 ====================
    # 保留的小数位数
    # 说明：所有情感分数都会四舍五入到指定小数位
    # 示例：1.23456 -> 1.235 (DECIMAL_PLACES=3)
    DECIMAL_PLACES = 3

    # ==================== 日志配置 ====================
    # 日志级别
    # 可选：DEBUG, INFO, WARNING, ERROR, CRITICAL
    # 推荐：INFO（记录关键信息）或 DEBUG（记录详细调试信息）
    LOG_LEVEL = "INFO"

    # 是否同时输出到控制台
    # True: 日志同时写入文件和显示在屏幕
    # False: 日志仅写入文件
    LOG_TO_CONSOLE = True

    # ==================== 辅助方法 ====================
    @classmethod
    def create_dirs(cls):
        """
        创建必要的输出目录
        说明：确保输出目录存在，不存在则自动创建
        """
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        """
        打印当前配置信息
        说明：在程序运行开始时，显示所有配置参数，便于确认
        """
        print("\n" + "=" * 80)
        print("M5 日情感聚合配置")
        print("=" * 80)
        print(f"【路径配置】")
        print(f"  输入文件: {cls.INPUT_FILE}")
        print(f"  输出文件: {cls.OUTPUT_FILE}")
        print(f"  日志文件: {cls.LOG_FILE}")
        print(f"\n【情感维度】")
        print(f"  维度列表: {cls.DIMENSIONS}")
        print(f"  维度数量: {len(cls.DIMENSIONS)}")
        print(f"\n【填充策略】")
        print(f"  移动平均窗口: {cls.MOVING_AVERAGE_WINDOW} 天")
        print(f"  全局均值填充: {'启用' if cls.USE_GLOBAL_MEAN_FALLBACK else '禁用'}")
        print(f"\n【输出格式】")
        print(f"  保留小数位: {cls.DECIMAL_PLACES}")
        print(f"\n【日志配置】")
        print(f"  日志级别: {cls.LOG_LEVEL}")
        print(f"  控制台输出: {'启用' if cls.LOG_TO_CONSOLE else '禁用'}")
        print("=" * 80 + "\n")

    @classmethod
    def validate(cls):
        """
        验证配置的合法性
        说明：检查配置参数是否合理，防止配置错误

        Raises:
            ValueError: 配置参数不合法时抛出
        """
        # 检查输入文件是否存在
        if not cls.INPUT_FILE.exists():
            raise FileNotFoundError(f"输入文件不存在: {cls.INPUT_FILE}")

        # 检查维度列表是否为空
        if not cls.DIMENSIONS:
            raise ValueError("情感维度列表不能为空")

        # 检查移动平均窗口
        if cls.MOVING_AVERAGE_WINDOW < 1:
            raise ValueError(f"移动平均窗口必须 >= 1，当前值: {cls.MOVING_AVERAGE_WINDOW}")

        # 检查小数位数
        if cls.DECIMAL_PLACES < 0:
            raise ValueError(f"小数位数不能为负，当前值: {cls.DECIMAL_PLACES}")

        # 检查日志级别
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if cls.LOG_LEVEL not in valid_log_levels:
            raise ValueError(
                f"无效的日志级别: {cls.LOG_LEVEL}，"
                f"有效值: {valid_log_levels}"
            )

        print("✓ 配置验证通过")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    """
    配置文件测试
    运行此文件可以查看当前配置并验证其合法性
    """
    # 打印配置
    M5Config.print_config()

    # 验证配置（需要输入文件存在才能通过）
    try:
        M5Config.validate()
        print("\n✓ 配置文件正常")
    except Exception as e:
        print(f"\n✗ 配置错误: {e}")