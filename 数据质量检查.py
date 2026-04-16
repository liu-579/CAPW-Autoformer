import pandas as pd
from sqlalchemy import create_engine

# ==========================================
# 1. 数据库与表配置 (请替换为你自己的实际信息)
# ==========================================
DB_USER = 'root'
DB_PASSWORD = 'password'
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_DATABASE = 'yq_data'
TABLE_NAME = 'history_hourly_eedscy'
TIME_COLUMN = 'datetime'  # 你的小时粒度时间列字段名


def check_hourly_data_continuity():
    # 创建 SQLAlchemy 数据库连接引擎
    # 这里使用 pymysql 作为 MySQL 驱动
    engine_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
    engine = create_engine(engine_url)

    print("⏳ 正在连接数据库并提取时间数据...")

    # ==========================================
    # 2. 读取数据
    # 为了节省内存和网络带宽，我们只 SELECT 时间列
    # ==========================================
    query = f"SELECT {TIME_COLUMN} FROM {TABLE_NAME}"

    try:
        # 将 SQL 查询结果直接加载为 Pandas DataFrame
        df = pd.read_sql(query, engine)

        if df.empty:
            print("⚠️ 数据表为空，无法进行连续性检查。")
            return

        # 确保该列被正确解析为 datetime 类型
        df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])

        # 可选：如果你的数据存在同一个小时有多条记录的情况，可以去重
        # df = df.drop_duplicates(subset=[TIME_COLUMN])

        # ==========================================
        # 3. 核心逻辑：比对时间序列
        # ==========================================
        min_time = df[TIME_COLUMN].min()
        max_time = df[TIME_COLUMN].max()

        actual_count = len(df)

        print("-" * 30)
        print(f"📈 数据概览:")
        print(f"起始时间: {min_time}")
        print(f"结束时间: {max_time}")
        print(f"实际数据量: {actual_count} 条")

        # 使用 pd.date_range 生成期望的连续时间范围，freq='h' 代表小时粒度
        expected_time_series = pd.date_range(start=min_time, end=max_time, freq='h')
        expected_count = len(expected_time_series)
        print(f"期望数据量: {expected_count} 条")
        print("-" * 30)

        # 找出缺失的时间点
        # 将实际时间和期望时间转换为 Set 集合，做差集运算
        actual_time_set = set(df[TIME_COLUMN])
        expected_time_set = set(expected_time_series)

        missing_times = sorted(list(expected_time_set - actual_time_set))

        # ==========================================
        # 4. 输出检查报告
        # ==========================================
        if not missing_times:
            print("✅ 检查通过：数据时间完全连续，中间无任何空缺！")
        else:
            print(f"❌ 发现数据缺失！在首尾区间内，共缺失了 {len(missing_times)} 个小时的数据。")
            print("以下是缺失的具体时间点（最多展示前 20 条）：")
            for t in missing_times[:20]:
                print(f"   -> {t}")

            if len(missing_times) > 20:
                print("   -> ... (更多缺失时间已省略)")

            # 你可以在这里将 missing_times 写入日志、发送告警邮件，或者存入异常监控表中
            # return missing_times

    except Exception as e:
        print(f"🚨 运行过程中出现错误: {e}")


if __name__ == "__main__":
    check_hourly_data_continuity()