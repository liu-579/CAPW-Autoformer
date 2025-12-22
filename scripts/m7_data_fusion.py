"""
数据融合与特征工程主程序
功能：将日度情感数据与物理特征数据融合，并构建时序衍生特征
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import warnings

# 导入配置文件
import config.m7_config as config

# 忽略 pandas 的链式赋值警告
warnings.filterwarnings('ignore')


class DataFusionProcessor:
    """数据融合处理器 - 负责情感数据与物理数据的融合及特征工程"""

    def __init__(self):
        """初始化处理器"""
        self.engine = None
        self.sentiment_data = None
        self.fused_data_list = []  # 存储所有景区的融合数据

    def log(self, message, level='INFO'):
        """
        打印日志信息

        参数:
            message (str): 日志内容
            level (str): 日志级别（INFO, WARNING, ERROR）
        """
        if config.VERBOSE_LOGGING:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] [{level}] {message}")

    def connect_database(self):
        """建立数据库连接"""
        self.log("正在连接数据库...")
        try:
            self.engine = create_engine(config.get_database_url())
            self.log(f"数据库连接成功：{config.DB_CONFIG['database']}")
        except Exception as e:
            self.log(f"数据库连接失败：{e}", level='ERROR')
            raise

    def load_sentiment_data(self):
        """
        步骤1：读取情感数据总表
        """
        self.log(f"正在读取情感数据表：{config.SENTIMENT_TABLE}...")

        try:
            query = f"SELECT * FROM {config.SENTIMENT_TABLE}"
            self.sentiment_data = pd.read_sql(query, self.engine)

            self.log(f"情感数据加载完成，共 {len(self.sentiment_data)} 条记录")
            self.log(f"包含景区：{self.sentiment_data[config.SCENIC_NAME_FIELD].unique().tolist()}")

            # 确保日期字段为 datetime 类型
            if config.DATE_FIELD in self.sentiment_data.columns:
                self.sentiment_data[config.DATE_FIELD] = pd.to_datetime(
                    self.sentiment_data[config.DATE_FIELD]
                )

        except Exception as e:
            self.log(f"情感数据加载失败：{e}", level='ERROR')
            raise

    def process_scenic_spot(self, scenic_name, physical_table):
        """
        处理单个景区的数据融合与特征工程

        参数:
            scenic_name (str): 景区中文名称
            physical_table (str): 对应的物理特征数据表名

        返回:
            pd.DataFrame: 融合并完成特征工程的数据
        """
        self.log(f"\n{'=' * 60}")
        self.log(f"正在处理景区：{scenic_name}")
        self.log(f"对应物理表：{physical_table}")
        self.log(f"{'=' * 60}")

        # Step A: 筛选情感数据
        self.log("Step A - 筛选情感数据...")
        sentiment_subset = self.sentiment_data[
            self.sentiment_data[config.SCENIC_NAME_FIELD] == scenic_name
            ].copy()

        self.log(f"情感数据筛选完成，共 {len(sentiment_subset)} 条记录")

        if len(sentiment_subset) == 0:
            self.log(f"警告：景区 {scenic_name} 没有情感数据，跳过处理", level='WARNING')
            return None

        # Step B: 读取物理特征数据
        self.log("Step B - 读取物理特征数据...")
        try:
            query = f"SELECT * FROM {physical_table}"
            physical_data = pd.read_sql(query, self.engine)

            self.log(f"物理数据读取完成，共 {len(physical_data)} 条记录")

            # 确保日期字段为 datetime 类型
            if config.DATE_FIELD in physical_data.columns:
                physical_data[config.DATE_FIELD] = pd.to_datetime(
                    physical_data[config.DATE_FIELD]
                )

        except Exception as e:
            self.log(f"物理数据读取失败：{e}", level='ERROR')
            return None

        # Step C: 数据融合（Inner Join）
        self.log("Step C - 执行数据融合（Inner Join）...")
        merged_data = sentiment_subset.merge(
            physical_data,
            on=config.DATE_FIELD,
            how='inner',
            suffixes=('_sentiment', '_physical')
        )

        self.log(f"数据融合完成，样本数：{len(merged_data)}")

        if len(merged_data) == 0:
            self.log(f"警告：景区 {scenic_name} 融合后无数据，跳过特征工程", level='WARNING')
            return None

        # 按日期排序（时序特征工程的前提）
        merged_data = merged_data.sort_values(by=config.DATE_FIELD).reset_index(drop=True)

        # Step D: 特征工程
        self.log("Step D - 开始特征工程...")
        feature_data = self.build_time_series_features(merged_data)

        # 添加景区标识
        if config.KEEP_SCENIC_NAME:
            feature_data[config.SCENIC_NAME_FIELD] = scenic_name

        if config.ADD_SOURCE_TABLE_COLUMN:
            feature_data['source_table'] = physical_table

        # 显示统计信息
        if config.SHOW_DATA_STATS:
            self.log(f"\n特征工程完成统计：")
            self.log(f"  原始样本数：{len(merged_data)}")
            self.log(f"  清洗后样本数：{len(feature_data)}")
            self.log(f"  特征列数量：{len(feature_data.columns)}")
            self.log(f"  日期范围：{feature_data[config.DATE_FIELD].min()} 至 {feature_data[config.DATE_FIELD].max()}")

        return feature_data

    def build_time_series_features(self, data):
        """
        构建时序衍生特征（滞后特征 + 移动平均特征）

        参数:
            data (pd.DataFrame): 融合后的原始数据

        返回:
            pd.DataFrame: 添加了时序特征的数据（已清除 NaN）
        """
        df = data.copy()

        # 遍历每个情感维度字段
        for col in config.SENTIMENT_COLS:
            if col not in df.columns:
                self.log(f"警告：字段 {col} 不存在，跳过特征生成", level='WARNING')
                continue

            # 1. 构建滞后特征（Lag Features）
            self.log(f"  正在为 {col} 构建滞后特征...")
            for lag in config.LAG_DAYS:
                lag_col_name = config.get_lag_feature_name(col, lag)
                df[lag_col_name] = df[col].shift(lag)

            # 2. 构建移动平均特征（Rolling Mean Features）
            # 注意：使用 shift(1) 确保不使用当天数据（避免数据泄露）
            self.log(f"  正在为 {col} 构建移动平均特征（窗口={config.ROLLING_WINDOW}天）...")
            rolling_col_name = config.get_rolling_feature_name(col, config.ROLLING_WINDOW)
            df[rolling_col_name] = (
                df[col]
                .shift(1)  # 先向下移动1天，避免用到当天数据
                .rolling(window=config.ROLLING_WINDOW, min_periods=1)
                .mean()
            )

        # 清除因滞后和移动平均产生的 NaN 行
        self.log("  正在清除特征工程产生的缺失值...")
        df_clean = df.dropna().reset_index(drop=True)

        return df_clean

    def process_all_scenic_spots(self):
        """
        步骤2：循环处理所有景区
        """
        self.log("\n" + "=" * 60)
        self.log(f"开始批量处理 {len(config.SCENIC_MAPPING)} 个景区")
        self.log("=" * 60 + "\n")

        success_count = 0
        fail_count = 0

        for scenic_name, physical_table in config.SCENIC_MAPPING.items():
            try:
                # 处理单个景区
                fused_data = self.process_scenic_spot(scenic_name, physical_table)

                if fused_data is not None and len(fused_data) > 0:
                    self.fused_data_list.append(fused_data)
                    success_count += 1
                else:
                    fail_count += 1

            except Exception as e:
                self.log(f"景区 {scenic_name} 处理失败：{e}", level='ERROR')
                fail_count += 1
                import traceback
                traceback.print_exc()

        self.log("\n" + "=" * 60)
        self.log(f"批量处理完成 - 成功：{success_count} 个，失败：{fail_count} 个")
        self.log("=" * 60 + "\n")

    def save_to_database(self):
        """
        步骤3：将融合数据回写到数据库
        """
        if len(self.fused_data_list) == 0:
            self.log("警告：没有数据可写入数据库", level='WARNING')
            return

        # 合并所有景区的数据
        self.log("正在合并所有景区的融合数据...")
        final_data = pd.concat(self.fused_data_list, ignore_index=True)

        self.log(f"合并完成，总样本数：{len(final_data)}")

        # 写入数据库
        self.log(f"正在写入数据库表：{config.OUTPUT_TABLE}...")

        try:
            final_data.to_sql(
                name=config.OUTPUT_TABLE,
                con=self.engine,
                if_exists=config.DB_WRITE_MODE,
                index=False
            )

            self.log(f"✅ 数据写入成功！表名：{config.OUTPUT_TABLE}，总记录数：{len(final_data)}")

            # 显示表结构
            if config.SHOW_DATA_STATS:
                self.log(f"\n输出表结构：")
                self.log(f"  列数：{len(final_data.columns)}")
                self.log(f"  列名：{final_data.columns.tolist()}")

        except Exception as e:
            self.log(f"数据写入失败：{e}", level='ERROR')
            raise

    def run(self):
        """
        主流程：执行完整的数据融合任务
        """
        try:
            # 连接数据库
            self.connect_database()

            # 加载情感数据
            self.load_sentiment_data()

            # 处理所有景区
            self.process_all_scenic_spots()

            # 保存到数据库
            self.save_to_database()

            self.log("\n" + "=" * 60)
            self.log("🎉 数据融合与特征工程任务全部完成！")
            self.log("=" * 60 + "\n")

        except Exception as e:
            self.log(f"\n任务执行失败：{e}", level='ERROR')
            raise

        finally:
            if self.engine:
                self.engine.dispose()
                self.log("数据库连接已关闭")


def main():
    """
    主函数入口
    """
    print("\n" + "=" * 60)
    print("🚀 数据融合与特征工程任务启动")
    print(f"情感数据源表：{config.SENTIMENT_TABLE}")
    print(f"输出目标表：{config.OUTPUT_TABLE}")
    print(f"待处理景区数量：{len(config.SCENIC_MAPPING)}")
    print("=" * 60 + "\n")

    # 创建处理器实例并运行
    processor = DataFusionProcessor()
    processor.run()


if __name__ == "__main__":
    main()