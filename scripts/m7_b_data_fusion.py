"""
数据融合与特征工程主程序（小时尺度版本）
功能：将日度情感数据广播到小时尺度，与小时物理特征数据融合，并构建时序衍生特征
版本：m7_b - 将每日的情感数据广播给当日的每小时
修改记录：已添加"周期性回填"策略，利用第二年数据修复第一年头部缺失值
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import warnings

# 导入配置文件
import config.m7_b_config as config

# 忽略 pandas 的链式赋值警告
warnings.filterwarnings('ignore')


class DataFusionProcessorHourly:
    """数据融合处理器（小时尺度）- 负责情感数据（日度）与物理数据（小时）的融合及特征工程"""

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
        步骤1：读取情感数据总表（日度数据）
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
        处理单个景区的数据融合与特征工程（小时尺度）

        参数:
            scenic_name (str): 景区中文名称
            physical_table (str): 对应的物理特征数据表名（小时尺度）

        返回:
            pd.DataFrame: 融合并完成特征工程的数据（小时尺度）
        """
        self.log(f"\n{'=' * 60}")
        self.log(f"正在处理景区：{scenic_name}")
        self.log(f"对应物理表：{physical_table}")
        self.log(f"{'=' * 60}")

        # Step A: 筛选情感数据（日度）
        self.log("Step A - 筛选情感数据（日度）...")
        sentiment_subset = self.sentiment_data[
            self.sentiment_data[config.SCENIC_NAME_FIELD] == scenic_name
        ].copy()

        self.log(f"情感数据筛选完成，共 {len(sentiment_subset)} 条记录")

        if len(sentiment_subset) == 0:
            self.log(f"警告：景区 {scenic_name} 没有情感数据，跳过处理", level='WARNING')
            return None

        # Step B: 读取物理特征数据（小时尺度）
        self.log("Step B - 读取物理特征数据（小时尺度）...")
        try:
            query = f"SELECT * FROM {physical_table}"
            physical_data = pd.read_sql(query, self.engine)

            self.log(f"物理数据读取完成，共 {len(physical_data)} 条记录")

            # 确保 datetime 字段为 datetime 类型
            if config.DATETIME_FIELD in physical_data.columns:
                physical_data[config.DATETIME_FIELD] = pd.to_datetime(
                    physical_data[config.DATETIME_FIELD]
                )
            else:
                self.log(f"错误：物理数据表缺少 datetime 字段", level='ERROR')
                return None

            # 从 datetime 字段提取日期用于融合
            physical_data[config.DATE_FIELD] = physical_data[config.DATETIME_FIELD].dt.date
            physical_data[config.DATE_FIELD] = pd.to_datetime(physical_data[config.DATE_FIELD])

            # 确保小时字段存在
            if config.HOUR_FIELD not in physical_data.columns:
                self.log(f"错误：物理数据表缺少小时字段 '{config.HOUR_FIELD}'", level='ERROR')
                return None

        except Exception as e:
            self.log(f"物理数据读取失败：{e}", level='ERROR')
            return None

        # Step C: 数据融合 - 将日度情感数据广播到小时尺度
        self.log("Step C - 执行数据融合（将日度情感数据广播到每小时）...")
        
        # 方法：通过日期字段进行左连接，将情感数据复制到每个小时
        merged_data = physical_data.merge(
            sentiment_subset,
            on=config.DATE_FIELD,
            how='left',  # 保留所有物理数据的小时记录
            suffixes=('_physical', '_sentiment')
        )

        self.log(f"数据融合完成，样本数：{len(merged_data)}")

        if len(merged_data) == 0:
            self.log(f"警告：景区 {scenic_name} 融合后无数据，跳过特征工程", level='WARNING')
            return None

        # 使用物理表中已有的 datetime 字段（已经包含了日期和小时信息）
        # 按日期时间排序（时序特征工程的前提）
        merged_data = merged_data.sort_values(by=config.DATETIME_FIELD).reset_index(drop=True)

        self.log(f"日期时间范围：{merged_data[config.DATETIME_FIELD].min()} 至 {merged_data[config.DATETIME_FIELD].max()}")

        # Step D: 特征工程（基于小时尺度）
        self.log("Step D - 开始特征工程（小时尺度）...")
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
            self.log(f"  时间范围：{feature_data[config.DATETIME_FIELD].min()} 至 {feature_data[config.DATETIME_FIELD].max()}")

        return feature_data

    def build_time_series_features(self, data):
        """
        构建时序衍生特征（滞后特征 + 移动平均特征）- 小时尺度版本
        修改版：使用"周期性回填"策略修复头部缺失值
        """
        df = data.copy()

        # =======================================================
        # 1. 基础数据填充策略 (修复原始数据中的空洞)
        # =======================================================
        # 策略：前向填充 (Forward Fill)
        # 逻辑：如果某小时数据缺失，假设它与前一小时持平。
        df = df.ffill()

        # 如果第一行就是空的（无法前向填充），则使用后向填充兜底
        df = df.bfill()

        # =======================================================
        # 2. 特征生成循环（基于小时尺度）
        # =======================================================
        for col in config.SENTIMENT_COLS:
            if col not in df.columns:
                self.log(f"警告：字段 {col} 不存在，跳过特征生成", level='WARNING')
                continue

            # 构建滞后特征 (Lag) - 小时尺度
            for lag in config.LAG_HOURS:
                lag_col_name = config.get_lag_feature_name(col, lag)
                df[lag_col_name] = df[col].shift(lag)

            # 构建移动平均特征 (Rolling) - 小时尺度
            rolling_col_name = config.get_rolling_feature_name(col, config.ROLLING_WINDOW)
            df[rolling_col_name] = (
                df[col]
                .shift(1)  # 保持防泄露逻辑
                .rolling(window=config.ROLLING_WINDOW, min_periods=1)
                .mean()
            )

        # =======================================================
        # 3. 最终策略：周期性回填 (Seasonal Backfill)
        # =======================================================

        # 业务逻辑：
        # 利用数据的强周期性（拥有两年数据），用"下一年同一时刻"的数据，
        # 来填补"第一年"缺失的滞后特征。
        # 目的：确保国庆七天的数据具有"国庆特征"，而不是被切除或被平日均值稀释。

        df_clean = df.copy()

        # 获取所有包含空值的行的索引（通常是前几小时到几天）
        nan_indices = df_clean[df_clean.isnull().any(axis=1)].index

        if len(nan_indices) > 0:
            self.log(f"  检测到 {len(nan_indices)} 行头部缺失值，尝试使用'下一年'数据进行周期性回填...")

        fill_count = 0

        for idx in nan_indices:
            # 获取当前行的日期时间
            current_datetime = df_clean.loc[idx, config.DATETIME_FIELD]

            # 计算"下一年"的日期时间 (处理闰年等情况)
            next_year_datetime = current_datetime + pd.DateOffset(years=1)

            # 在数据中查找下一年的那一行
            future_row = df_clean[df_clean[config.DATETIME_FIELD] == next_year_datetime]

            if not future_row.empty:
                # 找到对应行，开始回填
                future_idx = future_row.index[0]

                # 遍历该行所有列
                for col in df_clean.columns:
                    # 如果当前列是空的（需要填补）
                    if pd.isna(df_clean.loc[idx, col]):
                        # 且未来那行对应的值不是空的（有资源可填）
                        if not pd.isna(df_clean.loc[future_idx, col]):
                            df_clean.loc[idx, col] = df_clean.loc[future_idx, col]
                            fill_count += 1
            else:
                # 如果没找到下一年数据（可能数据不足两年，或该时刻缺失），不做操作，留给后续兜底
                pass

        if fill_count > 0:
            self.log(f"  周期性回填完成，共填补 {fill_count} 个空缺特征值")

        # =======================================================
        # 4. 兜底策略 (Fallback)
        # =======================================================
        # 如果还有空值（例如第二年也没有这一时刻），则使用均值填充作为最后的防线。
        # 这样确保了不会有任何 NaN 进入模型。

        cleaned_count = 0
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    mean_val = df_clean[col].mean()
                    df_clean[col].fillna(mean_val, inplace=True)
                else:
                    df_clean[col].bfill(inplace=True)
                cleaned_count += 1

        if cleaned_count > 0:
            self.log(f"  已对剩余空缺列执行均值/后向填充兜底")

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
    print("🚀 数据融合与特征工程任务启动（小时尺度版本）")
    print(f"情感数据源表：{config.SENTIMENT_TABLE}（日度）")
    print(f"物理数据源表：小时尺度")
    print(f"输出目标表：{config.OUTPUT_TABLE}（小时尺度）")
    print(f"待处理景区数量：{len(config.SCENIC_MAPPING)}")
    print("=" * 60 + "\n")

    # 创建处理器实例并运行
    processor = DataFusionProcessorHourly()
    processor.run()


if __name__ == "__main__":
    main()
