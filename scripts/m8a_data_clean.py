# -*- coding: utf-8 -*-
"""
M8A 数据清洗主程序
ETL Engineer: Chinese Text Data Cleaning Pipeline
功能: 处理中文文本脏数据,执行强力清洗,输出可读数值化宽表
"""

import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
import datetime
# 新增: 引入中国节假日库
import chinese_calendar

# 导入配置文件
import config.m8a_config as config  # 假设你的目录结构是直接引用，如果是在 config 文件夹下请保持原样

warnings.filterwarnings('ignore')


class ChineseDataCleaner:
    """中文数据清洗器 - 专门处理包含中文文本的脏数据"""

    def __init__(self):
        self.df = None
        self.scenic_name = config.TARGET_SCENIC_NAME
        self.encoding_mappings = {}  # 存储编码映射关系

    def connect_mysql(self):
        """建立 MySQL 连接"""
        print(f"\n{'=' * 60}")
        print("Step 0: 建立数据库连接")
        print(f"{'=' * 60}")

        try:
            connection_string = (
                f"mysql+pymysql://{config.DB_CONFIG['user']}:"
                f"{config.DB_CONFIG['password']}@"
                f"{config.DB_CONFIG['host']}:"
                f"{config.DB_CONFIG['port']}/"
                f"{config.DB_CONFIG['database']}?"
                f"charset={config.DB_CONFIG['charset']}"
            )
            self.engine = create_engine(connection_string)
            print(f"✓ 数据库连接成功: {config.DB_CONFIG['database']}")
            return True
        except Exception as e:
            print(f"✗ 数据库连接失败: {e}")
            return False

    def load_data(self):
        """Step 1: 定向读取与类型修复"""
        print(f"\n{'=' * 60}")
        print("Step 1: 定向读取与类型修复")
        print(f"{'=' * 60}")

        try:
            # 读取指定景区的数据
            query = f"""
            SELECT * FROM {config.SOURCE_TABLE}
            WHERE scenic_name = '{self.scenic_name}'
            """

            self.df = pd.read_sql(query, self.engine)
            print(f"✓ 读取数据成功: {len(self.df)} 行")

            # --- 新增: 确保 date 列是 datetime 类型 ---
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
                print(f"✓ Date列已转换为 datetime 类型")
            else:
                raise ValueError("数据中缺少 'date' 列，无法进行时间特征工程")

            # 强制转换 passenger_count 为 float
            if 'passenger_count' in self.df.columns:
                print("\n--- 修复 passenger_count 类型 (text → float) ---")

                # 简单处理非数值，强制转换
                self.df['passenger_count'] = pd.to_numeric(
                    self.df['passenger_count'],
                    errors='coerce'
                )
                print(f"✓ 转换后类型: {self.df['passenger_count'].dtype}")
                print(f"✓ 有效数值: {self.df['passenger_count'].notna().sum()} 行")

        except Exception as e:
            print(f"✗ 数据读取失败: {e}")
            raise

    def encode_chinese_features(self):
        """Step 2: 特征工程 (周期性编码与节假日处理)"""
        print(f"\n{'=' * 60}")
        print("Step 2: 高级特征工程 (周期性时间 & 中国节假日)")
        print(f"{'=' * 60}")

        # ==========================================
        # 任务 1: 月份特征 (周期性编码)
        # ==========================================
        print("\n--- 2.1 月份周期性编码 (Month Cyclic) ---")
        # 提取月份 (1-12)
        months = self.df['date'].dt.month

        # 周期性编码公式: sin(2 * pi * month / 12)
        self.df['month_sin'] = np.sin(2 * np.pi * months / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * months / 12)

        print(f"✓ 生成列: month_sin, month_cos")
        print(self.df[['date', 'month_sin', 'month_cos']].head(3))

        # ==========================================
        # 任务 2: 星期特征 (周期性编码)
        # ==========================================
        print("\n--- 2.2 星期周期性编码 (Weekday Cyclic) ---")
        # 提取星期 (0=周一, 6=周日)
        weekdays = self.df['date'].dt.dayofweek

        # 周期性编码公式: sin(2 * pi * weekday / 7)
        self.df['day_sin'] = np.sin(2 * np.pi * weekdays / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * weekdays / 7)

        print(f"✓ 生成列: day_sin, day_cos")
        print(self.df[['date', 'day_sin', 'day_cos']].head(3))

        # ==========================================
        # 任务 3: 是否休息日 (基于 chinese_calendar)
        # ==========================================
        print("\n--- 2.3 智能节假日判断 (Chinese Calendar) ---")
        print("正在计算复杂的调休逻辑，请稍候...")

        # 定义判断函数
        def check_is_day_off(date_val):
            # chinese_calendar.is_workday(date) 返回 True 表示是工作日（含补班）
            # 我们需要 is_day_off: 1=休息(含节假日), 0=上班(含补班)
            # 所以需要取反
            is_work = chinese_calendar.is_workday(date_val)
            return 0 if is_work else 1

        self.df['is_day_off'] = self.df['date'].apply(lambda x: check_is_day_off(x))

        count_off = self.df['is_day_off'].sum()
        count_work = len(self.df) - count_off
        print(f"✓ 生成列: is_day_off (处理了调休)")
        print(f"  - 休息状态 (1): {count_off} 天")
        print(f"  - 工作状态 (0): {count_work} 天")

        self.encoding_mappings['is_day_off'] = {0: '工作/补班', 1: '周末/节假日'}

        # ==========================================
        # 原有的 LabelEncoder (天气与风向)
        # ==========================================
        # 2.4 天气状况 LabelEncoder
        if 'weather_cond' in self.df.columns:
            print("\n--- 2.4 天气状况编码 (LabelEncoder) ---")
            self._encode_with_label_encoder('weather_cond', 'weather_cond_int')

        # 2.5 风向 LabelEncoder
        if 'wind_dir' in self.df.columns:
            print("\n--- 2.5 风向编码 (LabelEncoder) ---")
            self._encode_with_label_encoder('wind_dir', 'wind_dir_int')

    def _encode_with_label_encoder(self, col_name, new_col_name):
        """使用 LabelEncoder 编码,并打印映射关系"""
        # 处理空值: 填充为 "未知"
        temp_col = self.df[col_name].fillna('未知')

        # 执行编码
        le = LabelEncoder()
        self.df[new_col_name] = le.fit_transform(temp_col)

        # 记录映射关系
        mapping = {}
        for idx, label in enumerate(le.classes_):
            mapping[idx] = label

        self.encoding_mappings[col_name] = mapping
        print(f"✓ {col_name} 编码完成，类别数: {len(le.classes_)}")

    def strong_cleaning(self):
        """Step 3: 强力数据清洗"""
        print(f"\n{'=' * 60}")
        print("Step 3: 强力数据清洗")
        print(f"{'=' * 60}")

        initial_shape = self.df.shape
        print(f"清洗前: {initial_shape[0]} 行 × {initial_shape[1]} 列")

        # 3.1 删除全空列
        if config.DROP_EMPTY_COLS:
            self.df = self.df.dropna(axis=1, how='all')

        # 3.2 删除空行
        if config.DROP_EMPTY_ROWS:
            self.df = self.df.dropna()

        # 3.3 删除常数列 (排除保留列)
        if config.DROP_CONSTANT_COLS:
            print("\n--- 检查常数列 ---")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除保留列和新生成的特征
            preserve_cols = set(config.PRESERVE_COLS +
                                ['is_day_off', 'month_sin', 'month_cos', 'day_sin', 'day_cos'])

            check_cols = [col for col in numeric_cols if col not in preserve_cols]

            cols_to_drop = []
            for col in check_cols:
                if self.df[col].std() == 0:  # std=0 等同于方差=0
                    cols_to_drop.append(col)

            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
                print(f"✓ 已删除常数列: {cols_to_drop}")

        # 3.4 删除原始中文列
        print("\n--- 删除原始中文列 ---")
        # 配置文件中定义的 TEXT_COLS 里的列，如果在 df 中存在，则删除
        text_cols_to_drop = [col for col in config.TEXT_COLS.keys() if col in self.df.columns]
        if text_cols_to_drop:
            self.df = self.df.drop(columns=text_cols_to_drop)
            print(f"✓ 已删除原始文本列: {text_cols_to_drop}")

        final_shape = self.df.shape
        print(f"\n清洗后: {final_shape[0]} 行 × {final_shape[1]} 列")

    def output_results(self):
        """Step 4: 结果输出"""
        print(f"\n{'=' * 60}")
        print("Step 4: 结果输出")
        print(f"{'=' * 60}")

        # 4.1 保存到 MySQL
        table_name = f"{config.OUTPUT_TABLE_PREFIX}_{self.scenic_name}"
        try:
            self.df.to_sql(
                table_name,
                self.engine,
                if_exists='replace',
                index=False,
                chunksize=1000
            )
            print(f"✓ MySQL 保存成功: {table_name}")
        except Exception as e:
            print(f"✗ MySQL 保存失败: {e}")

        # 4.2 保存到 CSV
        csv_filename = f"{config.CSV_OUTPUT_PREFIX}_{self.scenic_name}.csv"
        try:
            self.df.to_csv(
                csv_filename,
                index=False,
                encoding=config.CSV_ENCODING
            )
            print(f"✓ CSV 保存成功: {csv_filename}")
        except Exception as e:
            print(f"✗ CSV 保存失败: {e}")

        # 4.3 打印预览
        print(f"\n数据列预览:")
        print(self.df.columns.tolist())
        print(f"\n前5行数据:")
        print(self.df.head())

    def run(self):
        """执行完整的清洗流程"""
        print(f"\n{'#' * 60}")
        print(f"# M8A 数据清洗流程启动 (Enhanced Feature Engineering)")
        print(f"# 目标景区: {self.scenic_name}")
        print(f"{'#' * 60}")

        if not self.connect_mysql():
            return False

        try:
            self.load_data()
            self.encode_chinese_features()
            self.strong_cleaning()
            self.output_results()

            print(f"\n{'#' * 60}")
            print(f"# 清洗流程完成!")
            print(f"{'#' * 60}\n")
            return True

        except Exception as e:
            print(f"\n✗ 清洗流程出错: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    cleaner = ChineseDataCleaner()
    cleaner.run()