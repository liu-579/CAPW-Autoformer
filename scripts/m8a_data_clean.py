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

# 导入配置文件
import config.m8a_config as config

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
            print(f"✓ 目标景区: {self.scenic_name}")
            print(f"✓ 原始列数: {len(self.df.columns)}")

            # 强制转换 passenger_count 为 float
            if 'passenger_count' in self.df.columns:
                print("\n--- 修复 passenger_count 类型 (text → float) ---")
                original_type = self.df['passenger_count'].dtype
                print(f"原始类型: {original_type}")

                # 尝试转换,记录失败的行
                conversion_errors = []
                for idx, value in self.df['passenger_count'].items():
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        if 'id' in self.df.columns:
                            conversion_errors.append({
                                'id': self.df.loc[idx, 'id'],
                                'row_index': idx,
                                'value': value
                            })
                        else:
                            conversion_errors.append({
                                'id': 'N/A',
                                'row_index': idx,
                                'value': value
                            })

                if conversion_errors:
                    print(f"\n⚠ 警告: 发现 {len(conversion_errors)} 行无法转换的数据:")
                    print("需要人工审核的记录ID:")
                    for error in conversion_errors[:10]:  # 最多显示10条
                        print(f"  - ID: {error['id']}, Row: {error['row_index']}, Value: '{error['value']}'")
                    if len(conversion_errors) > 10:
                        print(f"  ... 还有 {len(conversion_errors) - 10} 条记录")

                    # 询问是否继续
                    user_input = input("\n是否继续清洗(将这些行的passenger_count设为NaN)? (y/n): ")
                    if user_input.lower() != 'y':
                        print("清洗中止")
                        sys.exit(0)

                # 执行转换
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
        """Step 2: 中文特征编码"""
        print(f"\n{'=' * 60}")
        print("Step 2: 中文特征编码 (解决空值问题的关键)")
        print(f"{'=' * 60}")

        # 2.1 节假日二值化处理
        if 'holiday' in self.df.columns:
            print("\n--- 2.1 节假日二值化 (Holiday → is_holiday_int) ---")
            print(f"原始值分布:\n{self.df['holiday'].value_counts(dropna=False)}")

            # 执行二值化: 非节假日或空值→0, 其他→1
            self.df['is_holiday_int'] = self.df['holiday'].apply(
                lambda x: 0 if (pd.isna(x) or x == config.HOLIDAY_NORMAL_VALUE) else 1
            )

            print(f"\n编码后分布:")
            print(f"  0 (非节假日): {(self.df['is_holiday_int'] == 0).sum()} 行")
            print(f"  1 (节假日):   {(self.df['is_holiday_int'] == 1).sum()} 行")

            self.encoding_mappings['holiday'] = {
                0: '非节假日',
                1: '节假日(春节/国庆等)'
            }

        # 2.2 星期固定字典映射
        if 'day_of_week' in self.df.columns:
            print("\n--- 2.2 星期固定映射 (防止排序混乱) ---")
            print(f"原始值分布:\n{self.df['day_of_week'].value_counts(dropna=False)}")

            self.df['day_of_week_int'] = self.df['day_of_week'].map(
                config.DAY_OF_WEEK_MAPPING
            )

            print(f"\n映射规则:")
            for chinese, num in sorted(config.DAY_OF_WEEK_MAPPING.items(), key=lambda x: x[1]):
                count = (self.df['day_of_week_int'] == num).sum()
                print(f"  {num} = {chinese} ({count} 行)")

            # 检查未映射的值
            unmapped = self.df[self.df['day_of_week_int'].isna()]['day_of_week'].unique()
            if len(unmapped) > 0:
                print(f"\n⚠ 警告: 发现未映射的星期值: {unmapped}")

            self.encoding_mappings['day_of_week'] = config.DAY_OF_WEEK_MAPPING

        # 2.3 天气状况 LabelEncoder
        if 'weather_cond' in self.df.columns:
            print("\n--- 2.3 天气状况编码 (LabelEncoder) ---")
            self._encode_with_label_encoder('weather_cond', 'weather_cond_int')

        # 2.4 风向 LabelEncoder
        if 'wind_dir' in self.df.columns:
            print("\n--- 2.4 风向编码 (LabelEncoder) ---")
            self._encode_with_label_encoder('wind_dir', 'wind_dir_int')

    def _encode_with_label_encoder(self, col_name, new_col_name):
        """使用 LabelEncoder 编码,并打印映射关系"""
        print(f"原始值分布:\n{self.df[col_name].value_counts(dropna=False)}")

        # 处理空值: 填充为 "未知"
        temp_col = self.df[col_name].fillna('未知')

        # 执行编码
        le = LabelEncoder()
        self.df[new_col_name] = le.fit_transform(temp_col)

        # 打印映射关系
        print(f"\n编码映射关系:")
        mapping = {}
        for idx, label in enumerate(le.classes_):
            count = (self.df[new_col_name] == idx).sum()
            print(f"  {idx} = {label} ({count} 行)")
            mapping[idx] = label

        self.encoding_mappings[col_name] = mapping

    def strong_cleaning(self):
        """Step 3: 强力数据清洗"""
        print(f"\n{'=' * 60}")
        print("Step 3: 强力数据清洗")
        print(f"{'=' * 60}")

        initial_shape = self.df.shape
        print(f"清洗前: {initial_shape[0]} 行 × {initial_shape[1]} 列")

        # 3.1 删除全空列
        if config.DROP_EMPTY_COLS:
            print("\n--- 3.1 删除全空列 ---")
            empty_cols = self.df.columns[self.df.isna().all()].tolist()
            if empty_cols:
                print(f"发现全空列: {empty_cols}")
                self.df = self.df.drop(columns=empty_cols)
                print(f"✓ 已删除 {len(empty_cols)} 个全空列")
            else:
                print("✓ 未发现全空列")

        # 3.2 删除空行 (处理滞后特征产生的NaN)
        if config.DROP_EMPTY_ROWS:
            print("\n--- 3.2 删除空行 (滞后特征NaN) ---")
            rows_before = len(self.df)
            self.df = self.df.dropna()
            rows_after = len(self.df)
            deleted_rows = rows_before - rows_after
            if deleted_rows > 0:
                print(f"✓ 已删除 {deleted_rows} 行 (包含NaN的行)")
            else:
                print("✓ 未发现空行")

        # 3.3 删除常数列
        if config.DROP_CONSTANT_COLS:
            print("\n--- 3.3 删除常数列 (方差=0) ---")

            # 只检查数值列
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

            # 排除保留列
            preserve_cols = set(config.PRESERVE_COLS + ['is_holiday_int'])
            check_cols = [col for col in numeric_cols if col not in preserve_cols]

            constant_cols = []
            for col in check_cols:
                variance = self.df[col].var()
                if variance == config.VARIANCE_THRESHOLD:
                    constant_cols.append(col)
                    print(f"  - {col}: 方差={variance:.6f}, 唯一值={self.df[col].nunique()}")

            if constant_cols:
                self.df = self.df.drop(columns=constant_cols)
                print(f"✓ 已删除 {len(constant_cols)} 个常数列")
            else:
                print("✓ 未发现常数列")

        # 删除原始中文列 (保留编码后的列)
        print("\n--- 3.4 删除原始中文列 (保留编码列) ---")
        text_cols_to_drop = [col for col in config.TEXT_COLS.keys() if col in self.df.columns]
        if text_cols_to_drop:
            self.df = self.df.drop(columns=text_cols_to_drop)
            print(f"✓ 已删除原始中文列: {text_cols_to_drop}")

        final_shape = self.df.shape
        print(f"\n清洗后: {final_shape[0]} 行 × {final_shape[1]} 列")
        print(f"删除: {initial_shape[0] - final_shape[0]} 行, {initial_shape[1] - final_shape[1]} 列")

    def output_results(self):
        """Step 4: 结果输出"""
        print(f"\n{'=' * 60}")
        print("Step 4: 结果输出 (Human-Readable)")
        print(f"{'=' * 60}")

        # 4.1 保存到 MySQL
        table_name = f"{config.OUTPUT_TABLE_PREFIX}_{self.scenic_name}"
        print(f"\n--- 4.1 保存到 MySQL ---")
        print(f"目标表: {table_name}")

        try:
            self.df.to_sql(
                table_name,
                self.engine,
                if_exists='replace',
                index=False,
                chunksize=1000
            )
            print(f"✓ MySQL 保存成功: {len(self.df)} 行")
        except Exception as e:
            print(f"✗ MySQL 保存失败: {e}")

        # 4.2 保存到 CSV
        csv_filename = f"{config.CSV_OUTPUT_PREFIX}_{self.scenic_name}.csv"
        print(f"\n--- 4.2 保存到 CSV ---")
        print(f"文件名: {csv_filename}")

        try:
            self.df.to_csv(
                csv_filename,
                index=False,
                encoding=config.CSV_ENCODING
            )
            print(f"✓ CSV 保存成功 (带BOM头,防止Excel中文乱码)")
        except Exception as e:
            print(f"✗ CSV 保存失败: {e}")

        # 4.3 打印编码映射摘要
        if config.SHOW_ENCODING_MAPPING:
            print(f"\n{'=' * 60}")
            print("编码映射关系汇总 (供人工核对)")
            print(f"{'=' * 60}")
            for col_name, mapping in self.encoding_mappings.items():
                print(f"\n【{col_name}】")
                for code, label in sorted(mapping.items()):
                    print(f"  {code} → {label}")

        # 4.4 数据预览
        print(f"\n{'=' * 60}")
        print("清洗后数据预览 (前5行)")
        print(f"{'=' * 60}")
        print(self.df.head())

        print(f"\n{'=' * 60}")
        print("数据列名")
        print(f"{'=' * 60}")
        print(f"共 {len(self.df.columns)} 列:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"  {i}. {col}")

    def run(self):
        """执行完整的清洗流程"""
        print(f"\n{'#' * 60}")
        print(f"# M8A 数据清洗流程启动")
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