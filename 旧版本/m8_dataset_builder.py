"""
时序数据集构建主程序
功能：从MySQL读取融合数据，进行预处理、归一化和滑动窗口切片，生成.npy张量文件
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import json
from datetime import datetime
import warnings
import sys

# 导入配置文件
import 旧版本.m8_config as config

# 忽略警告
warnings.filterwarnings('ignore')


class TimeSeriesDatasetBuilder:
    """时序数据集构建器 - 负责数据预处理和张量文件生成"""

    def __init__(self):
        """初始化数据集构建器"""
        self.engine = None
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = None
        self.scaler = None
        self.feature_map = {}

        # 验证配置
        config.validate_config()

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

    def load_data(self):
        """
        Step 1: 定向数据加载
        从数据库读取指定景区的数据并按日期排序
        """
        self.log("\n" + "="*60)
        self.log("Step 1: 定向数据加载")
        self.log("="*60)

        self.log(f"目标景区：{config.TARGET_SCENIC_NAME}")
        self.log(f"数据表名：{config.INPUT_TABLE}")

        try:
            # 构建 SQL 查询语句（筛选指定景区）
            query = f"""
                SELECT * FROM {config.INPUT_TABLE}
                WHERE scenic_name = '{config.TARGET_SCENIC_NAME}'
                ORDER BY date ASC
            """

            self.log("正在从数据库读取数据...")
            self.raw_data = pd.read_sql(query, self.engine)

            # 检查数据是否为空
            if len(self.raw_data) == 0:
                error_msg = f"错误：景区 '{config.TARGET_SCENIC_NAME}' 在表 '{config.INPUT_TABLE}' 中无数据！"
                self.log(error_msg, level='ERROR')
                self.log("请检查以下内容：", level='ERROR')
                self.log("  1. 景区名称是否正确", level='ERROR')
                self.log("  2. 数据是否已通过 m7_data_fusion.py 写入数据库", level='ERROR')
                sys.exit(1)

            # 确保日期字段为 datetime 类型
            if 'date' in self.raw_data.columns:
                self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])

            self.log(f"✅ 数据加载成功！")
            self.log(f"  样本总数：{len(self.raw_data)}")
            self.log(f"  时间范围：{self.raw_data['date'].min()} 至 {self.raw_data['date'].max()}")
            self.log(f"  列数：{len(self.raw_data.columns)}")

            # 数据类型检查和转换
            self.log("\n正在检查和修复数据类型...")
            self._fix_data_types()

            if config.SHOW_DATA_STATS:
                self.log(f"\n  目标列 '{config.TARGET_COL}' 统计：")
                try:
                    self.log(f"    均值：{self.raw_data[config.TARGET_COL].mean():.2f}")
                    self.log(f"    最小值：{self.raw_data[config.TARGET_COL].min():.2f}")
                    self.log(f"    最大值：{self.raw_data[config.TARGET_COL].max():.2f}")
                except Exception as e:
                    self.log(f"    警告：无法计算统计信息 - {e}", level='WARNING')

        except Exception as e:
            self.log(f"数据加载失败：{e}", level='ERROR')
            raise

    def _fix_data_types(self):
        """
        修复数据类型问题
        将应该是数值型的列强制转换为数值类型
        """
        # 获取所有列名
        all_cols = self.raw_data.columns.tolist()

        # 需要排除的列（应保持原样的列）
        exclude_from_conversion = config.EXCLUDE_COLS + config.CATEGORICAL_COLS

        # 找出需要转换为数值的列
        numeric_cols = [col for col in all_cols if col not in exclude_from_conversion]

        self.log(f"  需要检查的数值列数量：{len(numeric_cols)}")

        conversion_count = 0
        error_cols = []

        for col in numeric_cols:
            # 跳过已经是数值类型的列
            if pd.api.types.is_numeric_dtype(self.raw_data[col]):
                continue

            # 检查列的数据类型
            col_dtype = self.raw_data[col].dtype

            # 如果是 object 或 string 类型，尝试转换
            if col_dtype == 'object' or pd.api.types.is_string_dtype(self.raw_data[col]):
                try:
                    # 先查看是否有异常值
                    sample_value = self.raw_data[col].iloc[0]

                    # 检测是否是超长字符串（可能是错误数据）
                    if isinstance(sample_value, str) and len(sample_value) > 50:
                        self.log(f"    警告：列 '{col}' 包含异常长字符串，尝试清理...", level='WARNING')
                        self.log(f"      示例值长度：{len(sample_value)} 字符", level='WARNING')

                        # 尝试清理：如果是拼接错误，尝试提取有效数字
                        # 这里使用 errors='coerce' 将无法转换的值设为 NaN
                        self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')

                        # 检查有多少 NaN
                        nan_count = self.raw_data[col].isna().sum()
                        if nan_count > 0:
                            self.log(f"      ⚠️  转换后产生 {nan_count} 个 NaN 值", level='WARNING')

                            # 如果 NaN 比例过高，可能需要删除该列
                            nan_ratio = nan_count / len(self.raw_data)
                            if nan_ratio > 0.5:
                                self.log(f"      ⚠️  NaN 占比 {nan_ratio:.1%}，建议检查源数据", level='WARNING')
                                error_cols.append(col)
                            else:
                                # 填充 NaN 值（使用中位数）
                                median_val = self.raw_data[col].median()
                                if pd.notna(median_val):
                                    self.raw_data[col].fillna(median_val, inplace=True)
                                    self.log(f"      已用中位数 {median_val:.2f} 填充 NaN", level='INFO')
                    else:
                        # 正常转换
                        self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')

                        # 填充可能产生的 NaN
                        if self.raw_data[col].isna().any():
                            median_val = self.raw_data[col].median()
                            if pd.notna(median_val):
                                self.raw_data[col].fillna(median_val, inplace=True)

                    conversion_count += 1
                    self.log(f"    ✓ 列 '{col}' 已转换为数值类型")

                except Exception as e:
                    self.log(f"    ✗ 列 '{col}' 转换失败：{e}", level='ERROR')
                    error_cols.append(col)

        self.log(f"\n  数据类型修复完成：")
        self.log(f"    成功转换：{conversion_count} 列")

        if error_cols:
            self.log(f"    转换失败：{len(error_cols)} 列", level='WARNING')
            self.log(f"    失败列名：{error_cols}", level='WARNING')

            # 询问是否继续（可选：自动删除问题列）
            self.log(f"\n  建议：检查以下列的源数据质量", level='WARNING')
            for col in error_cols[:5]:  # 只显示前5个
                sample_vals = self.raw_data[col].head(3).tolist()
                self.log(f"    {col}: {sample_vals}", level='WARNING')

    def preprocess_features(self):
        """
        Step 2: 特征预处理
        对类别型特征进行编码，并清理不需要的列
        """
        self.log("\n" + "="*60)
        self.log("Step 2: 特征预处理")
        self.log("="*60)

        df = self.raw_data.copy()

        # 2.1 对类别型特征进行 LabelEncoding
        self.log("正在对类别型特征进行 LabelEncoding...")
        encoders = {}

        for col in config.CATEGORICAL_COLS:
            if col in df.columns:
                self.log(f"  编码字段：{col}")
                encoder = LabelEncoder()

                # 处理缺失值
                df[col] = df[col].fillna('Unknown')

                # 执行编码
                df[col] = encoder.fit_transform(df[col].astype(str))
                encoders[col] = encoder

                self.log(f"    原始类别数：{len(encoder.classes_)}")
            else:
                self.log(f"  警告：字段 {col} 不存在，跳过", level='WARNING')

        # 2.2 删除排除列
        self.log("\n正在清理排除列...")
        cols_to_drop = [col for col in config.EXCLUDE_COLS if col in df.columns]

        if cols_to_drop:
            self.log(f"  删除列：{cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # 2.3 确保目标列存在
        if config.TARGET_COL not in df.columns:
            error_msg = f"错误：目标列 '{config.TARGET_COL}' 不存在！"
            self.log(error_msg, level='ERROR')
            sys.exit(1)

        # 2.4 最终数据类型检查
        self.log("\n正在进行最终数据类型检查...")
        non_numeric_cols = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
                self.log(f"  警告：列 '{col}' 仍不是数值类型 (dtype={df[col].dtype})", level='WARNING')

        if non_numeric_cols:
            self.log(f"\n  尝试强制转换剩余非数值列...", level='WARNING')
            for col in non_numeric_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # 填充 NaN
                    if df[col].isna().any():
                        median_val = df[col].median()
                        if pd.notna(median_val):
                            df[col].fillna(median_val, inplace=True)
                        else:
                            df[col].fillna(0, inplace=True)
                    self.log(f"    ✓ 列 '{col}' 强制转换成功")
                except Exception as e:
                    self.log(f"    ✗ 列 '{col}' 无法转换，将被删除：{e}", level='ERROR')
                    df = df.drop(columns=[col])

        # 2.5 保存特征列名（用于后续特征映射）
        self.feature_columns = df.columns.tolist()

        self.log(f"\n✅ 特征预处理完成！")
        self.log(f"  最终特征数：{len(self.feature_columns)}")

        if config.SHOW_FEATURE_LIST:
            self.log(f"  特征列表：{self.feature_columns}")

        # 检查是否有无穷大或 NaN 值
        if df.isnull().any().any():
            self.log(f"  警告：数据中仍存在 NaN 值，正在填充...", level='WARNING')
            df = df.fillna(df.median())

        if np.isinf(df.select_dtypes(include=[np.number]).values).any():
            self.log(f"  警告：数据中存在无穷大值，正在替换...", level='WARNING')
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(df.median())

        # 转换为 numpy 数组
        self.processed_data = df.values.astype(np.float32)

        self.log(f"  处理后数据形状：{self.processed_data.shape}")
        self.log(f"  数据类型：{self.processed_data.dtype}")

        return df

    def normalize_data(self):
        """
        Step 3: 严谨的归一化（防止数据泄露）
        仅使用训练集数据 fit scaler，然后 transform 全量数据
        """
        self.log("\n" + "="*60)
        self.log("Step 3: 数据归一化（防止数据泄露）")
        self.log("="*60)

        # 计算训练集分界索引
        train_size = int(len(self.processed_data) * config.TRAIN_RATIO)

        self.log(f"训练集样本数：{train_size}")
        self.log(f"全量数据样本数：{len(self.processed_data)}")

        # 初始化 MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=config.SCALER_FEATURE_RANGE)

        # 关键操作：仅使用训练集数据进行 fit
        self.log("正在使用训练集数据 fit scaler...")
        train_data = self.processed_data[:train_size]
        self.scaler.fit(train_data)

        # 对全量数据进行 transform
        self.log("正在对全量数据进行 transform...")
        self.processed_data = self.scaler.transform(self.processed_data)

        # 保存 scaler 对象
        scaler_filename = config.get_scaler_filename()
        self.log(f"正在保存 scaler 对象：{scaler_filename}")

        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)

        self.log(f"✅ 归一化完成！")
        self.log(f"  归一化范围：{config.SCALER_FEATURE_RANGE}")
        self.log(f"  Scaler 文件：{scaler_filename}")

        if config.SHOW_DATA_STATS:
            self.log(f"  归一化后数据统计：")
            self.log(f"    最小值：{self.processed_data.min():.4f}")
            self.log(f"    最大值：{self.processed_data.max():.4f}")
            self.log(f"    均值：{self.processed_data.mean():.4f}")

    def build_feature_map(self):
        """
        Step 4: 生成特征映射表
        区分情感特征、环境特征和目标特征的索引
        """
        self.log("\n" + "="*60)
        self.log("Step 4: 生成特征映射表")
        self.log("="*60)

        sentiment_idxs = []
        context_idxs = []
        target_idx = None

        for idx, col in enumerate(self.feature_columns):
            # 查找目标列索引
            if col == config.TARGET_COL:
                target_idx = idx
                self.log(f"  目标特征索引 [{idx}]: {col}")

            # 查找情感特征索引（列名包含 'score'）
            elif config.SENTIMENT_FEATURE_KEYWORD in col.lower():
                sentiment_idxs.append(idx)

            # 其他特征归类为环境特征
            else:
                context_idxs.append(idx)

        # 构建特征映射字典
        self.feature_map = {
            'sentiment_idxs': sentiment_idxs,
            'context_idxs': context_idxs,
            'target_idx': target_idx,
            'feature_names': self.feature_columns,
            'n_features': len(self.feature_columns)
        }

        # 保存为 JSON 文件
        feature_map_filename = config.get_feature_map_filename()
        self.log(f"\n正在保存特征映射表：{feature_map_filename}")

        with open(feature_map_filename, 'w', encoding='utf-8') as f:
            json.dump(self.feature_map, f, ensure_ascii=False, indent=2)

        self.log(f"✅ 特征映射表生成完成！")
        self.log(f"  情感特征数量：{len(sentiment_idxs)}")
        self.log(f"  环境特征数量：{len(context_idxs)}")
        self.log(f"  目标特征索引：{target_idx}")

        if config.SHOW_FEATURE_LIST:
            self.log(f"\n  情感特征列表：")
            for idx in sentiment_idxs[:10]:  # 只显示前10个
                self.log(f"    [{idx}] {self.feature_columns[idx]}")
            if len(sentiment_idxs) > 10:
                self.log(f"    ... 还有 {len(sentiment_idxs)-10} 个情感特征")

    def create_sliding_windows(self, data):
        """
        Step 5: 滑动窗口切片
        生成 (Samples, SEQ_LEN, Features) 的输入和 (Samples, PRED_LEN, 1) 的标签

        参数:
            data (np.ndarray): 归一化后的数据 (N, Features)

        返回:
            tuple: (X, Y) 其中 X.shape=(Samples, SEQ_LEN, Features), Y.shape=(Samples, PRED_LEN, 1)
        """
        self.log("\n" + "="*60)
        self.log("Step 5: 滑动窗口切片")
        self.log("="*60)

        seq_len = config.SEQ_LEN
        pred_len = config.PRED_LEN
        target_idx = self.feature_map['target_idx']

        self.log(f"输入序列长度 (SEQ_LEN)：{seq_len}")
        self.log(f"预测序列长度 (PRED_LEN)：{pred_len}")
        self.log(f"目标列索引：{target_idx}")

        X, Y = [], []

        # 滑动窗口切片
        for i in range(len(data) - seq_len - pred_len + 1):
            # 输入：过去 seq_len 天的所有特征
            x_window = data[i:i+seq_len, :]  # (SEQ_LEN, Features)

            # 标签：未来 pred_len 天的目标列
            y_window = data[i+seq_len:i+seq_len+pred_len, target_idx]  # (PRED_LEN,)

            X.append(x_window)
            Y.append(y_window)

        # 转换为 numpy 数组
        X = np.array(X, dtype=np.float32)  # (Samples, SEQ_LEN, Features)
        Y = np.array(Y, dtype=np.float32)  # (Samples, PRED_LEN)

        # 扩展 Y 的维度为 (Samples, PRED_LEN, 1)
        Y = np.expand_dims(Y, axis=-1)

        self.log(f"\n✅ 滑动窗口切片完成！")
        self.log(f"  输入张量 X.shape：{X.shape}")
        self.log(f"  标签张量 Y.shape：{Y.shape}")
        self.log(f"  总样本数：{len(X)}")

        return X, Y

    def split_and_save_datasets(self, X, Y):
        """
        Step 6: 数据集切分与保存
        按时间顺序（不打乱）切分为训练集、验证集、测试集并保存为 .npy 文件

        参数:
            X (np.ndarray): 输入张量 (Samples, SEQ_LEN, Features)
            Y (np.ndarray): 标签张量 (Samples, PRED_LEN, 1)
        """
        self.log("\n" + "="*60)
        self.log("Step 6: 数据集切分与保存")
        self.log("="*60)

        n_samples = len(X)

        # 计算切分点
        train_end = int(n_samples * config.TRAIN_RATIO)
        val_end = int(n_samples * (config.TRAIN_RATIO + config.VAL_RATIO))

        self.log(f"数据集切分方案：")
        self.log(f"  训练集：样本 0 ~ {train_end-1} (共 {train_end} 个)")
        self.log(f"  验证集：样本 {train_end} ~ {val_end-1} (共 {val_end-train_end} 个)")
        self.log(f"  测试集：样本 {val_end} ~ {n_samples-1} (共 {n_samples-val_end} 个)")

        # 切分数据
        train_x, train_y = X[:train_end], Y[:train_end]
        val_x, val_y = X[train_end:val_end], Y[train_end:val_end]
        test_x, test_y = X[val_end:], Y[val_end:]

        # 获取文件名
        filenames = config.get_dataset_filenames()

        # 保存数据集
        self.log(f"\n正在保存数据集文件...")

        datasets = {
            'train_x': train_x,
            'train_y': train_y,
            'val_x': val_x,
            'val_y': val_y,
            'test_x': test_x,
            'test_y': test_y
        }

        for key, data in datasets.items():
            filename = filenames[key]
            np.save(filename, data)
            self.log(f"  ✅ {filename} - Shape: {data.shape}")

        self.log(f"\n✅ 数据集保存完成！")

        # 输出汇总信息
        self.log("\n" + "="*60)
        self.log("数据集构建汇总")
        self.log("="*60)
        self.log(f"景区名称：{config.TARGET_SCENIC_NAME}")
        self.log(f"特征维度：{X.shape[-1]}")
        self.log(f"输入序列长度：{config.SEQ_LEN}")
        self.log(f"预测序列长度：{config.PRED_LEN}")
        self.log(f"\n数据集规模：")
        self.log(f"  训练集：X={train_x.shape}, Y={train_y.shape}")
        self.log(f"  验证集：X={val_x.shape}, Y={val_y.shape}")
        self.log(f"  测试集：X={test_x.shape}, Y={test_y.shape}")
        self.log("="*60 + "\n")

    def run(self):
        """
        主流程：执行完整的数据集构建任务
        """
        try:
            # 连接数据库
            self.connect_database()

            # Step 1: 加载数据
            self.load_data()

            # Step 2: 特征预处理
            self.preprocess_features()

            # Step 3: 数据归一化
            self.normalize_data()

            # Step 4: 构建特征映射表
            self.build_feature_map()

            # Step 5: 滑动窗口切片
            X, Y = self.create_sliding_windows(self.processed_data)

            # Step 6: 切分并保存数据集
            self.split_and_save_datasets(X, Y)

            self.log("="*60)
            self.log("🎉 数据集构建任务全部完成！")
            self.log("="*60 + "\n")

        except Exception as e:
            self.log(f"\n任务执行失败：{e}", level='ERROR')
            import traceback
            traceback.print_exc()
            sys.exit(1)

        finally:
            if self.engine:
                self.engine.dispose()
                self.log("数据库连接已关闭")


def main():
    """
    主函数入口
    """
    print("\n" + "="*60)
    print("🚀 时序数据集构建任务启动")
    print(f"目标景区：{config.TARGET_SCENIC_NAME}")
    print(f"输入表名：{config.INPUT_TABLE}")
    print(f"序列配置：SEQ_LEN={config.SEQ_LEN}, PRED_LEN={config.PRED_LEN}")
    print("="*60 + "\n")

    # 创建数据集构建器并运行
    builder = TimeSeriesDatasetBuilder()
    builder.run()


if __name__ == "__main__":
    main()