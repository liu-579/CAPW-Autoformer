# -*- coding: utf-8 -*-
"""
M8B 张量构建主程序
Data Pipeline Engineer: Leakage-Free Tensor Builder
功能: 防泄露归一化 + 滑动窗口切片 + 特征映射
"""

import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import pickle
import os
import warnings

# 导入配置文件
import config.m8b_config as config

warnings.filterwarnings('ignore')


class TensorBuilder:
    """张量构建器 - 实现防泄露的时序数据管道"""

    def __init__(self):
        self.df = None
        self.data = None
        self.scaler = None
        self.feature_columns = []
        self.feature_map = {}

        # 数据集划分索引
        self.train_end_idx = None
        self.val_end_idx = None

        # 张量
        self.X = None
        self.Y = None

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def connect_mysql(self):
        """建立 MySQL 连接"""
        print(f"\n{'=' * 70}")
        print("Step 0: 建立数据库连接")
        print(f"{'=' * 70}")

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

    def load_and_sort(self):
        """Step 1: 读取与排序"""
        print(f"\n{'=' * 70}")
        print("Step 1: 读取与排序 (时序数据的生命线)")
        print(f"{'=' * 70}")

        try:
            # 读取数据
            query = f"SELECT * FROM {config.INPUT_TABLE}"
            self.df = pd.read_sql(query, self.engine)
            print(f"✓ 读取数据成功: {len(self.df)} 行 × {len(self.df.columns)} 列")

            # 检查 date 列
            if 'date' not in self.df.columns:
                raise ValueError("数据表中缺少 'date' 列，无法进行时序排序")

            # 按 date 升序排序
            print(f"\n--- 时序排序 (按 date 升序) ---")
            print(f"排序前日期范围: {self.df['date'].min()} ~ {self.df['date'].max()}")
            self.df = self.df.sort_values('date').reset_index(drop=True)
            print(f"排序后日期范围: {self.df['date'].iloc[0]} ~ {self.df['date'].iloc[-1]}")
            print(f"✓ 数据已按时间顺序排列")

            # 丢弃元数据列
            print(f"\n--- 丢弃元数据列 ---")
            exclude_present = [col for col in config.EXCLUDE_COLS if col in self.df.columns]
            print(f"排除列: {exclude_present}")

            self.df = self.df.drop(columns=exclude_present)
            print(f"✓ 剩余列数: {len(self.df.columns)}")

            # 保存特征列名
            self.feature_columns = self.df.columns.tolist()
            print(f"\n特征列名:")
            for i, col in enumerate(self.feature_columns, 1):
                print(f"  {i}. {col}")

            # 转换为 Float32
            print(f"\n--- 转换为 {config.DTYPE.upper()} ---")
            original_dtypes = self.df.dtypes.value_counts()
            print(f"转换前类型分布:\n{original_dtypes}")

            self.data = self.df.astype(config.DTYPE).values
            print(f"✓ 转换完成")
            print(f"数据形状: {self.data.shape}")
            print(f"数据类型: {self.data.dtype}")

        except Exception as e:
            print(f"✗ 数据读取失败: {e}")
            raise

    def calculate_split_indices(self):
        """Step 2: 数据集划分 (Time-based Split)"""
        print(f"\n{'=' * 70}")
        print("Step 2: 数据集划分 (Time-based Split)")
        print(f"{'=' * 70}")

        total_samples = len(self.data)
        print(f"总样本数: {total_samples}")
        print(f"划分比例: Train={config.TRAIN_RATIO:.1%}, Val={config.VAL_RATIO:.1%}, Test={config.TEST_RATIO:.1%}")

        # 计算划分索引
        self.train_end_idx = int(total_samples * config.TRAIN_RATIO)
        self.val_end_idx = int(total_samples * (config.TRAIN_RATIO + config.VAL_RATIO))

        print(f"\n划分索引点:")
        print(f"  训练集: [0, {self.train_end_idx}) = {self.train_end_idx} 样本")
        print(f"  验证集: [{self.train_end_idx}, {self.val_end_idx}) = {self.val_end_idx - self.train_end_idx} 样本")
        print(f"  测试集: [{self.val_end_idx}, {total_samples}) = {total_samples - self.val_end_idx} 样本")

        # 显示日期范围（如果还保留在 df 中）
        if 'date' in self.df.columns:
            print(f"\n时间范围:")
            print(f"  训练集: {self.df['date'].iloc[0]} ~ {self.df['date'].iloc[self.train_end_idx - 1]}")
            print(
                f"  验证集: {self.df['date'].iloc[self.train_end_idx]} ~ {self.df['date'].iloc[self.val_end_idx - 1]}")
            print(f"  测试集: {self.df['date'].iloc[self.val_end_idx]} ~ {self.df['date'].iloc[-1]}")

        print(f"\n⚠ 注意: 此时暂不切分数据，仅记录索引点")

    def leakage_free_normalization(self):
        """Step 3: 防泄露归一化 (Leakage-Free Normalization)"""
        print(f"\n{'=' * 70}")
        print("Step 3: 防泄露归一化 (仅使用训练集统计量)")
        print(f"{'=' * 70}")

        # 初始化 Scaler
        if config.NORMALIZATION_METHOD == 'minmax':
            self.scaler = MinMaxScaler(feature_range=config.NORM_RANGE)
            print(f"归一化方法: MinMaxScaler{config.NORM_RANGE}")
        elif config.NORMALIZATION_METHOD == 'standard':
            self.scaler = StandardScaler()
            print(f"归一化方法: StandardScaler (零均值, 单位方差)")
        else:
            raise ValueError(f"不支持的归一化方法: {config.NORMALIZATION_METHOD}")

        # 关键操作: 仅使用训练集数据拟合
        print(f"\n--- 拟合 Scaler (仅使用训练集) ---")
        train_data = self.data[:self.train_end_idx]
        print(f"训练集形状: {train_data.shape}")

        self.scaler.fit(train_data)
        print(f"✓ Scaler 拟合完成")

        # 显示统计量
        if hasattr(self.scaler, 'data_min_'):
            print(f"\n训练集统计量 (前5个特征):")
            for i in range(min(5, len(self.feature_columns))):
                print(f"  {self.feature_columns[i]}:")
                print(f"    Min: {self.scaler.data_min_[i]:.4f}")
                print(f"    Max: {self.scaler.data_max_[i]:.4f}")
        elif hasattr(self.scaler, 'mean_'):
            print(f"\n训练集统计量 (前5个特征):")
            for i in range(min(5, len(self.feature_columns))):
                print(f"  {self.feature_columns[i]}:")
                print(f"    Mean: {self.scaler.mean_[i]:.4f}")
                print(f"    Std:  {self.scaler.scale_[i]:.4f}")

        # 对全量数据应用转换
        print(f"\n--- 转换全量数据 (应用训练集统计量) ---")
        data_before = self.data.copy()
        self.data = self.scaler.transform(self.data).astype(config.DTYPE)
        print(f"✓ 归一化完成")

        # 验证归一化效果
        print(f"\n归一化前后对比 (第1个样本的前3个特征):")
        for i in range(min(3, len(self.feature_columns))):
            print(f"  {self.feature_columns[i]}: {data_before[0, i]:.4f} → {self.data[0, i]:.4f}")

        # 保存 Scaler
        scaler_path = os.path.join(config.OUTPUT_DIR, config.SCALER_FILENAME)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"\n✓ Scaler 已保存: {scaler_path}")

    def generate_feature_map(self):
        """Step 4: 生成特征映射表 (Feature Map) - 支持分组广播"""
        print(f"\n{'=' * 70}")
        print("Step 4: 生成特征映射表 (Grouped Broadcasting Ready)")
        print(f"{'=' * 70}")

        target_idx = None

        # 1. 找到目标列索引
        for idx, col_name in enumerate(self.feature_columns):
            if col_name == config.TARGET_COLUMN:
                target_idx = idx
                break

        if target_idx is None:
            raise ValueError(f"未找到目标列: {config.TARGET_COLUMN}")

        print(f"✓ 目标列: {config.TARGET_COLUMN} (索引 {target_idx})")

        # ==========================================
        # 核心逻辑: 构建分组索引 (List of Lists)
        # ==========================================
        sentiment_groups = []  # 存储分组后的二维列表 [[...], [...]]
        all_sentiment_indices = set()  # 用于快速查找和互斥判断

        print(f"\n正在构建情感特征分组 (优先级顺序: {config.SENTIMENT_KEYWORDS})...")

        # 外层循环：遍历配置的关键词顺序 (确保组的顺序固定)
        for group_name in config.SENTIMENT_KEYWORDS:
            current_group_idxs = []

            # 内层循环：遍历所有特征列
            for idx, col_name in enumerate(self.feature_columns):
                # 跳过目标列
                if idx == target_idx:
                    continue

                # 跳过已经被分配到其他组的特征 (互斥逻辑)
                if idx in all_sentiment_indices:
                    continue

                # 匹配关键词 (不区分大小写)
                if group_name in col_name.lower():
                    current_group_idxs.append(idx)
                    all_sentiment_indices.add(idx)  # 标记已使用

            # 即使该组为空，也要保留一个空列表，以保持索引对齐
            sentiment_groups.append(current_group_idxs)

            print(f"  Group '{group_name}': 找到 {len(current_group_idxs)} 个特征")
            if len(current_group_idxs) > 0:
                example_cols = [self.feature_columns[i] for i in current_group_idxs[:3]]
                print(f"    示例: {example_cols} ...")

        # ==========================================
        # 识别上下文特征 (Context Features)
        # ==========================================
        context_idxs = []

        for idx, col_name in enumerate(self.feature_columns):
            if idx == target_idx:
                continue

            # 如果不是情感特征，则归为上下文特征
            if idx not in all_sentiment_indices:
                context_idxs.append(idx)

        # ==========================================
        # 构建并保存 Map
        # ==========================================
        self.feature_map = {
            'sentiment_groups': sentiment_groups,  # 二维列表
            'context_idxs': context_idxs,  # 一维列表
            'target_idx': target_idx,
            'feature_names': self.feature_columns,
            'total_features': len(self.feature_columns),
            'sentiment_keywords': config.SENTIMENT_KEYWORDS  # 记录元数据供核对
        }

        print(f"\n特征分类统计:")
        print(f"  情感分组数: {len(sentiment_groups)} 组")
        print(f"  情感特征总数: {len(all_sentiment_indices)} 个")
        print(f"  上下文特征数: {len(context_idxs)} 个")

        if context_idxs:
            print(f"    上下文示例: {[self.feature_columns[i] for i in context_idxs[:3]]}")

        # 保存特征映射
        feature_map_path = os.path.join(config.OUTPUT_DIR, config.FEATURE_MAP_FILENAME)
        with open(feature_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_map, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 特征映射已保存: {feature_map_path}")

    def sliding_window_slicing(self):
        """Step 5: 滑动窗口切片 (Sliding Window)"""
        print(f"\n{'=' * 70}")
        print("Step 5: 滑动窗口切片")
        print(f"{'=' * 70}")

        print(f"窗口参数:")
        print(f"  输入序列长度 (SEQ_LEN):  {config.SEQ_LEN}")
        print(f"  预测序列长度 (PRED_LEN): {config.PRED_LEN}")

        X_list = []
        Y_list = []

        total_samples = len(self.data)
        window_size = config.SEQ_LEN + config.PRED_LEN
        target_idx = self.feature_map['target_idx']

        print(f"\n生成滑动窗口:")
        print(f"  数据总长度: {total_samples}")
        print(f"  窗口总长度: {window_size} ({config.SEQ_LEN} + {config.PRED_LEN})")

        # 滑动窗口遍历
        valid_windows = 0
        for i in range(total_samples - window_size + 1):
            # X: 过去 SEQ_LEN 天的所有特征
            x_window = self.data[i: i + config.SEQ_LEN]

            # Y: 未来 PRED_LEN 天的 passenger_count
            y_window = self.data[i + config.SEQ_LEN: i + window_size, target_idx]

            X_list.append(x_window)
            Y_list.append(y_window)
            valid_windows += 1

        # 转换为 numpy 数组
        self.X = np.array(X_list, dtype=config.DTYPE)
        self.Y = np.array(Y_list, dtype=config.DTYPE)

        # Y 需要增加一个维度: (Samples, 7) → (Samples, 7, 1)
        self.Y = np.expand_dims(self.Y, axis=-1)

        print(f"\n✓ 滑动窗口生成完成")
        print(f"  有效窗口数: {valid_windows}")
        print(f"  X 形状: {self.X.shape} (Samples, SeqLen, Features)")
        print(f"  Y 形状: {self.Y.shape} (Samples, PredLen, 1)")

        if config.SHOW_SAMPLE_DATA:
            print(f"\n样本数据预览 (第1个窗口):")
            print(f"  X[0] 形状: {self.X[0].shape}")
            print(f"  X[0, 0, :5] (第1天的前5个特征):")
            print(f"    {self.X[0, 0, :5]}")
            print(f"  Y[0] 形状: {self.Y[0].shape}")
            print(f"  Y[0, :, 0] (未来7天的客流量):")
            print(f"    {self.Y[0, :, 0]}")

    def save_tensors(self):
        """Step 6: 保存张量"""
        print(f"\n{'=' * 70}")
        print("Step 6: 保存张量 (Train/Val/Test Split)")
        print(f"{'=' * 70}")

        total_windows = len(self.X)

        # 重新计算窗口级别的划分索引
        # 注意: 因为滑动窗口会损失 (SEQ_LEN + PRED_LEN - 1) 个样本
        # 所以需要按比例重新计算
        train_windows = int(total_windows * config.TRAIN_RATIO)
        val_windows = int(total_windows * (config.TRAIN_RATIO + config.VAL_RATIO))

        print(f"窗口级别划分:")
        print(f"  总窗口数: {total_windows}")
        print(f"  训练集窗口: [0, {train_windows}) = {train_windows}")
        print(f"  验证集窗口: [{train_windows}, {val_windows}) = {val_windows - train_windows}")
        print(f"  测试集窗口: [{val_windows}, {total_windows}) = {total_windows - val_windows}")

        # 切分数据
        X_train = self.X[:train_windows]
        Y_train = self.Y[:train_windows]

        X_val = self.X[train_windows:val_windows]
        Y_val = self.Y[train_windows:val_windows]

        X_test = self.X[val_windows:]
        Y_test = self.Y[val_windows:]

        # 保存张量
        tensors = {
            'train_x': X_train,
            'train_y': Y_train,
            'val_x': X_val,
            'val_y': Y_val,
            'test_x': X_test,
            'test_y': Y_test
        }

        print(f"\n保存张量文件:")
        for key, tensor in tensors.items():
            filename = config.TENSOR_FILES[key]
            filepath = os.path.join(config.OUTPUT_DIR, filename)
            np.save(filepath, tensor)
            print(f"  ✓ {filename}: {tensor.shape}")

        # 保存数据集信息
        dataset_info = {
            'seq_len': config.SEQ_LEN,
            'pred_len': config.PRED_LEN,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_features': self.X.shape[-1],
            'target_column': config.TARGET_COLUMN,
            'normalization': config.NORMALIZATION_METHOD,
            'dtype': config.DTYPE
        }

        info_path = os.path.join(config.OUTPUT_DIR, 'dataset_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 数据集信息已保存: {info_path}")

    def run(self):
        """执行完整的张量构建流程"""
        print(f"\n{'#' * 70}")
        print(f"# M8B 张量构建流程启动")
        print(f"# 防泄露归一化 + 滑动窗口切片")
        print(f"{'#' * 70}")

        if not self.connect_mysql():
            return False

        try:
            self.load_and_sort()
            self.calculate_split_indices()
            self.leakage_free_normalization()
            self.generate_feature_map()
            self.sliding_window_slicing()
            self.save_tensors()

            print(f"\n{'#' * 70}")
            print(f"# 张量构建流程完成!")
            print(f"# 输出目录: {config.OUTPUT_DIR}")
            print(f"{'#' * 70}\n")

            # 显示输出文件清单
            print("输出文件清单:")
            output_files = os.listdir(config.OUTPUT_DIR)
            for filename in sorted(output_files):
                filepath = os.path.join(config.OUTPUT_DIR, filename)
                if os.path.isfile(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"  - {filename} ({size_mb:.2f} MB)")

            return True

        except Exception as e:
            print(f"\n✗ 张量构建流程出错: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    builder = TensorBuilder()
    builder.run()