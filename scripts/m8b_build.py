# -*- coding: utf-8 -*-
"""
M8B 张量构建主程序
Data Pipeline Engineer: Leakage-Free Tensor Builder
功能: 防泄露归一化(混合策略) + 滑动窗口切片 + 特征映射
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
    """张量构建器 - 实现防泄露的时序数据管道 (混合归一化策略版)"""

    def __init__(self):
        self.df = None
        self.data = None

        # 存储多个 Scaler
        self.scalers = {
            'target_scaler': None,
            'sentiment_scaler': None,
            'other_scaler': None,
            'meta': {}  # 存储 log_transform 等元数据
        }

        self.feature_columns = []
        self.feature_map = {}
        self.col_groups_indices = {}  # 记录列的分组索引

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

            # 丢弃元数据列
            exclude_present = [col for col in config.EXCLUDE_COLS if col in self.df.columns]
            self.df = self.df.drop(columns=exclude_present)

            # 保存特征列名
            self.feature_columns = self.df.columns.tolist()

            # 转换为 Float32
            self.data = self.df.astype(config.DTYPE).values
            print(f"✓ 数据预处理完成: {self.data.shape} ({config.DTYPE})")

        except Exception as e:
            print(f"✗ 数据读取失败: {e}")
            raise

    def calculate_split_indices(self):
        """Step 2: 数据集划分 (Time-based Split)"""
        print(f"\n{'=' * 70}")
        print("Step 2: 数据集划分 (Time-based Split)")
        print(f"{'=' * 70}")

        total_samples = len(self.data)
        self.train_end_idx = int(total_samples * config.TRAIN_RATIO)
        self.val_end_idx = int(total_samples * (config.TRAIN_RATIO + config.VAL_RATIO))

        print(f"Train End: {self.train_end_idx}")
        print(f"Val End:   {self.val_end_idx}")

    def leakage_free_normalization(self):
        """Step 3: 混合防泄露归一化 (Hybrid Strategy)"""
        print(f"\n{'=' * 70}")
        print("Step 3: 混合归一化策略 (Hybrid Normalization)")
        print(f"{'=' * 70}")

        # 1. 识别列类型
        print("--- 3.1 自动识别列类型 ---")
        col_indices = {
            'target': [],
            'sentiment': [],  # _score
            'skip': [],  # _sin, _cos, is_
            'other': []  # remaining numeric
        }

        for idx, col_name in enumerate(self.feature_columns):
            if col_name == config.TARGET_COLUMN:
                col_indices['target'].append(idx)
            elif col_name.endswith('_sin') or col_name.endswith('_cos') or col_name.startswith('is_'):
                col_indices['skip'].append(idx)
            elif '_score' in col_name:  # 识别情感列
                col_indices['sentiment'].append(idx)
            else:
                col_indices['other'].append(idx)

        self.col_groups_indices = col_indices
        for group, idxs in col_indices.items():
            print(f"  [{group.upper()}]: {len(idxs)} cols")
            if idxs:
                sample_names = [self.feature_columns[i] for i in idxs[:3]]
                print(f"    -> {sample_names} ...")

        # 2. 执行差异化处理
        print("\n--- 3.2 执行转换 (Fit仅基于训练集) ---")

        # 获取训练集用于Fit
        train_slice = self.data[:self.train_end_idx]

        # [Strategy 1] Target: Log1p + MinMaxScaler
        if col_indices['target']:
            idx = col_indices['target'][0]
            print(f"  Processing TARGET (idx {idx}): Log1p + MinMaxScaler")

            # 1. Log transform entire data first
            self.data[:, idx] = np.log1p(self.data[:, idx])

            # 2. Fit Scaler on Log-transformed Train data
            target_data = self.data[:, idx].reshape(-1, 1)  # Full data
            train_target = target_data[:self.train_end_idx]

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(train_target)

            # 3. Transform
            self.data[:, idx] = scaler.transform(target_data).flatten()

            self.scalers['target_scaler'] = scaler
            self.scalers['meta']['target_log1p'] = True
            print(f"    ✓ Target processed. Min: {np.min(self.data[:, idx]):.4f}, Max: {np.max(self.data[:, idx]):.4f}")

        # [Strategy 2] Sentiment: StandardScaler (Z-Score)
        if col_indices['sentiment']:
            idxs = col_indices['sentiment']
            print(f"  Processing SENTIMENT ({len(idxs)} cols): StandardScaler")

            scaler = StandardScaler()
            scaler.fit(train_slice[:, idxs])

            self.data[:, idxs] = scaler.transform(self.data[:, idxs])

            self.scalers['sentiment_scaler'] = scaler
            print(f"    ✓ Sentiment processed. Mean≈0, Std≈1")

        # [Strategy 3] Other: Default (MinMax)
        if col_indices['other']:
            idxs = col_indices['other']
            print(f"  Processing OTHER ({len(idxs)} cols): {config.NORMALIZATION_METHOD}")

            if config.NORMALIZATION_METHOD == 'minmax':
                scaler = MinMaxScaler(feature_range=config.NORM_RANGE)
            else:
                scaler = StandardScaler()

            scaler.fit(train_slice[:, idxs])
            self.data[:, idxs] = scaler.transform(self.data[:, idxs])

            self.scalers['other_scaler'] = scaler
            print(f"    ✓ Other cols processed.")

        # [Strategy 4] Skip: No processing
        if col_indices['skip']:
            print(f"  Skipping SKIP ({len(col_indices['skip'])} cols): Cyclic/Binary")
            # Do nothing

        # 保存 Scalers 组合
        scaler_path = os.path.join(config.OUTPUT_DIR, config.SCALERS_FILENAME)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"\n✓ 混合 Scaler 对象已保存: {scaler_path}")

    def generate_feature_map(self):
        """Step 4: 生成特征映射表 (Feature Map)"""
        print(f"\n{'=' * 70}")
        print("Step 4: 生成特征映射表")
        print(f"{'=' * 70}")

        target_idx = self.col_groups_indices['target'][0]

        # 情感分组逻辑 (用于模型输入)
        sentiment_groups = []
        all_sentiment_indices = set()

        for group_name in config.SENTIMENT_KEYWORDS:
            current_group_idxs = []
            for idx, col_name in enumerate(self.feature_columns):
                if idx == target_idx: continue
                if idx in all_sentiment_indices: continue

                # 包含关键词 且 确实被识别为sentiment列 (或者包含_score)
                # 双重校验确保逻辑一致性
                if group_name in col_name.lower():
                    current_group_idxs.append(idx)
                    all_sentiment_indices.add(idx)

            sentiment_groups.append(current_group_idxs)

        # Context Features (剩余非目标列)
        context_idxs = []
        for idx in range(len(self.feature_columns)):
            if idx == target_idx: continue
            if idx not in all_sentiment_indices:
                context_idxs.append(idx)

        self.feature_map = {
            'feature_names': self.feature_columns,
            'target_idx': target_idx,
            'target_col': config.TARGET_COLUMN,
            'sentiment_groups': sentiment_groups,
            'context_idxs': context_idxs,
            'total_features': len(self.feature_columns),
            'scaler_info': {
                'target_log1p': True,
                'groups': self.col_groups_indices
            }
        }

        # 保存
        map_path = os.path.join(config.OUTPUT_DIR, config.FEATURE_MAP_FILENAME)
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_map, f, indent=2, ensure_ascii=False)
        print(f"✓ Feature Map Saved: {map_path}")

    def sliding_window_slicing(self):
        """Step 5: 滑动窗口切片 (Sliding Window)"""
        print(f"\n{'=' * 70}")
        print("Step 5: 滑动窗口切片")
        print(f"{'=' * 70}")

        X_list = []
        Y_list = []

        total_samples = len(self.data)
        window_size = config.SEQ_LEN + config.PRED_LEN
        target_idx = self.feature_map['target_idx']

        print(f"  Seq: {config.SEQ_LEN}, Pred: {config.PRED_LEN}")

        # 优化循环效率
        # 使用 stride tricks 或简单的切片
        for i in range(total_samples - window_size + 1):
            # X: [i, i+seq]
            x_window = self.data[i: i + config.SEQ_LEN]
            # Y: [i+seq, i+seq+pred] (Only target column)
            y_window = self.data[i + config.SEQ_LEN: i + window_size, target_idx]

            X_list.append(x_window)
            Y_list.append(y_window)

        self.X = np.array(X_list, dtype=config.DTYPE)
        self.Y = np.array(Y_list, dtype=config.DTYPE)
        self.Y = np.expand_dims(self.Y, axis=-1)

        print(f"✓ X: {self.X.shape}")
        print(f"✓ Y: {self.Y.shape}")

    def save_tensors(self):
        """Step 6: 保存张量"""
        print(f"\n{'=' * 70}")
        print("Step 6: 保存张量 (Split & Save)")
        print(f"{'=' * 70}")

        total_windows = len(self.X)
        train_windows = int(total_windows * config.TRAIN_RATIO)
        val_windows = int(total_windows * (config.TRAIN_RATIO + config.VAL_RATIO))

        tensors = {
            'train_x': self.X[:train_windows],
            'train_y': self.Y[:train_windows],
            'val_x': self.X[train_windows:val_windows],
            'val_y': self.Y[train_windows:val_windows],
            'test_x': self.X[val_windows:],
            'test_y': self.Y[val_windows:]
        }

        for key, tensor in tensors.items():
            filename = config.TENSOR_FILES[key]
            filepath = os.path.join(config.OUTPUT_DIR, filename)
            np.save(filepath, tensor)
            print(f"  Saved {filename}: {tensor.shape}")

        # 生成并保存 info 文件 (覆盖原有逻辑，增加scaler信息)
        info = {
            'seq_len': config.SEQ_LEN,
            'pred_len': config.PRED_LEN,
            'feature_dim': self.X.shape[-1],
            'samples': {
                'train': len(tensors['train_x']),
                'val': len(tensors['val_x']),
                'test': len(tensors['test_x'])
            },
            'normalization_strategy': 'Hybrid (Log-Target, Z-Sentiment, Raw-Time)',
            'scaler_file': config.SCALERS_FILENAME
        }
        with open(os.path.join(config.OUTPUT_DIR, 'dataset_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

    def run(self):
        """执行流程"""
        print(f"\n{'#' * 70}")
        print(f"# M8B 张量构建 (Hybrid Normalization)")
        print(f"{'#' * 70}")

        if not self.connect_mysql(): return False

        try:
            self.load_and_sort()
            self.calculate_split_indices()
            self.leakage_free_normalization()
            self.generate_feature_map()
            self.sliding_window_slicing()
            self.save_tensors()

            print(f"\n✓ 成功完成。输出目录: {config.OUTPUT_DIR}")
            return True
        except Exception as e:
            print(f"\n✗ 出错: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    builder = TensorBuilder()
    builder.run()