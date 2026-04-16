# -*- coding: utf-8 -*-
"""
M8B 张量构建主程序 V2 (小时级数据粒度)
Data Pipeline Engineer: Leakage-Free Tensor Builder (Hourly) V2
功能: 防泄露归一化(混合策略) + 滑动窗口切片 + 特征映射
改进:
  1. 类别特征使用 One-Hot 编码（解决 weather_cond_int, wind_dir_int 被错误缩放的问题）
  2. 移除冗余滞后特征（滑动窗口已提供历史信息）
  3. 添加缺失值检查与处理
"""

import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import json
import pickle
import os
import warnings
import re

# 导入配置文件（小时级 V2）
import config.m8b_config_h_2 as config

warnings.filterwarnings('ignore')


class TensorBuilderV2:
    """张量构建器 V2 - 实现防泄露的时序数据管道 (混合归一化策略版 - 小时级)
    
    改进点:
    - 类别特征 One-Hot 编码
    - 冗余滞后特征移除
    - 缺失值处理
    """

    def __init__(self):
        self.df = None
        self.data = None

        # 存储多个 Scaler
        self.scalers = {
            'target_scaler': None,
            'sentiment_scaler': None,
            'other_scaler': None,
            'onehot_encoder': None,  # [新增] One-Hot 编码器
            'meta': {}  # 存储 log_transform 等元数据
        }

        self.feature_columns = []
        self.feature_map = {}
        self.col_groups_indices = {}  # 记录列的分组索引

        # [新增] 记录移除的列
        self.removed_lag_cols = []
        self.categorical_cols_found = []

        # 数据集划分索引
        self.train_end_idx = None
        self.val_end_idx = None

        # 张量
        self.X = None
        self.Y = None

        # [新增] 保存原始 datetime 列用于推理时输出
        self.datetime_series = None

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
        """Step 1: 读取与排序 (小时级数据)"""
        print(f"\n{'=' * 70}")
        print("Step 1: 读取与排序 (时序数据的生命线 - 小时级)")
        print(f"{'=' * 70}")

        try:
            # 读取数据
            query = f"SELECT * FROM {config.INPUT_TABLE}"
            self.df = pd.read_sql(query, self.engine)
            print(f"✓ 读取数据成功: {len(self.df)} 行 × {len(self.df.columns)} 列")

            # 检查 datetime 列
            if 'datetime' not in self.df.columns:
                raise ValueError("数据表中缺少 'datetime' 列，无法进行时序排序")

            # 按 datetime 升序排序
            print(f"\n--- 时序排序 (按 datetime 升序) ---")
            print(f"排序前时间范围: {self.df['datetime'].min()} ~ {self.df['datetime'].max()}")
            self.df = self.df.sort_values('datetime').reset_index(drop=True)
            print(f"排序后时间范围: {self.df['datetime'].iloc[0]} ~ {self.df['datetime'].iloc[-1]}")

            # [新增] 按时间范围筛选（可选，默认不筛选）
            date_start = getattr(config, 'DATE_RANGE_START', None)
            date_end = getattr(config, 'DATE_RANGE_END', None)
            if date_start is not None or date_end is not None:
                print(f"\n--- [新增] 时间范围筛选 ---")
                # 确保 datetime 列为 datetime 类型
                self.df['datetime'] = pd.to_datetime(self.df['datetime'])
                before_count = len(self.df)
                if date_start is not None:
                    start_dt = pd.to_datetime(date_start)
                    self.df = self.df[self.df['datetime'] >= start_dt]
                    print(f"  起始日期筛选: >= {date_start}")
                if date_end is not None:
                    # 结束日期包含当天整天，所以加1天
                    end_dt = pd.to_datetime(date_end) + pd.Timedelta(days=1)
                    self.df = self.df[self.df['datetime'] < end_dt]
                    print(f"  结束日期筛选: <= {date_end} (含当天)")
                self.df = self.df.reset_index(drop=True)
                after_count = len(self.df)
                print(f"  筛选结果: {before_count} → {after_count} 行")
                print(f"  筛选后时间范围: {self.df['datetime'].iloc[0]} ~ {self.df['datetime'].iloc[-1]}")
            else:
                print(f"  时间范围筛选: 未启用（使用全部数据）")

            # [新增] 保存原始 datetime 列（在丢弃元数据列之前）
            if 'datetime' in self.df.columns:
                self.datetime_series = self.df['datetime'].copy()
                print(f"✓ 已保存原始 datetime 列 ({len(self.datetime_series)} 条)")

            # 丢弃元数据列
            exclude_present = [col for col in config.EXCLUDE_COLS if col in self.df.columns]
            self.df = self.df.drop(columns=exclude_present)
            print(f"✓ 移除元数据列: {len(exclude_present)} 列")

        except Exception as e:
            print(f"✗ 数据读取失败: {e}")
            raise

    def remove_lag_features(self):
        """Step 1.5: [新增] 移除冗余滞后特征"""
        print(f"\n{'=' * 70}")
        print("Step 1.5: 移除冗余滞后特征 (Redundancy Removal)")
        print(f"{'=' * 70}")

        if not config.REMOVE_LAG_FEATURES:
            print("  跳过: REMOVE_LAG_FEATURES = False")
            return

        # 识别滞后特征列
        lag_cols_to_remove = []
        for col in self.df.columns:
            for pattern in config.LAG_FEATURE_PATTERNS:
                if col.endswith(pattern):
                    lag_cols_to_remove.append(col)
                    break

        if lag_cols_to_remove:
            self.removed_lag_cols = lag_cols_to_remove
            self.df = self.df.drop(columns=lag_cols_to_remove)
            print(f"✓ 移除滞后特征: {len(lag_cols_to_remove)} 列")
            print(f"  示例: {lag_cols_to_remove[:5]} ...")
        else:
            print("  未发现滞后特征列")

    def check_missing_values(self):
        """Step 1.6: [新增] 缺失值检查与处理"""
        print(f"\n{'=' * 70}")
        print("Step 1.6: 缺失值检查与处理")
        print(f"{'=' * 70}")

        null_counts = self.df.isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            print(f"⚠ 发现缺失值: 共 {total_nulls} 个")
            null_cols = null_counts[null_counts > 0]
            for col, count in null_cols.items():
                print(f"    {col}: {count} 个缺失")

            # 使用前向填充 + 后向填充
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')
            
            # 再次检查
            remaining_nulls = self.df.isnull().sum().sum()
            if remaining_nulls > 0:
                # 如果还有缺失，用0填充
                self.df = self.df.fillna(0)
                print(f"  使用 ffill+bfill+0 填充后，剩余缺失: {remaining_nulls} → 0")
            else:
                print(f"✓ 缺失值已通过 ffill+bfill 填充")
        else:
            print("✓ 无缺失值")

    def process_categorical_features(self):
        """Step 1.7: [新增] 类别特征 One-Hot 编码"""
        print(f"\n{'=' * 70}")
        print("Step 1.7: 类别特征 One-Hot 编码")
        print(f"{'=' * 70}")

        # 识别存在的类别列
        categorical_cols = [col for col in config.CATEGORICAL_COLS if col in self.df.columns]
        
        if not categorical_cols:
            print("  未发现配置的类别特征列")
            return

        self.categorical_cols_found = categorical_cols
        print(f"  发现类别特征: {categorical_cols}")

        # 提取类别数据
        cat_data = self.df[categorical_cols].values

        # 创建 One-Hot 编码器
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit 编码器（使用全部数据以确保所有类别都被编码）
        encoder.fit(cat_data)
        
        # Transform
        onehot_data = encoder.transform(cat_data)
        
        # 生成新列名
        onehot_col_names = []
        for i, col in enumerate(categorical_cols):
            categories = encoder.categories_[i]
            for cat in categories:
                onehot_col_names.append(f"{col}_onehot_{int(cat)}")

        print(f"  One-Hot 编码后: {len(categorical_cols)} 列 → {len(onehot_col_names)} 列")
        print(f"  新列名示例: {onehot_col_names[:5]} ...")

        # 创建 One-Hot DataFrame
        onehot_df = pd.DataFrame(onehot_data, columns=onehot_col_names, index=self.df.index)

        # 移除原始类别列，添加 One-Hot 列
        self.df = self.df.drop(columns=categorical_cols)
        self.df = pd.concat([self.df, onehot_df], axis=1)

        # 保存编码器
        self.scalers['onehot_encoder'] = encoder
        self.scalers['meta']['categorical_cols'] = categorical_cols
        self.scalers['meta']['onehot_col_names'] = onehot_col_names

        print(f"✓ One-Hot 编码完成")

    def finalize_features(self):
        """Step 1.8: 最终化特征列"""
        print(f"\n{'=' * 70}")
        print("Step 1.8: 最终化特征列")
        print(f"{'=' * 70}")

        # 保存特征列名
        self.feature_columns = self.df.columns.tolist()

        # 转换为 Float32
        self.data = self.df.astype(config.DTYPE).values
        print(f"✓ 数据预处理完成: {self.data.shape} ({config.DTYPE})")
        print(f"  特征数: {len(self.feature_columns)}")

    def calculate_split_indices(self):
        """Step 2: 数据集划分 (Time-based Split)"""
        print(f"\n{'=' * 70}")
        print("Step 2: 数据集划分 (Time-based Split)")
        print(f"{'=' * 70}")

        total_samples = len(self.data)
        self.train_end_idx = int(total_samples * config.TRAIN_RATIO)
        self.val_end_idx = int(total_samples * (config.TRAIN_RATIO + config.VAL_RATIO))

        print(f"  总样本数: {total_samples}")
        print(f"  Train End: {self.train_end_idx}")
        print(f"  Val End:   {self.val_end_idx}")

    def leakage_free_normalization(self):
        """Step 3: 混合防泄露归一化 (Hybrid Strategy)"""
        print(f"\n{'=' * 70}")
        print("Step 3: 混合归一化策略 (Hybrid Normalization)")
        print(f"{'=' * 70}")

        # 1. 识别列类型
        print("--- 3.1 自动识别列类型 ---")
        passenger_derived_prefix = getattr(config, 'PASSENGER_DERIVED_PREFIX', 'passenger_count_')
        col_indices = {
            'target': [],
            'passenger_derived': [],  # 客流衍生特征 (滞后/滚动)，与 target 同源
            'sentiment': [],  # _score (不含滞后)
            'skip': [],  # _sin, _cos, is_, _onehot_
            'other': []  # remaining numeric
        }

        for idx, col_name in enumerate(self.feature_columns):
            if col_name == config.TARGET_COLUMN:
                col_indices['target'].append(idx)
            # 客流衍生特征：以 passenger_count_ 开头的滞后/滚动列
            elif col_name.startswith(passenger_derived_prefix):
                col_indices['passenger_derived'].append(idx)
            # 跳过: 周期编码、二值特征、One-Hot 编码
            elif (col_name.endswith('_sin') or col_name.endswith('_cos') or
                  col_name.startswith('is_') or '_onehot_' in col_name):
                col_indices['skip'].append(idx)
            # 情感评分（含基础评分和衍生特征如 _roll24h 等）
            elif '_score' in col_name:
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

        # [Strategy 1] Target: Log1p + Scaler (可配置 MinMax 或 Standard)
        if col_indices['target']:
            idx = col_indices['target'][0]
            
            # 获取配置的 scaler 类型，默认为 'minmax'
            target_scaler_type = getattr(config, 'TARGET_SCALER_TYPE', 'minmax').lower()
            
            print(f"  Processing TARGET (idx {idx}): Log1p + {target_scaler_type.upper()}Scaler")

            # 1. Log transform entire data first
            self.data[:, idx] = np.log1p(self.data[:, idx])

            # 2. Fit Scaler on Log-transformed Train data
            target_data = self.data[:, idx].reshape(-1, 1)  # Full data
            train_target = target_data[:self.train_end_idx]

            # 根据配置选择 Scaler 类型
            if target_scaler_type == 'standard':
                # StandardScaler: 输出范围约 [-2, 2]，高值区误差放大更可控
                scaler = StandardScaler()
                scaler_desc = "StandardScaler (Mean=0, Std=1)"
            else:
                # MinMaxScaler: 输出范围 [0, 1]，传统方法
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler_desc = "MinMaxScaler (Range=[0,1])"
            
            scaler.fit(train_target)

            # 3. Transform
            self.data[:, idx] = scaler.transform(target_data).flatten()

            self.scalers['target_scaler'] = scaler
            self.scalers['meta']['target_log1p'] = True
            self.scalers['meta']['target_scaler_type'] = target_scaler_type  # 记录 scaler 类型
            
            # 打印统计信息
            transformed_data = self.data[:, idx]
            print(f"    ✓ Target processed (Log1p + {scaler_desc})")
            print(f"      Mean: {np.mean(transformed_data):.4f}, Std: {np.std(transformed_data):.4f}")
            print(f"      Min: {np.min(transformed_data):.4f}, Max: {np.max(transformed_data):.4f}")

        # [Strategy 2] Passenger Derived: Log1p + Scaler (与 Target 相同处理)
        if col_indices['passenger_derived']:
            idxs = col_indices['passenger_derived']
            target_scaler_type = getattr(config, 'TARGET_SCALER_TYPE', 'minmax').lower()
            
            print(f"  Processing PASSENGER_DERIVED ({len(idxs)} cols): Log1p + {target_scaler_type.upper()}Scaler")
            
            # 1. Log transform entire data
            self.data[:, idxs] = np.log1p(self.data[:, idxs])
            
            # 2. Fit Scaler on Log-transformed Train data
            train_passenger_derived = train_slice[:, idxs]
            
            # 根据配置选择 Scaler 类型（与 Target 保持一致）
            if target_scaler_type == 'standard':
                scaler = StandardScaler()
                scaler_desc = "StandardScaler"
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaler_desc = "MinMaxScaler"
            
            scaler.fit(train_passenger_derived)
            
            # 3. Transform
            self.data[:, idxs] = scaler.transform(self.data[:, idxs])
            
            self.scalers['passenger_derived_scaler'] = scaler
            self.scalers['meta']['passenger_derived_log1p'] = True
            self.scalers['meta']['passenger_derived_scaler_type'] = target_scaler_type
            
            print(f"    ✓ Passenger derived processed (Log1p + {scaler_desc})")

        # [Strategy 3] Sentiment: StandardScaler (Z-Score)
        if col_indices['sentiment']:
            idxs = col_indices['sentiment']
            print(f"  Processing SENTIMENT ({len(idxs)} cols): StandardScaler")

            scaler = StandardScaler()
            scaler.fit(train_slice[:, idxs])

            self.data[:, idxs] = scaler.transform(self.data[:, idxs])

            self.scalers['sentiment_scaler'] = scaler
            print(f"    ✓ Sentiment processed. Mean≈0, Std≈1")

        # [Strategy 4] Other: Default (MinMax)
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

        # [Strategy 5] Skip: No processing (Cyclic/Binary/OneHot)
        if col_indices['skip']:
            print(f"  Skipping SKIP ({len(col_indices['skip'])} cols): Cyclic/Binary/OneHot")
            # Do nothing - these are already in appropriate range

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
                if group_name in col_name.lower() and '_score' in col_name:
                    current_group_idxs.append(idx)
                    all_sentiment_indices.add(idx)

            sentiment_groups.append(current_group_idxs)

        # Context Features (剩余非目标列)
        context_idxs = []
        for idx in range(len(self.feature_columns)):
            if idx == target_idx: continue
            if idx not in all_sentiment_indices:
                context_idxs.append(idx)

        # [新增] One-Hot 特征索引
        onehot_idxs = [idx for idx, col in enumerate(self.feature_columns) if '_onehot_' in col]

        self.feature_map = {
            'feature_names': self.feature_columns,
            'target_idx': target_idx,
            'target_col': config.TARGET_COLUMN,
            'sentiment_groups': sentiment_groups,
            'context_idxs': context_idxs,
            'onehot_idxs': onehot_idxs,  # [新增]
            'total_features': len(self.feature_columns),
            'scaler_info': {
                'target_log1p': True,
                'groups': self.col_groups_indices
            },
            'preprocessing_info': {  # [新增]
                'removed_lag_features': self.removed_lag_cols,
                'categorical_cols_encoded': self.categorical_cols_found,
                'version': 'v2'
            }
        }

        # 保存
        map_path = os.path.join(config.OUTPUT_DIR, config.FEATURE_MAP_FILENAME)
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_map, f, indent=2, ensure_ascii=False)
        print(f"✓ Feature Map Saved: {map_path}")

        # 打印摘要
        print(f"\n--- Feature Map 摘要 ---")
        print(f"  总特征数: {len(self.feature_columns)}")
        print(f"  情感分组: {[len(g) for g in sentiment_groups]}")
        print(f"  上下文特征: {len(context_idxs)}")
        print(f"  One-Hot 特征: {len(onehot_idxs)}")

    def sliding_window_slicing(self):
        """Step 5: 滑动窗口切片 (Sliding Window - 小时级)"""
        print(f"\n{'=' * 70}")
        print("Step 5: 滑动窗口切片 (小时级)")
        print(f"{'=' * 70}")

        X_list = []
        Y_list = []

        total_samples = len(self.data)
        window_size = config.SEQ_LEN + config.PRED_LEN
        target_idx = self.feature_map['target_idx']

        print(f"  Seq: {config.SEQ_LEN} 小时, Pred: {config.PRED_LEN} 小时")
        print(f"  Window Size: {window_size}")

        # 优化循环效率
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

        # [新增] 保存原始 datetime 信息（供 m11 推理使用）
        if self.datetime_series is not None:
            datetime_filename = getattr(config, 'DATETIME_FILENAME', 'datetime_index.npy')
            datetime_path = os.path.join(config.OUTPUT_DIR, datetime_filename)
            # 转为字符串数组保存，兼容性更好
            datetime_strs = self.datetime_series.astype(str).values
            np.save(datetime_path, datetime_strs)
            print(f"  Saved {datetime_filename}: {datetime_strs.shape}")
            print(f"    时间范围: {datetime_strs[0]} ~ {datetime_strs[-1]}")

        # 获取 target scaler 类型
        target_scaler_type = self.scalers['meta'].get('target_scaler_type', 'minmax')
        
        # 根据 scaler 类型生成描述
        if target_scaler_type == 'standard':
            scaler_desc = 'StandardScaler'
            scaler_note = '高值区误差放大更可控，输出范围约 [-2, 2]'
        else:
            scaler_desc = 'MinMaxScaler'
            scaler_note = '传统方法，输出范围 [0, 1]'
        
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
            # 动态生成归一化策略描述
            'normalization_strategy': f'Hybrid V2.1 (Log1p+{scaler_desc}-Target, Z-Sentiment, OneHot-Categorical, Raw-Time)',
            'target_transform': {
                'step1': 'log1p',
                'step2': scaler_desc,
                'scaler_type': target_scaler_type,
                'note': scaler_note
            },
            'scaler_file': config.SCALERS_FILENAME,
            'granularity': 'hourly',
            'version': 'v2.1',
            'improvements': [
                'Categorical features encoded with OneHot',
                'Redundant lag features removed',
                'Missing value handling added',
                f'Target uses Log1p + {scaler_desc}'
            ],
            'removed_lag_features_count': len(self.removed_lag_cols),
            'onehot_features_count': len([c for c in self.feature_columns if '_onehot_' in c])
        }
        with open(os.path.join(config.OUTPUT_DIR, 'dataset_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\n✓ Dataset Info 已保存")

    def run(self):
        """执行流程"""
        print(f"\n{'#' * 70}")
        print(f"# M8B 张量构建 V2 (Hybrid Normalization - 小时级)")
        print(f"# 改进: One-Hot类别编码 + 移除冗余滞后特征")
        print(f"{'#' * 70}")

        if not self.connect_mysql(): return False

        try:
            # Step 1: 数据加载与预处理
            self.load_and_sort()
            self.remove_lag_features()      # [新增] 移除滞后特征
            self.check_missing_values()     # [新增] 缺失值处理
            self.process_categorical_features()  # [新增] One-Hot 编码
            self.finalize_features()

            # Step 2-6: 原有流程
            self.calculate_split_indices()
            self.leakage_free_normalization()
            self.generate_feature_map()
            self.sliding_window_slicing()
            self.save_tensors()

            print(f"\n{'=' * 70}")
            print(f"✓ 成功完成。输出目录: {config.OUTPUT_DIR}")
            print(f"{'=' * 70}")
            
            # 打印改进摘要
            print(f"\n--- V2 改进摘要 ---")
            print(f"  移除滞后特征: {len(self.removed_lag_cols)} 列")
            print(f"  One-Hot 编码类别: {len(self.categorical_cols_found)} 列")
            print(f"  最终特征维度: {self.X.shape[-1]}")
            
            return True
        except Exception as e:
            print(f"\n✗ 出错: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    builder = TensorBuilderV2()
    builder.run()
