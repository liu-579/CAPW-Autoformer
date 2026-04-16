# -*- coding: utf-8 -*-
"""
M11 统一推理脚本 V2 (Unified Inference Script)
用途: 支持 5 种模型变体的统一推理、评估指标计算、CSV 导出、可视化
支持模型: transformer, autoformer, bp, lstm, autoformer_moe
改进:
  1. 支持选择推理数据集 (train/val/test/all)
  2. 支持分析时间窗口 (绘图+指标聚焦到指定日期范围)
  3. 指标分层输出 (整体/分 split/时间切片)
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import json
import importlib
from pathlib import Path
from tqdm import tqdm
import warnings
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from typing import Dict, List

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 导入推理配置
sys.path.append('.')
from config.m11_inference_config import InferenceConfig as cfg

# ==================== 模型注册表 ====================

MODEL_REGISTRY = {
    'transformer': {
        'model_module': 'scripts.m9_model_2',
        'config_module': 'config.m9_config',
        'config_class': 'ModelConfig',
    },
    'autoformer': {
        'model_module': 'scripts.m9_model_autoformer',
        'config_module': 'config.m9_config_autoformer',
        'config_class': 'ModelConfig',
    },
    'bp': {
        'model_module': 'scripts.m9_model_bp',
        'config_module': 'config.m9_config_bp',
        'config_class': 'ModelConfig',
    },
    'lstm': {
        'model_module': 'scripts.m9_model_lstm',
        'config_module': 'config.m9_config_lstm',
        'config_class': 'ModelConfig',
    },
    'autoformer_moe': {
        'model_module': 'scripts.m9_model_autoformer_moe',
        'config_module': 'config.m9_config_autoformer_moe',
        'config_class': 'ModelConfig',
    },
}

# 权重名称 (5 个情感组)
WEIGHT_NAMES = ['w_scenery', 'w_transportation', 'w_food', 'w_consumption', 'w_service']
WEIGHT_LABELS_CN = ['景色权重', '交通权重', '餐饮权重', '消费权重', '服务权重']

# 数据集划分名称 (有序)
SPLIT_ORDER = ['train', 'val', 'test']
SPLIT_LABELS_CN = {'train': '训练集', 'val': '验证集', 'test': '测试集', 'all': '全量'}


# ==================== 工具函数 ====================

def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model_and_config(model_type: str, dataset_dir: str, model_path: str):
    """
    根据模型类型动态加载模型构建函数和配置类
    并自动从 checkpoint 同目录的 experiment_results.csv 恢复训练时的超参数

    Returns: (build_model_fn, model_config_cls)
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"不支持的模型类型: {model_type}, 支持: {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[model_type]

    # 动态导入模型模块
    model_mod = importlib.import_module(entry['model_module'])
    build_model_fn = getattr(model_mod, 'build_model')

    # 动态导入配置模块
    config_mod = importlib.import_module(entry['config_module'])
    model_config_cls = getattr(config_mod, entry['config_class'])

    # 从数据集自动加载 seq_len / pred_len
    if hasattr(model_config_cls, 'from_dataset'):
        model_config_cls.from_dataset(dataset_dir, verbose=True)
    else:
        print(f"⚠️  {model_type} 的 ModelConfig 没有 from_dataset 方法，使用默认参数")

    # ---- 从 experiment_results.csv 自动恢复训练时超参数 ----
    _apply_checkpoint_hparams(model_config_cls, model_path)

    return build_model_fn, model_config_cls


def _apply_checkpoint_hparams(model_config_cls, model_path: str):
    """
    从 checkpoint 同目录的 experiment_results.csv 读取训练时的模型超参数,
    自动覆盖 ModelConfig 中的对应字段, 确保模型结构与权重文件完全匹配。
    """
    ckpt_dir = Path(model_path).parent
    exp_csv = ckpt_dir / 'experiment_results.csv'

    if not exp_csv.exists():
        print(f"  ⚠️  未找到 {exp_csv.name}，使用当前配置默认值构建模型")
        print(f"       若权重来自 HPO 训练，可能出现 size mismatch 错误")
        return

    print(f"\n  📄 检测到 experiment_results.csv，自动恢复训练时超参数...")

    df = pd.read_csv(exp_csv)
    if df.empty:
        print(f"  ⚠️  experiment_results.csv 为空，跳过")
        return

    row = df.iloc[-1]  # 取最后一行 (最终训练记录)

    # 需要覆盖的超参数映射: CSV列名 → (ModelConfig属性名, 类型转换函数)
    HPARAM_MAP = {
        'model_d_model':                ('d_model', int),
        'model_nhead':                  ('nhead', int),
        'model_num_encoder_layers':     ('num_encoder_layers', int),
        'model_num_decoder_layers':     ('num_decoder_layers', int),
        'model_dim_feedforward':        ('dim_feedforward', int),
        'model_dropout':                ('dropout', float),
        'model_moving_avg_kernel':      ('moving_avg_kernel', int),
        'model_autocorrelation_factor': ('autocorrelation_factor', int),
        'model_weight_net_hidden_dim':  ('weight_net_hidden_dim', int),
        'model_weight_net_num_heads':   ('weight_net_num_heads', int),
        'model_weight_net_num_layers':  ('weight_net_num_layers', int),
        'model_weight_net_dropout':     ('weight_net_dropout', float),
        'model_temperature_init':       ('temperature_init', float),
        'model_temperature_trainable':  ('temperature_trainable', lambda v: str(v).lower() == 'true'),
        'model_seq_len':                ('seq_len', int),
        'model_label_len':              ('label_len', int),
        'model_pred_len':               ('pred_len', int),
        'model_output_dim':             ('output_dim', int),
        'model_ablation_mode':          ('ablation_mode', str),
        'model_activation':             ('activation', str),
        'model_positional_encoding_type': ('positional_encoding_type', str),
    }

    # BP / LSTM 特有参数 (CSV 中可能不存在，安全跳过)
    BP_LSTM_EXTRAS = {
        'model_hidden_dims':       ('hidden_dims', lambda v: json.loads(str(v).replace("'", '"')) if pd.notna(v) else None),
        'model_use_batch_norm':    ('use_batch_norm', lambda v: str(v).lower() == 'true'),
        'model_use_residual':      ('use_residual', lambda v: str(v).lower() == 'true'),
        'model_lstm_num_layers':   ('lstm_num_layers', int),
        'model_lstm_bidirectional': ('lstm_bidirectional', lambda v: str(v).lower() == 'true'),
        'model_fc_hidden_dim':     ('fc_hidden_dim', int),
        'model_fc_num_layers':     ('fc_num_layers', int),
        'model_num_layers':        ('num_layers', int),
    }
    HPARAM_MAP.update(BP_LSTM_EXTRAS)

    applied = []
    for csv_col, (attr_name, type_fn) in HPARAM_MAP.items():
        if csv_col not in row.index:
            continue
        val = row[csv_col]
        if pd.isna(val):
            continue
        if not hasattr(model_config_cls, attr_name):
            continue

        old_val = getattr(model_config_cls, attr_name)
        try:
            new_val = type_fn(val)
        except (ValueError, TypeError):
            continue

        if new_val is None:
            continue

        if old_val != new_val:
            setattr(model_config_cls, attr_name, new_val)
            applied.append(f"    {attr_name}: {old_val} → {new_val}")

    if applied:
        print(f"  ✓ 已覆盖 {len(applied)} 个超参数:")
        for line in applied:
            print(line)
    else:
        print(f"  ✓ 所有超参数与默认配置一致，无需覆盖")


# ==================== 指标计算器 ====================

class MetricsCalculator:
    """评估指标计算器（与 m11_test.py 保持一致）"""

    def __init__(self, scaler_data):
        if isinstance(scaler_data, dict) and 'target_scaler' in scaler_data:
            self.scaler = scaler_data['target_scaler']
            self.meta = scaler_data.get('meta', {})
            self.is_log1p = self.meta.get('target_log1p', False)
        else:
            self.scaler = scaler_data
            self.is_log1p = False

        if hasattr(self.scaler, 'scale_'):
            self.scale_ = self.scaler.scale_[0]
            self.min_ = self.scaler.min_[0]
            self.scaler_type = 'minmax'
        elif hasattr(self.scaler, 'mean_'):
            self.mean_ = self.scaler.mean_[0]
            self.scale_ = self.scaler.scale_[0]
            self.scaler_type = 'standard'
        else:
            raise ValueError(f"不支持的 scaler 类型: {type(self.scaler)}")

    def inverse_transform(self, y):
        """反归一化"""
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        y = y.astype(np.float64)

        if self.scaler_type == 'minmax':
            y_restored = (y - self.min_) / self.scale_
        elif self.scaler_type == 'standard':
            y_restored = y * self.scale_ + self.mean_
        else:
            y_restored = y

        if self.is_log1p:
            y_restored = np.expm1(y_restored)

        y_restored = np.maximum(y_restored, 0)
        return y_restored

    def compute_metrics(self, preds, targets):
        """计算 5 项评估指标"""
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mae = np.mean(np.abs(preds - targets))
        epsilon = 1e-8
        mape = np.mean(np.abs((preds - targets) / (np.abs(targets) + epsilon))) * 100

        if len(preds) > 1:
            pearson_r = np.corrcoef(preds.flatten(), targets.flatten())[0, 1]
        else:
            pearson_r = 0.0

        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + epsilon))

        return {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'Pearson_R': float(pearson_r),
            'R2_Score': float(r2_score)
        }


# ==================== 权重提取 ====================

def extract_group_weights(model, expanded_weights: torch.Tensor) -> torch.Tensor:
    """
    从 expanded_weights (B, T, total_sentiment_dim) 提取 5 组原始权重 (B, T, 5)
    利用 model 的 weight_expansion_indices_buffer 逆映射
    """
    B, T, D = expanded_weights.shape
    wei_buf = model.weight_expansion_indices_buffer  # (total_sentiment_dim,)
    num_groups = model.num_sentiment_groups  # 5

    group_weights = torch.zeros(B, T, num_groups, device=expanded_weights.device)
    for g in range(num_groups):
        mask = (wei_buf == g)
        first_idx = mask.nonzero(as_tuple=False)[0, 0].item()
        group_weights[:, :, g] = expanded_weights[:, :, first_idx]

    return group_weights


def aggregate_over_time(values: np.ndarray, method: str) -> np.ndarray:
    """
    对时间维度进行聚合
    values: (N, T, D) 或 (N, T)
    返回: (N, D) 或 (N,)
    """
    if method == 'last':
        return values[:, -1] if values.ndim >= 2 else values
    elif method == 'mean':
        return values.mean(axis=1) if values.ndim >= 2 else values
    else:
        raise ValueError(f"不支持的聚合方式: {method}")


# ==================== 数据加载 (V2: 支持多 split) ====================

def load_split_data():
    """
    根据 RUN_SPLIT 配置加载对应的数据集

    Returns:
        data_x:          np.ndarray (N, seq_len, features)
        data_y:          np.ndarray (N, pred_len, 1)
        feature_map:     dict
        scaler_data:     dict
        datetime_index:  np.ndarray or None
        split_boundaries: dict {split_name: (start_idx, end_idx)} — 在 data_x 中的索引范围
        dataset_info:    dict — 来自 dataset_info.json
    """
    print("\n" + "=" * 70)
    print(f"Step 1: 加载数据 [RUN_SPLIT = {cfg.RUN_SPLIT}]")
    print("=" * 70)

    # ---- 加载公共文件 ----
    with open(cfg.DATA_DIR / 'feature_map.json', 'r', encoding='utf-8') as f:
        feature_map = json.load(f)

    with open(cfg.DATA_DIR / 'scalers.pkl', 'rb') as f:
        scaler_data = pickle.load(f)

    dataset_info = {}
    info_path = cfg.DATA_DIR / 'dataset_info.json'
    if info_path.exists():
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)

    # ---- 加载 datetime_index（可选）----
    datetime_path = cfg.DATA_DIR / cfg.DATETIME_FILENAME
    datetime_index = None
    if datetime_path.exists():
        datetime_index = np.load(datetime_path, allow_pickle=True)
        print(f"  datetime_index: {datetime_index.shape} "
              f"({datetime_index[0]} ~ {datetime_index[-1]})")
    else:
        print(f"  ⚠️ {cfg.DATETIME_FILENAME} 不存在，时间轴功能不可用")

    # ---- 根据 RUN_SPLIT 加载 x/y 数据 ----
    split_boundaries = {}

    if cfg.RUN_SPLIT == 'all':
        # 拼接所有三段
        arrays_x = []
        arrays_y = []
        offset = 0
        for split_name in SPLIT_ORDER:
            sx = np.load(cfg.DATA_DIR / f'{split_name}_x.npy')
            sy = np.load(cfg.DATA_DIR / f'{split_name}_y.npy')
            n = len(sx)
            split_boundaries[split_name] = (offset, offset + n)
            arrays_x.append(sx)
            arrays_y.append(sy)
            print(f"  {split_name}: {sx.shape} (窗口 {offset} ~ {offset + n - 1})")
            offset += n
        data_x = np.concatenate(arrays_x, axis=0)
        data_y = np.concatenate(arrays_y, axis=0)
    else:
        data_x = np.load(cfg.DATA_DIR / f'{cfg.RUN_SPLIT}_x.npy')
        data_y = np.load(cfg.DATA_DIR / f'{cfg.RUN_SPLIT}_y.npy')
        split_boundaries[cfg.RUN_SPLIT] = (0, len(data_x))
        print(f"  {cfg.RUN_SPLIT}: {data_x.shape}")

    print(f"\n  合计 data_x: {data_x.shape}")
    print(f"  合计 data_y: {data_y.shape}")
    print(f"  特征总数:    {feature_map['total_features']}")
    print(f"  情感组数:    {len(feature_map['sentiment_groups'])}")
    print(f"  环境特征数:  {len(feature_map['context_idxs'])}")
    print(f"✓ 数据加载完成")

    return data_x, data_y, feature_map, scaler_data, datetime_index, split_boundaries, dataset_info


def create_dataloader(data_x, data_y):
    """创建 DataLoader"""
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(data_x), torch.FloatTensor(data_y)),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    return loader


# ==================== 模型加载 ====================

def load_checkpoint(model, model_path: str, device: str):
    """加载模型权重 (兼容两种 checkpoint 格式)"""
    print(f"\n{'=' * 70}")
    print("Step 3: 加载模型权重")
    print("=" * 70)
    print(f"  路径: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'N/A'))
        print(f"  训练轮次:  {epoch}")
        print(f"  验证损失:  {val_loss}")
    else:
        model.load_state_dict(checkpoint)

    print(f"✓ 模型权重加载成功")
    return model


# ==================== 推理 ====================

@torch.no_grad()
def run_inference(model, data_loader, device: str, temperature: float):
    """
    在数据集上进行推理
    返回: preds (N, pred_len, 1), targets (N, pred_len, 1),
          group_weights (N, T, 5), inputs (N, seq_len, F)
    """
    print(f"\n{'=' * 70}")
    print("Step 4: 模型推理")
    print("=" * 70)

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_group_weights = []
    all_inputs = []

    for x, y in tqdm(data_loader, desc='推理进度'):
        x = x.to(device)
        y = y.to(device)

        preds, expanded_weights = model(x, temperature=temperature)
        group_w = extract_group_weights(model, expanded_weights)

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
        all_group_weights.append(group_w.cpu())
        all_inputs.append(x.cpu())

    preds_tensor = torch.cat(all_preds, dim=0)          # (N, pred_len, 1)
    targets_tensor = torch.cat(all_targets, dim=0)       # (N, pred_len, 1)
    weights_tensor = torch.cat(all_group_weights, dim=0) # (N, T, 5)
    inputs_tensor = torch.cat(all_inputs, dim=0)         # (N, seq_len, F)

    print(f"  预测输出:   {preds_tensor.shape}")
    print(f"  真实标签:   {targets_tensor.shape}")
    print(f"  组权重:     {weights_tensor.shape}")
    print(f"  输入特征:   {inputs_tensor.shape}")
    print(f"✓ 推理完成，共 {len(preds_tensor)} 个样本")

    return preds_tensor, targets_tensor, weights_tensor, inputs_tensor


# ==================== CSV 导出 (V2: 含 datetime + split 列) ====================

def get_context_features_for_csv(inputs_np: np.ndarray, feature_map: dict):
    """从输入数据中提取指定的环境特征列"""
    feature_names = feature_map.get('feature_names', [])
    result = {}
    for feat_name in cfg.CSV_CONTEXT_FEATURES:
        if feat_name in feature_names:
            feat_idx = feature_names.index(feat_name)
            feat_values = inputs_np[:, :, feat_idx]  # (N, seq_len)
            feat_agg = aggregate_over_time(feat_values, cfg.CONTEXT_AGG_METHOD)
            result[feat_name] = feat_agg
        else:
            print(f"  ⚠️  特征 '{feat_name}' 不在 feature_map 中，跳过")
    return result


def _compute_window_offset(split_name: str, dataset_info: dict) -> int:
    """
    计算某个 split 的窗口在全局 datetime_index 中的起始偏移。

    滑动窗口从全局数据创建后按比例划分:
      - train 窗口从全局位置 0 开始
      - val   窗口从全局位置 train_count 开始
      - test  窗口从全局位置 train_count + val_count 开始
    """
    samples = dataset_info.get('samples', {})
    train_n = samples.get('train', 0)
    val_n = samples.get('val', 0)

    if split_name == 'train':
        return 0
    elif split_name == 'val':
        return train_n
    elif split_name == 'test':
        return train_n + val_n
    else:
        return 0  # 'all' 拼接后从 0 开始


def build_and_save_csv(preds_real, targets_real, weights_np, inputs_np,
                       feature_map, datetime_index, split_boundaries,
                       dataset_info, seq_len):
    """
    构建并保存 CSV 文件
    列: datetime, split, real_flow, pred_flow, context_features..., w_*
    """
    print(f"\n{'=' * 70}")
    print("Step 6: 导出 CSV")
    print("=" * 70)

    N_orig = preds_real.shape[0]  # 原始窗口数

    # ---- 1. 处理预测步 ----
    if cfg.PRED_STEP_MODE == 'first':
        if preds_real.ndim > 1 and preds_real.shape[1] > 1:
            pred_col = preds_real[:, 0].flatten()
            real_col = targets_real[:, 0].flatten()
        else:
            pred_col = preds_real.flatten()
            real_col = targets_real.flatten()
        pred_len_expand = 1
    elif cfg.PRED_STEP_MODE == 'mean':
        pred_col = preds_real.mean(axis=1).flatten() if preds_real.ndim > 1 else preds_real.flatten()
        real_col = targets_real.mean(axis=1).flatten() if targets_real.ndim > 1 else targets_real.flatten()
        pred_len_expand = 1
    elif cfg.PRED_STEP_MODE == 'all':
        pred_col = preds_real.flatten()
        real_col = targets_real.flatten()
        pred_len_expand = preds_real.shape[1] if preds_real.ndim > 1 else 1
        weights_np = np.repeat(weights_np, pred_len_expand, axis=0)
        inputs_np = np.repeat(inputs_np, pred_len_expand, axis=0)
    else:
        pred_col = preds_real.flatten()
        real_col = targets_real.flatten()
        pred_len_expand = 1

    N = len(pred_col)

    # ---- 2. 计算 split 标签 ----
    split_labels = np.empty(N_orig, dtype=object)
    for sname, (s_start, s_end) in split_boundaries.items():
        split_labels[s_start:s_end] = sname
    if pred_len_expand > 1:
        split_labels = np.repeat(split_labels, pred_len_expand)

    # ---- 3. 计算 datetime 列 ----
    datetime_col = None
    if datetime_index is not None and len(dataset_info) > 0:
        try:
            # 计算每个 split 在全局 datetime_index 中的窗口偏移
            # 当 RUN_SPLIT='all' 时，拼接顺序是 train→val→test，全局窗口 j 对应 datetime_index[j + seq_len]
            # 当 RUN_SPLIT='test' 时，窗口 j 对应 datetime_index[window_offset + j + seq_len]

            if cfg.RUN_SPLIT == 'all':
                # 全量拼接: 全局窗口 j → datetime_index[j + seq_len]
                global_offset = 0
            else:
                global_offset = _compute_window_offset(cfg.RUN_SPLIT, dataset_info)

            datetime_col = []
            for j in range(N_orig):
                if cfg.PRED_STEP_MODE == 'all':
                    for step in range(pred_len_expand):
                        dt_idx = global_offset + j + seq_len + step
                        if dt_idx < len(datetime_index):
                            datetime_col.append(datetime_index[dt_idx])
                        else:
                            datetime_col.append('')
                else:
                    dt_idx = global_offset + j + seq_len
                    if dt_idx < len(datetime_index):
                        datetime_col.append(datetime_index[dt_idx])
                    else:
                        datetime_col.append('')
            datetime_col = np.array(datetime_col)
            print(f"  ✓ 已计算 datetime 列 ({len(datetime_col)} 条)")
        except Exception as e:
            print(f"  ⚠️ 计算 datetime 列时出错: {e}，跳过")
            datetime_col = None

    # ---- 4. 构建 DataFrame ----
    df_data = {}
    if datetime_col is not None:
        df_data['datetime'] = datetime_col[:N]
    df_data['split'] = split_labels[:N]
    df_data['real_flow'] = real_col[:N]
    df_data['pred_flow'] = pred_col[:N]
    df = pd.DataFrame(df_data)

    # 添加环境特征
    context_feats = get_context_features_for_csv(inputs_np, feature_map)
    for feat_name, feat_values in context_feats.items():
        df[feat_name] = feat_values[:N]

    # 添加 5 个动态权重
    weights_agg = aggregate_over_time(weights_np, cfg.WEIGHT_AGG_METHOD)
    for i, wname in enumerate(WEIGHT_NAMES):
        df[wname] = weights_agg[:N, i]

    # ---- 5. 保存 ----
    df.to_csv(cfg.CSV_FILE, index=False, encoding='utf-8-sig')

    print(f"  行数:   {len(df)}")
    print(f"  列数:   {len(df.columns)}")
    print(f"  列名:   {list(df.columns)}")
    # 打印各 split 样本数
    for sname in df['split'].unique():
        print(f"    [{sname}] {len(df[df['split'] == sname])} 行")
    print(f"  文件:   {cfg.CSV_FILE}")
    print(f"✓ CSV 导出完成")

    return df


# ==================== 时间筛选工具 ====================

def filter_by_analysis_date(df: pd.DataFrame) -> pd.DataFrame:
    """根据 ANALYSIS_DATE_START/END 配置筛选 DataFrame"""
    if 'datetime' not in df.columns:
        return df

    start = getattr(cfg, 'ANALYSIS_DATE_START', None)
    end = getattr(cfg, 'ANALYSIS_DATE_END', None)

    if start is None and end is None:
        return df

    filtered = df.copy()
    filtered['datetime'] = pd.to_datetime(filtered['datetime'])

    if start is not None:
        filtered = filtered[filtered['datetime'] >= pd.to_datetime(start)]
    if end is not None:
        end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)
        filtered = filtered[filtered['datetime'] < end_dt]

    return filtered.reset_index(drop=True)


# ==================== 分层指标计算 (V2) ====================

def _print_metrics(metrics: dict, indent: str = "  "):
    """格式化打印一组指标"""
    print(f"{indent}RMSE:       {metrics['RMSE']:.4f}")
    print(f"{indent}MAE:        {metrics['MAE']:.4f}")
    print(f"{indent}MAPE:       {metrics['MAPE']:.2f}%")
    print(f"{indent}Pearson R:  {metrics['Pearson_R']:.4f}")
    print(f"{indent}R2 Score:   {metrics['R2_Score']:.4f}")


def compute_layered_metrics(df: pd.DataFrame, metrics_calc: MetricsCalculator) -> dict:
    """
    计算分层指标:
      1. overall:    当前 RUN_SPLIT 范围内的整体指标
      2. per_split:  按 train/val/test 分别计算
      3. time_slice: 按 ANALYSIS_DATE 时间窗口计算（如果设置了的话）
    """
    print(f"\n{'=' * 70}")
    print("Step 5: 计算分层评估指标")
    print("=" * 70)

    results = {}

    # ---------- 1. 整体指标 ----------
    print(f"\n  --- 1. 整体指标 [{cfg.RUN_SPLIT}] ({len(df)} 样本) ---")
    overall_m = metrics_calc.compute_metrics(
        df['pred_flow'].values, df['real_flow'].values
    )
    results['overall'] = overall_m
    _print_metrics(overall_m, indent="    ")

    # ---------- 2. 分 split 指标 ----------
    split_names = [s for s in SPLIT_ORDER if s in df['split'].unique()]
    per_split = {}
    if len(split_names) > 1 or (len(split_names) == 1 and cfg.RUN_SPLIT == 'all'):
        print(f"\n  --- 2. 分数据集指标 ---")
        for sname in split_names:
            sub = df[df['split'] == sname]
            if len(sub) == 0:
                continue
            m = metrics_calc.compute_metrics(
                sub['pred_flow'].values, sub['real_flow'].values
            )
            per_split[sname] = m
            cn = SPLIT_LABELS_CN.get(sname, sname)
            print(f"\n    [{cn}] ({len(sub)} 样本)")
            _print_metrics(m, indent="      ")
    else:
        # 单 split 模式: per_split 里只有当前 split
        sname = split_names[0] if split_names else cfg.RUN_SPLIT
        per_split[sname] = overall_m
        print(f"\n  --- 2. 分数据集指标: 单 split [{sname}]，同整体 ---")

    results['per_split'] = per_split

    # ---------- 3. 时间切片指标 ----------
    start = getattr(cfg, 'ANALYSIS_DATE_START', None)
    end = getattr(cfg, 'ANALYSIS_DATE_END', None)

    if (start or end) and 'datetime' in df.columns:
        slice_df = filter_by_analysis_date(df)
        start_str = start or '数据起始'
        end_str = end or '数据结束'

        if len(slice_df) > 0:
            slice_m = metrics_calc.compute_metrics(
                slice_df['pred_flow'].values, slice_df['real_flow'].values
            )
            results['time_slice'] = {
                'date_range': f"{start_str} ~ {end_str}",
                'sample_count': int(len(slice_df)),
                'metrics': slice_m
            }
            print(f"\n  --- 3. 时间切片指标 [{start_str} ~ {end_str}] ({len(slice_df)} 样本) ---")
            _print_metrics(slice_m, indent="    ")

            # 按 split 细分切片（如果切片跨 split）
            slice_splits = [s for s in SPLIT_ORDER if s in slice_df['split'].unique()]
            if len(slice_splits) > 1:
                slice_per_split = {}
                for sname in slice_splits:
                    sub = slice_df[slice_df['split'] == sname]
                    if len(sub) > 0:
                        slice_per_split[sname] = {
                            'sample_count': int(len(sub)),
                            'metrics': metrics_calc.compute_metrics(
                                sub['pred_flow'].values, sub['real_flow'].values
                            )
                        }
                results['time_slice']['per_split'] = slice_per_split
        else:
            print(f"\n  --- 3. 时间切片: [{start_str} ~ {end_str}] 范围内无数据 ---")
            results['time_slice'] = None
    else:
        if start or end:
            print(f"\n  --- 3. 时间切片: datetime 列不可用，跳过 ---")
        else:
            print(f"\n  --- 3. 时间切片: 未设置分析时间窗口 ---")
        results['time_slice'] = None

    print(f"\n{'=' * 70}")
    return results


# ==================== 指标 JSON 保存 ====================

def save_metrics_json(layered_metrics: dict):
    """保存分层评估指标到 JSON"""
    print(f"\n{'=' * 70}")
    print("Step 7: 保存评估指标 JSON")
    print("=" * 70)

    results = {
        'model_type': cfg.MODEL_TYPE,
        'scenic_name': cfg.scenic_name,
        'run_split': cfg.RUN_SPLIT,
        'overall_metrics': layered_metrics['overall'],
        'per_split_metrics': layered_metrics['per_split'],
        'time_slice_metrics': layered_metrics.get('time_slice'),
        'inference_config': {
            'model_path': str(cfg.MODEL_PATH),
            'data_dir': str(cfg.DATA_DIR),
            'temperature': cfg.TEMPERATURE,
            'weight_agg_method': cfg.WEIGHT_AGG_METHOD,
            'pred_step_mode': cfg.PRED_STEP_MODE,
            'analysis_date_start': cfg.ANALYSIS_DATE_START,
            'analysis_date_end': cfg.ANALYSIS_DATE_END,
            'inference_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }

    with open(cfg.METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"  文件: {cfg.METRICS_FILE}")
    print(f"✓ 分层指标已保存")


# ==================== 可视化 (V2: 支持时间切片) ====================

def _build_time_axis(df: pd.DataFrame):
    """
    从 DataFrame 构建时间 x 轴。
    返回: (x_axis, use_datetime: bool)
    """
    if 'datetime' not in df.columns:
        return np.arange(len(df)), False

    try:
        time_labels = pd.to_datetime(df['datetime'])
        if time_labels.isna().any():
            return np.arange(len(df)), False
        return time_labels.tolist(), True
    except Exception:
        return np.arange(len(df)), False


def _format_datetime_axis(ax, fig, time_labels):
    """格式化日期 x 轴"""
    fig.autofmt_xdate()
    if len(time_labels) > 1:
        total_days = (time_labels[-1] - time_labels[0]).days
    else:
        total_days = 0

    if total_days <= 7:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    elif total_days <= 60:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())


def _plot_comparison_core(df: pd.DataFrame, save_path, title_suffix: str = ''):
    """绘制对比图核心逻辑"""
    preds_plot = df['pred_flow'].values
    targets_plot = df['real_flow'].values

    fig, ax = plt.subplots(figsize=cfg.FIGURE_SIZE)
    x_axis, use_dt = _build_time_axis(df)

    ax.plot(x_axis, targets_plot, label='真实值', alpha=0.7, color='blue', linewidth=1.5)
    ax.plot(x_axis, preds_plot, label='预测值', alpha=0.7, color='red', linestyle='--', linewidth=1.5)

    if use_dt:
        ax.set_xlabel('时间', fontsize=12)
        _format_datetime_axis(ax, fig, x_axis)
    else:
        ax.set_xlabel('样本索引', fontsize=12)

    ax.set_ylabel('游客流量 (人次)', fontsize=12)
    title = f'{cfg.test_description} - 预测对比图 [{cfg.MODEL_TYPE.upper()}]'
    if title_suffix:
        title += f' {title_suffix}'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.FIGURE_DPI)
    if cfg.SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"  文件: {save_path}")


def _plot_weights_core(df: pd.DataFrame, save_path, title_suffix: str = ''):
    """绘制权重图核心逻辑"""
    fig, ax = plt.subplots(figsize=cfg.FIGURE_SIZE)
    x_axis, use_dt = _build_time_axis(df)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for i, wname in enumerate(WEIGHT_NAMES):
        if wname in df.columns:
            ax.plot(x_axis, df[wname].values, label=WEIGHT_LABELS_CN[i],
                    alpha=0.8, linewidth=1.2, color=colors[i])

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='基准线 (1.0)')

    if use_dt:
        ax.set_xlabel('时间', fontsize=12)
        _format_datetime_axis(ax, fig, x_axis)
    else:
        ax.set_xlabel('样本索引', fontsize=12)

    ax.set_ylabel('权重值', fontsize=12)
    title = f'情感维度动态权重变化 [{cfg.MODEL_TYPE.upper()}]'
    if title_suffix:
        title += f' {title_suffix}'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 2.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.FIGURE_DPI)
    if cfg.SHOW_PLOTS:
        plt.show()
    plt.close()
    print(f"  文件: {save_path}")


def plot_all_charts(df: pd.DataFrame):
    """
    统一绘图入口:
      1. 全量对比图 + 全量权重图（始终生成）
      2. 若设置了 ANALYSIS_DATE，额外生成切片对比图 + 切片权重图
    """
    print(f"\n{'=' * 70}")
    print("Step 8: 绘制可视化图表")
    print("=" * 70)

    # ---- 全量图（可能受 PLOT_SAMPLE_LIMIT 限制）----
    plot_df = df.copy()
    if cfg.PLOT_SAMPLE_LIMIT > 0:
        plot_df = plot_df.iloc[:cfg.PLOT_SAMPLE_LIMIT]
        print(f"  全量图限制显示前 {len(plot_df)} 个样本")

    print(f"\n  --- 全量对比图 ---")
    _plot_comparison_core(plot_df, cfg.PLOT_COMPARISON)
    print(f"\n  --- 全量权重图 ---")
    _plot_weights_core(plot_df, cfg.PLOT_WEIGHTS)

    # ---- 切片图（如果设置了分析时间窗口）----
    start = getattr(cfg, 'ANALYSIS_DATE_START', None)
    end = getattr(cfg, 'ANALYSIS_DATE_END', None)

    if (start or end) and 'datetime' in df.columns:
        slice_df = filter_by_analysis_date(df)
        start_str = start or '起始'
        end_str = end or '结束'

        if len(slice_df) > 0:
            suffix = f'({start_str}~{end_str})'
            print(f"\n  --- 切片对比图 [{suffix}] ({len(slice_df)} 样本) ---")
            _plot_comparison_core(slice_df, cfg.PLOT_COMPARISON_SLICE, title_suffix=suffix)
            print(f"\n  --- 切片权重图 [{suffix}] ({len(slice_df)} 样本) ---")
            _plot_weights_core(slice_df, cfg.PLOT_WEIGHTS_SLICE, title_suffix=suffix)
        else:
            print(f"\n  ⚠️ 时间切片 [{start_str}~{end_str}] 内无数据，跳过切片图")

    print(f"\n✓ 可视化完成")


# ==================== 报告 ====================

def _format_metrics_for_report(metrics: dict, indent: str = "  ") -> List[str]:
    """将指标格式化为报告行"""
    return [
        f"{indent}RMSE:       {metrics['RMSE']:.4f}",
        f"{indent}MAE:        {metrics['MAE']:.4f}",
        f"{indent}MAPE:       {metrics['MAPE']:.2f}%",
        f"{indent}Pearson R:  {metrics['Pearson_R']:.4f}",
        f"{indent}R2 Score:   {metrics['R2_Score']:.4f}",
    ]


def generate_report(layered_metrics: dict, df: pd.DataFrame):
    """生成推理报告 (.txt)"""
    print(f"\n{'=' * 70}")
    print("Step 9: 生成推理报告")
    print("=" * 70)

    lines = []
    lines.append("=" * 70)
    lines.append(f"M11 统一推理报告 V2")
    lines.append("=" * 70)
    lines.append(f"")
    lines.append(f"推理时间:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"模型类型:    {cfg.MODEL_TYPE}")
    lines.append(f"场景名称:    {cfg.scenic_name}")
    lines.append(f"测试描述:    {cfg.test_description}")
    lines.append(f"推理数据集:  {cfg.RUN_SPLIT}")
    lines.append(f"")

    # 配置信息
    lines.append(f"{'=' * 70}")
    lines.append(f"配置信息")
    lines.append(f"{'=' * 70}")
    lines.append(f"数据目录:    {cfg.DATA_DIR}")
    lines.append(f"模型路径:    {cfg.MODEL_PATH}")
    lines.append(f"推理温度:    {cfg.TEMPERATURE}")
    lines.append(f"权重聚合:    {cfg.WEIGHT_AGG_METHOD}")
    lines.append(f"预测步模式:  {cfg.PRED_STEP_MODE}")
    if cfg.ANALYSIS_DATE_START or cfg.ANALYSIS_DATE_END:
        lines.append(f"分析时间窗:  {cfg.ANALYSIS_DATE_START or '起始'} ~ {cfg.ANALYSIS_DATE_END or '结束'}")
    lines.append(f"")

    # 1. 整体指标
    lines.append(f"{'=' * 70}")
    lines.append(f"整体评估指标 [{cfg.RUN_SPLIT}] ({len(df)} 样本)")
    lines.append(f"{'=' * 70}")
    lines.extend(_format_metrics_for_report(layered_metrics['overall']))
    lines.append(f"")

    # 2. 分 split 指标
    per_split = layered_metrics.get('per_split', {})
    if per_split:
        lines.append(f"{'=' * 70}")
        lines.append(f"分数据集评估指标")
        lines.append(f"{'=' * 70}")
        for sname in SPLIT_ORDER:
            if sname not in per_split:
                continue
            cn = SPLIT_LABELS_CN.get(sname, sname)
            sub_count = len(df[df['split'] == sname]) if 'split' in df.columns else '?'
            lines.append(f"")
            lines.append(f"  [{cn}] ({sub_count} 样本)")
            lines.extend(_format_metrics_for_report(per_split[sname], indent="    "))
        lines.append(f"")

    # 3. 时间切片指标
    time_slice = layered_metrics.get('time_slice')
    if time_slice:
        lines.append(f"{'=' * 70}")
        lines.append(f"时间切片评估指标 [{time_slice['date_range']}] ({time_slice['sample_count']} 样本)")
        lines.append(f"{'=' * 70}")
        lines.extend(_format_metrics_for_report(time_slice['metrics']))
        # 切片内分 split
        if 'per_split' in time_slice:
            for sname, info in time_slice['per_split'].items():
                cn = SPLIT_LABELS_CN.get(sname, sname)
                lines.append(f"")
                lines.append(f"  [{cn}] ({info['sample_count']} 样本)")
                lines.extend(_format_metrics_for_report(info['metrics'], indent="    "))
        lines.append(f"")

    # 4. 权重统计
    lines.append(f"{'=' * 70}")
    lines.append(f"动态权重统计")
    lines.append(f"{'=' * 70}")
    for i, wname in enumerate(WEIGHT_NAMES):
        if wname in df.columns:
            w_col = df[wname].values
            lines.append(f"  {WEIGHT_LABELS_CN[i]} ({wname}):")
            lines.append(f"    均值: {w_col.mean():.4f}  标准差: {w_col.std():.4f}  "
                         f"最小: {w_col.min():.4f}  最大: {w_col.max():.4f}")
    lines.append(f"")

    # 输出文件
    lines.append(f"{'=' * 70}")
    lines.append(f"输出文件")
    lines.append(f"{'=' * 70}")
    lines.append(f"CSV:          {cfg.CSV_FILE}")
    lines.append(f"指标JSON:     {cfg.METRICS_FILE}")
    lines.append(f"对比图:       {cfg.PLOT_COMPARISON}")
    lines.append(f"权重图:       {cfg.PLOT_WEIGHTS}")
    if cfg.ANALYSIS_DATE_START or cfg.ANALYSIS_DATE_END:
        lines.append(f"切片对比图:   {cfg.PLOT_COMPARISON_SLICE}")
        lines.append(f"切片权重图:   {cfg.PLOT_WEIGHTS_SLICE}")
    lines.append(f"报告:         {cfg.REPORT_FILE}")
    lines.append(f"")
    lines.append(f"{'=' * 70}")
    lines.append(f"推理完成")
    lines.append(f"{'=' * 70}")

    report_text = '\n'.join(lines)

    with open(cfg.REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✓ 报告已保存到 {cfg.REPORT_FILE}")


# ==================== 主函数 ====================

def main():
    """统一推理主流程 V2"""
    print("\n" + "#" * 70)
    print("#  M11 统一推理系统 V2 (Unified Inference)")
    print("#  支持: 多 split 推理 | 时间切片分析 | 分层指标")
    print("#" * 70)

    start_time = datetime.now()

    # ---- Step 0: 初始化 ----
    set_seed(cfg.SEED)
    cfg.validate()
    cfg.setup_dirs()
    cfg.print_config()

    # ---- Step 1: 加载数据 ----
    (data_x, data_y, feature_map, scaler_data,
     datetime_index, split_boundaries, dataset_info) = load_split_data()

    data_loader = create_dataloader(data_x, data_y)

    # 获取 seq_len
    seq_len = dataset_info.get('seq_len', data_x.shape[1] if data_x.ndim >= 2 else 48)

    # ---- Step 2: 构建模型 ----
    print(f"\n{'=' * 70}")
    print(f"Step 2: 构建模型 [{cfg.MODEL_TYPE}]")
    print("=" * 70)

    build_model_fn, model_config_cls = load_model_and_config(
        cfg.MODEL_TYPE, str(cfg.DATA_DIR), str(cfg.MODEL_PATH)
    )
    model = build_model_fn(feature_map, model_config_cls)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量:     {total_params:,}")
    print(f"  可训练参数:   {trainable_params:,}")

    ablation_mode = getattr(model_config_cls, 'ablation_mode', 'N/A (transformer)')
    print(f"  消融模式:     {ablation_mode}")
    print(f"✓ 模型构建完成")

    # ---- Step 3: 加载权重 ----
    model = load_checkpoint(model, str(cfg.MODEL_PATH), cfg.DEVICE)

    # ---- Step 4: 推理 ----
    preds_tensor, targets_tensor, weights_tensor, inputs_tensor = run_inference(
        model, data_loader, cfg.DEVICE, cfg.TEMPERATURE
    )

    # 转换为 numpy
    weights_np = weights_tensor.numpy()   # (N, T, 5)
    inputs_np = inputs_tensor.numpy()     # (N, seq_len, F)

    # 反归一化
    metrics_calc = MetricsCalculator(scaler_data)
    preds_real = metrics_calc.inverse_transform(preds_tensor)
    targets_real = metrics_calc.inverse_transform(targets_tensor)

    # ---- Step 5: 计算分层指标 (需要先构建 CSV 用的 DataFrame) ----
    # 先构建但不打印指标，统一在 compute_layered_metrics 中处理

    # ---- Step 6: CSV 导出 ----
    df = build_and_save_csv(
        preds_real, targets_real, weights_np, inputs_np,
        feature_map, datetime_index, split_boundaries,
        dataset_info, seq_len
    )

    # ---- Step 5 (实际执行): 分层指标 ----
    layered_metrics = compute_layered_metrics(df, metrics_calc)

    # ---- Step 7: 保存指标 JSON ----
    save_metrics_json(layered_metrics)

    # ---- Step 8: 绘制可视化图表 ----
    plot_all_charts(df)

    # ---- Step 9: 生成报告 ----
    generate_report(layered_metrics, df)

    # ---- 完成 ----
    elapsed = datetime.now() - start_time
    print(f"\n{'#' * 70}")
    print(f"#  推理完成！总耗时: {elapsed}")
    print(f"#  所有结果已保存到: {cfg.OUTPUT_DIR}")
    print(f"{'#' * 70}\n")


if __name__ == '__main__':
    main()
