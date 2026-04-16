# -*- coding: utf-8 -*-
"""
M9 LSTM 动态融合预测器
LSTM-based Dynamic Fusion Predictor

基于 LSTM 的时序预测模型，替换 Autoformer / Vanilla Transformer 骨干。
保留 ContextAwareWeightNet 动态权重融合机制。

核心组件:
1. LSTMBackbone: 双向 LSTM + 全连接输出头
2. ContextAwareWeightNet: 环境感知权重网络 (复用)
3. DynamicFusionPredictor: 动态融合预测器 (顶层模型)

架构:
    Input → WeightNet(context) → sentiment × weights
    → [weighted_sentiment | context] → LSTM → FC → pred_len output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Dict, Optional, List

from config.m9_config_lstm import ModelConfig


# ============================================================================
# Part 1: ContextAwareWeightNet (复用自 m9_model_autoformer.py)
# ============================================================================

class ContextAwareWeightNet(nn.Module):
    """
    环境感知的动态权重生成网络 (复用自原版 m9_model_2.py)

    两阶段设计:
    Phase A: 特征交互 (Self-Attention)
    Phase B: 权重生成 (MLP + 1+Tanh 残差激活)

    权重范围: (0, 2), 初始接近 1.0
    """

    def __init__(self, context_dim: int, num_sentiment_groups: int, config: ModelConfig):
        super().__init__()

        self.context_dim = context_dim
        self.num_sentiment_groups = num_sentiment_groups
        self.config = config

        # Phase A: 特征交互层 (轻量级 Transformer)
        self.context_embedding = nn.Linear(context_dim, config.weight_net_hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.weight_net_hidden_dim,
            nhead=config.weight_net_num_heads,
            dim_feedforward=config.weight_net_hidden_dim * 4,
            dropout=config.weight_net_dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=False
        )

        self.context_interaction = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.weight_net_num_layers
        )

        # Phase B: 权重生成层
        self.weight_generator = nn.Sequential(
            nn.Linear(config.weight_net_hidden_dim, config.weight_net_hidden_dim),
            nn.LayerNorm(config.weight_net_hidden_dim),
            self._get_activation(config.activation),
            nn.Dropout(config.weight_net_dropout),
            nn.Linear(config.weight_net_hidden_dim, num_sentiment_groups)
        )

        # Temperature 参数
        if config.temperature_trainable:
            self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        else:
            self.register_buffer('temperature', torch.tensor(config.temperature_init))

    def _get_activation(self, name: str) -> nn.Module:
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, context_features: torch.Tensor,
                temperature: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            context_features: (Batch, Seq, Context_Dim)
            temperature: 可选的温度覆盖值
        Returns:
            weights: (Batch, Seq, Num_Groups), 范围在 (0, 2)
        """
        context_embedded = self.context_embedding(context_features)
        context_interacted = self.context_interaction(context_embedded)
        logits = self.weight_generator(context_interacted)

        temp = temperature if temperature is not None else self.temperature
        weights = 1.0 + torch.tanh(logits / temp)

        return weights


# ============================================================================
# Part 2: LSTMBackbone (替代 AutoformerBackbone / TimeSeriesBackbone)
# ============================================================================

class LSTMBackbone(nn.Module):
    """
    LSTM 时序预测骨干网络
    替代 Autoformer (Encoder-Decoder) / Vanilla Transformer Encoder

    架构:
    1. Input Projection: Linear(input_dim → d_model)
    2. LSTM: 多层双向 LSTM
    3. FC Output Head: 全连接网络将 LSTM 输出映射为 (pred_len, output_dim)

    特点:
    - 双向 LSTM 同时捕捉前向/后向时序依赖
    - 全连接输出头支持多步预测 (multi-step forecasting)
    - 结构简洁, 训练速度快, 适合作为消融实验基线
    """

    def __init__(self, input_dim: int, seq_len: int, config: ModelConfig):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.config = config
        self.pred_len = config.pred_len
        self.output_dim = config.output_dim
        self.hidden_size = config.d_model
        self.num_layers = config.lstm_num_layers
        self.bidirectional = config.lstm_bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        # Input Projection
        self.input_projection = nn.Linear(input_dim, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0.0,
            bidirectional=config.lstm_bidirectional
        )

        # LSTM 输出维度 = hidden_size * num_directions
        lstm_output_dim = config.d_model * self.num_directions

        # FC Output Head: 将 LSTM 最后时刻 (或全序列) 映射到 (pred_len * output_dim)
        fc_layers = []
        current_dim = lstm_output_dim

        for i in range(config.fc_num_layers - 1):
            fc_layers.extend([
                nn.Linear(current_dim, config.fc_hidden_dim),
                nn.LayerNorm(config.fc_hidden_dim),
                nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            current_dim = config.fc_hidden_dim

        # 最后一层: 映射到 pred_len * output_dim
        fc_layers.append(nn.Linear(current_dim, self.pred_len * self.output_dim))

        self.fc_head = nn.Sequential(*fc_layers)

        # 可选: 对 LSTM 输出做 LayerNorm
        self.output_norm = nn.LayerNorm(lstm_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LSTM 前向传播

        Args:
            x: (B, seq_len, input_dim) - 融合后的输入特征

        Returns:
            y_pred: (B, pred_len, output_dim) - 预测输出
        """
        B = x.size(0)

        # Step 1: Input Projection
        x_proj = self.input_projection(x)  # (B, seq_len, d_model)
        x_proj = self.input_norm(x_proj)

        # Step 2: LSTM Forward
        # lstm_out: (B, seq_len, hidden_size * num_directions)
        # h_n: (num_layers * num_directions, B, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x_proj)

        # Step 3: 使用最后一个时间步的输出作为序列表示
        # lstm_out[:, -1, :] 包含了双向 LSTM 前向最后步 + 后向第一步
        last_output = lstm_out[:, -1, :]  # (B, hidden_size * num_directions)
        last_output = self.output_norm(last_output)

        # Step 4: FC Head → (B, pred_len * output_dim)
        fc_out = self.fc_head(last_output)

        # Step 5: Reshape 为 (B, pred_len, output_dim)
        y_pred = fc_out.view(B, self.pred_len, self.output_dim)

        return y_pred


# ============================================================================
# Part 3: DynamicFusionPredictor (LSTM 版)
# ============================================================================

class DynamicFusionPredictor(nn.Module):
    """
    动态融合预测器 (LSTM 版)

    与 Autoformer 版 / 原版 m9_model_2.py 保持相同的外部接口:
    - forward(x, temperature) -> (y_pred, expanded_weights)
    - freeze_backbone() / freeze_weight_net() / unfreeze_all()
    - compute_weight_l1_loss() / compute_group_diversity_loss()

    内部将骨干网络替换为 LSTMBackbone

    消融模式 (ablation_mode):
    - 'full':           全开 — 情感数据 × 权重生成网络权重 → 主干网络
    - 'no_weight_net':  无权重网络 — 情感数据 × 1 → 主干网络 (权重恒为1)
    - 'no_sentiment':   全关 — 仅环境特征进入主干网络, 情感与权重网络均不参与
    """

    def __init__(self, feature_map: Dict, config: ModelConfig):
        super().__init__()

        self.feature_map = feature_map
        self.config = config
        config.validate()

        # 消融模式
        self.ablation_mode = getattr(config, 'ablation_mode', 'full')

        # 解析分组信息
        self.sentiment_groups = feature_map['sentiment_groups']
        self.num_sentiment_groups = len(self.sentiment_groups)

        # 扁平化索引
        flat_sentiment_idxs = []
        for group in self.sentiment_groups:
            flat_sentiment_idxs.extend(group)
        self.flat_sentiment_idxs = torch.tensor(flat_sentiment_idxs, dtype=torch.long)
        self.total_sentiment_dim = len(flat_sentiment_idxs)

        # 权重扩展索引
        weight_expansion_indices = []
        for group_idx, group in enumerate(self.sentiment_groups):
            weight_expansion_indices.extend([group_idx] * len(group))
        self.weight_expansion_indices = torch.tensor(weight_expansion_indices, dtype=torch.long)

        # 环境特征索引
        self.context_idxs = torch.tensor(feature_map['context_idxs'], dtype=torch.long)
        self.context_dim = len(self.context_idxs)
        self.target_idx = feature_map['target_idx']
        self.total_dim = feature_map['total_features']

        # 注册 buffer
        self.register_buffer('flat_sentiment_idxs_buffer', self.flat_sentiment_idxs)
        self.register_buffer('context_idxs_buffer', self.context_idxs)
        self.register_buffer('weight_expansion_indices_buffer', self.weight_expansion_indices)

        # 子模块初始化
        # 权重网络: 'no_sentiment' 模式下仍创建 (保持模型结构一致), 但 forward 中不使用
        self.weight_net = ContextAwareWeightNet(
            context_dim=self.context_dim,
            num_sentiment_groups=self.num_sentiment_groups,
            config=config
        )

        # 【核心替换】使用 LSTMBackbone
        # 'no_sentiment' 模式: 仅环境特征进入主干, input_dim = context_dim
        # 其他模式: 情感特征 + 环境特征, input_dim = total_sentiment_dim + context_dim
        if self.ablation_mode == 'no_sentiment':
            backbone_input_dim = self.context_dim
        else:
            backbone_input_dim = self.total_sentiment_dim + self.context_dim

        self.backbone = LSTMBackbone(
            input_dim=backbone_input_dim,
            seq_len=config.seq_len,
            config=config
        )

        # 打印消融模式信息
        _ablation_desc = {
            'full': '全开 (情感×权重网络→主干)',
            'no_weight_net': '无权重网络 (情感×1→主干)',
            'no_sentiment': '全关 (仅环境特征→主干)'
        }
        print(f"  📋 消融模式: {self.ablation_mode} — {_ablation_desc.get(self.ablation_mode, '未知')}")
        print(f"     Backbone input_dim = {backbone_input_dim}")

    def forward(self, x: torch.Tensor,
                temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 (支持三种消融模式)

        Args:
            x: (B, T, F) - 输入特征
            temperature: 温度参数

        Returns:
            y_pred: (B, pred_len, output_dim) - 预测输出
            expanded_weights: (B, T, total_sentiment_dim) - 动态权重
                - 'full' 模式: 权重网络生成的真实权重
                - 'no_weight_net' 模式: 全1权重
                - 'no_sentiment' 模式: 全0权重 (仅作占位符, 不参与计算)
        """
        B, T, F = x.shape

        # Step 1: Slicing — 提取情感特征和环境特征
        x_sentiment = torch.index_select(x, dim=2, index=self.flat_sentiment_idxs_buffer)
        x_context = torch.index_select(x, dim=2, index=self.context_idxs_buffer)

        # ==================== 消融分支 ====================

        if self.ablation_mode == 'full':
            # ---- 全开模式: 情感 × 权重网络权重 → 主干 ----
            # Step 2: Weight Generation (1+Tanh 激活)
            raw_weights = self.weight_net(x_context, temperature)

            # Step 3: Weight Expansion
            indices_expanded = self.weight_expansion_indices_buffer.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            expanded_weights = torch.gather(raw_weights, dim=-1, index=indices_expanded)

            # Step 4: Fusion
            x_sentiment_weighted = x_sentiment * expanded_weights

            # Step 5: Re-assemble
            x_fused = torch.cat([x_sentiment_weighted, x_context], dim=2)

        elif self.ablation_mode == 'no_weight_net':
            # ---- 无权重网络模式: 情感 × 1 → 主干 (权重恒为1) ----
            expanded_weights = torch.ones(B, T, self.total_sentiment_dim,
                                          device=x.device, dtype=x.dtype)

            # 情感数据不加权, 直接拼接环境特征
            x_fused = torch.cat([x_sentiment, x_context], dim=2)

        elif self.ablation_mode == 'no_sentiment':
            # ---- 全关模式: 仅环境特征 → 主干 (无情感, 无权重) ----
            expanded_weights = torch.zeros(B, T, self.total_sentiment_dim,
                                           device=x.device, dtype=x.dtype)

            # 仅环境特征进入主干
            x_fused = x_context

        else:
            raise ValueError(f"不支持的消融模式: {self.ablation_mode}")

        # Step 6: Predict (使用 LSTMBackbone)
        y_pred = self.backbone(x_fused)

        return y_pred, expanded_weights

    # ==================== 辅助方法 (与原版完全一致) ====================

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_weight_net(self):
        for param in self.weight_net.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_weight_statistics(self, expanded_weights: torch.Tensor) -> Dict:
        stats = {
            'mean': expanded_weights.mean(dim=(0, 1)).detach().cpu().numpy(),
            'std': expanded_weights.std(dim=(0, 1)).detach().cpu().numpy(),
            'max': expanded_weights.max(dim=2)[0].mean(dim=(0, 1)).detach().cpu().item(),
            'min': expanded_weights.min(dim=2)[0].mean(dim=(0, 1)).detach().cpu().item(),
            'sparsity': (expanded_weights < 0.01).float().mean().detach().cpu().item()
        }
        return stats

    def get_group_weight_statistics(self, expanded_weights: torch.Tensor) -> Dict:
        group_stats = {}
        for group_idx, group in enumerate(self.sentiment_groups):
            group_mask = (self.weight_expansion_indices_buffer == group_idx)
            group_weights = expanded_weights[:, :, group_mask]
            group_stats[f'group_{group_idx}'] = {
                'mean': group_weights.mean().detach().cpu().item(),
                'std': group_weights.std().detach().cpu().item(),
                'size': len(group)
            }
        return group_stats

    def compute_weight_l1_loss(self, expanded_weights: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(expanded_weights))

    def compute_group_diversity_loss(self, expanded_weights: torch.Tensor) -> torch.Tensor:
        group_means = []
        for group_idx in range(self.num_sentiment_groups):
            group_mask = (self.weight_expansion_indices_buffer == group_idx)
            group_weights = expanded_weights[:, :, group_mask]
            group_means.append(group_weights.mean())
        group_means = torch.stack(group_means)
        diversity_loss = -torch.var(group_means)
        return diversity_loss


# ============================================================================
# Part 4: build_model 入口函数
# ============================================================================

def build_model(feature_map: Dict, config: ModelConfig = None) -> DynamicFusionPredictor:
    """
    构建 LSTM 动态融合预测模型

    接口与 m9_model_autoformer.py / m9_model_2.py 完全一致，可直接替换使用。

    Args:
        feature_map: 特征映射字典
        config: ModelConfig 配置类

    Returns:
        model: DynamicFusionPredictor (LSTM 版)
    """
    if config is None:
        config = ModelConfig

    required_fields = ['sentiment_groups', 'context_idxs', 'target_idx', 'total_features']
    for field in required_fields:
        if field not in feature_map:
            raise ValueError(f"feature_map 缺少必需字段: {field}")

    model = DynamicFusionPredictor(feature_map, config)
    return model


# ============================================================================
# Part 5: 测试代码
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LSTM 动态融合预测器 架构测试 (含消融模式)")
    print("=" * 70 + "\n")

    # 测试用 feature_map
    feature_map_test = {
        'sentiment_groups': [list(range(0, 5)), list(range(5, 10))],
        'context_idxs': list(range(10, 15)),
        'target_idx': 15,
        'feature_names': [f'feature_{i}' for i in range(16)],
        'total_features': 16
    }

    x_test = torch.randn(4, 30, 16)

    # ==================== 测试三种消融模式 ====================
    ablation_modes = ['full', 'no_weight_net', 'no_sentiment']
    ablation_descs = {
        'full': '全开 (情感×权重网络→主干)',
        'no_weight_net': '无权重网络 (情感×1→主干)',
        'no_sentiment': '全关 (仅环境特征→主干)'
    }

    for mode in ablation_modes:
        print(f"\n{'─' * 70}")
        print(f"  测试消融模式: {mode} — {ablation_descs[mode]}")
        print(f"{'─' * 70}")

        # 设置较短的序列长度进行测试
        ModelConfig.seq_len = 30
        ModelConfig.pred_len = 7
        ModelConfig.ablation_mode = mode

        config = ModelConfig
        model = build_model(feature_map_test, config)

        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")

        # 测试前向传播
        y_pred, expanded_weights = model(x_test)

        print(f"  输入形状: {x_test.shape}")
        print(f"  输出形状: {y_pred.shape}  (期望: [4, {ModelConfig.pred_len}, {ModelConfig.output_dim}])")
        print(f"  权重形状: {expanded_weights.shape}")

        if mode == 'full':
            print(f"  权重范围: Min={expanded_weights.min().item():.4f}, "
                  f"Max={expanded_weights.max().item():.4f}, "
                  f"Mean={expanded_weights.mean().item():.4f} (初始应接近 1.0)")
        elif mode == 'no_weight_net':
            assert (expanded_weights == 1.0).all(), "no_weight_net 模式权重应全为 1"
            print(f"  权重验证: 全部为 1.0 ✓")
        elif mode == 'no_sentiment':
            assert (expanded_weights == 0.0).all(), "no_sentiment 模式权重应全为 0"
            print(f"  权重验证: 全部为 0.0 ✓ (占位符)")

        print(f"  ✓ 模式 '{mode}' 测试通过")

    # 测试各子模块
    print(f"\n{'─' * 70}")
    print(f"子模块测试:")
    print(f"  LSTMBackbone: ✓")
    print(f"  ContextAwareWeightNet: ✓")
    print(f"  DynamicFusionPredictor (3种消融模式): ✓")

    print(f"\n{'=' * 70}")
    print("✓ 所有测试通过！LSTM 模型可正常使用。")
    print(f"{'=' * 70}\n")
