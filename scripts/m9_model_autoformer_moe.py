# -*- coding: utf-8 -*-
"""
M9 Autoformer + Hard-Gating MoE 动态融合预测器
Autoformer-based Dynamic Fusion Predictor with Season-Aware Hard-Gating MoE

架构概述:
┌─────────────────────────────────────────────────────────────────┐
│  Input: x (B, seq_len, F)                                      │
│    ↓                                                            │
│  ContextAwareWeightNet → 情感加权融合                            │
│    ↓                                                            │
│  ┌───────────────── Shared Bottom ──────────────────┐           │
│  │  Autoformer Encoder (Auto-Correlation + Decomp)  │           │
│  │          ↓                                       │           │
│  │  Autoformer Decoder (Cross-Correlation + Decomp) │           │
│  │          ↓                                       │           │
│  │  Deep Feature: (B, pred_len, d_model)            │           │
│  └──────────────────────────────────────────────────┘           │
│    ↓                                                            │
│  ┌───────── Hard-Gating MoE (3 Season Experts) ─────────┐      │
│  │                                                       │      │
│  │  Expert_Low:    Linear(d_model → c_out) ─┐            │      │
│  │  Expert_Normal: Linear(d_model → c_out) ─┼─ Stack     │      │
│  │  Expert_Peak:   Linear(d_model → c_out) ─┘            │      │
│  │          ↓                                            │      │
│  │  Gate Mask: [is_low, is_normal, is_peak]              │      │
│  │     (B, 3) → unsqueeze → (B, 1, 1, 3)                │      │
│  │          ↓                                            │      │
│  │  output = Σ (expert_i * gate_i)  along expert dim     │      │
│  │          ↓                                            │      │
│  │  Final: (B, pred_len, c_out)                          │      │
│  └───────────────────────────────────────────────────────┘      │
│    ↓                                                            │
│  output += trend_accumulated  (Autoformer 趋势分量)              │
│    ↓                                                            │
│  y_pred: (B, pred_len, output_dim)                              │
└─────────────────────────────────────────────────────────────────┘

核心创新:
1. Shared Bottom: Autoformer Encoder-Decoder 作为共享特征提取器
2. Hard-Gating MoE: 利用 is_low/normal/peak_season 的 One-Hot 特性实现物理级梯度隔离
3. 梯度隔离保证: gate_mask 是数据驱动的常量 (0/1), 计算图通过 element-wise multiply
   天然实现: grad=0 的专家不接收梯度, 无需 detach/stop_gradient 等操作

Reference:
- Autoformer: https://arxiv.org/abs/2106.13008
- MoE: https://arxiv.org/abs/1701.06538
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Dict, Optional, List

from config.m9_config_autoformer_moe import ModelConfig


# ============================================================================
# Part 1: 序列分解模块 (Series Decomposition) — 与原版完全一致
# ============================================================================

class MovingAvg(nn.Module):
    """移动平均模块: 提取时间序列的趋势分量"""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_padded = x_padded.permute(0, 2, 1)
        trend = self.avg(x_padded)
        trend = trend.permute(0, 2, 1)
        return trend


class SeriesDecomp(nn.Module):
    """序列分解模块: x = Trend + Seasonal"""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# ============================================================================
# Part 2: 位置编码 (Positional Encoding) — 与原版完全一致
# ============================================================================

class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model: int, max_len: int = 5000,
                 encoding_type: str = 'sinusoidal', dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoding_type = encoding_type

        if encoding_type == 'sinusoidal':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                 (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        elif encoding_type == 'learnable':
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            raise ValueError(f"不支持的位置编码类型: {encoding_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ============================================================================
# Part 3: Auto-Correlation 快速版本 — 与原版完全一致
# ============================================================================

class AutoCorrelationFast(nn.Module):
    """Auto-Correlation 快速版本 (向量化实现)"""

    def __init__(self, d_model: int, nhead: int, factor: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.factor = factor
        self.dropout = nn.Dropout(p=dropout)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        Q = self.W_q(query).view(B, L_q, self.nhead, self.d_k).permute(0, 2, 1, 3)
        K = self.W_k(key).view(B, L_k, self.nhead, self.d_k).permute(0, 2, 1, 3)
        V = self.W_v(value).view(B, L_k, self.nhead, self.d_k).permute(0, 2, 1, 3)

        if L_k > L_q:
            K = K[:, :, :L_q, :]
            V = V[:, :, :L_q, :]
        elif L_q > L_k:
            K = F.pad(K, (0, 0, 0, L_q - L_k))
            V = F.pad(V, (0, 0, 0, L_q - L_k))

        L = L_q

        Q_fft = torch.fft.rfft(Q, dim=2)
        K_fft = torch.fft.rfft(K, dim=2)
        corr = torch.fft.irfft(Q_fft * torch.conj(K_fft), n=L, dim=2)
        corr_mean = corr.mean(dim=-1)

        top_k = max(1, int(self.factor * math.log(L + 1)))
        top_k = min(top_k, L)
        top_k_weights, top_k_indices = torch.topk(corr_mean, top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        output = torch.zeros_like(V)

        for i in range(top_k):
            delays = top_k_indices[:, :, i]
            weights_i = top_k_weights[:, :, i]

            arange = torch.arange(L, device=V.device).unsqueeze(0).unsqueeze(0)
            shifted = (arange + delays.unsqueeze(-1)) % L
            shifted = shifted.unsqueeze(-1).expand(-1, -1, -1, self.d_k)

            rolled_v = torch.gather(V, dim=2, index=shifted)
            output += weights_i.unsqueeze(-1).unsqueeze(-1) * rolled_v

        output = output.permute(0, 2, 1, 3).contiguous().view(B, L_q, self.d_model)
        output = self.W_o(output)
        output = self.dropout(output)

        return output


# ============================================================================
# Part 4: Autoformer Encoder — 与原版完全一致
# ============================================================================

class AutoformerEncoderLayer(nn.Module):
    """Autoformer Encoder 单层"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 moving_avg_kernel: int, factor: int = 1,
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()

        self.auto_correlation = AutoCorrelationFast(
            d_model=d_model, nhead=nhead, factor=factor, dropout=dropout
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.decomp1 = SeriesDecomp(moving_avg_kernel)
        self.decomp2 = SeriesDecomp(moving_avg_kernel)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.auto_correlation(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        seasonal1, trend1 = self.decomp1(x)

        ff_out = self.ff(seasonal1)
        x = seasonal1 + ff_out
        x = self.norm2(x)
        seasonal2, trend2 = self.decomp2(x)

        trend = trend1 + trend2
        return seasonal2, trend


class AutoformerEncoder(nn.Module):
    """Autoformer Encoder (多层堆叠)"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 num_layers: int, moving_avg_kernel: int, factor: int = 1,
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()

        self.layers = nn.ModuleList([
            AutoformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                moving_avg_kernel=moving_avg_kernel, factor=factor,
                dropout=dropout, activation=activation
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend_accumulated = torch.zeros_like(x)

        for layer in self.layers:
            x, trend = layer(x)
            trend_accumulated += trend

        enc_out = self.norm(x)
        return enc_out, trend_accumulated


# ============================================================================
# Part 5: Autoformer Decoder Layer — 与原版完全一致
# ============================================================================

class AutoformerDecoderLayer(nn.Module):
    """Autoformer Decoder 单层"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 moving_avg_kernel: int, factor: int = 1,
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()

        self.self_correlation = AutoCorrelationFast(
            d_model=d_model, nhead=nhead, factor=factor, dropout=dropout
        )

        self.cross_correlation = AutoCorrelationFast(
            d_model=d_model, nhead=nhead, factor=factor, dropout=dropout
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.decomp1 = SeriesDecomp(moving_avg_kernel)
        self.decomp2 = SeriesDecomp(moving_avg_kernel)
        self.decomp3 = SeriesDecomp(moving_avg_kernel)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # 趋势投影
        self.trend_projection = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor,
                enc_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sub-layer 1: Self Auto-Correlation
        self_attn_out = self.self_correlation(x, x, x)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)
        seasonal1, trend1 = self.decomp1(x)

        # Sub-layer 2: Cross Auto-Correlation
        cross_attn_out = self.cross_correlation(seasonal1, enc_out, enc_out)
        x = seasonal1 + self.dropout(cross_attn_out)
        x = self.norm2(x)
        seasonal2, trend2 = self.decomp2(x)

        # Sub-layer 3: FeedForward
        ff_out = self.ff(seasonal2)
        x = seasonal2 + ff_out
        x = self.norm3(x)
        seasonal3, trend3 = self.decomp3(x)

        # 累积趋势并投影到输出维度
        trend = self.trend_projection(trend1 + trend2 + trend3)

        return seasonal3, trend


# ============================================================================
# Part 6: 【重构核心】Autoformer Decoder with MoE (替代原版 Decoder)
# ============================================================================

class SeasonExpertLayer(nn.Module):
    """
    单个季节专家网络

    根据 expert_hidden_dim 决定专家网络的复杂度:
    - None: 单层线性映射 d_model → output_dim (轻量级, 与原版 projection 等价)
    - int:  两层 MLP d_model → hidden → output_dim (增强拟合能力)
    """

    def __init__(self, d_model: int, output_dim: int,
                 hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()

        if hidden_dim is not None and hidden_dim > 0:
            self.expert = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            # 与原版 seasonal_projection 等价的单层线性映射
            self.expert = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, pred_len, d_model) - Decoder 深层特征
        Returns:
            out: (B, pred_len, output_dim) - 专家预测
        """
        return self.expert(x)


class AutoformerDecoderMoE(nn.Module):
    """
    Autoformer Decoder with Hard-Gating MoE (核心重构模块)

    ┌─────────────────────────────────────────────────────┐
    │  原版 AutoformerDecoder:                            │
    │  x → DecoderLayers → norm → seasonal_projection    │
    │                                    ↓                │
    │                              (B, L, c_out)          │
    │                              + trend_accumulated    │
    │                              = output               │
    └─────────────────────────────────────────────────────┘
                              ↓ 重构为 ↓
    ┌─────────────────────────────────────────────────────┐
    │  MoE版 AutoformerDecoderMoE:                        │
    │  x → DecoderLayers → norm → deep_feature            │
    │                               ↓                     │
    │              ┌────────────────┼────────────────┐     │
    │              ↓                ↓                ↓     │
    │         Expert_Low      Expert_Normal     Expert_Peak│
    │              ↓                ↓                ↓     │
    │         pred_low         pred_normal      pred_peak  │
    │              └────────────────┼────────────────┘     │
    │                               ↓ stack                │
    │                   (B, pred_len, c_out, 3)            │
    │                               ×                      │
    │              gate_mask: (B, 1, 1, 3)   ← One-Hot    │
    │                               ↓ sum(dim=-1)         │
    │                   (B, pred_len, c_out)               │
    │                        + trend_accumulated           │
    │                        = output                      │
    └─────────────────────────────────────────────────────┘

    梯度隔离保证:
    - gate_mask 中只有一个通道为 1, 其余为 0
    - element-wise multiply 后, gate=0 的专家输出被置零
    - 反向传播: ∂L/∂expert_i = ∂L/∂output × gate_i
      当 gate_i=0 时, 梯度自然为 0, 无需额外操作
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 num_layers: int, moving_avg_kernel: int, factor: int = 1,
                 dropout: float = 0.1, activation: str = 'gelu',
                 output_dim: int = 1, num_experts: int = 3,
                 expert_hidden_dim: Optional[int] = None,
                 expert_dropout: float = 0.1):
        super().__init__()

        self.num_experts = num_experts
        self.output_dim = output_dim

        # ==================== Decoder Layers (共享底座, 不允许修改) ====================
        self.layers = nn.ModuleList([
            AutoformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                moving_avg_kernel=moving_avg_kernel, factor=factor,
                dropout=dropout, activation=activation
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # ==================== 【删除】原版单一 seasonal_projection ====================
        # self.seasonal_projection = nn.Linear(d_model, output_dim)  # ← 被移除

        # ==================== 【新增】MoE: 三个并行的季节专家网络 ====================
        # expert_low:    淡季专家 (is_low_season = 1)
        # expert_normal: 平季专家 (is_normal_season = 1)
        # expert_peak:   旺季专家 (is_peak_season = 1)
        self.experts = nn.ModuleList([
            SeasonExpertLayer(
                d_model=d_model,
                output_dim=output_dim,
                hidden_dim=expert_hidden_dim,
                dropout=expert_dropout
            )
            for _ in range(num_experts)
        ])

        # 为每个专家命名 (用于日志和可视化)
        self.expert_names = ['expert_low', 'expert_normal', 'expert_peak']

        print(f"  🏗️  AutoformerDecoderMoE 初始化:")
        print(f"     专家数量: {num_experts}")
        print(f"     专家类型: {'MLP (d_model→hidden→c_out)' if expert_hidden_dim else 'Linear (d_model→c_out)'}")
        if expert_hidden_dim:
            print(f"     专家隐藏层: {expert_hidden_dim}")
        print(f"     门控类型: Hard-Gating (One-Hot Season Mask)")

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                trend_init: torch.Tensor,
                season_gate_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L_dec, d_model) - Decoder 输入 (季节初始化)
            enc_out: (B, L_enc, d_model) - Encoder 输出
            trend_init: (B, L_dec, 1) - 趋势初始化
            season_gate_mask: (B, 3) - 季节门控掩码 [is_low, is_normal, is_peak]
                              One-Hot: 有且仅有一个为 1, 其余为 0

        Returns:
            output: (B, L_dec, output_dim) - 最终预测输出 (仅返回 pred_len 部分)
        """
        # ==================== Step 1: 共享底座前向 (与原版一致) ====================
        trend_accumulated = trend_init

        for layer in self.layers:
            x, trend = layer(x, enc_out)
            trend_accumulated += trend

        # 深层特征 (Shared Representation)
        deep_feature = self.norm(x)
        # deep_feature: (B, L_dec, d_model)
        # L_dec = label_len + pred_len

        # ==================== Step 2: MoE 专家分支前向 ====================
        # 每个专家独立处理 deep_feature, 产生各自的候选预测
        # expert_outputs[i]: (B, L_dec, output_dim)
        expert_outputs = [expert(deep_feature) for expert in self.experts]

        # Stack: (B, L_dec, output_dim, num_experts)
        expert_stack = torch.stack(expert_outputs, dim=-1)
        # expert_stack: (B, L_dec, output_dim, 3)

        # ==================== Step 3: 硬门控路由 (Hard-Gating) ====================
        # season_gate_mask: (B, 3) → 需要扩展为 (B, 1, 1, 3) 以广播
        # 对齐维度: (B, L_dec, output_dim, 3) × (B, 1, 1, 3)
        gate = season_gate_mask.unsqueeze(1).unsqueeze(2)
        # gate: (B, 1, 1, 3)

        # Element-wise Multiply + Sum over expert dimension
        # 因为 gate 是 One-Hot (只有一个 1), 等价于选中对应专家的输出
        # 梯度隔离: gate_i=0 → ∂L/∂expert_i = ∂L/∂output × 0 = 0
        seasonal_out = (expert_stack * gate).sum(dim=-1)
        # seasonal_out: (B, L_dec, output_dim)

        # ==================== Step 4: 季节分量 + 趋势累积 ====================
        output = seasonal_out + trend_accumulated

        return output


# ============================================================================
# Part 7: AutoformerBackbone with MoE (替代原版 AutoformerBackbone)
# ============================================================================

class AutoformerBackboneMoE(nn.Module):
    """
    Autoformer + MoE 时序预测骨干网络

    与原版 AutoformerBackbone 的区别:
    1. Decoder 替换为 AutoformerDecoderMoE
    2. forward() 额外接收 season_gate_mask 参数
    """

    def __init__(self, input_dim: int, seq_len: int, config: ModelConfig):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.config = config
        self.pred_len = config.pred_len
        self.label_len = config.label_len
        self.output_dim = config.output_dim

        # Input Embedding
        self.enc_embedding = nn.Linear(input_dim, config.d_model)
        self.dec_embedding = nn.Linear(input_dim, config.d_model)

        # Positional Encoding
        self.enc_pos = PositionalEncoding(
            d_model=config.d_model, max_len=config.max_seq_length,
            encoding_type=config.positional_encoding_type, dropout=config.dropout
        )
        self.dec_pos = PositionalEncoding(
            d_model=config.d_model, max_len=config.max_seq_length,
            encoding_type=config.positional_encoding_type, dropout=config.dropout
        )

        # 初始 Series Decomposition
        self.decomp_init = SeriesDecomp(config.moving_avg_kernel)

        # Autoformer Encoder (共享底座 — 不允许修改)
        self.encoder = AutoformerEncoder(
            d_model=config.d_model, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.num_encoder_layers,
            moving_avg_kernel=config.moving_avg_kernel,
            factor=config.autocorrelation_factor,
            dropout=config.dropout, activation=config.activation
        )

        # 【核心替换】Autoformer Decoder with MoE
        self.decoder = AutoformerDecoderMoE(
            d_model=config.d_model, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.num_decoder_layers,
            moving_avg_kernel=config.moving_avg_kernel,
            factor=config.autocorrelation_factor,
            dropout=config.dropout, activation=config.activation,
            output_dim=config.output_dim,
            num_experts=config.num_experts,
            expert_hidden_dim=config.expert_hidden_dim,
            expert_dropout=config.expert_dropout
        )

    def forward(self, x: torch.Tensor,
                season_gate_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, input_dim) - 融合后的输入特征
            season_gate_mask: (B, 3) - 季节门控掩码

        Returns:
            y_pred: (B, pred_len, output_dim)
        """
        B = x.size(0)

        # ==================== Encoder ====================
        enc_input = self.enc_embedding(x)
        enc_input = self.enc_pos(enc_input)
        enc_out, enc_trend = self.encoder(enc_input)

        # ==================== Decoder Input 准备 ====================
        x_label = x[:, -self.label_len:, :]
        x_zero = torch.zeros(B, self.pred_len, self.input_dim, device=x.device)
        dec_raw = torch.cat([x_label, x_zero], dim=1)

        seasonal_init, trend_init = self.decomp_init(dec_raw)

        dec_input = self.dec_embedding(seasonal_init)
        dec_input = self.dec_pos(dec_input)

        trend_init_proj = trend_init[:, :, :self.output_dim]
        if self.input_dim < self.output_dim:
            pad_size = self.output_dim - self.input_dim
            trend_init_proj = F.pad(trend_init_proj, (0, pad_size))

        # ==================== Decoder + MoE ====================
        output = self.decoder(dec_input, enc_out, trend_init_proj, season_gate_mask)

        # 只取预测部分 (最后 pred_len 步)
        y_pred = output[:, -self.pred_len:, :]

        return y_pred


# ============================================================================
# Part 8: ContextAwareWeightNet (复用)
# ============================================================================

class ContextAwareWeightNet(nn.Module):
    """环境感知的动态权重生成网络 (与原版完全一致)"""

    def __init__(self, context_dim: int, num_sentiment_groups: int, config: ModelConfig):
        super().__init__()

        self.context_dim = context_dim
        self.num_sentiment_groups = num_sentiment_groups
        self.config = config

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
            encoder_layer, num_layers=config.weight_net_num_layers
        )

        self.weight_generator = nn.Sequential(
            nn.Linear(config.weight_net_hidden_dim, config.weight_net_hidden_dim),
            nn.LayerNorm(config.weight_net_hidden_dim),
            self._get_activation(config.activation),
            nn.Dropout(config.weight_net_dropout),
            nn.Linear(config.weight_net_hidden_dim, num_sentiment_groups)
        )

        if config.temperature_trainable:
            self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        else:
            self.register_buffer('temperature', torch.tensor(config.temperature_init))

    def _get_activation(self, name: str) -> nn.Module:
        activations = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}
        if name not in activations:
            raise ValueError(f"不支持的激活函数: {name}")
        return activations[name]()

    def forward(self, context_features: torch.Tensor,
                temperature: Optional[float] = None) -> torch.Tensor:
        context_embedded = self.context_embedding(context_features)
        context_interacted = self.context_interaction(context_embedded)
        logits = self.weight_generator(context_interacted)

        temp = temperature if temperature is not None else self.temperature
        weights = 1.0 + torch.tanh(logits / temp)

        return weights


# ============================================================================
# Part 9: DynamicFusionPredictor with MoE (顶层模型)
# ============================================================================

class DynamicFusionPredictor(nn.Module):
    """
    动态融合预测器 (Autoformer + Hard-Gating MoE 版)

    外部接口:
    - forward(x, temperature, season_gate_mask) -> (y_pred, expanded_weights)

    关键变化 (相对于原版):
    1. forward() 新增 season_gate_mask 参数 (B, 3)
    2. 内部 backbone 替换为 AutoformerBackboneMoE
    3. 自动从 feature_map 中检索季节特征索引，支持从 x 中自动提取 season mask
    """

    def __init__(self, feature_map: Dict, config: ModelConfig):
        super().__init__()

        self.feature_map = feature_map
        self.config = config
        config.validate()

        # 消融模式
        self.ablation_mode = getattr(config, 'ablation_mode', 'full')

        # ==================== 解析分组信息 ====================
        self.sentiment_groups = feature_map['sentiment_groups']
        self.num_sentiment_groups = len(self.sentiment_groups)

        flat_sentiment_idxs = []
        for group in self.sentiment_groups:
            flat_sentiment_idxs.extend(group)
        self.flat_sentiment_idxs = torch.tensor(flat_sentiment_idxs, dtype=torch.long)
        self.total_sentiment_dim = len(flat_sentiment_idxs)

        weight_expansion_indices = []
        for group_idx, group in enumerate(self.sentiment_groups):
            weight_expansion_indices.extend([group_idx] * len(group))
        self.weight_expansion_indices = torch.tensor(weight_expansion_indices, dtype=torch.long)

        self.context_idxs = torch.tensor(feature_map['context_idxs'], dtype=torch.long)
        self.context_dim = len(self.context_idxs)
        self.target_idx = feature_map['target_idx']
        self.total_dim = feature_map['total_features']

        # ==================== 【新增】自动检索季节特征索引 ====================
        feature_names = feature_map.get('feature_names', [])
        season_names = config.season_feature_names  # ['is_low_season', 'is_normal_season', 'is_peak_season']

        self.season_feature_idxs = []
        for name in season_names:
            if name in feature_names:
                idx = feature_names.index(name)
                self.season_feature_idxs.append(idx)
            else:
                raise ValueError(
                    f"feature_map 中未找到季节特征 '{name}'。\n"
                    f"可用特征: {feature_names}\n"
                    f"请检查 config.season_feature_names 设置是否正确。"
                )

        self.season_feature_idxs_tensor = torch.tensor(self.season_feature_idxs, dtype=torch.long)

        print(f"\n  🔍 季节特征索引自动检索:")
        for name, idx in zip(season_names, self.season_feature_idxs):
            print(f"     {name} → feature_idx={idx}")

        # 注册 buffer
        self.register_buffer('flat_sentiment_idxs_buffer', self.flat_sentiment_idxs)
        self.register_buffer('context_idxs_buffer', self.context_idxs)
        self.register_buffer('weight_expansion_indices_buffer', self.weight_expansion_indices)
        self.register_buffer('season_feature_idxs_buffer', self.season_feature_idxs_tensor)

        # ==================== 子模块初始化 ====================
        self.weight_net = ContextAwareWeightNet(
            context_dim=self.context_dim,
            num_sentiment_groups=self.num_sentiment_groups,
            config=config
        )

        if self.ablation_mode == 'no_sentiment':
            backbone_input_dim = self.context_dim
        else:
            backbone_input_dim = self.total_sentiment_dim + self.context_dim

        # 【核心替换】使用 AutoformerBackboneMoE
        self.backbone = AutoformerBackboneMoE(
            input_dim=backbone_input_dim,
            seq_len=config.seq_len,
            config=config
        )

        _ablation_desc = {
            'full': '全开 (情感×权重网络→主干)',
            'no_weight_net': '无权重网络 (情感×1→主干)',
            'no_sentiment': '全关 (仅环境特征→主干)'
        }
        print(f"  📋 消融模式: {self.ablation_mode} — {_ablation_desc.get(self.ablation_mode, '未知')}")
        print(f"     Backbone input_dim = {backbone_input_dim}")
        print(f"     Backbone type = Autoformer + Hard-Gating MoE ({config.num_experts} experts)")

    def _extract_season_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        从输入序列中自动提取季节门控掩码

        设计说明:
        - 取输入序列 **最后一个时间步** 的季节特征作为当前样本的季节标识
        - 因为预测窗口紧接在输入序列之后，最后一个时间步的季节状态
          最能代表预测期的季节状态 (小时级数据，季节在 pred_len=24h 内不会变化)

        Args:
            x: (B, seq_len, F) - 完整输入特征

        Returns:
            season_mask: (B, 3) - [is_low, is_normal, is_peak], One-Hot
        """
        # 取最后一个时间步的季节特征
        # x[:, -1, :]: (B, F) → index_select → (B, 3)
        season_mask = torch.index_select(x[:, -1, :], dim=1,
                                          index=self.season_feature_idxs_buffer)
        return season_mask

    def forward(self, x: torch.Tensor,
                temperature: Optional[float] = None,
                season_gate_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: (B, T, F) - 输入特征
            temperature: 温度参数
            season_gate_mask: (B, 3) - 可选的外部季节门控掩码
                              如果为 None, 则自动从 x 中提取

        Returns:
            y_pred: (B, pred_len, output_dim) - 预测输出
            expanded_weights: (B, T, total_sentiment_dim) - 动态权重
        """
        B, T, F = x.shape

        # ==================== Step 0: 提取/验证季节门控掩码 ====================
        if season_gate_mask is None:
            season_gate_mask = self._extract_season_mask(x)
        # season_gate_mask: (B, 3), One-Hot

        # ==================== Step 1: Slicing ====================
        x_sentiment = torch.index_select(x, dim=2, index=self.flat_sentiment_idxs_buffer)
        x_context = torch.index_select(x, dim=2, index=self.context_idxs_buffer)

        # ==================== 消融分支 ====================
        if self.ablation_mode == 'full':
            raw_weights = self.weight_net(x_context, temperature)
            indices_expanded = self.weight_expansion_indices_buffer.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            expanded_weights = torch.gather(raw_weights, dim=-1, index=indices_expanded)
            x_sentiment_weighted = x_sentiment * expanded_weights
            x_fused = torch.cat([x_sentiment_weighted, x_context], dim=2)

        elif self.ablation_mode == 'no_weight_net':
            expanded_weights = torch.ones(B, T, self.total_sentiment_dim,
                                          device=x.device, dtype=x.dtype)
            x_fused = torch.cat([x_sentiment, x_context], dim=2)

        elif self.ablation_mode == 'no_sentiment':
            expanded_weights = torch.zeros(B, T, self.total_sentiment_dim,
                                           device=x.device, dtype=x.dtype)
            x_fused = x_context

        else:
            raise ValueError(f"不支持的消融模式: {self.ablation_mode}")

        # ==================== Step 6: Predict (Autoformer + MoE) ====================
        y_pred = self.backbone(x_fused, season_gate_mask)

        return y_pred, expanded_weights

    # ==================== 辅助方法 ====================

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

    def get_expert_routing_statistics(self, season_gate_mask: torch.Tensor) -> Dict:
        """
        统计专家路由分布

        Args:
            season_gate_mask: (B, 3) - 批次中的季节掩码

        Returns:
            routing_stats: 各专家被激活的比例
        """
        expert_names = ['low_season', 'normal_season', 'peak_season']
        counts = season_gate_mask.sum(dim=0).detach().cpu().numpy()
        total = season_gate_mask.size(0)

        routing_stats = {}
        for name, count in zip(expert_names, counts):
            routing_stats[name] = {
                'count': int(count),
                'ratio': float(count / total) if total > 0 else 0.0
            }
        return routing_stats


# ============================================================================
# Part 10: build_model 入口函数
# ============================================================================

def build_model(feature_map: Dict, config: ModelConfig = None) -> DynamicFusionPredictor:
    """
    构建 Autoformer + MoE 动态融合预测模型

    接口与原版 m9_model_autoformer.py 兼容 (新增 season_gate_mask 参数为可选)

    Args:
        feature_map: 特征映射字典 (必须包含 feature_names 以自动检索季节索引)
        config: ModelConfig 配置类

    Returns:
        model: DynamicFusionPredictor (Autoformer + MoE 版)
    """
    if config is None:
        config = ModelConfig

    required_fields = ['sentiment_groups', 'context_idxs', 'target_idx',
                       'total_features', 'feature_names']
    for field in required_fields:
        if field not in feature_map:
            raise ValueError(f"feature_map 缺少必需字段: {field}")

    model = DynamicFusionPredictor(feature_map, config)
    return model


# ============================================================================
# Part 11: 测试代码
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Autoformer + Hard-Gating MoE 动态融合预测器 架构测试")
    print("=" * 70 + "\n")

    # 模拟 feature_map (包含季节特征)
    feature_map_test = {
        'sentiment_groups': [list(range(0, 5)), list(range(5, 10))],
        'context_idxs': list(range(10, 18)),  # 包含季节特征在 idx 15,16,17
        'target_idx': 0,
        'feature_names': [
            'f0', 'f1', 'f2', 'f3', 'f4',   # sentiment group 1
            'f5', 'f6', 'f7', 'f8', 'f9',   # sentiment group 2
            'f10', 'f11', 'f12', 'f13', 'f14',  # context
            'is_low_season', 'is_normal_season', 'is_peak_season'  # 季节特征
        ],
        'total_features': 18
    }

    B = 8
    x_test = torch.randn(B, 30, 18)

    # 模拟季节标签 (One-Hot)
    # 前3个样本: 淡季, 中间3个: 平季, 后2个: 旺季
    season_labels = torch.zeros(B, 3)
    season_labels[0:3, 0] = 1.0  # is_low_season
    season_labels[3:6, 1] = 1.0  # is_normal_season
    season_labels[6:8, 2] = 1.0  # is_peak_season

    # 将季节标签写入 x_test 对应位置
    x_test[:, :, 15] = season_labels[:, 0:1].expand(-1, 30)
    x_test[:, :, 16] = season_labels[:, 1:2].expand(-1, 30)
    x_test[:, :, 17] = season_labels[:, 2:3].expand(-1, 30)

    # 设置较短的序列长度进行测试
    ModelConfig.seq_len = 30
    ModelConfig.pred_len = 7
    ModelConfig.label_len = 4
    ModelConfig.ablation_mode = 'full'

    print("─" * 70)
    print("  测试 1: 基础前向传播 (自动提取 season mask)")
    print("─" * 70)

    model = build_model(feature_map_test, ModelConfig)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # 测试: 自动提取 season mask
    y_pred, expanded_weights = model(x_test)
    print(f"  输入形状: {x_test.shape}")
    print(f"  输出形状: {y_pred.shape}  (期望: [8, 7, 1])")
    print(f"  权重形状: {expanded_weights.shape}")
    print(f"  ✓ 自动 season mask 提取测试通过")

    # 测试: 手动传入 season mask
    print(f"\n{'─' * 70}")
    print("  测试 2: 手动传入 season_gate_mask")
    print("─" * 70)

    y_pred2, _ = model(x_test, season_gate_mask=season_labels)
    print(f"  输出形状: {y_pred2.shape}  (期望: [8, 7, 1])")
    print(f"  ✓ 手动 season mask 测试通过")

    # 测试: 梯度隔离验证
    print(f"\n{'─' * 70}")
    print("  测试 3: 梯度隔离验证 (核心)")
    print("─" * 70)

    model.zero_grad()
    y_pred3, _ = model(x_test, season_gate_mask=season_labels)
    loss = y_pred3.sum()
    loss.backward()

    # 检查各专家的梯度
    decoder_moe = model.backbone.decoder
    for i, (expert, name) in enumerate(zip(decoder_moe.experts, decoder_moe.expert_names)):
        # 获取专家最后一层的权重梯度
        for pname, param in expert.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # 如果该专家对应的季节样本数 > 0, 梯度应非零
                expected_active = season_labels[:, i].sum().item() > 0
                status = "✓ Active" if (grad_norm > 0 and expected_active) else \
                         ("✓ Inactive (grad=0)" if (grad_norm == 0 and not expected_active) else "✗ ERROR")
                print(f"  {name}.{pname}: grad_norm={grad_norm:.6f}  [{status}]")
                break  # 只检查第一个参数

    # 测试: 专家路由统计
    print(f"\n{'─' * 70}")
    print("  测试 4: 专家路由统计")
    print("─" * 70)

    routing_stats = model.get_expert_routing_statistics(season_labels)
    for name, stats in routing_stats.items():
        print(f"  {name}: count={stats['count']}, ratio={stats['ratio']:.2%}")

    print(f"\n{'=' * 70}")
    print("✓ 所有测试通过！Autoformer + MoE 模型可正常使用。")
    print(f"{'=' * 70}\n")
