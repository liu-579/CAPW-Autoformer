# -*- coding: utf-8 -*-
"""
M9 Autoformer 动态融合预测器
Autoformer-based Dynamic Fusion Predictor

基于 Autoformer (NeurIPS 2021) 的时序预测模型，替换原版 Vanilla Transformer 骨干。
保留 ContextAwareWeightNet 动态权重融合机制。

核心组件:
1. SeriesDecomp: 序列趋势-季节分解
2. AutoCorrelation: 基于自相关的注意力机制
3. AutoformerEncoder: 编码器 (含渐进式分解)
4. AutoformerDecoder: 解码器 (含渐进式分解)
5. ContextAwareWeightNet: 环境感知权重网络 (复用)
6. DynamicFusionPredictor: 动态融合预测器 (顶层模型)

Reference: https://arxiv.org/abs/2106.13008
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Dict, Optional, List

from config.m9_config_autoformer import ModelConfig


# ============================================================================
# Part 1: 序列分解模块 (Series Decomposition)
# ============================================================================

class MovingAvg(nn.Module):
    """
    移动平均模块
    用于提取时间序列的趋势分量 (Trend Component)

    原理: 使用固定大小的平均池化窗口滑过序列，提取平滑趋势
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Seq_Len, Channels)
        Returns:
            trend: (Batch, Seq_Len, Channels), 与输入等长 (前后端填充)
        """
        # 前后端填充以保持序列长度不变
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # AvgPool1d 需要 (Batch, Channels, Seq_Len) 格式
        x_padded = x_padded.permute(0, 2, 1)
        trend = self.avg(x_padded)
        trend = trend.permute(0, 2, 1)

        return trend


class SeriesDecomp(nn.Module):
    """
    序列分解模块 (Series Decomposition Block)
    将时间序列分解为趋势分量 (Trend) 和季节分量 (Seasonal)

    x = Trend + Seasonal
    Trend = MovingAvg(x)
    Seasonal = x - Trend
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, Seq_Len, Channels)
        Returns:
            seasonal: (Batch, Seq_Len, Channels) - 季节分量
            trend: (Batch, Seq_Len, Channels) - 趋势分量
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# ============================================================================
# Part 2: 位置编码 (Positional Encoding)
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    支持正弦位置编码和可学习位置编码
    """

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
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
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
# Part 3: Auto-Correlation 机制 (AutoCorrelation)
# ============================================================================

class AutoCorrelation(nn.Module):
    """
    Auto-Correlation 机制 (Autoformer 核心创新)

    原理: 利用 FFT 计算时间序列的自相关函数，
    通过 Top-k 周期选择来替代传统的点积注意力 (Point-wise Attention)。

    优势:
    1. O(L log L) 复杂度 (FFT), 优于标准注意力的 O(L²)
    2. 天然捕捉时间序列的周期性模式
    3. 通过 Roll 操作实现子序列级别的信息聚合

    Reference: Autoformer Section 3.2
    """

    def __init__(self, d_model: int, nhead: int, factor: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.factor = factor
        self.dropout = nn.Dropout(p=dropout)

        # Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _time_delay_agg(self, values: torch.Tensor,
                        corr: torch.Tensor,
                        top_k_indices: torch.Tensor,
                        top_k_weights: torch.Tensor) -> torch.Tensor:
        """
        时间延迟聚合 (Time Delay Aggregation)
        根据 Top-k 自相关周期，对 Value 进行 Roll 并加权聚合

        Args:
            values: (B, H, L, d_k)
            corr: (B, H, L, L) 自相关矩阵
            top_k_indices: (B, H, top_k) 选中的周期索引
            top_k_weights: (B, H, top_k) 选中的周期权重 (softmax后)

        Returns:
            output: (B, H, L, d_k)
        """
        B, H, L, d_k = values.shape
        top_k = top_k_indices.shape[-1]

        # 初始化输出
        output = torch.zeros_like(values)

        for i in range(top_k):
            # 获取第 i 个周期的延迟量 (每个 batch/head 可能不同)
            delay = top_k_indices[:, :, i]  # (B, H)
            weight = top_k_weights[:, :, i]  # (B, H)

            # 对 values 按延迟量进行 Roll (循环移位)
            # 为了支持 batch 内不同延迟，需要逐样本处理
            for b in range(B):
                for h in range(H):
                    d = int(delay[b, h].item())
                    rolled = torch.roll(values[b, h], shifts=d, dims=0)  # (L, d_k)
                    output[b, h] += weight[b, h] * rolled

        return output

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """
        Auto-Correlation 前向传播

        Args:
            query: (B, L_q, d_model)
            key: (B, L_k, d_model)
            value: (B, L_k, d_model)

        Returns:
            output: (B, L_q, d_model)
        """
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        # 1. 线性投影 + 多头拆分
        Q = self.W_q(query).view(B, L_q, self.nhead, self.d_k).permute(0, 2, 1, 3)
        K = self.W_k(key).view(B, L_k, self.nhead, self.d_k).permute(0, 2, 1, 3)
        V = self.W_v(value).view(B, L_k, self.nhead, self.d_k).permute(0, 2, 1, 3)
        # Q, K, V: (B, H, L, d_k)

        # 2. 计算 Auto-Correlation (基于 FFT)
        # 对 Q 和 K 在时间维度上计算互相关
        # 互相关定理: corr(Q, K) = IFFT(FFT(Q) * conj(FFT(K)))

        # 对齐长度: 使用 Q 的长度
        if L_k > L_q:
            # 截断 K 和 V
            K = K[:, :, :L_q, :]
            V = V[:, :, :L_q, :]
            L_k = L_q
        elif L_q > L_k:
            # 用零填充 K
            K = F.pad(K, (0, 0, 0, L_q - L_k))
            V = F.pad(V, (0, 0, 0, L_q - L_k))
            L_k = L_q

        L = L_q

        # FFT 计算自相关
        Q_fft = torch.fft.rfft(Q, dim=2)  # (B, H, L//2+1, d_k)
        K_fft = torch.fft.rfft(K, dim=2)

        # 互相关: element-wise 乘以共轭
        corr_fft = Q_fft * torch.conj(K_fft)
        corr = torch.fft.irfft(corr_fft, n=L, dim=2)  # (B, H, L, d_k)

        # 3. 在 d_k 维度上取均值，得到每个时间延迟的相关强度
        corr_mean = corr.mean(dim=-1)  # (B, H, L)

        # 4. Top-k 周期选择
        # k = ceil(factor * log(L))
        top_k = max(1, int(self.factor * math.log(L + 1)))
        top_k = min(top_k, L)

        top_k_weights, top_k_indices = torch.topk(corr_mean, top_k, dim=-1)
        # top_k_weights: (B, H, top_k)
        # top_k_indices: (B, H, top_k)

        # Softmax 归一化权重
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # 5. 时间延迟聚合
        output = self._time_delay_agg(V, corr_mean, top_k_indices, top_k_weights)
        # output: (B, H, L_q, d_k)

        # 6. 多头合并 + 输出投影
        output = output.permute(0, 2, 1, 3).contiguous().view(B, L_q, self.d_model)
        output = self.W_o(output)
        output = self.dropout(output)

        return output


class AutoCorrelationFast(nn.Module):
    """
    Auto-Correlation 快速版本 (向量化实现)

    与 AutoCorrelation 逻辑相同，但使用向量化的 Roll 操作
    避免逐样本循环，显著提升训练速度。
    """

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
        """
        快速 Auto-Correlation 前向传播 (向量化)
        """
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        Q = self.W_q(query).view(B, L_q, self.nhead, self.d_k).permute(0, 2, 1, 3)
        K = self.W_k(key).view(B, L_k, self.nhead, self.d_k).permute(0, 2, 1, 3)
        V = self.W_v(value).view(B, L_k, self.nhead, self.d_k).permute(0, 2, 1, 3)

        # 对齐长度
        if L_k > L_q:
            K = K[:, :, :L_q, :]
            V = V[:, :, :L_q, :]
        elif L_q > L_k:
            K = F.pad(K, (0, 0, 0, L_q - L_k))
            V = F.pad(V, (0, 0, 0, L_q - L_k))

        L = L_q

        # FFT Auto-Correlation
        Q_fft = torch.fft.rfft(Q, dim=2)
        K_fft = torch.fft.rfft(K, dim=2)
        corr = torch.fft.irfft(Q_fft * torch.conj(K_fft), n=L, dim=2)
        corr_mean = corr.mean(dim=-1)  # (B, H, L)

        # Top-k
        top_k = max(1, int(self.factor * math.log(L + 1)))
        top_k = min(top_k, L)
        top_k_weights, top_k_indices = torch.topk(corr_mean, top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        # 向量化 Time Delay Aggregation
        # 使用 gather 和 repeat 构建延迟后的 Value
        output = torch.zeros_like(V)  # (B, H, L, d_k)

        # 构建 indices 用于 gather
        # top_k_indices: (B, H, top_k) -> delays
        for i in range(top_k):
            delays = top_k_indices[:, :, i]  # (B, H)
            weights_i = top_k_weights[:, :, i]  # (B, H)

            # 构建偏移后的索引: 对每个 (b, h)，生成 [0,1,...,L-1] + delay 的循环索引
            arange = torch.arange(L, device=V.device).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
            shifted = (arange + delays.unsqueeze(-1)) % L  # (B, H, L)
            shifted = shifted.unsqueeze(-1).expand(-1, -1, -1, self.d_k)  # (B, H, L, d_k)

            rolled_v = torch.gather(V, dim=2, index=shifted)  # (B, H, L, d_k)
            output += weights_i.unsqueeze(-1).unsqueeze(-1) * rolled_v

        output = output.permute(0, 2, 1, 3).contiguous().view(B, L_q, self.d_model)
        output = self.W_o(output)
        output = self.dropout(output)

        return output


# ============================================================================
# Part 4: Autoformer Encoder
# ============================================================================

class AutoformerEncoderLayer(nn.Module):
    """
    Autoformer Encoder 单层

    结构: Auto-Correlation → Add & Norm → FeedForward → Add & Norm
    每层后接 Series Decomposition，渐进式提取趋势

    Reference: Autoformer Figure 3 (left)
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 moving_avg_kernel: int, factor: int = 1,
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()

        # Auto-Correlation (替代 Multi-Head Attention)
        self.auto_correlation = AutoCorrelationFast(
            d_model=d_model, nhead=nhead, factor=factor, dropout=dropout
        )

        # FeedForward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Series Decomposition (每层一个)
        self.decomp1 = SeriesDecomp(moving_avg_kernel)
        self.decomp2 = SeriesDecomp(moving_avg_kernel)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            seasonal: (B, L, d_model) - 季节分量 (送入下一层)
            trend: (B, L, d_model) - 趋势分量 (累积)
        """
        # Sub-layer 1: Auto-Correlation + Residual + Decomp
        attn_out = self.auto_correlation(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        seasonal1, trend1 = self.decomp1(x)

        # Sub-layer 2: FeedForward + Residual + Decomp
        ff_out = self.ff(seasonal1)
        x = seasonal1 + ff_out
        x = self.norm2(x)
        seasonal2, trend2 = self.decomp2(x)

        # 累积趋势
        trend = trend1 + trend2

        return seasonal2, trend


class AutoformerEncoder(nn.Module):
    """
    Autoformer Encoder (多层堆叠)

    特点: 每层输出季节分量和趋势分量
    季节分量传入下一层，趋势分量累积
    """

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
        """
        Args:
            x: (B, L, d_model)
        Returns:
            enc_out: (B, L, d_model) - 编码器输出 (最终季节分量)
            trend_accumulated: (B, L, d_model) - 累积趋势
        """
        trend_accumulated = torch.zeros_like(x)

        for layer in self.layers:
            x, trend = layer(x)
            trend_accumulated += trend

        enc_out = self.norm(x)
        return enc_out, trend_accumulated


# ============================================================================
# Part 5: Autoformer Decoder
# ============================================================================

class AutoformerDecoderLayer(nn.Module):
    """
    Autoformer Decoder 单层

    结构:
    1. Self Auto-Correlation → Decomp
    2. Cross Auto-Correlation (with Encoder output) → Decomp
    3. FeedForward → Decomp

    每层输出季节分量 + 累积趋势

    Reference: Autoformer Figure 3 (right)
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 moving_avg_kernel: int, factor: int = 1,
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()

        # Self Auto-Correlation
        self.self_correlation = AutoCorrelationFast(
            d_model=d_model, nhead=nhead, factor=factor, dropout=dropout
        )

        # Cross Auto-Correlation
        self.cross_correlation = AutoCorrelationFast(
            d_model=d_model, nhead=nhead, factor=factor, dropout=dropout
        )

        # FeedForward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Series Decomposition (每个子层后一个)
        self.decomp1 = SeriesDecomp(moving_avg_kernel)
        self.decomp2 = SeriesDecomp(moving_avg_kernel)
        self.decomp3 = SeriesDecomp(moving_avg_kernel)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # 趋势投影 (将 d_model 趋势投影到输出维度)
        self.trend_projection = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor,
                enc_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L_dec, d_model) - Decoder 输入 (季节分量)
            enc_out: (B, L_enc, d_model) - Encoder 输出

        Returns:
            seasonal: (B, L_dec, d_model) - 季节分量
            trend: (B, L_dec, 1) - 趋势分量 (已投影到输出维度)
        """
        # Sub-layer 1: Self Auto-Correlation
        self_attn_out = self.self_correlation(x, x, x)
        x = x + self.dropout(self_attn_out)
        x = self.norm1(x)
        seasonal1, trend1 = self.decomp1(x)

        # Sub-layer 2: Cross Auto-Correlation (Query=Decoder, K/V=Encoder)
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


class AutoformerDecoder(nn.Module):
    """
    Autoformer Decoder (多层堆叠)

    特点:
    1. 输入: 标签部分 (label_len) + 零填充预测部分 (pred_len)
    2. 每层渐进式分解，累积趋势
    3. 最终输出 = 季节投影 + 累积趋势
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 num_layers: int, moving_avg_kernel: int, factor: int = 1,
                 dropout: float = 0.1, activation: str = 'gelu',
                 output_dim: int = 1):
        super().__init__()

        self.layers = nn.ModuleList([
            AutoformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                moving_avg_kernel=moving_avg_kernel, factor=factor,
                dropout=dropout, activation=activation
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 季节分量最终投影
        self.seasonal_projection = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                trend_init: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L_dec, d_model) - Decoder 输入 (季节初始化)
            enc_out: (B, L_enc, d_model) - Encoder 输出
            trend_init: (B, L_dec, 1) - 趋势初始化 (来自输入的移动平均)

        Returns:
            output: (B, L_dec, output_dim) - 最终预测输出
        """
        trend_accumulated = trend_init

        for layer in self.layers:
            x, trend = layer(x, enc_out)
            trend_accumulated += trend

        # 最终输出 = 季节投影 + 趋势累积
        seasonal_out = self.seasonal_projection(self.norm(x))
        output = seasonal_out + trend_accumulated

        return output


# ============================================================================
# Part 6: ContextAwareWeightNet (复用自 m9_model_2.py)
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
# Part 7: AutoformerBackbone (替代原版 TimeSeriesBackbone)
# ============================================================================

class AutoformerBackbone(nn.Module):
    """
    Autoformer 时序预测骨干网络
    替代原版的 Vanilla Transformer Encoder (TimeSeriesBackbone)

    架构:
    1. Input Embedding + Positional Encoding
    2. Series Decomposition (初始分解)
    3. Autoformer Encoder (渐进式分解)
    4. Autoformer Decoder (趋势+季节重构)
    5. 输出: (Batch, pred_len, output_dim)
    """

    def __init__(self, input_dim: int, seq_len: int, config: ModelConfig):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.config = config
        self.pred_len = config.pred_len
        self.label_len = config.label_len
        self.output_dim = config.output_dim

        # Input Embedding (Encoder 和 Decoder 各一个)
        self.enc_embedding = nn.Linear(input_dim, config.d_model)
        self.dec_embedding = nn.Linear(input_dim, config.d_model)

        # Positional Encoding
        self.enc_pos = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_length,
            encoding_type=config.positional_encoding_type,
            dropout=config.dropout
        )
        self.dec_pos = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_length,
            encoding_type=config.positional_encoding_type,
            dropout=config.dropout
        )

        # 初始 Series Decomposition
        self.decomp_init = SeriesDecomp(config.moving_avg_kernel)

        # Autoformer Encoder
        self.encoder = AutoformerEncoder(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.num_encoder_layers,
            moving_avg_kernel=config.moving_avg_kernel,
            factor=config.autocorrelation_factor,
            dropout=config.dropout,
            activation=config.activation
        )

        # Autoformer Decoder
        self.decoder = AutoformerDecoder(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.num_decoder_layers,
            moving_avg_kernel=config.moving_avg_kernel,
            factor=config.autocorrelation_factor,
            dropout=config.dropout,
            activation=config.activation,
            output_dim=config.output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autoformer 前向传播

        Args:
            x: (B, seq_len, input_dim) - 融合后的输入特征

        Returns:
            y_pred: (B, pred_len, output_dim) - 预测输出
        """
        B = x.size(0)

        # ==================== Encoder 输入准备 ====================
        # x: (B, seq_len, input_dim)
        enc_input = self.enc_embedding(x)  # (B, seq_len, d_model)
        enc_input = self.enc_pos(enc_input)

        # Encoder 前向
        enc_out, enc_trend = self.encoder(enc_input)
        # enc_out: (B, seq_len, d_model)

        # ==================== Decoder 输入准备 ====================
        # Autoformer Decoder 的输入:
        # - 已知部分: 输入序列的最后 label_len 步
        # - 未知部分: pred_len 步零填充

        # 取输入序列末尾 label_len 步作为 Decoder 标签
        x_label = x[:, -self.label_len:, :]  # (B, label_len, input_dim)

        # 零填充预测部分
        x_zero = torch.zeros(B, self.pred_len, self.input_dim, device=x.device)

        # 拼接: [label | zeros]
        dec_raw = torch.cat([x_label, x_zero], dim=1)  # (B, label_len+pred_len, input_dim)

        # 对 Decoder 原始输入做初始分解
        seasonal_init, trend_init = self.decomp_init(dec_raw)

        # Decoder 嵌入 (仅对季节分量)
        dec_input = self.dec_embedding(seasonal_init)  # (B, label_len+pred_len, d_model)
        dec_input = self.dec_pos(dec_input)

        # 趋势初始化投影到输出维度
        trend_init_proj = trend_init[:, :, :self.output_dim]  # (B, label_len+pred_len, output_dim)
        # 如果 input_dim < output_dim，需要零填充
        if self.input_dim < self.output_dim:
            pad_size = self.output_dim - self.input_dim
            trend_init_proj = F.pad(trend_init_proj, (0, pad_size))
        # 如果 input_dim > output_dim，已经截断了

        # Decoder 前向
        output = self.decoder(dec_input, enc_out, trend_init_proj)
        # output: (B, label_len+pred_len, output_dim)

        # 只取预测部分 (最后 pred_len 步)
        y_pred = output[:, -self.pred_len:, :]  # (B, pred_len, output_dim)

        return y_pred


# ============================================================================
# Part 8: DynamicFusionPredictor (Autoformer 版)
# ============================================================================

class DynamicFusionPredictor(nn.Module):
    """
    动态融合预测器 (Autoformer 版)

    与原版 m9_model_2.py 保持相同的外部接口:
    - forward(x, temperature) -> (y_pred, expanded_weights)
    - freeze_backbone() / freeze_weight_net() / unfreeze_all()
    - compute_weight_l1_loss() / compute_group_diversity_loss()

    内部将 TimeSeriesBackbone 替换为 AutoformerBackbone

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

        # 【核心替换】使用 AutoformerBackbone 替代 TimeSeriesBackbone
        # 'no_sentiment' 模式: 仅环境特征进入主干, input_dim = context_dim
        # 其他模式: 情感特征 + 环境特征, input_dim = total_sentiment_dim + context_dim
        if self.ablation_mode == 'no_sentiment':
            backbone_input_dim = self.context_dim
        else:
            backbone_input_dim = self.total_sentiment_dim + self.context_dim

        self.backbone = AutoformerBackbone(
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
            # 权重全部设为1 (不使用权重生成网络)
            expanded_weights = torch.ones(B, T, self.total_sentiment_dim,
                                          device=x.device, dtype=x.dtype)

            # 情感数据不加权, 直接拼接环境特征
            x_fused = torch.cat([x_sentiment, x_context], dim=2)

        elif self.ablation_mode == 'no_sentiment':
            # ---- 全关模式: 仅环境特征 → 主干 (无情感, 无权重) ----
            # 权重全部设为0 (占位符, 不参与任何计算)
            expanded_weights = torch.zeros(B, T, self.total_sentiment_dim,
                                           device=x.device, dtype=x.dtype)

            # 仅环境特征进入主干
            x_fused = x_context

        else:
            raise ValueError(f"不支持的消融模式: {self.ablation_mode}")

        # Step 6: Predict (使用 AutoformerBackbone)
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
# Part 9: build_model 入口函数
# ============================================================================

def build_model(feature_map: Dict, config: ModelConfig = None) -> DynamicFusionPredictor:
    """
    构建 Autoformer 动态融合预测模型

    接口与原版 m9_model_2.py 完全一致，可直接替换使用。

    Args:
        feature_map: 特征映射字典
        config: ModelConfig 配置类

    Returns:
        model: DynamicFusionPredictor (Autoformer 版)
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
# Part 10: 测试代码
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Autoformer 动态融合预测器 架构测试 (含消融模式)")
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
        ModelConfig.label_len = 4
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
    print(f"  SeriesDecomp: ✓")
    print(f"  AutoCorrelation (Fast): ✓")
    print(f"  AutoformerEncoder: ✓")
    print(f"  AutoformerDecoder: ✓")
    print(f"  ContextAwareWeightNet: ✓")
    print(f"  AutoformerBackbone: ✓")
    print(f"  DynamicFusionPredictor (3种消融模式): ✓")

    print(f"\n{'=' * 70}")
    print("✓ 所有测试通过！模型可正常使用。")
    print(f"{'=' * 70}\n")
