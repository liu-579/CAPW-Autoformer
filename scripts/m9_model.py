# -*- coding: utf-8 -*-
"""
M9 动态融合预测模型 (升级版)
Dynamic Fusion Predictor with Grouped Sentiment Weighting
创新点: 环境特征生成分组权重，通过预计算索引实现高效广播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Dict, Optional, List

from config.m9_config import ModelConfig


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
            # 正弦位置编码 (Attention is All You Need)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                 (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)

            self.register_buffer('pe', pe)

        elif encoding_type == 'learnable':
            # 可学习位置编码
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        else:
            raise ValueError(f"不支持的位置编码类型: {encoding_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, Seq, d_model)
        Returns:
            x + pe: (Batch, Seq, d_model)
        """
        seq_len = x.size(1)

        if self.encoding_type == 'sinusoidal':
            x = x + self.pe[:, :seq_len, :]
        else:
            x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class ContextAwareWeightNet(nn.Module):
    """
    创新点二：环境感知的动态权重生成网络
    "看环境脸色，给情感打分"

    两阶段设计:
    Phase A: 特征交互 (Self-Attention)
    Phase B: 权重生成 (MLP + Temperature Softmax)

    【升级】输出维度修改为 num_sentiment_groups (通常为5)
    """

    def __init__(self, context_dim: int, num_sentiment_groups: int, config: ModelConfig):
        super().__init__()

        self.context_dim = context_dim
        self.num_sentiment_groups = num_sentiment_groups
        self.config = config

        # Phase A: 特征交互层 (轻量级 Transformer)
        # 目的: 让 "雨天" 和 "节假日" 等特征发生化学反应
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
        # 目的: 将情境向量映射为情感组权重
        self.weight_generator = nn.Sequential(
            nn.Linear(config.weight_net_hidden_dim, config.weight_net_hidden_dim),
            nn.LayerNorm(config.weight_net_hidden_dim),
            self._get_activation(config.activation),
            nn.Dropout(config.weight_net_dropout),
            nn.Linear(config.weight_net_hidden_dim, num_sentiment_groups)
        )

        # Temperature 参数 (用于控制 Softmax 平滑度)
        if config.temperature_trainable:
            self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        else:
            self.register_buffer('temperature', torch.tensor(config.temperature_init))

    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
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
        前向传播: 环境特征 -> 分组权重

        Args:
            context_features: (Batch, Seq, Context_Dim)
            temperature: 可选的温度覆盖值 (用于推理时调节)

        Returns:
            weights: (Batch, Seq, Num_Groups), 每个时间步的权重和为1
        """
        # Phase A: 特征交互
        # 嵌入到隐藏空间
        context_embedded = self.context_embedding(context_features)  # (B, S, H)

        # Transformer 交互: 让特征相互作用
        context_interacted = self.context_interaction(context_embedded)  # (B, S, H)

        # Phase B: 权重生成
        # 生成 logits
        logits = self.weight_generator(context_interacted)  # (B, S, Num_Groups)

        # Temperature Softmax
        temp = temperature if temperature is not None else self.temperature
        weights = F.softmax(logits / temp, dim=-1)  # (B, S, Num_Groups)

        return weights


class TimeSeriesBackbone(nn.Module):
    """
    创新点三：时序预测骨干网络
    基于 Transformer Encoder 的时序特征提取器
    """

    def __init__(self, input_dim: int, seq_len: int, config: ModelConfig):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.config = config

        # Input Embedding: 将原始特征映射到 d_model 维度
        if config.input_embedding_type == 'linear':
            self.input_embedding = nn.Linear(input_dim, config.d_model)
        elif config.input_embedding_type == 'conv1d':
            self.input_embedding = nn.Conv1d(
                input_dim, config.d_model, kernel_size=3, padding=1
            )
        else:
            raise ValueError(f"不支持的嵌入类型: {config.input_embedding_type}")

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_length,
            encoding_type=config.positional_encoding_type,
            dropout=config.dropout
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=False
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        )

        # Output MLP: 从 Transformer 输出到最终预测
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model * seq_len, config.dim_feedforward),
            nn.LayerNorm(config.dim_feedforward),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.pred_len * config.output_dim)
        )

    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 时序特征 -> 未来预测

        Args:
            x: (Batch, Seq, Input_Dim)

        Returns:
            y_pred: (Batch, Pred_Len, Output_Dim)
        """
        batch_size = x.size(0)

        # Input Embedding
        if self.config.input_embedding_type == 'linear':
            x_embedded = self.input_embedding(x)  # (B, S, d_model)
        else:  # conv1d
            x = x.transpose(1, 2)  # (B, Input_Dim, S)
            x_embedded = self.input_embedding(x)  # (B, d_model, S)
            x_embedded = x_embedded.transpose(1, 2)  # (B, S, d_model)

        # Positional Encoding
        x_pos = self.positional_encoding(x_embedded)  # (B, S, d_model)

        # Transformer Encoder
        x_encoded = self.transformer_encoder(x_pos)  # (B, S, d_model)

        # Flatten and Project to Output
        x_flat = x_encoded.reshape(batch_size, -1)  # (B, S * d_model)
        y_pred = self.output_projection(x_flat)  # (B, Pred_Len * Output_Dim)

        # Reshape to (Batch, Pred_Len, Output_Dim)
        y_pred = y_pred.view(batch_size, self.config.pred_len, self.config.output_dim)

        return y_pred


class DynamicFusionPredictor(nn.Module):
    """
    动态融合预测器 (升级版 - 支持分组权重广播)

    核心流程:
    1. Slicing:     根据 feature_map 拆分特征
    2. Weighting:   环境特征生成分组权重 (5维)
    3. Expansion:   将5维权重扩展为N维 (通过预计算索引)
    4. Fusion:      加权融合情感特征
    5. Re-assemble: 拼接加权情感 + 原始环境
    6. Predict:     Transformer 骨干网络预测
    """

    def __init__(self, feature_map: Dict, config: ModelConfig):
        super().__init__()

        self.feature_map = feature_map
        self.config = config

        # 验证配置
        config.validate()

        # ==================== 解析分组信息 (新增) ====================

        # 读取分组信息
        self.sentiment_groups = feature_map['sentiment_groups']
        self.num_sentiment_groups = len(self.sentiment_groups)

        print(f"\n{'=' * 70}")
        print("DynamicFusionPredictor 初始化 (分组权重版)")
        print(f"{'=' * 70}")
        print(f"情感维度分组数: {self.num_sentiment_groups}")
        for i, group in enumerate(self.sentiment_groups):
            print(f"  组 {i}: {len(group)} 个特征 (索引: {group[:3]}... 共{len(group)}个)")

        # ==================== 构建扁平化索引 ====================

        # 扁平化所有情感特征的绝对索引
        flat_sentiment_idxs = []
        for group in self.sentiment_groups:
            flat_sentiment_idxs.extend(group)

        self.flat_sentiment_idxs = torch.tensor(flat_sentiment_idxs, dtype=torch.long)
        self.total_sentiment_dim = len(flat_sentiment_idxs)

        print(f"\n扁平化后情感特征总数: {self.total_sentiment_dim}")
        print(f"  示例索引: {flat_sentiment_idxs[:10]}...")

        # ==================== 预计算权重扩展索引 (核心创新) ====================

        # 构建 weight_expansion_indices
        # 逻辑: 对于每个情感特征，记录它属于哪个组 (0~num_groups-1)
        weight_expansion_indices = []
        for group_idx, group in enumerate(self.sentiment_groups):
            # 该组的每个特征都对应 group_idx 这个权重
            weight_expansion_indices.extend([group_idx] * len(group))

        self.weight_expansion_indices = torch.tensor(weight_expansion_indices, dtype=torch.long)

        print(f"\n权重扩展索引 (前20个): {weight_expansion_indices[:20]}")
        print(f"  形状: [{self.total_sentiment_dim}]")
        print(f"  含义: 第i个情感特征使用第 weight_expansion_indices[i] 个权重")

        # ==================== 解析环境特征索引 ====================

        self.context_idxs = torch.tensor(feature_map['context_idxs'], dtype=torch.long)
        self.context_dim = len(self.context_idxs)
        self.target_idx = feature_map['target_idx']
        self.total_dim = feature_map['total_features']

        print(f"\n环境特征维度: {self.context_dim}")
        print(f"总特征维度:   {self.total_dim}")
        print(f"目标列索引:   {self.target_idx}")

        # ==================== 注册为 buffer ====================

        # 注册为 buffer (不参与梯度计算，但会随模型移动到 GPU)
        self.register_buffer('flat_sentiment_idxs_buffer', self.flat_sentiment_idxs)
        self.register_buffer('context_idxs_buffer', self.context_idxs)
        self.register_buffer('weight_expansion_indices_buffer', self.weight_expansion_indices)

        # ==================== 子模块初始化 ====================

        # 子模块 1: 环境感知权重网络 (输出维度改为 num_sentiment_groups)
        self.weight_net = ContextAwareWeightNet(
            context_dim=self.context_dim,
            num_sentiment_groups=self.num_sentiment_groups,
            config=config
        )

        # 子模块 2: 时序预测骨干网络
        # 注意: 输入维度 = 情感总维度 + 环境维度
        self.backbone = TimeSeriesBackbone(
            input_dim=self.total_sentiment_dim + self.context_dim,
            seq_len=30,  # 从 m8b 的 SEQ_LEN
            config=config
        )

        print(f"\n✓ 模型初始化完成")
        print(f"  - 权重网络参数量: {sum(p.numel() for p in self.weight_net.parameters()):,}")
        print(f"  - 骨干网络参数量: {sum(p.numel() for p in self.backbone.parameters()):,}")
        print(f"  - 总参数量:       {sum(p.numel() for p in self.parameters()):,}")
        print(f"{'=' * 70}\n")

    def forward(self, x: torch.Tensor, temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 (升级版 - 支持分组权重广播)

        Args:
            x: (Batch, Seq_Len, Feature_Dim) - 完整特征张量
            temperature: 可选温度参数 (用于权重 Softmax)

        Returns:
            y_pred: (Batch, Pred_Len, Output_Dim) - 预测值
            expanded_weights: (Batch, Seq_Len, Total_Sentiment_Dim) - 扩展后的权重 (用于可解释性)
        """
        B, T, F = x.shape  # Batch, Time (Seq), Features

        # ==================== Step 1: Slicing (特征切分) ====================

        # 使用预计算的扁平化索引切分情感特征
        # Shape: [B, T, F] -> [B, T, Total_Sentiment_Dim]
        x_sentiment = torch.index_select(x, dim=2, index=self.flat_sentiment_idxs_buffer)

        # 切分环境特征
        # Shape: [B, T, F] -> [B, T, Context_Dim]
        x_context = torch.index_select(x, dim=2, index=self.context_idxs_buffer)

        # ==================== Step 2: Weight Generation (分组权重生成) ====================

        # 环境特征 -> 权重网络 -> 分组权重
        # Shape: [B, T, Context_Dim] -> [B, T, Num_Groups]
        raw_weights = self.weight_net(x_context, temperature)

        # ==================== Step 3: Weight Expansion (权重扩展 - 核心创新) ====================

        # 目标: 将 [B, T, Num_Groups] 扩展为 [B, T, Total_Sentiment_Dim]
        # 方法: 使用 torch.gather 根据预计算的索引进行"取值广播"

        # 步骤 3.1: 扩展索引张量的维度以匹配 raw_weights
        # [Total_Sentiment_Dim] -> [1, 1, Total_Sentiment_Dim] -> [B, T, Total_Sentiment_Dim]
        indices_expanded = self.weight_expansion_indices_buffer.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

        # 步骤 3.2: 使用 gather 在最后一维根据索引取值
        # 原理: expanded_weights[b, t, i] = raw_weights[b, t, indices_expanded[b, t, i]]
        # Shape: [B, T, Num_Groups] + [B, T, Total_Sentiment_Dim] -> [B, T, Total_Sentiment_Dim]
        expanded_weights = torch.gather(raw_weights, dim=-1, index=indices_expanded)

        # 验证形状
        assert expanded_weights.shape == x_sentiment.shape, \
            f"权重扩展后形状不匹配: {expanded_weights.shape} vs {x_sentiment.shape}"

        # ==================== Step 4: Fusion (加权融合) ====================

        # 逐元素乘法: 情感特征 * 扩展后的权重
        # Shape: [B, T, Total_Sentiment_Dim] * [B, T, Total_Sentiment_Dim]
        x_sentiment_weighted = x_sentiment * expanded_weights

        # ==================== Step 5: Re-assemble (特征重组) ====================

        # 拼接加权情感特征和原始环境特征
        # Shape: [B, T, Total_Sentiment_Dim] + [B, T, Context_Dim] -> [B, T, Total_Sentiment_Dim + Context_Dim]
        x_fused = torch.cat([x_sentiment_weighted, x_context], dim=2)

        # ==================== Step 6: Predict (时序预测) ====================

        # 融合特征 -> Transformer 骨干网络 -> 预测值
        # Shape: [B, T, Total_Sentiment_Dim + Context_Dim] -> [B, Pred_Len, Output_Dim]
        y_pred = self.backbone(x_fused)

        # 返回预测值和扩展后的权重 (扩展后的权重用于可解释性分析)
        return y_pred, expanded_weights

    # ==================== 三阶段训练支持方法 ====================

    def freeze_backbone(self):
        """冻结骨干网络参数 (用于阶段2训练)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ 骨干网络已冻结")

    def freeze_weight_net(self):
        """冻结权重网络参数 (用于阶段1训练)"""
        for param in self.weight_net.parameters():
            param.requires_grad = False
        print("✓ 权重网络已冻结")

    def unfreeze_all(self):
        """解冻所有参数 (用于阶段3联合训练)"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ 所有参数已解冻")

    def get_trainable_params(self):
        """获取可训练参数列表"""
        return filter(lambda p: p.requires_grad, self.parameters())

    def count_trainable_params(self):
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ==================== 可解释性分析支持方法 ====================

    def get_weight_statistics(self, expanded_weights: torch.Tensor) -> Dict:
        """
        计算扩展后权重的统计信息 (用于可解释性分析)

        Args:
            expanded_weights: (Batch, Seq, Total_Sentiment_Dim)

        Returns:
            stats: 包含均值、方差、最大值等统计信息的字典
        """
        stats = {
            'mean': expanded_weights.mean(dim=(0, 1)).detach().cpu().numpy(),
            'std': expanded_weights.std(dim=(0, 1)).detach().cpu().numpy(),
            'max': expanded_weights.max(dim=2)[0].mean(dim=(0, 1)).detach().cpu().item(),
            'min': expanded_weights.min(dim=2)[0].mean(dim=(0, 1)).detach().cpu().item(),
            'sparsity': (expanded_weights < 0.01).float().mean().detach().cpu().item()
        }
        return stats

    def get_group_weight_statistics(self, expanded_weights: torch.Tensor) -> Dict:
        """
        计算每个情感组的平均权重 (用于可解释性分析)

        Args:
            expanded_weights: (Batch, Seq, Total_Sentiment_Dim)

        Returns:
            group_stats: 每个组的平均权重统计
        """
        group_stats = {}

        for group_idx, group in enumerate(self.sentiment_groups):
            # 计算该组所有特征的平均权重
            group_mask = (self.weight_expansion_indices_buffer == group_idx)
            group_weights = expanded_weights[:, :, group_mask]

            group_stats[f'group_{group_idx}'] = {
                'mean': group_weights.mean().detach().cpu().item(),
                'std': group_weights.std().detach().cpu().item(),
                'size': len(group)
            }

        return group_stats

    def compute_weight_l1_loss(self, expanded_weights: torch.Tensor) -> torch.Tensor:
        """
        计算权重的 L1 正则化损失 (用于稀疏化)

        Args:
            expanded_weights: (Batch, Seq, Total_Sentiment_Dim)

        Returns:
            l1_loss: 标量张量
        """
        return torch.mean(torch.abs(expanded_weights))

    def compute_group_diversity_loss(self, expanded_weights: torch.Tensor) -> torch.Tensor:
        """
        计算组间权重多样性损失 (鼓励不同组的权重有差异)

        Args:
            expanded_weights: (Batch, Seq, Total_Sentiment_Dim)

        Returns:
            diversity_loss: 标量张量 (越小说明组间差异越大)
        """
        # 计算每个组的平均权重
        group_means = []
        for group_idx in range(self.num_sentiment_groups):
            group_mask = (self.weight_expansion_indices_buffer == group_idx)
            group_weights = expanded_weights[:, :, group_mask]
            group_means.append(group_weights.mean())

        group_means = torch.stack(group_means)  # [Num_Groups]

        # 计算方差的负值 (方差越大，多样性越好)
        diversity_loss = -torch.var(group_means)

        return diversity_loss


# ==================== 模型构建函数 ====================

def build_model(feature_map: Dict, config: ModelConfig = None) -> DynamicFusionPredictor:
    """
    构建模型的工厂函数

    Args:
        feature_map: 特征映射字典 (来自 m8b，包含 sentiment_groups)
        config: 模型配置 (默认使用 ModelConfig)

    Returns:
        model: DynamicFusionPredictor 实例
    """
    if config is None:
        config = ModelConfig

    # 验证 feature_map 包含必需字段
    required_fields = ['sentiment_groups', 'context_idxs', 'target_idx', 'total_features']
    for field in required_fields:
        if field not in feature_map:
            raise ValueError(f"feature_map 缺少必需字段: {field}")

    model = DynamicFusionPredictor(feature_map, config)
    return model


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("模型架构测试 (分组权重广播版)")
    print("=" * 70 + "\n")

    # 模拟 feature_map (包含 sentiment_groups)
    feature_map_test = {
        'sentiment_groups': [
            list(range(0, 5)),  # 组0: 特征 0-4
            list(range(5, 10)),  # 组1: 特征 5-9
            list(range(10, 15)),  # 组2: 特征 10-14
            list(range(15, 20)),  # 组3: 特征 15-19
            list(range(20, 25)),  # 组4: 特征 20-24
        ],
        'context_idxs': list(range(25, 35)),  # 环境特征: 25-34
        'target_idx': 34,
        'feature_names': [f'feature_{i}' for i in range(35)],
        'total_features': 35
    }

    # 构建模型
    config = ModelConfig
    config.print_config()

    model = build_model(feature_map_test, config)

    # 测试前向传播
    batch_size = 4
    seq_len = 30
    feature_dim = 35

    x_test = torch.randn(batch_size, seq_len, feature_dim)

    print("\n" + "=" * 70)
    print("前向传播测试")
    print("=" * 70)
    print(f"\n输入形状: {x_test.shape}")

    y_pred, expanded_weights = model(x_test)

    print(f"\n输出形状:")
    print(f"  - y_pred:           {y_pred.shape}")
    print(f"  - expanded_weights: {expanded_weights.shape}")

    # 验证权重扩展逻辑
    print(f"\n" + "=" * 70)
    print("权重扩展逻辑验证")
    print("=" * 70)

    # 手动检查：组0的所有特征应该有相同的权重
    group_0_weights = expanded_weights[0, 0, 0:5]
    print(f"\n组0 (前5个特征) 的权重:")
    print(f"  {group_0_weights.detach().cpu().numpy()}")
    print(f"  标准差 (应接近0): {group_0_weights.std().item():.6f}")

    # 测试权重统计
    print(f"\n" + "=" * 70)
    print("权重统计")
    print("=" * 70)

    stats = model.get_weight_statistics(expanded_weights)
    print(f"\n整体统计:")
    print(f"  - 平均值 (前5): {stats['mean'][:5]}")
    print(f"  - 标准差 (前5): {stats['std'][:5]}")
    print(f"  - 稀疏度:       {stats['sparsity']:.4f}")

    group_stats = model.get_group_weight_statistics(expanded_weights)
    print(f"\n分组统计:")
    for group_name, stat in group_stats.items():
        print(f"  {group_name}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, size={stat['size']}")

    # 测试冻结/解冻
    print(f"\n" + "=" * 70)
    print("参数冻结测试")
    print("=" * 70)
    print(f"\n  初始可训练参数: {model.count_trainable_params():,}")

    model.freeze_backbone()
    print(f"  冻结骨干后:     {model.count_trainable_params():,}")

    model.freeze_weight_net()
    print(f"  全部冻结后:     {model.count_trainable_params():,}")

    model.unfreeze_all()
    print(f"  解冻后:         {model.count_trainable_params():,}")

    # 测试损失函数
    print(f"\n" + "=" * 70)
    print("损失函数测试")
    print("=" * 70)

    l1_loss = model.compute_weight_l1_loss(expanded_weights)
    print(f"\n  L1 Loss:         {l1_loss.item():.6f}")

    diversity_loss = model.compute_group_diversity_loss(expanded_weights)
    print(f"  Diversity Loss:  {diversity_loss.item():.6f}")

    print("\n" + "=" * 70)
    print("✓ 所有测试通过!")
    print("=" * 70 + "\n")