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
    (保持原样)
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
    Phase B: 权重生成 (MLP + Residual Activation)

    【架构升级说明】
    原逻辑: Softmax (权重和为1) -> 容易导致特征值被压缩过小，引发梯度消失。
    新逻辑: 1.0 + Tanh (残差激活) -> 初始权重接近1.0 (保持特征原样)，允许学习出 >1.0 (放大) 或 <1.0 (抑制) 的权重。
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

        # Temperature 参数 (在 Tanh 模式下用于调节激活函数的饱和度)
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
            temperature: 可选的温度覆盖值

        Returns:
            weights: (Batch, Seq, Num_Groups), 范围在 (0, 2) 之间
        """
        # Phase A: 特征交互
        context_embedded = self.context_embedding(context_features)  # (B, S, H)
        context_interacted = self.context_interaction(context_embedded)  # (B, S, H)

        # Phase B: 权重生成 (生成 Logits)
        logits = self.weight_generator(context_interacted)  # (B, S, Num_Groups)

        # 获取温度参数
        temp = temperature if temperature is not None else self.temperature

        # -----------------------------------------------------------
        # 【核心修改】 激活机制升级
        # 原代码: weights = F.softmax(logits / temp, dim=-1)
        # 修改为: 1.0 + tanh(logits / temp)
        # -----------------------------------------------------------

        # 使用残差方式生成权重:
        # 1. logits / temp: 温度越低，tanh 越接近阶跃函数；温度越高，tanh 越平滑
        # 2. tanh: 输出范围 (-1, 1)
        # 3. 1.0 + tanh: 输出范围 (0, 2)
        #    - 结果接近 1.0: 特征保持原样 (此时网络认为该特征不需要特别处理)
        #    - 结果 > 1.0: 特征被放大 (Signal Amplification)
        #    - 结果 < 1.0: 特征被抑制 (Signal Suppression)

        weights = 1.0 + torch.tanh(logits / temp)

        return weights


class TimeSeriesBackbone(nn.Module):
    """
    创新点三：时序预测骨干网络
    基于 Transformer Encoder 的时序特征提取器
    (保持原样)
    """

    def __init__(self, input_dim: int, seq_len: int, config: ModelConfig):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.config = config

        # Input Embedding
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

        # Output MLP
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model * seq_len, config.dim_feedforward),
            nn.LayerNorm(config.dim_feedforward),
            self._get_activation(config.activation),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.pred_len * config.output_dim)
        )

    def _get_activation(self, name: str) -> nn.Module:
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # Input Embedding
        if self.config.input_embedding_type == 'linear':
            x_embedded = self.input_embedding(x)
        else:
            x = x.transpose(1, 2)
            x_embedded = self.input_embedding(x)
            x_embedded = x_embedded.transpose(1, 2)

        x_pos = self.positional_encoding(x_embedded)
        x_encoded = self.transformer_encoder(x_pos)
        x_flat = x_encoded.reshape(batch_size, -1)
        y_pred = self.output_projection(x_flat)
        y_pred = y_pred.view(batch_size, self.config.pred_len, self.config.output_dim)

        return y_pred


class DynamicFusionPredictor(nn.Module):
    """
    动态融合预测器 (升级版 - 支持分组权重广播)
    (保持原样，仅 forward 中的注释逻辑随之更新)
    """

    def __init__(self, feature_map: Dict, config: ModelConfig):
        super().__init__()

        self.feature_map = feature_map
        self.config = config
        config.validate()

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
        self.weight_net = ContextAwareWeightNet(
            context_dim=self.context_dim,
            num_sentiment_groups=self.num_sentiment_groups,
            config=config
        )

        self.backbone = TimeSeriesBackbone(
            input_dim=self.total_sentiment_dim + self.context_dim,
            seq_len=config.seq_len,
            config=config
        )

    def forward(self, x: torch.Tensor, temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        """
        B, T, F = x.shape

        # Step 1: Slicing
        x_sentiment = torch.index_select(x, dim=2, index=self.flat_sentiment_idxs_buffer)
        x_context = torch.index_select(x, dim=2, index=self.context_idxs_buffer)

        # Step 2: Weight Generation (使用新的 1+Tanh 激活)
        raw_weights = self.weight_net(x_context, temperature)

        # Step 3: Weight Expansion
        indices_expanded = self.weight_expansion_indices_buffer.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        expanded_weights = torch.gather(raw_weights, dim=-1, index=indices_expanded)

        # Step 4: Fusion
        # 这里的 expanded_weights 现在是 (0, 2) 之间的缩放因子
        # 1.0 代表不改变原始特征幅度
        x_sentiment_weighted = x_sentiment * expanded_weights

        # Step 5: Re-assemble
        x_fused = torch.cat([x_sentiment_weighted, x_context], dim=2)

        # Step 6: Predict
        y_pred = self.backbone(x_fused)

        return y_pred, expanded_weights

    # ==================== 辅助方法保持原样 ====================

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
            # 修改 sparsity 定义：这里检查是否接近 1.0 (无操作) 还是被抑制 (接近0)
            # 我们可以定义 "active" 为偏离 1.0 的程度，或者维持原有的 "接近0" 检测
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
        # 对于 1+Tanh 模式，我们可能更希望权重接近 1.0 (不干预)
        # 所以 L1 Loss 可以改为 abs(weights - 1.0)
        # 但为了保持兼容性，如果你希望稀疏化(变为0)，则保持 abs(weights)
        # 这里建议根据你的业务需求。如果希望特征选择（去掉某些特征），保持 abs(weights)。
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


def build_model(feature_map: Dict, config: ModelConfig = None) -> DynamicFusionPredictor:
    if config is None:
        config = ModelConfig
    required_fields = ['sentiment_groups', 'context_idxs', 'target_idx', 'total_features']
    for field in required_fields:
        if field not in feature_map:
            raise ValueError(f"feature_map 缺少必需字段: {field}")
    model = DynamicFusionPredictor(feature_map, config)
    return model


if __name__ == "__main__":
    # 测试代码保持原样，验证新逻辑下的输出范围
    print("\n" + "=" * 70)
    print("模型架构测试 (Residual Tanh 激活版)")
    print("=" * 70 + "\n")

    feature_map_test = {
        'sentiment_groups': [list(range(0, 5)), list(range(5, 10))],
        'context_idxs': list(range(10, 15)),
        'target_idx': 15,
        'feature_names': [f'feature_{i}' for i in range(16)],
        'total_features': 16
    }

    config = ModelConfig
    model = build_model(feature_map_test, config)

    x_test = torch.randn(4, 30, 16)
    y_pred, expanded_weights = model(x_test)

    print(f"权重范围检查:")
    print(f"  Min: {expanded_weights.min().item():.4f} (理论 > 0)")
    print(f"  Max: {expanded_weights.max().item():.4f} (理论 < 2)")
    print(f"  Mean: {expanded_weights.mean().item():.4f} (初始应接近 1.0)")