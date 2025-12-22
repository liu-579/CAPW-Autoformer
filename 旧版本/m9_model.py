"""
m9_model.py - 动态融合预测模型
===================================================================
核心创新：基于上下文感知的动态权重生成网络

包含：
    1. ContextAwareWeightNet - 上下文感知权重生成网络
    2. TimeSeriesBackbone - 时序预测骨干网络
    3. DynamicFusionPredictor - 完整的动态融合预测模型
    4. 数据加载器
    5. 训练器

作者：[您的姓名]
日期：2025-12-18
版本：v3.0 - 最终整合版
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import Tuple, Optional, Dict
from tqdm import tqdm


# ============================================================================
# 1. 数据加载模块
# ============================================================================

class TimeSeriesDataset(Dataset):
    """
    时序预测数据集

    加载预处理好的 NumPy 张量数据
    """

    def __init__(self, x_path: str, y_path: str):
        """
        参数：
            x_path: 输入特征的 .npy 文件路径
            y_path: 标签的 .npy 文件路径
        """
        self.x = np.load(x_path)  # (Samples, 30, Feature_Dim)
        self.y = np.load(y_path)  # (Samples, 7, 1)

        assert self.x.shape[0] == self.y.shape[0], \
            f"样本数不匹配: X={self.x.shape[0]}, Y={self.y.shape[0]}"

        self.num_samples = self.x.shape[0]
        self.seq_len = self.x.shape[1]
        self.feature_dim = self.x.shape[2]
        self.pred_len = self.y.shape[1]

        print(f"✅ 数据集加载成功:")
        print(f"   样本数: {self.num_samples}")
        print(f"   输入形状: {self.x.shape}")
        print(f"   标签形状: {self.y.shape}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
            x: [Seq_Len, Feature_Dim]
            y: [Pred_Len, 1]
        """
        x = torch.from_numpy(self.x[idx]).float()
        y = torch.from_numpy(self.y[idx]).float()
        return x, y


class DataLoaderFactory:
    """
    数据加载器工厂

    负责：
        1. 加载特征映射表 (feature_map.json)
        2. 加载归一化器 (scaler.pkl)
        3. 创建训练/验证/测试的 DataLoader
    """

    def __init__(self, config):
        """
        参数：
            config: ModelConfig 对象
        """
        self.config = config
        self.data_paths = config.data_config.get_all_paths()

        # 加载特征映射表
        self.feature_map = self._load_feature_map()
        config.load_feature_map(self.feature_map)

        # 加载归一化器（可选）
        self.scaler = self._load_scaler()

    def _load_feature_map(self) -> Dict:
        """加载特征映射表 JSON 文件"""
        feature_map_path = self.data_paths['feature_map']

        if not os.path.exists(feature_map_path):
            raise FileNotFoundError(f"特征映射表文件不存在: {feature_map_path}")

        with open(feature_map_path, 'r', encoding='utf-8') as f:
            feature_map = json.load(f)

        required_keys = ['sentiment_idxs', 'context_idxs']
        for key in required_keys:
            if key not in feature_map:
                raise KeyError(f"特征映射表缺少必要的键: {key}")

        print(f"✅ 特征映射表加载成功: {feature_map_path}")
        return feature_map

    def _load_scaler(self) -> Optional[object]:
        """加载归一化器 pickle 文件（如果存在）"""
        scaler_path = self.data_paths['scaler']

        if not os.path.exists(scaler_path):
            print(f"⚠️  归一化器文件不存在: {scaler_path}")
            return None

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        print(f"✅ 归一化器加载成功: {scaler_path}")
        return scaler

    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """创建训练/验证/测试的 DataLoader"""
        dataloaders = {}

        # 训练集
        train_dataset = TimeSeriesDataset(
            self.data_paths['train_x'],
            self.data_paths['train_y']
        )
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )

        # 验证集
        val_dataset = TimeSeriesDataset(
            self.data_paths['val_x'],
            self.data_paths['val_y']
        )
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        # 测试集
        test_dataset = TimeSeriesDataset(
            self.data_paths['test_x'],
            self.data_paths['test_y']
        )
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        print("\n" + "="*80)
        print("DataLoader 创建成功:")
        print(f"  训练集批次数: {len(dataloaders['train'])}")
        print(f"  验证集批次数: {len(dataloaders['val'])}")
        print(f"  测试集批次数: {len(dataloaders['test'])}")
        print("="*80)

        return dataloaders


# ============================================================================
# 2. 上下文感知权重生成网络
# ============================================================================

class ContextAwareWeightNet(nn.Module):
    """
    上下文感知的动态权重生成网络

    输入：
        context_features: [Batch, Seq_Len, Context_Dim]
    输出：
        weights: [Batch, Seq_Len, Sentiment_Dim]
    """

    def __init__(self, config):
        super(ContextAwareWeightNet, self).__init__()

        self.context_dim = config.context_dim
        self.sentiment_dim = config.sentiment_dim
        self.hidden_dim = config.hidden_dim

        # 特征交互模块
        self.input_projection = nn.Linear(self.context_dim, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.weight_net_dropout,
            activation=config.weight_net_activation,
            batch_first=True
        )

        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )

        self.context_aggregation = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.weight_net_dropout)
        )

        # 权重生成模块
        self.weight_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.weight_net_dropout),
            nn.Linear(self.hidden_dim // 2, self.sentiment_dim)
        )

        self.register_buffer(
            'temperature',
            torch.tensor(config.default_temperature, dtype=torch.float32)
        )

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, context_features: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        # 投影到隐藏空间
        hidden = self.input_projection(context_features)

        # Self-Attention 编码
        encoded = self.context_encoder(hidden)
        encoded = self.layer_norm(encoded)

        # 提取情境向量
        context_vector = self.context_aggregation(encoded)

        # 生成权重 logits
        weight_logits = self.weight_generator(context_vector)

        # Softmax 归一化
        temp = temperature if temperature is not None else self.temperature
        weights = F.softmax(weight_logits / temp, dim=-1)

        return weights


# ============================================================================
# 3. 时序预测骨干网络
# ============================================================================

class PositionalEncoding(nn.Module):
    """标准的正弦位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesBackbone(nn.Module):
    """时序预测骨干网络"""

    def __init__(self, config):
        super(TimeSeriesBackbone, self).__init__()

        self.input_dim = config.total_dim
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.d_model = config.d_model

        # 输入嵌入
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = PositionalEncoding(
            self.d_model,
            config.backbone_dropout,
            max_len=config.max_position_embeddings
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.backbone_dropout,
            activation=config.backbone_activation,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.backbone_dropout),
            nn.Linear(self.d_model // 2, 1)
        )

        # 时序映射
        if self.seq_len != self.pred_len:
            self.temporal_projection = nn.Linear(self.seq_len, self.pred_len)
        else:
            self.temporal_projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入嵌入
        embedded = self.input_embedding(x)
        embedded = self.positional_encoding(embedded)

        # Transformer 编码
        encoded = self.transformer_encoder(embedded)

        # 输出投影
        output = self.output_projection(encoded)

        # 时序调整
        if self.temporal_projection is not None:
            output = output.transpose(1, 2)
            output = self.temporal_projection(output)
            output = output.transpose(1, 2)

        return output


# ============================================================================
# 4. 动态融合预测模型
# ============================================================================

class DynamicFusionPredictor(nn.Module):
    """
    动态融合预测模型

    核心流程：
        1. 特征切分（使用 torch.index_select）
        2. 权重生成（基于环境特征）
        3. 特征融合（加权情感特征）
        4. 时序预测（骨干网络）
    """

    def __init__(self, config):
        super(DynamicFusionPredictor, self).__init__()

        self.config = config

        # 特征索引（从配置读取）
        self.sentiment_indices = torch.tensor(config.sentiment_indices, dtype=torch.long)
        self.context_indices = torch.tensor(config.context_indices, dtype=torch.long)

        self.sentiment_dim = config.sentiment_dim
        self.context_dim = config.context_dim
        self.total_dim = config.total_dim

        # 注册为 buffer（不参与梯度更新）
        self.register_buffer('sentiment_indices_buffer', self.sentiment_indices)
        self.register_buffer('context_indices_buffer', self.context_indices)

        # 子模块
        self.weight_net = ContextAwareWeightNet(config)
        self.backbone = TimeSeriesBackbone(config)

        # 静态权重模式
        self.use_static_weights = False
        self.register_buffer(
            'static_weights',
            torch.ones(self.sentiment_dim) / self.sentiment_dim
        )

    def forward(self, x: torch.Tensor, temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数：
            x: [Batch, Seq_Len, Total_Dim] - 完整输入特征
        返回：
            y_pred: [Batch, Pred_Len, 1] - 预测值
            dynamic_weights: [Batch, Seq_Len, Sentiment_Dim] - 动态权重
        """
        batch_size, seq_len, _ = x.shape

        # Step 1: 特征切分（使用 index_select）
        x_sentiment = torch.index_select(x, dim=2, index=self.sentiment_indices_buffer)
        x_context = torch.index_select(x, dim=2, index=self.context_indices_buffer)

        # Step 2: 动态权重生成
        if self.use_static_weights:
            dynamic_weights = self.static_weights.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_len, -1
            )
        else:
            dynamic_weights = self.weight_net(x_context, temperature)

        # Step 3: 特征融合
        x_sentiment_weighted = x_sentiment * dynamic_weights
        x_fused = torch.cat([x_sentiment_weighted, x_context], dim=-1)

        # Step 4: 时序预测
        y_pred = self.backbone(x_fused)

        return y_pred, dynamic_weights

    # 训练控制接口
    def freeze_backbone(self):
        """冻结骨干网络"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✅ 骨干网络已冻结")

    def freeze_weight_net(self):
        """冻结权重网络"""
        for param in self.weight_net.parameters():
            param.requires_grad = False
        print("✅ 权重生成网络已冻结")

    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        print("✅ 所有参数已解冻")

    def set_static_weights(self, weights: Optional[torch.Tensor] = None):
        """启用静态权重模式"""
        self.use_static_weights = True
        if weights is not None:
            weights = weights / weights.sum()
            self.static_weights = weights
        print(f"✅ 已启用静态权重模式")

    def disable_static_weights(self):
        """禁用静态权重模式"""
        self.use_static_weights = False
        print("✅ 已禁用静态权重模式")

    def get_trainable_params(self):
        """获取可训练参数统计"""
        weight_net_params = sum(p.numel() for p in self.weight_net.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'weight_net': weight_net_params,
            'backbone': backbone_params,
            'total': total_params
        }


# ============================================================================
# 5. 训练器
# ============================================================================

class Trainer:
    """模型训练器"""

    def __init__(self, model, config, dataloaders):
        self.model = model.to(config.device)
        self.config = config
        self.dataloaders = dataloaders
        self.device = config.device

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 损失函数
        self.criterion = nn.MSELoss()

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1}")
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)

            # 前向传播
            y_pred, weights = self.model(x)

            # 计算损失
            loss = self.criterion(y_pred, y)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

            self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(self.dataloaders['train'])
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """验证"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in self.dataloaders['val']:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred, _ = self.model(x)
                loss = self.criterion(y_pred, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.dataloaders['val'])
        self.val_losses.append(avg_val_loss)

        return avg_val_loss

    def train(self, num_epochs):
        """完整训练流程"""
        print("\n" + "="*80)
        print("开始训练")
        print("="*80)

        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss = self.validate()

            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证损失: {val_loss:.6f}")

            # 保存最优模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"  ✅ 验证损失改善! 保存模型...")
                # TODO: 添加模型保存逻辑

        print("\n" + "="*80)
        print("训练完成!")
        print("="*80)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    from 旧版本.m9_config import ModelConfig

    print("="*80)
    print("DynamicFusionPredictor 模型测试（v3.0）")
    print("="*80)

    # 1. 创建配置
    config = ModelConfig()

    # 2. 模拟加载特征映射
    mock_feature_map = {
        'sentiment_idxs': [0, 1, 2, 3, 4],
        'context_idxs': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    }
    config.load_feature_map(mock_feature_map)

    # 3. 创建模型
    model = DynamicFusionPredictor(config)
    print(f"\n模型参数: {model.get_trainable_params()}")

    # 4. 测试数据
    batch_size = 8
    x = torch.randn(batch_size, config.seq_len, config.total_dim)

    print(f"\n输入形状: {x.shape}")

    # 5. 前向传播
    y_pred, weights = model(x)

    print(f"预测形状: {y_pred.shape}")
    print(f"权重形状: {weights.shape}")
    print(f"权重示例: {weights[0, 0].detach().cpu().numpy()}")
    print(f"权重和: {weights[0, 0].sum().item():.6f}")

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)