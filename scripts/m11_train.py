
# -*- coding: utf-8 -*-
"""
M11 统一联合训练脚本 (Unified Joint Training Script)
=====================================================
版本: M11 (基于 M10 优化注释版本)
作者: Data Pipeline Engineer
用途: 训练时序预测模型，使用 CCC Loss 和自适应峰值加权机制

核心特性:
---------
1. **CCC Loss (一致性相关系数损失)**
   - 统一解决 RMSE 与 Pearson 指标冲突问题
   - 同时考虑数值准确性和趋势相关性
   - 公式: CCC = (2*cov) / (var_pred + var_target + (mean_pred - mean_target)^2)
   - Loss = 1 - CCC，取值范围 [0, 2]，0 表示完美拟合

2. **自适应峰值加权 (Adaptive Peak Weighting)**
   - 解决模型对高数值样本的削峰问题
   - 动态计算峰值阈值: Threshold = Mean + Sigma * Std
   - 对超过阈值的样本施加更高的损失权重

3. **温度退火机制 (Temperature Annealing)**
   - 控制注意力权重的锐化过程
   - 训练初期: 高温 (5.0) → 权重平滑，探索特征空间
   - 训练后期: 低温 (0.01) → 权重尖锐，专注关键特征

4. **早停策略 (Early Stopping)**
   - 基于验证集 CCC Loss 判断
   - 防止过拟合，节省训练时间

修改记录:
---------
- M11: 优化注释结构，增加详细说明和使用指南
- M10: 引入 CCC Loss 和自适应峰值加权机制
- M9: 基础模型架构

使用方法:
---------
1. 确保已运行 M8B 数据构建脚本，生成训练数据
2. 修改 config/m11_config.py 中的路径和参数
3. 运行: python scripts/m11_train.py
4. 训练完成后，模型保存在 data/output/m11_checkpoints_xxx/ 目录
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import warnings
import sys
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 导入模型和配置
sys.path.append('.')
from m9_model_2 import build_model
from config.m9_config import ModelConfig
from config.m11_config import TrainingConfig as cfg


class TemperatureScheduler:
    """
    温度退火调度器 (Temperature Annealing Scheduler)
    
    功能:
    -----
    控制注意力权重的锐化过程，实现从平滑到尖锐的渐进式转变。
    
    原理:
    -----
    - 高温 (temp_start): Softmax 输出平滑，权重分布均匀
    - 低温 (temp_end): Softmax 输出尖锐，权重集中在少数特征
    - 退火过程: 随训练进行，温度从 temp_start 逐渐降至 temp_end
    
    参数:
    -----
    temp_start : float
        初始温度 (建议 5.0-10.0)
    temp_end : float
        最终温度 (建议 0.01-0.1)
    num_epochs : int
        总训练轮数
    schedule : str
        退火策略，可选 'linear' (线性) 或 'cosine' (余弦)
    
    示例:
    -----
    >>> scheduler = TemperatureScheduler(5.0, 0.01, 100, 'linear')
    >>> temp = scheduler.get_temperature(50)  # 第 50 轮的温度
    """

    def __init__(self, temp_start, temp_end, num_epochs, schedule='linear'):
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.num_epochs = num_epochs
        self.schedule = schedule

    def get_temperature(self, epoch):
        """
        获取当前轮次的温度值
        
        参数:
        -----
        epoch : int
            当前训练轮次 (从 0 开始)
        
        返回:
        -----
        float
            当前温度值
        """
        # 计算训练进度 (0.0 到 1.0)
        progress = min(epoch / self.num_epochs, 1.0)
        
        if self.schedule == 'linear':
            # 线性退火: temp = start + (end - start) * progress
            temp = self.temp_start + (self.temp_end - self.temp_start) * progress
        elif self.schedule == 'cosine':
            # 余弦退火: 前期下降慢，后期下降快
            temp = self.temp_end + (self.temp_start - self.temp_end) * \
                   (1 + np.cos(np.pi * progress)) / 2
        else:
            raise ValueError(f"不支持的退火策略: {self.schedule}")
        
        # 确保温度不低于最终温度
        return max(temp, self.temp_end)


class MetricsCalculator:
    """
    评估指标计算器 (Metrics Calculator)
    
    功能:
    -----
    1. 反归一化: 将模型输出从归一化空间转换回真实空间
    2. 计算评估指标: RMSE, MAE, MAPE, Pearson R
    
    支持的归一化方式:
    -----------------
    - MinMaxScaler: 线性缩放到 [0, 1]
    - StandardScaler: Z-Score 标准化
    - Log1p 变换: 对数变换 (用于目标变量)
    
    参数:
    -----
    scaler_data : dict or Scaler
        归一化器对象或包含归一化器的字典
        如果是字典，应包含:
        - 'target_scaler': 目标变量的归一化器
        - 'meta': 元数据，包含 'target_log1p' 标志
    
    示例:
    -----
    >>> with open('scalers.pkl', 'rb') as f:
    ...     scaler_data = pickle.load(f)
    >>> calc = MetricsCalculator(scaler_data)
    >>> preds_real = calc.inverse_transform(preds_normalized)
    >>> metrics = calc.compute_metrics(preds_real, targets_real)
    """

    def __init__(self, scaler_data):
        # 解析 scaler_data 结构
        if isinstance(scaler_data, dict) and 'target_scaler' in scaler_data:
            # 新版格式: 包含多个 scaler 的字典
            self.scaler = scaler_data['target_scaler']
            self.meta = scaler_data.get('meta', {})
            self.is_log1p = self.meta.get('target_log1p', False)
        else:
            # 旧版格式: 单个 scaler 对象
            self.scaler = scaler_data
            self.is_log1p = False

        # 识别 scaler 类型并提取参数
        if hasattr(self.scaler, 'scale_'):
            # MinMaxScaler
            self.scale_ = self.scaler.scale_[0]
            self.min_ = self.scaler.min_[0]
            self.scaler_type = 'minmax'
        elif hasattr(self.scaler, 'mean_'):
            # StandardScaler
            self.mean_ = self.scaler.mean_[0]
            self.scale_ = self.scaler.scale_[0]
            self.scaler_type = 'standard'
        else:
            raise ValueError(f"不支持的 scaler 类型: {type(self.scaler)}")

    def inverse_transform(self, y):
        """
        反归一化: 将归一化后的数据转换回真实空间
        
        转换步骤:
        ---------
        1. 反归一化 (MinMax 或 Standard)
        2. 反对数变换 (如果使用了 log1p)
        3. 截断负值 (确保非负)
        
        参数:
        -----
        y : torch.Tensor or np.ndarray
            归一化后的数据
        
        返回:
        -----
        np.ndarray
            真实空间的数据
        """
        # 转换为 numpy 数组
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        y = y.astype(np.float64)

        # 步骤 1: 反归一化
        if self.scaler_type == 'minmax':
            # MinMaxScaler 反变换: y_real = (y_norm - min) / scale
            y_restored = (y - self.min_) / self.scale_
        elif self.scaler_type == 'standard':
            # StandardScaler 反变换: y_real = y_norm * scale + mean
            y_restored = y * self.scale_ + self.mean_
        else:
            y_restored = y

        # 步骤 2: 反对数变换
        if self.is_log1p:
            # log1p 的反变换: expm1(x) = exp(x) - 1
            y_restored = np.expm1(y_restored)

        # 步骤 3: 截断负值 (游客数量不能为负)
        y_restored = np.maximum(y_restored, 0)
        
        return y_restored

    def compute_metrics(self, preds, targets):
        """
        计算评估指标
        
        指标说明:
        ---------
        - RMSE (Root Mean Squared Error): 均方根误差，衡量预测值与真实值的偏差
        - MAE (Mean Absolute Error): 平均绝对误差，对异常值不敏感
        - MAPE (Mean Absolute Percentage Error): 平均绝对百分比误差，相对误差
        - Pearson R: 皮尔逊相关系数，衡量预测值与真实值的线性相关性
        
        参数:
        -----
        preds : np.ndarray
            预测值 (真实空间)
        targets : np.ndarray
            真实值 (真实空间)
        
        返回:
        -----
        dict
            包含各项指标的字典
        """
        # RMSE: 均方根误差
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        
        # MAE: 平均绝对误差
        mae = np.mean(np.abs(preds - targets))
        
        # MAPE: 平均绝对百分比误差
        epsilon = 1e-8  # 防止除零
        mape = np.mean(np.abs((preds - targets) / (np.abs(targets) + epsilon))) * 100
        
        # Pearson R: 皮尔逊相关系数
        pearson_r = np.corrcoef(preds.flatten(), targets.flatten())[0, 1] if len(preds) > 1 else 0.0

        return {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'Pearson_R': float(pearson_r)
        }


class CCCLoss(nn.Module):
    """
    一致性相关系数损失 (Concordance Correlation Coefficient Loss)
    
    功能:
    -----
    统一解决 RMSE 与 Pearson 指标冲突问题，同时考虑:
    1. 数值准确性: 均值偏差、方差偏差
    2. 趋势相关性: Pearson 相关系数
    
    公式:
    -----
    CCC = (2 * cov(pred, target)) / 
          (var(pred) + var(target) + (mean(pred) - mean(target))^2)
    
    Loss = 1 - CCC
    
    取值范围:
    ---------
    - CCC ∈ [-1, 1]: 1 表示完美一致，-1 表示完全相反
    - Loss ∈ [0, 2]: 0 表示完美拟合，2 表示完全相反
    
    优势:
    -----
    1. 避免 RMSE 与 Pearson 的权重平衡问题
    2. 单一损失函数，训练更稳定
    3. 对尺度不敏感，适用于不同量级的数据
    
    参数:
    -----
    eps : float
        数值稳定性参数，防止除零 (默认 1e-8)
    
    示例:
    -----
    >>> criterion = CCCLoss()
    >>> loss = criterion(predictions, targets)
    """

    def __init__(self, eps=1e-8):
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        计算 CCC Loss
        
        参数:
        -----
        pred : torch.Tensor
            预测值，形状 (Batch, ...) 或 (Batch, PredLen, 1)
        target : torch.Tensor
            真实值，形状与 pred 相同
        
        返回:
        -----
        torch.Tensor
            标量损失值
        """
        # 确保输入维度一致，展平多余维度
        if pred.dim() > 2:
            pred = pred.squeeze(-1)  # (Batch, PredLen, 1) → (Batch, PredLen)
        if target.dim() > 2:
            target = target.squeeze(-1)

        # 步骤 1: 计算统计量 (在 Batch 维度上聚合)
        mean_pred = torch.mean(pred, dim=0)      # 预测值的均值
        mean_target = torch.mean(target, dim=0)  # 真实值的均值

        # 计算方差 (unbiased=False 表示总体方差，与均值计算方式一致)
        var_pred = torch.var(pred, dim=0, unbiased=False)
        var_target = torch.var(target, dim=0, unbiased=False)

        # 步骤 2: 计算协方差 (Covariance)
        pred_centered = pred - mean_pred      # 中心化预测值
        target_centered = target - mean_target  # 中心化真实值
        covariance = torch.mean(pred_centered * target_centered, dim=0)

        # 步骤 3: 计算 CCC
        # CCC = (2 * cov) / (var_pred + var_target + (mu_pred - mu_target)^2)
        numerator = 2 * covariance
        denominator = var_pred + var_target + (mean_pred - mean_target) ** 2 + self.eps

        ccc = numerator / denominator

        # 步骤 4: 计算 Loss (1 - CCC)
        # 如果有多个时间步或特征，取平均
        loss = 1.0 - ccc
        return loss.mean()


class EarlyStopping:
    """
    早停机制 (Early Stopping)
    
    功能:
    -----
    监控验证集性能，当性能不再提升时提前终止训练，防止过拟合。
    
    判断逻辑:
    ---------
    1. 记录历史最佳性能 (best_score)
    2. 如果当前性能未超过 best_score + min_delta，计数器 +1
    3. 如果计数器达到 patience，触发早停
    4. 如果性能提升，重置计数器
    
    参数:
    -----
    patience : int
        耐心值，性能多少轮不提升后触发早停 (默认 20)
    min_delta : float
        最小改善阈值，性能提升小于该值视为没有改善 (默认 1e-4)
    verbose : bool
        是否打印早停信息 (默认 True)
    
    属性:
    -----
    early_stop : bool
        是否触发早停
    counter : int
        当前计数器值
    best_score : float
        历史最佳性能
    
    示例:
    -----
    >>> early_stopping = EarlyStopping(patience=20, min_delta=1e-4)
    >>> for epoch in range(num_epochs):
    ...     val_loss = validate()
    ...     early_stopping(val_loss)
    ...     if early_stopping.early_stop:
    ...         break
    """

    def __init__(self, patience=20, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_metric):
        """
        更新早停状态
        
        参数:
        -----
        val_metric : float
            验证集指标 (Loss，越小越好)
        """
        # 将 Loss 转换为 score (越大越好)
        score = -val_metric

        if self.best_score is None:
            # 第一次调用，初始化 best_score
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            # 性能没有显著提升
            self.counter += 1
            if self.verbose:
                print(f'   ⏳ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # 达到耐心值，触发早停
                self.early_stop = True
        else:
            # 性能提升，重置计数器
            self.best_score = score
            self.counter = 0


class UnifiedTrainer:
    """
    统一联合训练器 (Unified Joint Trainer)
    
    功能:
    -----
    实现端到端的模型训练流程，包括:
    1. CCC Loss 计算
    2. 自适应峰值加权 (可选)
    3. 温度退火
    4. 早停机制
    5. 模型保存与评估
    
    训练流程:
    ---------
    1. 初始化: 创建优化器、调度器、早停器
    2. 预计算: 分析训练集分布，确定峰值阈值
    3. 训练循环:
       a. 前向传播: 计算预测值
       b. 损失计算: CCC Loss + 自适应加权 MSE (可选) + L1 正则化
       c. 反向传播: 更新模型参数
       d. 验证: 计算验证集性能
       e. 保存: 保存最佳模型
       f. 早停: 判断是否提前终止
    4. 评估: 加载最佳模型，在测试集上评估
    
    参数:
    -----
    model : nn.Module
        待训练的模型
    train_loader : DataLoader
        训练集数据加载器
    val_loader : DataLoader
        验证集数据加载器
    test_loader : DataLoader
        测试集数据加载器
    metrics_calculator : MetricsCalculator
        评估指标计算器
    config : TrainingConfig
        训练配置对象
    
    示例:
    -----
    >>> trainer = UnifiedTrainer(model, train_loader, val_loader, 
    ...                          test_loader, metrics_calc, cfg)
    >>> trainer.train()
    """

    def __init__(self, model, train_loader, val_loader, test_loader,
                 metrics_calculator, config=cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics_calc = metrics_calculator
        self.config = config

        # 1. 初始化损失函数: CCC Loss
        self.criterion = CCCLoss()
        print(f"🔧 Loss Function: CCCLoss (Concordance Correlation Coefficient)")

        # 2. 初始化自适应峰值加权机制
        # 在训练开始前，统计训练集分布，确定峰值阈值
        self.device = config.DEVICE

        if getattr(self.config, 'USE_ADAPTIVE_WEIGHT', False):
            # 启用自适应加权，计算动态阈值
            self.peak_threshold = self._calculate_adaptive_threshold()
        else:
            # 禁用自适应加权
            print("ℹ️ 自适应加权已禁用 (Config.USE_ADAPTIVE_WEIGHT = False)")
            self.peak_threshold = float('inf')  # 阈值无限大，不会触发加权

        # 3. 初始化早停机制
        if config.EARLY_STOP:
            print(f"🔧 早停机制已启用: Patience={config.PATIENCE}, Delta={config.MIN_DELTA}")
            self.early_stopping = EarlyStopping(
                patience=config.PATIENCE,
                min_delta=config.MIN_DELTA
            )
        else:
            self.early_stopping = None

        # 4. 初始化优化器和学习率调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 5. 初始化温度调度器
        self.temp_scheduler = TemperatureScheduler(
            config.TEMP_START, config.TEMP_END, config.NUM_EPOCHS, config.TEMP_SCHEDULE
        )

        # 6. 初始化训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_log = []

    def _calculate_adaptive_threshold(self):
        """
        计算自适应峰值阈值
        
        功能:
        -----
        遍历训练集，计算目标变量的均值和标准差，
        根据公式 Threshold = Mean + Sigma * Std 确定峰值阈值。
        
        返回:
        -----
        torch.Tensor
            峰值阈值 (在 device 上)
        
        说明:
        -----
        - 该阈值用于识别"高峰"样本
        - 超过阈值的样本将在训练时被施加更高的损失权重
        - 阈值基于归一化后的数据计算 (Model Input Space)
        """
        sigma = getattr(self.config, 'PEAK_SIGMA', 1.5)
        weight = getattr(self.config, 'PEAK_PENALTY_WEIGHT', 5.0)

        print(f"\n📊 正在分析训练集分布以设定自适应阈值 (Sigma={sigma})...")

        all_targets = []
        # 遍历训练集，收集所有标签
        for _, y in self.train_loader:
            all_targets.append(y)

        all_targets = torch.cat(all_targets, dim=0)

        # 计算统计量
        train_mean = torch.mean(all_targets).item()
        train_std = torch.std(all_targets).item()

        # 计算动态阈值
        threshold = train_mean + sigma * train_std

        print(f"   Stats (Normalized): Mean={train_mean:.4f}, Std={train_std:.4f}")
        print(f"   🎯 动态峰值阈值 (Threshold): {threshold:.4f}")
        print(f"   说明: 超过此值的样本将被施加 {weight} 倍权重的惩罚")
        print(f"{'-' * 70}")

        # 将阈值转为 tensor 并移至 device
        return torch.tensor(threshold).to(self.device)

    def _create_optimizer(self):
        """
        创建优化器
        
        返回:
        -----
        torch.optim.Optimizer
            优化器对象
        """
        if self.config.OPTIMIZER == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                betas=self.config.BETAS, 
                eps=self.config.EPS
            )
        else:
            return torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config.LEARNING_RATE
            )

    def _create_scheduler(self):
        """
        创建学习率调度器
        
        返回:
        -----
        torch.optim.lr_scheduler._LRScheduler or None
            学习率调度器对象
        """
        if self.config.SCHEDULER == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.NUM_EPOCHS, 
                eta_min=self.config.LR_MIN
            )
        return None

    def train_epoch(self, epoch):
        """
        训练一个 epoch
        
        流程:
        -----
        1. 设置模型为训练模式
        2. 获取当前温度值
        3. 遍历训练集:
           a. 前向传播
           b. 计算损失 (CCC Loss + 自适应加权 MSE + L1 正则化)
           c. 反向传播
           d. 梯度裁剪
           e. 更新参数
        4. 返回平均损失
        
        参数:
        -----
        epoch : int
            当前训练轮次
        
        返回:
        -----
        tuple
            (平均损失, 当前温度)
        """
        # 设置模型为训练模式
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        # 获取当前温度
        current_temp = self.temp_scheduler.get_temperature(epoch)
        total_loss = 0.0

        # 监控各项损失
        total_ccc_loss = 0.0
        total_peak_loss = 0.0

        # 创建进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.NUM_EPOCHS}')

        for x, y in pbar:
            # 数据移至 device
            x = x.to(self.device)
            y = y.to(self.device)

            # 梯度清零
            self.optimizer.zero_grad()

            # 前向传播
            predictions, expanded_weights = self.model(x, temperature=current_temp)

            # ==================== 损失计算 ====================

            # 1. 基础损失: CCC Loss
            # 功能: 统一考虑数值准确性和趋势相关性
            # 范围: 0 (完美) 到 2 (完全反向)
            ccc_loss = self.criterion(predictions, y)
            loss = ccc_loss

            # 2. 自适应加权 MSE (可选)
            # 功能: 解决峰值削峰问题，强迫模型关注高数值样本
            peak_loss_val = 0.0
            if getattr(self.config, 'USE_ADAPTIVE_WEIGHT', False):
                # 计算原始 MSE
                pred_view = predictions.view_as(y)
                mse_raw = (pred_view - y) ** 2

                # 初始化权重向量 (默认为 1.0)
                weights = torch.ones_like(mse_raw)

                # 识别高峰样本: y > threshold
                peak_mask = y > self.peak_threshold

                # 获取惩罚倍数
                penalty = getattr(self.config, 'PEAK_PENALTY_WEIGHT', 5.0)

                # 对高峰样本施加惩罚权重
                weights[peak_mask] = penalty

                # 计算加权 MSE
                weighted_mse = (weights * mse_raw).mean()

                # 获取加权 MSE 的系数
                lambda_peak = getattr(self.config, 'LOSS_WEIGHT_PEAK_MSE', 0.5)

                # 添加到总损失
                peak_loss_term = lambda_peak * weighted_mse
                loss += peak_loss_term