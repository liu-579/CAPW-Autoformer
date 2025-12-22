# -*- coding: utf-8 -*-
"""
M10 三阶段渐进式训练脚本 (Fixed Version)
Three-Phase Progressive Training Pipeline
用途: 加载 m8b 数据，训练 m9 模型，保存最佳模型
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

warnings.filterwarnings('ignore')

# 导入模型和配置
sys.path.append('.')
from m9_model import build_model
from config.m9_config import ModelConfig
from config.m10_config import TrainingConfig as cfg


class TemperatureScheduler:
    """温度退火调度器"""

    def __init__(self, temp_start, temp_end, num_epochs, schedule='linear'):
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.num_epochs = num_epochs
        self.schedule = schedule

    def get_temperature(self, epoch):
        """获取当前 epoch 的温度"""
        # 确保 epoch 不超过 num_epochs (虽然逻辑上不应该，但防止越界)
        progress = min(epoch / self.num_epochs, 1.0)

        if self.schedule == 'linear':
            temp = self.temp_start + (self.temp_end - self.temp_start) * progress
        elif self.schedule == 'cosine':
            temp = self.temp_end + (self.temp_start - self.temp_end) * \
                   (1 + np.cos(np.pi * progress)) / 2
        else:
            raise ValueError(f"不支持的退火策略: {self.schedule}")

        return max(temp, self.temp_end)


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self, scaler, target_idx):
        """
        Args:
            scaler: sklearn 归一化器
            target_idx: 目标列索引
        """
        self.scaler = scaler
        self.target_idx = target_idx

        # 提取目标列的归一化参数
        if hasattr(scaler, 'mean_'):
            # StandardScaler
            self.target_mean = scaler.mean_[target_idx]
            self.target_scale = scaler.scale_[target_idx]
            self.scaler_type = 'standard'
        elif hasattr(scaler, 'min_'):
            # MinMaxScaler
            self.target_min = scaler.min_[target_idx]
            self.target_scale = scaler.scale_[target_idx]
            self.target_data_min = scaler.data_min_[target_idx]
            self.target_data_max = scaler.data_max_[target_idx]
            self.scaler_type = 'minmax'
        else:
            raise ValueError(f"不支持的 scaler 类型: {type(scaler)}")

        print(f"✓ 指标计算器初始化:")
        print(f"  - Scaler 类型: {self.scaler_type}")
        print(f"  - 目标列索引: {target_idx}")

    def inverse_transform(self, y):
        """
        反归一化 - 只对目标列进行反归一化
        Args:
            y: torch.Tensor [B, T, 1] 或 [B*T, 1]
        Returns:
            y_real: numpy array [N,]
        """
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # 展平
        y_flat = y.reshape(-1)  # [N,]

        # 根据 scaler 类型反归一化
        if self.scaler_type == 'standard':
            # StandardScaler: x_real = x_norm * scale + mean
            y_real = y_flat * self.target_scale + self.target_mean
        elif self.scaler_type == 'minmax':
            # MinMaxScaler: x_real = (x_norm - min) / scale
            y_real = y_flat / self.target_scale + self.target_data_min

        return y_real

    def compute_metrics(self, preds, targets):
        """
        计算评估指标
        Args:
            preds: numpy array - 预测值
            targets: numpy array - 真实值
        Returns:
            metrics: dict - 包含 RMSE, MAE, MAPE, Pearson_R
        """
        # RMSE
        rmse = np.sqrt(np.mean((preds - targets) ** 2))

        # MAE
        mae = np.mean(np.abs(preds - targets))

        # MAPE (避免除零)
        epsilon = 1e-8
        mape = np.mean(np.abs((preds - targets) / (np.abs(targets) + epsilon))) * 100

        # Pearson 相关系数
        pearson_r = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0.0

        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Pearson_R': pearson_r
        }


class ThreePhaseTrainer:
    """三阶段渐进式训练器"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 metrics_calculator, config=cfg):
        """
        Args:
            model: DynamicFusionPredictor
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            metrics_calculator: MetricsCalculator 实例
            config: TrainingConfig 配置类
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics_calc = metrics_calculator
        self.config = config

        # 损失函数
        self.mse_loss = nn.MSELoss()

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 温度调度器
        self.temp_scheduler = TemperatureScheduler(
            config.TEMP_START,
            config.TEMP_END,
            config.NUM_EPOCHS,
            config.TEMP_SCHEDULE
        )

        # 训练状态
        self.current_epoch = 0
        self.current_phase = None
        self.current_temperature = config.TEMP_START
        self.best_val_rmse = float('inf')
        self.patience_counter = 0
        self.training_log = []

        # 设备
        self.device = config.DEVICE

    def _create_optimizer(self):
        """创建优化器"""
        if self.config.OPTIMIZER == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                betas=self.config.BETAS,
                eps=self.config.EPS
            )
        elif self.config.OPTIMIZER == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                betas=self.config.BETAS,
                eps=self.config.EPS
            )
        elif self.config.OPTIMIZER == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                momentum=0.9
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.OPTIMIZER}")

    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.SCHEDULER == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.NUM_EPOCHS,
                eta_min=self.config.LR_MIN
            )
        elif self.config.SCHEDULER == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.LR_STEP_SIZE,
                gamma=self.config.LR_GAMMA
            )
        elif self.config.SCHEDULER == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.LR_FACTOR,
                patience=self.config.LR_PATIENCE,
                verbose=True
            )
        elif self.config.SCHEDULER == 'none':
            return None
        else:
            raise ValueError(f"不支持的调度器: {self.config.SCHEDULER}")

    def set_training_phase(self, epoch):
        """根据 epoch 设置训练阶段并冻结/解冻参数"""
        if epoch < self.config.PHASE_1_EPOCHS:
            if self.current_phase != 'Phase1':
                print(f"\n{'=' * 70}")
                print(f"🔥 进入 Phase 1: 预热骨干网络 (Epoch {epoch})")
                print(f"   策略: 冻结权重网络，只训练 Transformer 骨干")
                print(f"   目标: 学习基本时序模式")
                print(f"{'=' * 70}\n")

                self.model.unfreeze_all()
                self.model.freeze_weight_net()
                self.current_phase = 'Phase1'

                # 打印可训练参数量
                trainable = self.model.count_trainable_params()
                print(f"   可训练参数: {trainable:,}")

        elif epoch < self.config.PHASE_1_EPOCHS + self.config.PHASE_2_EPOCHS:
            if self.current_phase != 'Phase2':
                print(f"\n{'=' * 70}")
                print(f"⚡ 进入 Phase 2: 权重网络学习 (Epoch {epoch})")
                print(f"   策略: 冻结骨干网络，只训练权重生成网络")
                print(f"   目标: 学习情境感知的动态权重")
                print(f"{'=' * 70}\n")

                self.model.unfreeze_all()
                self.model.freeze_backbone()
                self.current_phase = 'Phase2'

                trainable = self.model.count_trainable_params()
                print(f"   可训练参数: {trainable:,}")

                if trainable == 0:
                    raise RuntimeError("Phase 2 初始化失败：所有参数均被冻结！")

        else:
            if self.current_phase != 'Phase3':
                print(f"\n{'=' * 70}")
                print(f"🚀 进入 Phase 3: 端到端联合微调 (Epoch {epoch})")
                print(f"   策略: 解冻所有参数")
                print(f"   目标: 全局优化，融会贯通")
                print(f"{'=' * 70}\n")

                self.model.unfreeze_all()
                self.current_phase = 'Phase3'

                trainable = self.model.count_trainable_params()
                print(f"   可训练参数: {trainable:,}")

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()

        # 1. 设置训练阶段 (关键修复: 确保每轮开始都检查阶段)
        self.set_training_phase(epoch)

        # 2. 计算当前 epoch 的温度 (关键修复: 随 epoch 更新)
        current_temp = self.temp_scheduler.get_temperature(epoch)
        self.current_temperature = current_temp

        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0
        total_div = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.NUM_EPOCHS}')

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device)  # [B, T, F]
            y = y.to(self.device)  # [B, T, 1]

            self.optimizer.zero_grad()

            # 前向传播 (关键修复: 传入温度，只接收两个返回值)
            predictions, expanded_weights = self.model(x, temperature=current_temp)

            # 确保预测值需要梯度
            if not predictions.requires_grad:
                raise RuntimeError(
                    f"预测值不需要梯度! 当前阶段: {self.current_phase}\n"
                    f"可训练参数数量: {self.model.count_trainable_params()}"
                )

            # --- 损失计算 ---

            # 1. MSE Loss (主任务)
            mse_loss = F.mse_loss(predictions, y)

            # 2. L1 Loss (稀疏性)
            l1_loss = self.model.compute_weight_l1_loss(expanded_weights)

            # 3. Diversity Loss (多样性，仅在 Phase 1/3 计算，Phase 2 冻结骨干时不影响)
            # 在 Phase 2，虽然权重网在更新，但我们主要关注它对 Loss 的下降，多样性是辅助
            diversity_loss = torch.tensor(0.0, device=self.device)
            if self.config.USE_DIVERSITY_LOSS:
                # 只有当权重确实参与梯度计算时才算 loss
                if expanded_weights.requires_grad:
                    diversity_loss = self.model.compute_group_diversity_loss(expanded_weights)

            # 总损失组合 (修复: 使用 Config 中的 Lambda)
            loss = (mse_loss +
                    self.config.L1_LAMBDA * l1_loss +
                    self.config.DIVERSITY_LAMBDA * diversity_loss)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.MAX_GRAD_NORM
                )

            self.optimizer.step()

            # 累计损失
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()
            total_div += diversity_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MSE': f'{mse_loss.item():.4f}',
                'Temp': f'{current_temp:.2f}'
            })

        n_batches = len(self.train_loader)
        return (total_loss / n_batches,
                total_mse / n_batches,
                total_l1 / n_batches,
                total_div / n_batches,
                current_temp)

    @torch.no_grad()
    def validate(self, data_loader, phase='Val'):
        """验证/测试"""
        self.model.eval()

        all_preds = []
        all_targets = []
        all_weights = []

        # 验证/测试时使用最终的低温 (模拟实际应用场景)
        eval_temp = self.config.TEMP_END

        for x, y in tqdm(data_loader, desc=f'{phase} Evaluation', leave=False):
            x = x.to(self.device)
            y = y.to(self.device)

            # 前向传播
            preds, weights = self.model(x, temperature=eval_temp)

            all_preds.append(preds)
            all_targets.append(y)
            all_weights.append(weights)

        # 合并所有批次
        all_preds = torch.cat(all_preds, dim=0)  # [N, Pred_Len, 1]
        all_targets = torch.cat(all_targets, dim=0)  # [N, Pred_Len, 1]
        all_weights = torch.cat(all_weights, dim=0)  # [N, Seq_Len, Sentiment_Dim]

        # 反归一化
        preds_real = self.metrics_calc.inverse_transform(all_preds)
        targets_real = self.metrics_calc.inverse_transform(all_targets)

        # 计算指标
        metrics = self.metrics_calc.compute_metrics(preds_real, targets_real)

        # 保存权重统计 (仅测试集)
        if phase == 'Test' and self.config.SAVE_ATTENTION_WEIGHTS:
            weight_stats = self.model.get_weight_statistics(all_weights)
            group_stats = self.model.get_group_weight_statistics(all_weights)
            metrics['weight_stats'] = weight_stats
            metrics['group_stats'] = group_stats

        return metrics, (preds_real, targets_real) if phase == 'Test' else (None, None)

    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_rmse': self.best_val_rmse,
            'val_metrics': val_metrics,
            'config': {
                'LEARNING_RATE': self.config.LEARNING_RATE,
                'BATCH_SIZE': self.config.BATCH_SIZE,
                'NUM_EPOCHS': self.config.NUM_EPOCHS,
                'PHASE_1_EPOCHS': self.config.PHASE_1_EPOCHS,
                'PHASE_2_EPOCHS': self.config.PHASE_2_EPOCHS,
            }
        }

        # 保存最新模型
        torch.save(checkpoint, self.config.SAVE_LATEST_MODEL)

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.config.SAVE_BEST_MODEL)
            print(f"✓ 保存最佳模型 (RMSE: {val_metrics['RMSE']:.4f})")

    def train(self):
        """完整训练流程"""
        print(f"\n{'=' * 70}")
        print("开始三阶段渐进式训练")
        print(f"{'=' * 70}")
        print(f"总轮数:   {self.config.NUM_EPOCHS}")
        print(f"Phase 1:  {self.config.PHASE_1_EPOCHS} epochs (预热骨干)")
        print(f"Phase 2:  {self.config.PHASE_2_EPOCHS} epochs (权重学习)")
        print(
            f"Phase 3:  {self.config.NUM_EPOCHS - self.config.PHASE_1_EPOCHS - self.config.PHASE_2_EPOCHS} epochs (联合微调)")
        print(f"{'=' * 70}\n")

        start_time = datetime.now()

        for epoch in range(self.config.NUM_EPOCHS):
            # 训练一个 epoch
            train_loss, train_mse, train_l1, train_div, temp = self.train_epoch(epoch)

            # 验证
            val_metrics, _ = self.validate(self.val_loader, phase='Val')

            # 更新学习率调度器
            if self.scheduler is not None:
                if self.config.SCHEDULER == 'plateau':
                    self.scheduler.step(val_metrics['RMSE'])
                else:
                    self.scheduler.step()

            # 记录日志
            log_entry = {
                'Epoch': epoch + 1,
                'Phase': self.current_phase,
                'Train_Loss': train_loss,
                'Train_MSE': train_mse,
                'Train_L1': train_l1,
                'Train_Diversity': train_div,
                'Temperature': temp,
                'LR': self.optimizer.param_groups[0]['lr'],
                **{f'Val_{k}': v for k, v in val_metrics.items()
                   if k not in ['weight_stats', 'group_stats']}
            }
            self.training_log.append(log_entry)

            # 打印验证结果
            print(f"\n📊 Epoch {epoch + 1} ({self.current_phase}) 验证结果:")
            print(f"   RMSE:      {val_metrics['RMSE']:.4f}")
            print(f"   MAE:       {val_metrics['MAE']:.4f}")
            print(f"   MAPE:      {val_metrics['MAPE']:.2f}%")
            print(f"   Pearson R: {val_metrics['Pearson_R']:.4f}")

            # 早停检查
            is_best = val_metrics['RMSE'] < self.best_val_rmse
            if is_best:
                self.best_val_rmse = val_metrics['RMSE']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 保存检查点
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)

            # 早停
            if self.config.EARLY_STOP and self.patience_counter >= self.config.PATIENCE:
                print(f"\n⚠️  早停触发! 验证集 {self.config.PATIENCE} 轮无改善")
                break

        # 保存训练日志
        log_df = pd.DataFrame(self.training_log)
        log_df.to_csv(self.config.SAVE_LOG, index=False)
        print(f"\n✓ 训练日志已保存: {self.config.SAVE_LOG}")

        # 计算训练时长
        training_time = datetime.now() - start_time
        print(f"\n⏱️  总训练时长: {training_time}")

        # 最终测试
        self.final_test()

    def final_test(self):
        """加载最佳模型并在测试集上评估"""
        print(f"\n{'=' * 70}")
        print("🎯 最终测试阶段")
        print(f"{'=' * 70}\n")

        # 加载最佳模型
        if self.config.SAVE_BEST_MODEL.exists():
            checkpoint = torch.load(self.config.SAVE_BEST_MODEL, weights_only=False)    # 添加 weights_only=False 以允许加载包含 numpy 数据的字典
            # checkpoint = torch.load(self.config.SAVE_BEST_MODEL)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 加载最佳模型 (Epoch {checkpoint['epoch'] + 1}, "
                  f"Val RMSE: {checkpoint['best_val_rmse']:.4f})")
        else:
            print("⚠️ 未找到最佳模型文件，使用当前模型进行测试")

        # 测试集评估
        test_metrics, (preds, targets) = self.validate(self.test_loader, phase='Test')

        # 打印测试结果
        print(f"\n{'=' * 70}")
        print("📈 测试集最终结果")
        print(f"{'=' * 70}")
        print(f"RMSE:      {test_metrics['RMSE']:.4f}")
        print(f"MAE:       {test_metrics['MAE']:.4f}")
        print(f"MAPE:      {test_metrics['MAPE']:.2f}%")
        print(f"Pearson R: {test_metrics['Pearson_R']:.4f}")
        print(f"{'=' * 70}\n")

        # 打印权重统计
        if 'weight_stats' in test_metrics:
            print(f"\n{'=' * 70}")
            print("🔍 权重统计分析")
            print(f"{'=' * 70}")

            stats = test_metrics['weight_stats']
            print(f"\n整体统计:")
            print(f"  - 平均权重:  {stats['mean'].mean():.6f}")
            print(f"  - 权重标准差: {stats['std'].mean():.6f}")
            print(f"  - 稀疏度:     {stats['sparsity']:.2%}")

            group_stats = test_metrics['group_stats']
            print(f"\n分组统计:")
            for i, (group_name, stat) in enumerate(group_stats.items()):
                print(f"  组 {i} ({group_name}): mean={stat['mean']:.4f}, "
                      f"std={stat['std']:.4f}, size={stat['size']}")

            print(f"{'=' * 70}\n")

        # 保存测试结果
        test_result = {
            'test_metrics': {k: float(v) if not isinstance(v, dict) else v
                             for k, v in test_metrics.items()
                             if k not in ['weight_stats', 'group_stats']},
            'group_stats': test_metrics.get('group_stats', {}),
        }

        with open(self.config.SAVE_TEST_RESULT, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, indent=4, ensure_ascii=False)

        print(f"✓ 测试结果已保存: {self.config.SAVE_TEST_RESULT}")

        # 保存预测结果
        if self.config.SAVE_PRED_VS_TRUE:
            np.savez(
                self.config.SAVE_DIR / 'predictions.npz',
                predictions=preds,
                targets=targets
            )
            print(f"✓ 预测值已保存: {self.config.SAVE_DIR / 'predictions.npz'}")


# ==================== 数据加载函数 ====================

def load_data(config=cfg):
    """加载训练数据"""
    print("\n" + "=" * 70)
    print("📂 加载数据...")
    print("=" * 70)

    # 加载 numpy 数据
    try:
        train_x = np.load(config.TRAIN_X)
        train_y = np.load(config.TRAIN_Y)
        val_x = np.load(config.VAL_X)
        val_y = np.load(config.VAL_Y)
        test_x = np.load(config.TEST_X)
        test_y = np.load(config.TEST_Y)
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到数据文件。请检查路径: {config.DATA_DIR}")
        raise e

    print(f"\n✓ 数据形状:")
    print(f"  - 训练集: X={train_x.shape}, Y={train_y.shape}")
    print(f"  - 验证集: X={val_x.shape}, Y={val_y.shape}")
    print(f"  - 测试集: X={test_x.shape}, Y={test_y.shape}")

    # 加载 feature_map
    with open(config.FEATURE_MAP, 'r', encoding='utf-8') as f:
        feature_map = json.load(f)

    print(f"\n✓ 特征映射:")
    print(f"  - 总特征数:     {feature_map['total_features']}")
    print(f"  - 情感组数:     {len(feature_map['sentiment_groups'])}")
    print(f"  - 环境特征数:   {len(feature_map['context_idxs'])}")
    print(f"  - 目标列索引:   {feature_map['target_idx']}")

    # 加载归一化器
    with open(config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print(f"\n✓ 归一化器已加载")
    print("=" * 70 + "\n")

    return train_x, train_y, val_x, val_y, test_x, test_y, feature_map, scaler


def create_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y, config=cfg):
    """创建 DataLoader"""
    print("🔄 创建 DataLoader...")

    # 转换为 Tensor
    train_dataset = TensorDataset(
        torch.FloatTensor(train_x),
        torch.FloatTensor(train_y)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_x),
        torch.FloatTensor(val_y)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_x),
        torch.FloatTensor(test_y)
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    print(f"✓ DataLoader 创建完成")
    return train_loader, val_loader, test_loader


def create_model(feature_map, config=cfg):
    """创建模型"""
    print("🏗️  创建模型...")

    # 使用 m9 的 build_model 函数
    model = build_model(feature_map, ModelConfig)
    model = model.to(config.DEVICE)

    print(f"✓ 模型已移动到: {config.DEVICE}\n")

    return model


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ 随机种子已设置: {seed}")


# ==================== 主函数 ====================

def main():
    """主训练流程"""
    print("\n" + "=" * 70)
    print("M10 三阶段渐进式训练 (Fixed)")
    print("Dynamic Fusion Predictor - Three-Phase Training")
    print("=" * 70 + "\n")

    # 设置随机种子
    set_seed(cfg.SEED)

    # 验证配置
    cfg.validate()
    cfg.print_config()

    # 创建保存目录
    cfg.setup_dirs()

    # 加载数据
    train_x, train_y, val_x, val_y, test_x, test_y, feature_map, scaler = load_data()

    # 创建 DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_x, train_y, val_x, val_y, test_x, test_y
    )

    # 创建模型
    model = create_model(feature_map)

    # 创建指标计算器
    metrics_calculator = MetricsCalculator(scaler, feature_map['target_idx'])

    # 创建训练器
    trainer = ThreePhaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        metrics_calculator=metrics_calculator,
        config=cfg
    )

    # 开始训练
    trainer.train()

    print("\n" + "=" * 70)
    print("🎉 训练完成!")
    print(f"最佳模型已保存至: {cfg.SAVE_BEST_MODEL}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()