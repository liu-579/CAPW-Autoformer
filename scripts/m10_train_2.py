# -*- coding: utf-8 -*-
"""
M10 统一联合训练脚本 (CCC Loss Version + Adaptive Peak Weighting)
Unified Joint Training Pipeline with CCC Loss & Adaptive MSE
用途: 训练模型，使用 CCC Loss 解决 RMSE 与 Pearson 指标冲突问题，
      并引入自适应加权机制解决极值削峰(Peak Shaving)问题。
修改记录:
1. 引入 CCCLoss 替代 MSE + Pearson 组合损失
2. 统一早停和最佳模型判断标准为 CCC Loss
3. 【新增】启动时自动统计训练集分布，计算动态峰值阈值 (Mean + N*Std)
4. 【新增】自适应加权 MSE：对超过阈值的样本施加重罚，强迫模型拟合高峰
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
from config.m10_config import TrainingConfig as cfg


class TemperatureScheduler:
    """温度退火调度器"""

    def __init__(self, temp_start, temp_end, num_epochs, schedule='linear'):
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.num_epochs = num_epochs
        self.schedule = schedule

    def get_temperature(self, epoch):
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
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mae = np.mean(np.abs(preds - targets))
        epsilon = 1e-8
        mape = np.mean(np.abs((preds - targets) / (np.abs(targets) + epsilon))) * 100
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
    Loss = 1 - CCC
    兼顾数值准确性 (均值/方差偏差) 和 趋势相关性 (Pearson)
    """

    def __init__(self, eps=1e-8):
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        计算 CCC Loss
        Args:
            pred: 预测值 (Batch, ...)
            target: 真实值 (Batch, ...)
        Returns:
            loss: scalar
        """
        # 确保输入维度一致，展平不需要的维度，保留 Batch (dim=0)
        if pred.dim() > 2:
            pred = pred.squeeze(-1)
        if target.dim() > 2:
            target = target.squeeze(-1)

        # 1. 计算统计量 (在 Batch 维度 dim=0 上聚合)
        mean_pred = torch.mean(pred, dim=0)
        mean_target = torch.mean(target, dim=0)

        # unbiased=False 对应总体方差 (div by N)，与 mean (div by N) 保持一致
        var_pred = torch.var(pred, dim=0, unbiased=False)
        var_target = torch.var(target, dim=0, unbiased=False)

        # 2. 计算协方差 (Covariance)
        pred_centered = pred - mean_pred
        target_centered = target - mean_target
        covariance = torch.mean(pred_centered * target_centered, dim=0)

        # 3. 计算 CCC 公式
        # CCC = (2 * cov) / (var_pred + var_target + (mu_pred - mu_target)^2)
        numerator = 2 * covariance
        denominator = var_pred + var_target + (mean_pred - mean_target) ** 2 + self.eps

        ccc = numerator / denominator

        # 4. 计算 Loss (1 - CCC) 并取所有特征/时间步的平均
        loss = 1.0 - ccc
        return loss.mean()


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=20, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_metric):
        # 这里的 val_metric 是 Loss，越小越好
        # 转换为 score (越大越好)
        score = -val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            # 性能没有显著提升
            self.counter += 1
            if self.verbose:
                print(f'   ⏳ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 性能提升
            self.best_score = score
            self.counter = 0


class UnifiedTrainer:
    """
    统一联合训练器 (Unified Joint Trainer)
    包含: CCC Loss + 自适应峰值加权 (Adaptive Peak Weighting)
    """

    def __init__(self, model, train_loader, val_loader, test_loader,
                 metrics_calculator, config=cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics_calc = metrics_calculator
        self.config = config

        # 1. 基础 Loss: CCC Loss
        self.criterion = CCCLoss()
        print(f"🔧 Loss Function: CCCLoss (Concordance Correlation Coefficient)")

        # 2. 【新增】自适应阈值计算 (Pre-calculation)
        # 在训练开始前，统计训练集的分布，确定什么是"高峰"
        # 依赖 config 中的 PEAK_SIGMA (例如 1.5)
        self.device = config.DEVICE

        if getattr(self.config, 'USE_ADAPTIVE_WEIGHT', False):
            self.peak_threshold = self._calculate_adaptive_threshold()
        else:
            print("ℹ️ 自适应加权已禁用 (Config.USE_ADAPTIVE_WEIGHT = False)")
            self.peak_threshold = float('inf')  # 阈值无限大，实际上不会触发加权

        # 早停对象
        if config.EARLY_STOP:
            print(f"🔧 早停机制已启用: Patience={config.PATIENCE}, Delta={config.MIN_DELTA}")
            self.early_stopping = EarlyStopping(
                patience=config.PATIENCE,
                min_delta=config.MIN_DELTA
            )
        else:
            self.early_stopping = None

        # 优化器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.temp_scheduler = TemperatureScheduler(
            config.TEMP_START, config.TEMP_END, config.NUM_EPOCHS, config.TEMP_SCHEDULE
        )

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_log = []

    def _calculate_adaptive_threshold(self):
        """
        【新增】遍历训练集，计算 Mean/Std，动态设定峰值阈值
        """
        sigma = getattr(self.config, 'PEAK_SIGMA', 1.5)
        weight = getattr(self.config, 'PEAK_PENALTY_WEIGHT', 5.0)

        print(f"\n📊 正在分析训练集分布以设定自适应阈值 (Sigma={sigma})...")

        all_targets = []
        # 临时遍历一次 DataLoader 获取所有标签
        # 注意：这里获取的是归一化后的 y (Model Input Space)
        for _, y in self.train_loader:
            all_targets.append(y)

        all_targets = torch.cat(all_targets, dim=0)

        # 计算统计量
        train_mean = torch.mean(all_targets).item()
        train_std = torch.std(all_targets).item()

        # 定义动态阈值
        threshold = train_mean + sigma * train_std

        print(f"   Stats (Normalized): Mean={train_mean:.4f}, Std={train_std:.4f}")
        print(f"   🎯 动态峰值阈值 (Threshold): {threshold:.4f}")
        print(f"   说明: 超过此值的样本将被施加 {weight} 倍权重的惩罚")
        print(f"{'-' * 70}")

        # 将阈值转为 tensor 并存入 device，方便后续比较
        return torch.tensor(threshold).to(self.device)

    def _create_optimizer(self):
        if self.config.OPTIMIZER == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                betas=self.config.BETAS, eps=self.config.EPS
            )
        else:
            return torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)

    def _create_scheduler(self):
        if self.config.SCHEDULER == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.NUM_EPOCHS, eta_min=self.config.LR_MIN
            )
        return None

    def train_epoch(self, epoch):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        current_temp = self.temp_scheduler.get_temperature(epoch)
        total_loss = 0.0

        # 监控指标
        total_ccc_loss = 0.0
        total_peak_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.NUM_EPOCHS}')

        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            predictions, expanded_weights = self.model(x, temperature=current_temp)

            # ==================== Loss 计算逻辑 ====================

            # 1. 基础 CCC Loss (关注趋势和整体分布)
            # 范围: 0 (完美) 到 2 (完全反向)
            ccc_loss = self.criterion(predictions, y)
            loss = ccc_loss

            # 2. 【新增】自适应加权 MSE (关注峰值绝对精度)
            # 仅当启用且 config 存在参数时执行
            peak_loss_val = 0.0
            if getattr(self.config, 'USE_ADAPTIVE_WEIGHT', False):
                # 原始 MSE (Batch, 1)
                # 确保维度对齐
                pred_view = predictions.view_as(y)
                mse_raw = (pred_view - y) ** 2

                # 初始化权重向量，默认为 1.0
                weights = torch.ones_like(mse_raw)

                # 筛选高峰样本：y > threshold
                # peak_threshold 已经在 device 上
                peak_mask = y > self.peak_threshold

                # 获取惩罚倍数
                penalty = getattr(self.config, 'PEAK_PENALTY_WEIGHT', 5.0)

                # 对高峰样本施加惩罚权重
                weights[peak_mask] = penalty

                # 计算加权后的 MSE (Mean reduction)
                weighted_mse = (weights * mse_raw).mean()

                # 获取加权 MSE 在总 Loss 中的系数
                lambda_peak = getattr(self.config, 'LOSS_WEIGHT_PEAK_MSE', 0.5)

                # 添加到总 Loss
                peak_loss_term = lambda_peak * weighted_mse
                loss += peak_loss_term

                peak_loss_val = peak_loss_term.item()

            # 3. L1 正则化 (稀疏性约束)
            if self.config.L1_LAMBDA > 0:
                l1_loss = self.model.compute_weight_l1_loss(expanded_weights)
                loss += self.config.L1_LAMBDA * l1_loss

            # ==================== 反向传播 ====================

            loss.backward()

            if self.config.GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.MAX_GRAD_NORM
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_ccc_loss += ccc_loss.item()
            total_peak_loss += peak_loss_val

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CCC': f'{ccc_loss.item():.4f}',
                'PeakMSE': f'{peak_loss_val:.4f}'  # 显示加权项 Loss
            })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, current_temp

    @torch.no_grad()
    def validate(self, data_loader, phase='Val'):
        self.model.eval()
        all_preds = []
        all_targets = []

        eval_temp = self.config.TEMP_END

        for x, y in tqdm(data_loader, desc=f'{phase} Eval', leave=False):
            x = x.to(self.device)
            y = y.to(self.device)
            preds, _ = self.model(x, temperature=eval_temp)
            all_preds.append(preds)
            all_targets.append(y)

        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)

        # 计算验证集的 Loss (统一使用 CCC Loss 作为核心指标)
        val_ccc_loss = self.criterion(all_preds_tensor, all_targets_tensor)

        # 评估指标 (Real Space) - 依然计算 RMSE 和 Pearson 供观察
        preds_real = self.metrics_calc.inverse_transform(all_preds_tensor)
        targets_real = self.metrics_calc.inverse_transform(all_targets_tensor)

        metrics = self.metrics_calc.compute_metrics(preds_real, targets_real)

        # 返回 metrics, preds, targets, loss
        return metrics, preds_real, targets_real, val_ccc_loss.item()

    def save_best_model(self, val_loss):
        """保存最佳模型 (基于 CCC Loss)"""
        # Loss (1-CCC) 越小越好
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            save_path = self.config.SAVE_DIR / 'best_model_unified.pth'

            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_ccc_loss': val_loss
            }
            torch.save(checkpoint, save_path)

    def train(self):
        print(f"\n{'=' * 70}")
        print("开始联合训练 (Unified Joint Training with CCC + Adaptive Weighted MSE)")
        print(f"策略: End-to-End Training | Main Loss: CCC")
        if getattr(self.config, 'USE_ADAPTIVE_WEIGHT', False):
            print(f"增强: Adaptive Peak Weighting Enabled (Threshold based on Stats)")
        print(f"{'=' * 70}\n")

        start_time = datetime.now()

        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            train_loss, temp = self.train_epoch(epoch)

            # 验证
            val_metrics, _, _, val_loss = self.validate(self.val_loader, phase='Val')

            if self.scheduler is not None:
                self.scheduler.step()

            log_entry = {
                'Epoch': epoch + 1,
                'Train_Loss': train_loss,
                'Val_Loss': val_loss,  # CCC Loss
                'Val_RMSE': val_metrics['RMSE'],  # Real Space
                'Val_Pearson': val_metrics['Pearson_R']  # Real Space
            }
            self.training_log.append(log_entry)

            print(f"📊 Epoch {epoch + 1}: "
                  f"Loss={train_loss:.4f}, "
                  f"Val_Loss(CCC)={val_loss:.4f}, "
                  f"Val_RMSE={val_metrics['RMSE']:.2f}, "
                  f"Pearson={val_metrics['Pearson_R']:.4f}")

            # 保存最佳模型
            self.save_best_model(val_loss)

            # 早停
            if self.early_stopping:
                self.early_stopping(val_loss)

                if self.early_stopping.early_stop:
                    print(f"\n🛑 早停触发！在第 {epoch + 1} 轮停止训练。")
                    break

        # 保存日志
        log_df = pd.DataFrame(self.training_log)
        log_df.to_csv(self.config.SAVE_LOG, index=False)
        print(f"\n⏱️  总训练时长: {datetime.now() - start_time}")

        self.evaluate_final_model()

    def evaluate_final_model(self):
        model_path = self.config.SAVE_DIR / 'best_model_unified.pth'
        print(f"\n🔎 正在评估最佳模型: {model_path}")

        if not model_path.exists():
            print("⚠️  未找到模型文件。")
            return

        checkpoint = torch.load(model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics, preds, targets, _ = self.validate(self.test_loader, phase='Test')

        print(f"   📈 测试集最终结果 (Real Space):")
        print(f"      RMSE:      {test_metrics['RMSE']:.4f}")
        print(f"      MAE:       {test_metrics['MAE']:.4f}")
        print(f"      MAPE:      {test_metrics['MAPE']:.2f}%")
        print(f"      Pearson R: {test_metrics['Pearson_R']:.4f}")

        with open(self.config.SAVE_DIR / 'result_final.json', 'w') as f:
            json.dump(test_metrics, f, indent=4)

        self.plot_results(preds, targets, 'Final_Unified', test_metrics['RMSE'])

    def plot_results(self, preds, targets, name, rmse):
        """
        绘制对比图
        修正：处理多步预测 (Pred_Len > 1) 的情况，避免 flatten 导致数据重复
        """
        # 检查是否是多步预测 (Batch, Pred_Len, 1) 或 (Batch, Pred_Len)
        # 假设 preds 是 numpy array (由 inverse_transform 返回)

        if preds.ndim > 1 and preds.shape[1] > 1:
            print(f"   ℹ️ 检测到多步预测 (Pred_Len={preds.shape[1]})，仅提取 't+1' (第一步) 用于绘图...")
            # 只取预测序列的第 0 个时间步 (即预测的明天)
            # 这样长度就变回了 Batch Size
            preds_to_plot = preds[:, 0]

            # 对 targets 做同样处理
            if targets.ndim > 1 and targets.shape[1] > 1:
                targets_to_plot = targets[:, 0]
            else:
                targets_to_plot = targets.flatten()
        else:
            # 单步预测，直接展平
            preds_to_plot = preds.flatten()
            targets_to_plot = targets.flatten()

        plt.figure(figsize=(15, 6))
        # 使用处理后的数据绘图
        plt.plot(targets_to_plot, label='True Values (t+1)', alpha=0.7, color='blue', linewidth=1)
        plt.plot(preds_to_plot, label='Predictions (t+1)', alpha=0.7, color='red', linestyle='--', linewidth=1)

        plt.title(f'{name} Prediction (Test Set, RMSE: {rmse:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.SAVE_DIR / f'vis_trend_{name.lower()}.png', dpi=300)
        plt.close()
        print(f"   🖼️  图表已保存 (长度: {len(preds_to_plot)})")


# ==================== 数据加载与主函数 ====================

def load_data(config=cfg):
    print("\n📂 加载数据...")
    train_x = np.load(config.TRAIN_X)
    train_y = np.load(config.TRAIN_Y)
    val_x = np.load(config.VAL_X)
    val_y = np.load(config.VAL_Y)
    test_x = np.load(config.TEST_X)
    test_y = np.load(config.TEST_Y)

    with open(config.FEATURE_MAP, 'r', encoding='utf-8') as f:
        feature_map = json.load(f)

    with open(config.SCALER_PATH, 'rb') as f:
        scaler_data = pickle.load(f)

    return train_x, train_y, val_x, val_y, test_x, test_y, feature_map, scaler_data


def create_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y, config=cfg):
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y)),
                              batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y)),
                            batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y)),
                             batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    return train_loader, val_loader, test_loader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    print(f"\n{'=' * 70}\nM10 统一联合训练 (Adaptive Peak Weighting)\n{'=' * 70}")
    set_seed(cfg.SEED)
    cfg.setup_dirs()

    train_x, train_y, val_x, val_y, test_x, test_y, feature_map, scaler_data = load_data()
    train_loader, val_loader, test_loader = create_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y)

    model = build_model(feature_map, ModelConfig).to(cfg.DEVICE)
    metrics_calc = MetricsCalculator(scaler_data)

    trainer = UnifiedTrainer(model, train_loader, val_loader, test_loader, metrics_calc, cfg)
    trainer.train()


if __name__ == '__main__':
    main()