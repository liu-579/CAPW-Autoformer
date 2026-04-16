# -*- coding: utf-8 -*-
"""
M10 Autoformer + Hard-Gating MoE 训练脚本
Autoformer + Season-Aware MoE Training Pipeline

用途: 训练 Autoformer + MoE 模型
与 m10_train_2.py 的区别:
1. 导入 m9_model_autoformer_moe.build_model (MoE 版 Autoformer)
2. 导入 config.m9_config_autoformer_moe.ModelConfig (MoE 配置)
3. SAVE_DIR 修改为 MoE 专用目录
4. 训练/验证循环中: season_gate_mask 由模型内部自动从 x 中提取, 训练逻辑无需改动
5. 新增: 每个 epoch 结束后打印专家路由分布统计

关键设计:
- season_gate_mask 的提取逻辑已内置在 DynamicFusionPredictor._extract_season_mask() 中
- 训练器的 forward 调用 model(x, temperature=temp) 即可, season_gate_mask 自动提取
- 无需修改 DataLoader 或 Dataset, 完全向后兼容
"""

import copy
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
import platform

warnings.filterwarnings('ignore')

# ==================== 【核心修改】导入 MoE 版模型和配置 ====================
sys.path.append('.')
from m9_model_autoformer_moe import build_model               # MoE 版 Autoformer
from config.m9_config_autoformer_moe import ModelConfig        # MoE 版配置
from config.m10_config import TrainingConfig as cfg            # 训练配置 (复用)


# ==================== 【修改】覆盖保存目录为 MoE 专用 ====================
cfg.SAVE_DIR = Path(cfg.BASE_DIR / f'data/output/h_out_{cfg.scenic_name}/m10_checkpoints_{cfg.scenic_name}_autoformer_moe_h_4d')


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
    """评估指标计算器 (支持 MinMaxScaler 和 StandardScaler)"""

    def __init__(self, scaler_data):
        if isinstance(scaler_data, dict) and 'target_scaler' in scaler_data:
            self.scaler = scaler_data['target_scaler']
            self.meta = scaler_data.get('meta', {})
            self.is_log1p = self.meta.get('target_log1p', False)
            self.scaler_type = self.meta.get('target_scaler_type', None)
        else:
            self.scaler = scaler_data
            self.is_log1p = False
            self.scaler_type = None

        if self.scaler_type is None:
            if hasattr(self.scaler, 'min_'):
                self.scaler_type = 'minmax'
            elif hasattr(self.scaler, 'mean_'):
                self.scaler_type = 'standard'
            else:
                raise ValueError(f"不支持的 scaler 类型: {type(self.scaler)}")

        if self.scaler_type == 'minmax':
            self.scale_ = self.scaler.scale_[0]
            self.min_ = self.scaler.min_[0]
        elif self.scaler_type == 'standard':
            self.mean_ = self.scaler.mean_[0]
            self.std_ = self.scaler.scale_[0]

        print(f"📊 MetricsCalculator 初始化:")
        print(f"   Scaler 类型: {self.scaler_type}")
        print(f"   Log1p 变换: {self.is_log1p}")

    def inverse_transform(self, y):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        y = y.astype(np.float64)

        if self.scaler_type == 'minmax':
            y_restored = (y - self.min_) / self.scale_
        elif self.scaler_type == 'standard':
            y_restored = y * self.std_ + self.mean_
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

        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + epsilon)

        return {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'Pearson_R': float(pearson_r),
            'R2': float(r2)
        }


class HPOPruneTrial(RuntimeError):
    """用于在超参数优化过程中显式中止当前 trial。"""


class CCCLoss(nn.Module):
    """一致性相关系数损失 (CCC Loss)"""

    def __init__(self, eps=1e-8):
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        if pred.dim() > 2:
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            else:
                pred = pred.reshape(pred.shape[0], -1)
        if target.dim() > 2:
            if target.shape[-1] == 1:
                target = target.squeeze(-1)
            else:
                target = target.reshape(target.shape[0], -1)

        mean_pred = torch.mean(pred, dim=0)
        mean_target = torch.mean(target, dim=0)

        var_pred = torch.var(pred, dim=0, unbiased=False)
        var_target = torch.var(target, dim=0, unbiased=False)

        pred_centered = pred - mean_pred
        target_centered = target - mean_target
        covariance = torch.mean(pred_centered * target_centered, dim=0)

        numerator = 2 * covariance
        denominator = var_pred + var_target + (mean_pred - mean_target) ** 2 + self.eps

        ccc = numerator / denominator
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

    def __call__(self, val_metric):
        score = -val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'   ⏳ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class UnifiedTrainer:
    """
    统一联合训练器 (Autoformer + MoE Version)

    与原版训练器的唯一区别:
    - model.forward() 内部自动从 x 提取 season_gate_mask
    - 新增: epoch 结束后打印专家路由分布统计
    - 新增: 结果保存中记录 MoE 相关配置和路由统计
    """

    def __init__(self, model, train_loader, val_loader, test_loader,
                 metrics_calculator, config=cfg, trial=None,
                 trial_metric='best_val_ccc_loss', prune_enabled=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics_calc = metrics_calculator
        self.config = config
        self.run_name = getattr(config, 'RUN_NAME', datetime.now().strftime('run_%Y%m%d_%H%M%S'))
        self.experiment_log_csv = Path(getattr(config, 'EXPERIMENT_LOG_CSV', config.SAVE_DIR / 'experiment_results.csv'))
        self.training_started_at = datetime.now()
        self.training_finished_at = None
        self.final_test_metrics = {}
        self.final_test_ccc_loss = None
        self.final_epoch_reached = 0
        self.best_epoch = None
        self.trial = trial
        self.trial_metric = trial_metric
        self.prune_enabled = prune_enabled

        self.criterion = CCCLoss()
        print(f"🔧 Loss Function: CCCLoss (Concordance Correlation Coefficient)")
        print(f"🧠 Backbone: Autoformer + Hard-Gating MoE ({ModelConfig.num_experts} Season Experts)")

        self.device = config.DEVICE

        if getattr(self.config, 'USE_ADAPTIVE_WEIGHT', False):
            self.peak_threshold = self._calculate_adaptive_threshold()
        else:
            print("ℹ️ 自适应加权已禁用 (Config.USE_ADAPTIVE_WEIGHT = False)")
            self.peak_threshold = float('inf')

        if getattr(self.config, 'USE_ADAPTIVE_WEIGHT', False):
            _lambda = getattr(self.config, 'LOSS_WEIGHT_PEAK_MSE', 0.5)
            print(f"📐 模型选择策略: Composite Loss = CCC + {_lambda} × WeightedMSE (与训练目标对齐)")
        else:
            print(f"📐 模型选择策略: Pure CCC Loss")

        if config.EARLY_STOP:
            print(f"🔧 早停机制已启用: Patience={config.PATIENCE}, Delta={config.MIN_DELTA}")
            self.early_stopping = EarlyStopping(
                patience=config.PATIENCE,
                min_delta=config.MIN_DELTA
            )
        else:
            self.early_stopping = None

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        self.temp_scheduler = TemperatureScheduler(
            config.TEMP_START, config.TEMP_END, config.NUM_EPOCHS, config.TEMP_SCHEDULE
        )

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_ccc = float('inf')
        self.training_log = []

        # 【新增】累积路由统计
        self.epoch_routing_counts = np.zeros(ModelConfig.num_experts)

    def _calculate_adaptive_threshold(self):
        sigma = getattr(self.config, 'PEAK_SIGMA', 1.5)
        weight = getattr(self.config, 'PEAK_PENALTY_WEIGHT', 5.0)

        print(f"\n📊 正在分析训练集分布以设定自适应阈值 (Sigma={sigma})...")

        all_targets = []
        for _, y in self.train_loader:
            all_targets.append(y)

        all_targets = torch.cat(all_targets, dim=0)

        train_mean = torch.mean(all_targets).item()
        train_std = torch.std(all_targets, unbiased=False).item()

        threshold = train_mean + sigma * train_std

        print(f"   Stats (Normalized): Mean={train_mean:.4f}, Std={train_std:.4f}")
        print(f"   🎯 动态峰值阈值 (Threshold): {threshold:.4f}")
        print(f"   说明: 超过此值的样本将被施加 {weight} 倍权重的惩罚")
        print(f"{'-' * 70}")

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

    def _report_trial_result(self, epoch, train_loss, val_composite_loss,
                             val_metrics, val_ccc_loss=None):
        if self.trial is None:
            return

        metric_candidates = {
            'best_val_ccc_loss': val_composite_loss,
            'val_loss': val_composite_loss,
            'val_ccc_loss': val_ccc_loss if val_ccc_loss is not None else val_composite_loss,
            'val_rmse': val_metrics.get('RMSE'),
            'val_pearson': val_metrics.get('Pearson_R'),
            'train_loss': train_loss,
        }
        report_value = metric_candidates.get(self.trial_metric)
        if report_value is None:
            raise KeyError(f"不支持的 trial_metric: {self.trial_metric}")

        report_value = float(report_value)
        self.trial.report(report_value, step=epoch + 1)
        print(f"🧪 Trial 上报: step={epoch + 1}, metric={self.trial_metric}, value={report_value:.6f}")

        if self.prune_enabled and self.trial.should_prune():
            raise HPOPruneTrial(
                f"Trial 在 epoch {epoch + 1} 被剪枝: {self.trial_metric}={report_value:.6f}"
            )

    def train_epoch(self, epoch):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        current_temp = self.temp_scheduler.get_temperature(epoch)
        total_loss = 0.0
        total_ccc_loss = 0.0
        total_peak_loss = 0.0

        # 【新增】epoch 级路由统计
        epoch_routing = np.zeros(ModelConfig.num_experts)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config.NUM_EPOCHS}')

        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # 模型内部自动从 x 中提取 season_gate_mask
            predictions, expanded_weights = self.model(x, temperature=current_temp)

            # 【新增】收集路由统计 (不影响梯度)
            with torch.no_grad():
                season_mask = self.model._extract_season_mask(x)
                epoch_routing += season_mask.sum(dim=0).cpu().numpy()

            # ==================== Loss 计算 (与原版完全一致) ====================
            ccc_loss = self.criterion(predictions, y)
            loss = ccc_loss

            peak_loss_val = 0.0
            if getattr(self.config, 'USE_ADAPTIVE_WEIGHT', False):
                pred_view = predictions.view_as(y)
                mse_raw = (pred_view - y) ** 2

                weights = torch.ones_like(mse_raw)
                peak_mask = y > self.peak_threshold
                penalty = getattr(self.config, 'PEAK_PENALTY_WEIGHT', 5.0)
                weights[peak_mask] = penalty
                weighted_mse = (weights * mse_raw).mean()

                lambda_peak = getattr(self.config, 'LOSS_WEIGHT_PEAK_MSE', 0.5)
                peak_loss_term = lambda_peak * weighted_mse
                loss += peak_loss_term
                peak_loss_val = peak_loss_term.item()

            if self.config.L1_LAMBDA > 0:
                l1_loss = self.model.compute_weight_l1_loss(expanded_weights)
                loss += self.config.L1_LAMBDA * l1_loss

            loss.backward()

            if self.config.GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.MAX_GRAD_NORM
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_ccc_loss += ccc_loss.item()
            total_peak_loss += peak_loss_val

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CCC': f'{ccc_loss.item():.4f}',
                'PeakMSE': f'{peak_loss_val:.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)

        # 【新增】打印本 epoch 的路由分布
        total_samples = epoch_routing.sum()
        expert_names = ['Low', 'Normal', 'Peak']
        routing_str = " | ".join([
            f"{name}: {int(count)} ({count/total_samples*100:.1f}%)"
            for name, count in zip(expert_names, epoch_routing)
        ])
        print(f"   🔀 Expert Routing: {routing_str}")

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
            # season_gate_mask 由模型自动提取
            preds, _ = self.model(x, temperature=eval_temp)
            all_preds.append(preds)
            all_targets.append(y)

        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)

        val_ccc_loss = self.criterion(all_preds_tensor, all_targets_tensor).item()

        val_composite_loss = val_ccc_loss
        if getattr(self.config, 'USE_ADAPTIVE_WEIGHT', False):
            pred_view = all_preds_tensor.view_as(all_targets_tensor)
            mse_raw = (pred_view - all_targets_tensor) ** 2
            weights = torch.ones_like(mse_raw)
            peak_mask = all_targets_tensor > self.peak_threshold
            penalty = getattr(self.config, 'PEAK_PENALTY_WEIGHT', 5.0)
            weights[peak_mask] = penalty
            weighted_mse = (weights * mse_raw).mean()
            lambda_peak = getattr(self.config, 'LOSS_WEIGHT_PEAK_MSE', 0.5)
            val_composite_loss += lambda_peak * weighted_mse.item()

        preds_real = self.metrics_calc.inverse_transform(all_preds_tensor)
        targets_real = self.metrics_calc.inverse_transform(all_targets_tensor)

        metrics = self.metrics_calc.compute_metrics(preds_real, targets_real)

        return metrics, preds_real, targets_real, val_ccc_loss, val_composite_loss

    def save_best_model(self, val_composite_loss, val_ccc_loss):
        if val_composite_loss < self.best_val_loss:
            self.best_val_loss = val_composite_loss
            self.best_val_ccc = val_ccc_loss
            self.best_epoch = self.current_epoch + 1
            save_path = self.config.SAVE_DIR / 'best_model_unified.pth'

            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_composite_loss': val_composite_loss,
                'best_ccc_loss': val_ccc_loss
            }
            torch.save(checkpoint, save_path)

    def _get_latest_learning_rate(self):
        return float(self.optimizer.param_groups[0]['lr'])

    def _build_experiment_record(self):
        model_params = ModelConfig.to_dict() if hasattr(ModelConfig, 'to_dict') else {}
        train_params = self.config.to_dict() if hasattr(self.config, 'to_dict') else {}

        best_log_entry = None
        if self.training_log:
            best_log_entry = min(self.training_log, key=lambda item: item['Val_Loss'])

        record = {
            'run_name': self.run_name,
            'started_at': self.training_started_at.strftime('%Y-%m-%d %H:%M:%S'),
            'finished_at': self.training_finished_at.strftime('%Y-%m-%d %H:%M:%S') if self.training_finished_at else '',
            'duration_seconds': round((self.training_finished_at - self.training_started_at).total_seconds(), 2) if self.training_finished_at else None,
            'platform': platform.platform(),
            'python_device': self.config.DEVICE,
            'dataset_dir': str(self.config.DATA_DIR),
            'save_dir': str(self.config.SAVE_DIR),
            'train_samples': len(self.train_loader.dataset),
            'val_samples': len(self.val_loader.dataset),
            'test_samples': len(self.test_loader.dataset),
            'epochs_planned': self.config.NUM_EPOCHS,
            'epochs_completed': self.final_epoch_reached,
            'best_epoch': self.best_epoch,
            'best_val_ccc_loss': self.best_val_loss,
            'best_val_pure_ccc': self.best_val_ccc,
            'final_test_ccc_loss': self.final_test_ccc_loss,
            'final_test_rmse': self.final_test_metrics.get('RMSE'),
            'final_test_mae': self.final_test_metrics.get('MAE'),
            'final_test_mape': self.final_test_metrics.get('MAPE'),
            'final_test_pearson_r': self.final_test_metrics.get('Pearson_R'),
            'final_test_r2': self.final_test_metrics.get('R2'),
            'peak_threshold': float(self.peak_threshold.detach().cpu().item()) if isinstance(self.peak_threshold, torch.Tensor) else float(self.peak_threshold),
            'best_train_loss': best_log_entry['Train_Loss'] if best_log_entry else None,
            'best_val_rmse': best_log_entry['Val_RMSE'] if best_log_entry else None,
            'best_val_pearson': best_log_entry['Val_Pearson'] if best_log_entry else None,
            'last_learning_rate': self._get_latest_learning_rate(),
            'training_log_csv': str(self.config.SAVE_LOG),
            'best_model_path': str(self.config.SAVE_DIR / 'best_model_unified.pth')
        }

        record.update({f'model_{k}': v for k, v in model_params.items()})
        record.update({f'train_{k}': v for k, v in train_params.items()})
        return record

    def _append_experiment_record(self):
        record = self._build_experiment_record()
        record_df = pd.DataFrame([record])
        csv_path = self.experiment_log_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        if csv_path.exists():
            history_df = pd.read_csv(csv_path)
            record_df = pd.concat([history_df, record_df], ignore_index=True)

        record_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"🧾 实验记录已追加保存: {csv_path}")
        return record

    def train(self):
        print(f"\n{'=' * 70}")
        print("开始联合训练 (Autoformer + Hard-Gating MoE + CCC + Adaptive Weighted MSE)")
        print(f"策略: End-to-End Training | Main Loss: CCC | Backbone: Autoformer + MoE")
        print(f"专家: {ModelConfig.num_experts} Season Experts (Low / Normal / Peak)")
        print(f"门控: Hard-Gating (One-Hot Season Mask, Gradient Isolated)")
        if getattr(self.config, 'USE_ADAPTIVE_WEIGHT', False):
            print(f"增强: Adaptive Peak Weighting Enabled (Threshold based on Stats)")
        print(f"{'=' * 70}\n")

        start_time = datetime.now()

        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            self.final_epoch_reached = epoch + 1
            train_loss, temp = self.train_epoch(epoch)

            val_metrics, _, _, val_ccc_loss, val_composite_loss = self.validate(self.val_loader, phase='Val')

            if self.scheduler is not None:
                self.scheduler.step()

            log_entry = {
                'Epoch': epoch + 1,
                'Train_Loss': train_loss,
                'Val_Loss': val_composite_loss,
                'Val_CCC_Loss': val_ccc_loss,
                'Val_RMSE': val_metrics['RMSE'],
                'Val_Pearson': val_metrics['Pearson_R'],
                'Val_R2': val_metrics['R2']
            }
            self.training_log.append(log_entry)
            self._report_trial_result(epoch, train_loss, val_composite_loss,
                                      val_metrics, val_ccc_loss=val_ccc_loss)

            print(f"📊 Epoch {epoch + 1}: "
                  f"Loss={train_loss:.4f}, "
                  f"Val_Sel={val_composite_loss:.4f}, "
                  f"Val_CCC={val_ccc_loss:.4f}, "
                  f"Val_RMSE={val_metrics['RMSE']:.2f}, "
                  f"Pearson={val_metrics['Pearson_R']:.4f}, "
                  f"R²={val_metrics['R2']:.4f}")

            self.save_best_model(val_composite_loss, val_ccc_loss)

            if self.early_stopping:
                self.early_stopping(val_composite_loss)

                if self.early_stopping.early_stop:
                    print(f"\n🛑 早停触发！在第 {epoch + 1} 轮停止训练。")
                    break

        log_df = pd.DataFrame(self.training_log)
        log_df.to_csv(self.config.SAVE_LOG, index=False)
        self.training_finished_at = datetime.now()
        print(f"\n⏱️  总训练时长: {self.training_finished_at - start_time}")

        print("\n🎨 正在绘制训练曲线...")
        self.plot_training_curves()

        self.evaluate_final_model()
        return self._append_experiment_record()

    def evaluate_final_model(self):
        model_path = self.config.SAVE_DIR / 'best_model_unified.pth'
        print(f"\n🔎 正在评估最佳模型 (Autoformer + MoE): {model_path}")

        if not model_path.exists():
            print("⚠️  未找到模型文件。")
            return

        checkpoint = torch.load(model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics, preds, targets, test_ccc_loss, _ = self.validate(self.test_loader, phase='Test')
        self.final_test_metrics = test_metrics
        self.final_test_ccc_loss = test_ccc_loss

        print(f"   📈 测试集最终结果 (Real Space) [Autoformer + Hard-Gating MoE]:")
        print(f"      RMSE:      {test_metrics['RMSE']:.4f}")
        print(f"      MAE:       {test_metrics['MAE']:.4f}")
        print(f"      MAPE:      {test_metrics['MAPE']:.2f}%")
        print(f"      Pearson R: {test_metrics['Pearson_R']:.4f}")
        print(f"      R²:        {test_metrics['R2']:.4f}")

        with open(self.config.SAVE_DIR / 'result_final.json', 'w') as f:
            json.dump(test_metrics, f, indent=4)

        self.plot_results(preds, targets, 'Final_Autoformer_MoE', test_metrics['RMSE'])

    def plot_training_curves(self):
        if not self.training_log:
            print("⚠️  训练日志为空，跳过曲线绘制。")
            return

        log_df = pd.DataFrame(self.training_log)
        epochs = log_df['Epoch'].values

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Training Curves (Autoformer + Hard-Gating MoE)', fontsize=16, fontweight='bold')

        ax1 = axes[0, 0]
        ax1.plot(epochs, log_df['Train_Loss'].values, label='Train Loss',
                 color='steelblue', linewidth=1.5)
        ax1.plot(epochs, log_df['Val_Loss'].values, label='Val Selection Loss',
                 color='darkorange', linewidth=1.5, linestyle='--')
        if 'Val_CCC_Loss' in log_df.columns:
            ax1.plot(epochs, log_df['Val_CCC_Loss'].values, label='Val CCC Loss',
                     color='forestgreen', linewidth=1.2, linestyle=':')
        ax1.set_title('Loss (Selection & CCC)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(epochs, log_df['Val_RMSE'].values, label='Val RMSE',
                 color='crimson', linewidth=1.5)
        ax2.set_title('Validation RMSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(epochs, log_df['Val_Pearson'].values, label='Val Pearson R',
                 color='seagreen', linewidth=1.5)
        ax3.set_title('Validation Pearson R')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Pearson R')
        ax3.set_ylim(-1, 1)
        ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        if 'Val_R2' in log_df.columns:
            ax4.plot(epochs, log_df['Val_R2'].values, label='Val R²',
                     color='mediumpurple', linewidth=1.5)
            ax4.set_title('Validation R²')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('R²')
            ax4.axhline(y=0, color='gray', linestyle=':', linewidth=1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.set_visible(False)

        if self.best_epoch is not None:
            for ax in [ax1, ax2, ax3, ax4]:
                if ax.get_visible():
                    ax.axvline(x=self.best_epoch, color='gold', linewidth=1.5,
                               linestyle='-.', label=f'Best Epoch ({self.best_epoch})')
                    ax.legend()

        plt.tight_layout()
        save_path = self.config.SAVE_DIR / 'vis_training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📈 训练曲线图已保存: {save_path}")

    def plot_results(self, preds, targets, name, rmse):
        if preds.ndim > 1 and preds.shape[1] > 1:
            print(f"   ℹ️ 检测到多步预测 (Pred_Len={preds.shape[1]})，仅提取 't+1' 用于绘图...")
            preds_to_plot = preds[:, 0]
            if targets.ndim > 1 and targets.shape[1] > 1:
                targets_to_plot = targets[:, 0]
            else:
                targets_to_plot = targets.flatten()
        else:
            preds_to_plot = preds.flatten()
            targets_to_plot = targets.flatten()

        plt.figure(figsize=(15, 6))
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


def snapshot_config(config_cls):
    snapshot = {}
    skipped_keys = []

    for key, value in vars(config_cls).items():
        if key.startswith('__'):
            continue
        if callable(value) or isinstance(value, (classmethod, staticmethod, property)):
            skipped_keys.append(key)
            continue
        try:
            snapshot[key] = copy.deepcopy(value)
        except Exception as exc:
            skipped_keys.append(f"{key}<{type(value).__name__}>:{exc}")

    print(f"🧩 snapshot_config[{config_cls.__name__}] 已捕获 {len(snapshot)} 个属性")
    if skipped_keys:
        print(f"   ↪ 跳过属性: {skipped_keys}")
    return snapshot


def restore_config(config_cls, snapshot):
    current_keys = [key for key in vars(config_cls).keys() if not key.startswith('__')]
    skipped_keys = []

    for key in current_keys:
        value = vars(config_cls).get(key)
        if callable(value) or isinstance(value, (classmethod, staticmethod, property)):
            skipped_keys.append(key)
            continue
        if key not in snapshot:
            delattr(config_cls, key)

    for key, value in snapshot.items():
        setattr(config_cls, key, copy.deepcopy(value))

    print(f"♻️ restore_config[{config_cls.__name__}] 已恢复 {len(snapshot)} 个属性")
    if skipped_keys:
        print(f"   ↪ 保留描述符属性: {skipped_keys}")


def apply_overrides(config_cls, overrides=None):
    if not overrides:
        return
    for key, value in overrides.items():
        if not hasattr(config_cls, key):
            raise AttributeError(f"{config_cls.__name__} 不存在属性: {key}")
        setattr(config_cls, key, value)


def refresh_training_paths(config=cfg):
    config.TRAIN_X = config.DATA_DIR / 'train_x.npy'
    config.TRAIN_Y = config.DATA_DIR / 'train_y.npy'
    config.VAL_X = config.DATA_DIR / 'val_x.npy'
    config.VAL_Y = config.DATA_DIR / 'val_y.npy'
    config.TEST_X = config.DATA_DIR / 'test_x.npy'
    config.TEST_Y = config.DATA_DIR / 'test_y.npy'
    config.FEATURE_MAP = config.DATA_DIR / 'feature_map.json'
    config.SCALER_PATH = config.DATA_DIR / 'scalers.pkl'
    config.SAVE_LOG = config.SAVE_DIR / 'training_log.csv'
    config.SAVE_LATEST_MODEL = config.SAVE_DIR / 'latest_model.pth'
    config.EXPERIMENT_LOG_CSV = config.SAVE_DIR / 'experiment_results.csv'
    config.SAVE_BEST_MODEL_PHASE1 = config.SAVE_DIR / 'best_model_phase1.pth'
    config.SAVE_BEST_MODEL_PHASE2 = config.SAVE_DIR / 'best_model_phase2.pth'
    config.SAVE_BEST_MODEL_PHASE3 = config.SAVE_DIR / 'best_model_phase3.pth'


def main(training_overrides=None, model_overrides=None, reset_model_config=True,
         restore_after_run=False, trial=None, trial_metric='best_val_ccc_loss',
         prune_enabled=False):
    print(f"\n{'=' * 70}\nM10 Autoformer + Hard-Gating MoE 训练 (Adaptive Peak Weighting)\n{'=' * 70}")

    cfg_snapshot = snapshot_config(cfg)
    model_snapshot = snapshot_config(ModelConfig)

    try:
        if reset_model_config:
            ModelConfig.reset_to_default()

        apply_overrides(cfg, training_overrides)
        refresh_training_paths(cfg)
        apply_overrides(ModelConfig, model_overrides)

        set_seed(cfg.SEED)
        cfg.validate()
        cfg.setup_dirs()

        print(f"\n📂 数据集目录: {cfg.DATA_DIR}")
        ModelConfig.from_dataset(cfg.DATA_DIR, verbose=True)

        train_x, train_y, val_x, val_y, test_x, test_y, feature_map, scaler_data = load_data(cfg)
        train_loader, val_loader, test_loader = create_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y, cfg)

        model = build_model(feature_map, ModelConfig).to(cfg.DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n🧠 Autoformer + MoE 模型:")
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数量: {trainable_params:,}")

        # 打印各专家参数量
        decoder_moe = model.backbone.decoder
        for i, (expert, name) in enumerate(zip(decoder_moe.experts, decoder_moe.expert_names)):
            expert_params = sum(p.numel() for p in expert.parameters())
            print(f"   {name} 参数量: {expert_params:,}")

        metrics_calc = MetricsCalculator(scaler_data)

        trainer = UnifiedTrainer(
            model,
            train_loader,
            val_loader,
            test_loader,
            metrics_calc,
            cfg,
            trial=trial,
            trial_metric=trial_metric,
            prune_enabled=prune_enabled
        )
        return trainer.train()
    finally:
        if restore_after_run:
            restore_config(cfg, cfg_snapshot)
            restore_config(ModelConfig, model_snapshot)
            refresh_training_paths(cfg)

if __name__ == '__main__':
    main()
