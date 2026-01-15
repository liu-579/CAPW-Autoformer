# -*- coding: utf-8 -*-
"""
M10 三阶段渐进式训练脚本 (Final Thesis Version)
Three-Phase Progressive Training Pipeline with Phase-wise Evaluation & Visualization
用途: 训练模型，分别保存三个阶段的最佳模型，并生成对比图表
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
from m9_model import build_model
from config.m9_config_flash import ModelConfig
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
    """评估指标计算器 (适配 Log1p + Hybrid Scaling)"""

    def __init__(self, scaler_data):
        # 1. 解析 m8b 传入的 Scaler 字典
        if isinstance(scaler_data, dict) and 'target_scaler' in scaler_data:
            self.scaler = scaler_data['target_scaler']
            self.meta = scaler_data.get('meta', {})
            self.is_log1p = self.meta.get('target_log1p', False)
            print(f"🔧 MetricsCalculator 初始化: 检测到 Log1p={self.is_log1p}")
        else:
            # 兼容旧版单一 Scaler 对象
            self.scaler = scaler_data
            self.is_log1p = False
            print(f"🔧 MetricsCalculator 初始化: Legacy模式 (No Log1p)")

        # 2. 预提取反归一化参数 (Target Scaler 仅有一列，无需 target_idx)
        # sklearn scaler transform: y = (x * scale_) + min_
        # inverse: x = (y - min_) / scale_
        if hasattr(self.scaler, 'scale_'):
            self.scale_ = self.scaler.scale_[0]
            self.min_ = self.scaler.min_[0]
            self.scaler_type = 'minmax'  # 或 standard，参数名通用
        elif hasattr(self.scaler, 'mean_'):
            self.mean_ = self.scaler.mean_[0]
            self.scale_ = self.scaler.scale_[0]
            self.scaler_type = 'standard'
        else:
            raise ValueError(f"不支持的 scaler 类型: {type(self.scaler)}")

    def inverse_transform(self, y):
        """执行双重反变换: Scaling -> Log1p -> Real"""
        # 1. 转为 Numpy
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # 确保是 float 类型
        y = y.astype(np.float64)

        # 2. Inverse Scaling (恢复到 Log 空间的值)
        if self.scaler_type == 'minmax':
            y_restored = (y - self.min_) / self.scale_
        elif self.scaler_type == 'standard':
            y_restored = y * self.scale_ + self.mean_
        else:
            y_restored = y

        # 3. Inverse Log (Expm1)
        if self.is_log1p:
            # expm1(x) = exp(x) - 1
            y_restored = np.expm1(y_restored)

        # 4. 物理约束: 人数不能为负
        y_restored = np.maximum(y_restored, 0)

        return y_restored

    def compute_metrics(self, preds, targets):
        """计算真实空间的指标"""
        # 输入已经是 inverse_transform 过的真实值
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mae = np.mean(np.abs(preds - targets))

        epsilon = 1e-8
        # MAPE 需要防除零
        mape = np.mean(np.abs((preds - targets) / (np.abs(targets) + epsilon))) * 100

        pearson_r = np.corrcoef(preds.flatten(), targets.flatten())[0, 1] if len(preds) > 1 else 0.0

        return {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'Pearson_R': float(pearson_r)
        }
class PearsonCorrelationLoss(nn.Module):
    """
    可微分的 Pearson 相关系数损失函数
    Loss = 1 - r
    """
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, pred, target):
        # 展平输入 (Batch_Size, ...) -> (Batch_Size, )
        pred = pred.view(-1)
        target = target.view(-1)

        # 避免计算中的数值不稳定 (除零错误)
        epsilon = 1e-8

        # 计算均值
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)

        # 中心化
        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        # 计算协方差 (分子)
        covariance = torch.sum(pred_centered * target_centered)

        # 计算标准差乘积 (分母)
        pred_std = torch.sqrt(torch.sum(pred_centered ** 2) + epsilon)
        target_std = torch.sqrt(torch.sum(target_centered ** 2) + epsilon)

        # 计算 Pearson r
        pearson_r = covariance / (pred_std * target_std)

        # 我们希望 r 越大越好 (接近1)，所以 Loss = 1 - r
        # 范围 [0, 2], 0 表示完全正相关
        loss = 1.0 - pearson_r
        return loss

class ThreePhaseTrainer:
    """三阶段渐进式训练器"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 metrics_calculator, config=cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics_calc = metrics_calculator
        self.config = config

        # Loss 计算在 Log 空间进行 (无需修改)
        self.mse_criterion = nn.MSELoss()
        self.pearson_criterion = PearsonCorrelationLoss()
        self.w_mse = config.LOSS_WEIGHT_MSE
        self.w_pearson = config.LOSS_WEIGHT_PEARSON


        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.temp_scheduler = TemperatureScheduler(
            config.TEMP_START, config.TEMP_END, config.NUM_EPOCHS, config.TEMP_SCHEDULE
        )

        self.current_epoch = 0
        self.current_phase = None
        self.current_temperature = config.TEMP_START

        self.best_rmse_phase1 = float('inf')
        self.best_rmse_phase2 = float('inf')
        self.best_rmse_phase3 = float('inf')

        self.training_log = []
        self.device = config.DEVICE

    def _create_optimizer(self):
        backbone_params = []
        weight_net_params = []

        for name, param in self.model.named_parameters():
            if 'weight_net' in name:
                weight_net_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {'params': backbone_params, 'name': 'backbone'},
            {'params': weight_net_params, 'name': 'weight_net'}
        ]

        if self.config.OPTIMIZER == 'adamw':
            return torch.optim.AdamW(
                param_groups, lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                betas=self.config.BETAS, eps=self.config.EPS
            )
        else:
            return torch.optim.Adam(param_groups, lr=self.config.LEARNING_RATE)

    def _create_scheduler(self):
        if self.config.SCHEDULER == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.NUM_EPOCHS, eta_min=self.config.LR_MIN
            )
        return None

    def set_training_phase(self, epoch):
        current_base_lr = self.optimizer.param_groups[1]['lr']

        if epoch < self.config.PHASE_1_EPOCHS:
            # Phase 1: 预热骨干
            if self.current_phase != 'Phase1':
                print(f"\n{'=' * 70}")
                print(f"🔥 Phase 1: 预热骨干网络 (Epoch {epoch})")
                self.current_phase = 'Phase1'
                self.model.unfreeze_all()
                self.model.freeze_weight_net()

            self.optimizer.param_groups[0]['lr'] = current_base_lr
            self.optimizer.param_groups[1]['lr'] = 0.0

        elif epoch < self.config.PHASE_1_EPOCHS + self.config.PHASE_2_EPOCHS:
            # Phase 2: 权重学习
            if self.current_phase != 'Phase2':
                print(f"\n{'=' * 70}")
                print(f"⚡ Phase 2: 权重网络学习 (Epoch {epoch})")
                self.current_phase = 'Phase2'
                self.model.unfreeze_all()

            self.optimizer.param_groups[0]['lr'] = current_base_lr * 0.1
            self.optimizer.param_groups[1]['lr'] = current_base_lr

        else:
            # Phase 3: 联合微调
            if self.current_phase != 'Phase3':
                print(f"\n{'=' * 70}")
                print(f"🚀 Phase 3: 联合微调 (Epoch {epoch})")
                self.current_phase = 'Phase3'
                self.model.unfreeze_all()

            self.optimizer.param_groups[0]['lr'] = current_base_lr
            self.optimizer.param_groups[1]['lr'] = current_base_lr

    def train_epoch(self, epoch):
        self.model.train()
        self.set_training_phase(epoch)
        current_temp = self.temp_scheduler.get_temperature(epoch)
        self.current_temperature = current_temp

        total_loss = 0.0
        total_mse_part = 0.0  # 记录分项 Loss 以便观察
        total_pearson_part = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.NUM_EPOCHS}')

        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)  # y is in Log space

            self.optimizer.zero_grad()
            predictions, expanded_weights = self.model(x, temperature=current_temp)

            # ========== 修改点：计算组合损失 ==========

            # 1. 计算 MSE 部分
            mse_loss = self.mse_criterion(predictions, y)

            # 2. 计算 Pearson 部分 (注意: 如果 BatchSize=1，Pearson无法计算，会报错或返回NaN)
            if x.size(0) > 1:
                pearson_loss = self.pearson_criterion(predictions, y)
            else:
                pearson_loss = torch.tensor(0.0, device=self.device)

            # 3. 组合
            loss = (self.w_mse * mse_loss) + (self.w_pearson * pearson_loss)

            # 4. (可选) 加入正则化项
            if self.config.L1_LAMBDA > 0:
                l1_loss = self.model.compute_weight_l1_loss(expanded_weights)
                loss += self.config.L1_LAMBDA * l1_loss

            # =======================================
            loss.backward()

            if self.config.GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.MAX_GRAD_NORM
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_mse_part += mse_loss.item()
            total_pearson_part += pearson_loss.item()

            # 更新进度条显示，让你能实时看到 Pearson Loss 是否在下降
            pbar.set_postfix({
                'L_Tot': f'{loss.item():.4f}',
                'L_MSE': f'{mse_loss.item():.4f}',
                'L_P': f'{pearson_loss.item():.4f}',  # 越接近0说明相关性越高
                'LR': f"{self.optimizer.param_groups[0]['lr']:.1e}"
            })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, current_temp

    @torch.no_grad()
    def validate(self, data_loader, phase='Val'):
        """验证/测试 (自动执行双重反变换)"""
        self.model.eval()
        all_preds = []
        all_targets = []

        # 验证时固定温度
        eval_temp = self.config.TEMP_END

        for x, y in tqdm(data_loader, desc=f'{phase} Eval', leave=False):
            x = x.to(self.device)
            y = y.to(self.device)
            preds, _ = self.model(x, temperature=eval_temp)
            all_preds.append(preds)
            all_targets.append(y)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 核心修改: 执行双重反变换 (Log Space -> Real Space)
        preds_real = self.metrics_calc.inverse_transform(all_preds)
        targets_real = self.metrics_calc.inverse_transform(all_targets)

        metrics = self.metrics_calc.compute_metrics(preds_real, targets_real)

        return metrics, preds_real, targets_real

    def save_phase_best(self, val_rmse, phase):
        save_path = None
        is_best = False

        if phase == 'Phase1':
            if val_rmse < self.best_rmse_phase1:
                self.best_rmse_phase1 = val_rmse
                save_path = self.config.SAVE_BEST_MODEL_PHASE1
                is_best = True
        elif phase == 'Phase2':
            if val_rmse < self.best_rmse_phase2:
                self.best_rmse_phase2 = val_rmse
                save_path = self.config.SAVE_BEST_MODEL_PHASE2
                is_best = True
        elif phase == 'Phase3':
            if val_rmse < self.best_rmse_phase3:
                self.best_rmse_phase3 = val_rmse
                save_path = self.config.SAVE_BEST_MODEL_PHASE3
                is_best = True

        if is_best and save_path:
            checkpoint = {
                'epoch': self.current_epoch,
                'phase': phase,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_rmse': val_rmse
            }
            torch.save(checkpoint, save_path)

    def train(self):
        print(f"\n{'=' * 70}")
        print("开始三阶段渐进式训练 (Log-Space Loss, Real-Space Metrics)")
        print(f"{'=' * 70}\n")

        start_time = datetime.now()

        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            train_loss, temp = self.train_epoch(epoch)

            # 这里的 val_metrics 已经是基于真实人数计算的了
            val_metrics, _, _ = self.validate(self.val_loader, phase='Val')

            if self.scheduler is not None:
                self.scheduler.step()

            log_entry = {
                'Epoch': epoch + 1,
                'Phase': self.current_phase,
                'Train_Loss': train_loss,
                'Val_RMSE': val_metrics['RMSE'],
                'Val_Pearson': val_metrics['Pearson_R']
            }
            self.training_log.append(log_entry)

            print(f"📊 Epoch {epoch + 1} ({self.current_phase}): "
                  f"Loss={train_loss:.4f} (Log), "
                  f"Val_RMSE={val_metrics['RMSE']:.2f} (Real), "
                  f"Pearson={val_metrics['Pearson_R']:.4f}")

            self.save_phase_best(val_metrics['RMSE'], self.current_phase)

        log_df = pd.DataFrame(self.training_log)
        log_df.to_csv(self.config.SAVE_LOG, index=False)
        print(f"\n⏱️  总训练时长: {datetime.now() - start_time}")

        # 评估各阶段
        self.evaluate_phase_model('Phase1', self.config.SAVE_BEST_MODEL_PHASE1)
        self.evaluate_phase_model('Phase2', self.config.SAVE_BEST_MODEL_PHASE2)
        self.evaluate_phase_model('Phase3', self.config.SAVE_BEST_MODEL_PHASE3)

    def evaluate_phase_model(self, phase_name, model_path):
        print(f"\n🔎 正在评估 {phase_name} 最佳模型...")
        if not model_path.exists():
            print(f"⚠️  未找到 {phase_name} 模型文件，跳过。")
            return

        checkpoint = torch.load(model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 测试集评估 (metrics 返回的是 Real Space)
        test_metrics, preds, targets = self.validate(self.test_loader, phase='Test')

        print(f"   📈 {phase_name} 测试集结果 (Real Space):")
        print(f"      RMSE:      {test_metrics['RMSE']:.4f}")
        print(f"      MAE:       {test_metrics['MAE']:.4f}")
        print(f"      MAPE:      {test_metrics['MAPE']:.2f}%")
        print(f"      Pearson R: {test_metrics['Pearson_R']:.4f}")

        # 保存结果
        with open(self.config.SAVE_DIR / f'result_{phase_name.lower()}.json', 'w') as f:
            json.dump(test_metrics, f, indent=4)

        # 绘图 (preds, targets 已经是真实数值)
        self.plot_results(preds, targets, phase_name, test_metrics['RMSE'])

    def plot_results(self, preds, targets, phase_name, rmse):
        """绘制对比图 (确保输入是 1D array)"""
        preds = preds.flatten()
        targets = targets.flatten()

        # 1. 趋势图
        plt.figure(figsize=(15, 6))
        plt.plot(targets, label='True Values', alpha=0.7, color='blue', linewidth=1)
        plt.plot(preds, label='Predictions', alpha=0.7, color='red', linestyle='--', linewidth=1)
        plt.title(f'{phase_name} Prediction vs True (Real Space, RMSE: {rmse:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.SAVE_DIR / f'vis_trend_{phase_name.lower()}.png', dpi=300)
        plt.close()

        # 2. 散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, preds, alpha=0.3, s=10, c='purple')
        lim_min = min(targets.min(), preds.min())
        lim_max = max(targets.max(), preds.max())
        plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.75)
        plt.xlabel('True Values (Real)')
        plt.ylabel('Predictions (Real)')
        plt.title(f'{phase_name} Scatter Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.config.SAVE_DIR / f'vis_scatter_{phase_name.lower()}.png', dpi=300)
        plt.close()

        print(f"   🖼️  图表已保存")


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

    # 读取 Scalers 字典
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
    print(f"\n{'=' * 70}\nM10 三阶段渐进式训练 (Thesis Final)\n{'=' * 70}")
    set_seed(cfg.SEED)
    cfg.setup_dirs()

    # 加载数据
    train_x, train_y, val_x, val_y, test_x, test_y, feature_map, scaler_data = load_data()
    train_loader, val_loader, test_loader = create_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y)

    # 构建模型
    model = build_model(feature_map, ModelConfig).to(cfg.DEVICE)

    # 初始化计算器 (传入完整的 scaler_data 字典)
    metrics_calc = MetricsCalculator(scaler_data)

    trainer = ThreePhaseTrainer(model, train_loader, val_loader, test_loader, metrics_calc, cfg)
    trainer.train()


if __name__ == '__main__':
    main()