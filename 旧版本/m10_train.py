"""
m10_train.py - 三阶段渐进式训练主程序
===================================================================
核心流程：
    1. 加载 m8 的数据（.npy 格式）
    2. 实例化 m9 的 DynamicFusionPredictor 模型
    3. 执行三阶段训练策略
    4. 保存最佳模型和训练日志

作者：[您的姓名]
日期：2025-12-18
版本：v1.0 - 硕士论文核心训练脚本
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 导入配置和模型
from 旧版本.m10_config import TrainConfig
from 旧版本.m9_config import ModelConfig
from m9_model import DynamicFusionPredictor


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int):
    """设置随机种子，保证实验可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"✅ 随机种子已设置: {seed}")


def compute_temperature(epoch: int, total_epochs: int, temp_start: float, temp_end: float) -> float:
    """
    线性温度退火

    参数：
        epoch: 当前轮数
        total_epochs: 总轮数
        temp_start: 起始温度
        temp_end: 结束温度

    返回：
        当前温度值
    """
    progress = epoch / total_epochs
    current_temp = temp_start - (temp_start - temp_end) * progress
    return max(current_temp, temp_end)  # 确保不低于最小温度


def get_current_phase(epoch: int, phase_1_epochs: int, phase_2_epochs: int) -> int:
    """
    判断当前训练阶段

    返回：
        1: Phase 1 (预热骨干)
        2: Phase 2 (学习权重)
        3: Phase 3 (联合微调)
    """
    if epoch < phase_1_epochs:
        return 1
    elif epoch < phase_1_epochs + phase_2_epochs:
        return 2
    else:
        return 3


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """
    计算评估指标

    参数：
        preds: 预测值 [N, pred_len, 1]
        targets: 真实值 [N, pred_len, 1]

    返回：
        包含 MSE, RMSE, MAE, Pearson-r 的字典
    """
    # 展平为 1D 数组
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    # MSE
    mse = np.mean((preds_flat - targets_flat) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(preds_flat - targets_flat))

    # Pearson Correlation
    if len(preds_flat) > 1:
        pearson_r, _ = pearsonr(preds_flat, targets_flat)
    else:
        pearson_r = 0.0

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Pearson_r': pearson_r
    }


# ============================================================================
# 数据加载器
# ============================================================================

class DataLoader_M8:
    """
    加载 m8 模块输出的数据

    数据格式：
        - train_x_{景区名}.npy: [N, seq_len, feature_dim]
        - train_y_{景区名}.npy: [N, pred_len, 1]
        - feature_map_{景区名}.json: {'sentiment_idxs': [...], 'context_idxs': [...]}
        - scaler_{景区名}.pkl: sklearn 归一化器
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.scenic_name = config.TARGET_SCENIC_NAME
        self.data_dir = config.DATA_DIR

        # 加载数据
        self.train_x, self.train_y = self._load_data('train')
        self.val_x, self.val_y = self._load_data('val')
        self.test_x, self.test_y = self._load_data('test')

        # 加载特征映射表
        self.feature_map = self._load_feature_map()

        # 加载归一化器（可选）
        self.scaler = self._load_scaler()

        print(f"\n✅ 数据加载完成:")
        print(f"   训练集: {self.train_x.shape}")
        print(f"   验证集: {self.val_x.shape}")
        print(f"   测试集: {self.test_x.shape}")

    def _load_data(self, split: str):
        """加载指定数据集"""

        x_path = os.path.join(self.data_dir, f"tourism_seq_{self.scenic_name}_{split}_x.npy")
        y_path = os.path.join(self.data_dir, f"tourism_seq_{self.scenic_name}_{split}_y.npy")

        if not os.path.exists(x_path):
            raise FileNotFoundError(f"数据文件不存在: {x_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"数据文件不存在: {y_path}")

        x = np.load(x_path)
        y = np.load(y_path)

        return x, y

    def _load_feature_map(self) -> dict:
        """加载特征映射表 JSON"""
        feature_map_path = os.path.join(
            self.data_dir,
            f"tourism_seq_{self.scenic_name}_feature_map.json"
        )

        if not os.path.exists(feature_map_path):
            raise FileNotFoundError(f"特征映射表文件不存在: {feature_map_path}")

        with open(feature_map_path, 'r', encoding='utf-8') as f:
            feature_map = json.load(f)

        # 验证必要的键
        required_keys = ['sentiment_idxs', 'context_idxs']
        for key in required_keys:
            if key not in feature_map:
                raise KeyError(f"特征映射表缺少必要的键: {key}")

        print(f"✅ 特征映射表加载成功:")
        print(f"   情感特征索引: {feature_map['sentiment_idxs']}")
        print(f"   环境特征索引: {feature_map['context_idxs']}")

        return feature_map

    def _load_scaler(self):
        """加载归一化器（可选）"""
        scaler_path = os.path.join(self.data_dir, f"scaler_{self.scenic_name}.pkl")

        if not os.path.exists(scaler_path):
            print(f"⚠️  归一化器文件不存在（可选）: {scaler_path}")
            return None

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        print(f"✅ 归一化器加载成功")
        return scaler

    def create_dataloaders(self) -> dict:
        """创建 PyTorch DataLoader"""

        # 转换为 Tensor
        train_x_tensor = torch.from_numpy(self.train_x).float()
        train_y_tensor = torch.from_numpy(self.train_y).float()
        val_x_tensor = torch.from_numpy(self.val_x).float()
        val_y_tensor = torch.from_numpy(self.val_y).float()
        test_x_tensor = torch.from_numpy(self.test_x).float()
        test_y_tensor = torch.from_numpy(self.test_y).float()

        # 创建 TensorDataset
        train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
        val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
        test_dataset = TensorDataset(test_x_tensor, test_y_tensor)

        # 创建 DataLoader
        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY,
                drop_last=True
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY
            )
        }

        print(f"\n✅ DataLoader 创建成功:")
        print(f"   训练集批次数: {len(dataloaders['train'])}")
        print(f"   验证集批次数: {len(dataloaders['val'])}")
        print(f"   测试集批次数: {len(dataloaders['test'])}")

        return dataloaders


# ============================================================================
# 三阶段训练器
# ============================================================================

class ThreePhaseTrainer:
    """
    三阶段渐进式训练器

    Phase 1: 预热骨干网络（冻结权重生成网络）
    Phase 2: 学习动态权重规则（冻结骨干网络）
    Phase 3: 端到端联合微调（解冻所有参数）
    """

    def __init__(
            self,
            model: DynamicFusionPredictor,
            dataloaders: dict,
            config: TrainConfig
    ):
        self.model = model.to(config.DEVICE)
        self.dataloaders = dataloaders
        self.config = config
        self.device = config.DEVICE

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.BETAS,
            eps=config.EPS
        )

        # 学习率调度器
        if config.USE_LR_SCHEDULER and config.LR_SCHEDULER_TYPE == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS,
                eta_min=config.LR_MIN
            )
        else:
            self.scheduler = None

        # 损失函数
        self.criterion = nn.MSELoss()

        # 训练历史
        self.history = {
            'epoch': [],
            'phase': [],
            'train_loss': [],
            'train_loss_mse': [],
            'train_loss_reg': [],
            'val_loss': [],
            'val_mse': [],
            'val_rmse': [],
            'val_mae': [],
            'val_pearson_r': [],
            'temperature': [],
            'learning_rate': []
        }

        # 早停监控
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

        # 权重历史（可选）
        self.weight_history = [] if config.SAVE_WEIGHT_HISTORY else None

    def _adjust_phase(self, epoch: int):
        """
        根据当前 epoch 调整训练阶段

        核心逻辑：
            Phase 1 (0 ~ PHASE_1_EPOCHS): 冻结权重网络，训练骨干
            Phase 2 (PHASE_1_EPOCHS ~ PHASE_1+PHASE_2): 冻结骨干，训练权重网络
            Phase 3 (剩余轮数): 解冻所有参数，联合微调
        """
        current_phase = get_current_phase(
            epoch,
            self.config.PHASE_1_EPOCHS,
            self.config.PHASE_2_EPOCHS
        )

        if current_phase == 1:
            # Phase 1: 预热骨干网络
            self.model.unfreeze_all()
            self.model.freeze_weight_net()
            phase_name = "Phase 1: 预热 Backbone（冻结权重网络）"

        elif current_phase == 2:
            # Phase 2: 学习动态权重规则
            self.model.unfreeze_all()
            self.model.freeze_backbone()
            phase_name = "Phase 2: 学习动态权重规则（冻结 Backbone）"

        else:
            # Phase 3: 联合微调
            self.model.unfreeze_all()
            phase_name = "Phase 3: 端到端联合微调"

        print(f"\n{'=' * 80}")
        print(f"[Epoch {epoch + 1}/{self.config.NUM_EPOCHS}] {phase_name}")
        print(f"{'=' * 80}")

        return current_phase

    def _train_one_epoch(self, epoch: int) -> dict:
        """训练一个 epoch"""
        self.model.train()

        # 计算当前温度（线性退火）
        current_temp = compute_temperature(
            epoch,
            self.config.NUM_EPOCHS,
            self.config.TEMP_START,
            self.config.TEMP_END
        )

        epoch_loss = 0.0
        epoch_loss_mse = 0.0
        epoch_loss_reg = 0.0

        pbar = tqdm(
            self.dataloaders['train'],
            desc=f"Training (T={current_temp:.2f})",
            leave=False
        )

        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device)  # [Batch, Seq_Len, Feature_Dim]
            y = y.to(self.device)  # [Batch, Pred_Len, 1]

            # ========== 前向传播 ==========
            # preds: [Batch, Pred_Len, 1]
            # dynamic_weights: [Batch, Seq_Len, Sentiment_Dim]
            preds, dynamic_weights = self.model(x, temperature=current_temp)

            # ========== 复合损失函数 ==========
            # 1. 预测误差（MSE）
            loss_mse = self.criterion(preds, y)

            # 2. L1 正则化（权重稀疏化约束）
            # 目的：让模型倾向于选择少数关键的情感维度，避免权重过于平均
            loss_reg = torch.mean(torch.abs(dynamic_weights)) * self.config.L1_LAMBDA

            # 3. 总损失
            total_loss = loss_mse + loss_reg

            # ========== 反向传播 ==========
            self.optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRADIENT_CLIP
            )

            self.optimizer.step()

            # ========== 统计损失 ==========
            epoch_loss += total_loss.item()
            epoch_loss_mse += loss_mse.item()
            epoch_loss_reg += loss_reg.item()

            # 更新进度条
            if batch_idx % self.config.LOG_INTERVAL == 0:
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'mse': loss_mse.item(),
                    'reg': loss_reg.item()
                })

            # 保存权重历史（可选，仅保存第一个样本）
            if self.weight_history is not None and batch_idx == 0:
                self.weight_history.append({
                    'epoch': epoch,
                    'weights': dynamic_weights[0].detach().cpu().numpy()  # [Seq_Len, Sentiment_Dim]
                })

        # 计算平均损失
        num_batches = len(self.dataloaders['train'])
        avg_loss = epoch_loss / num_batches
        avg_loss_mse = epoch_loss_mse / num_batches
        avg_loss_reg = epoch_loss_reg / num_batches

        return {
            'total_loss': avg_loss,
            'mse_loss': avg_loss_mse,
            'reg_loss': avg_loss_reg,
            'temperature': current_temp
        }

    def _validate(self) -> dict:
        """验证"""
        self.model.eval()

        all_preds = []
        all_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for x, y in self.dataloaders['val']:
                x = x.to(self.device)
                y = y.to(self.device)

                # 前向传播（验证时使用固定温度）
                preds, _ = self.model(x, temperature=self.config.TEMP_END)

                # 计算损失
                loss = self.criterion(preds, y)
                val_loss += loss.item()

                # 收集预测结果
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        # 计算平均损失
        avg_val_loss = val_loss / len(self.dataloaders['val'])

        # 合并所有预测结果
        all_preds = np.concatenate(all_preds, axis=0)  # [N, pred_len, 1]
        all_targets = np.concatenate(all_targets, axis=0)  # [N, pred_len, 1]

        # 计算评估指标
        metrics = compute_metrics(all_preds, all_targets)
        metrics['val_loss'] = avg_val_loss

        return metrics

    def _early_stopping_check(self, val_loss: float) -> bool:
        """
        早停检查

        返回：
            True: 应该停止训练
            False: 继续训练
        """
        if val_loss < self.best_val_loss - self.config.MIN_DELTA:
            # 验证损失有明显改善
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = self.model.state_dict().copy()
            return False
        else:
            # 验证损失没有改善
            self.patience_counter += 1
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\n⚠️  早停触发! 验证损失连续 {self.config.PATIENCE} 轮无改善")
                return True
            return False

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        if is_best:
            # 保存最佳模型
            save_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                f"best_model_{self.config.TARGET_SCENIC_NAME}.pth"
            )
            torch.save(checkpoint, save_path)
            print(f"✅ 最佳模型已保存: {save_path}")
        else:
            # 定期保存
            save_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(checkpoint, save_path)
            print(f"💾 检查点已保存: {save_path}")

    def train(self):
        """完整的三阶段训练流程"""
        print("\n" + "=" * 80)
        print("开始三阶段渐进式训练")
        print("=" * 80)

        for epoch in range(self.config.NUM_EPOCHS):
            # ========== Step 1: 调整训练阶段 ==========
            current_phase = self._adjust_phase(epoch)

            # ========== Step 2: 训练一个 epoch ==========
            train_metrics = self._train_one_epoch(epoch)

            # ========== Step 3: 验证 ==========
            if (epoch + 1) % self.config.VAL_INTERVAL == 0:
                val_metrics = self._validate()

                # 打印指标
                print(f"\n【Epoch {epoch + 1}/{self.config.NUM_EPOCHS} 训练结果】")
                print(f"  训练损失: {train_metrics['total_loss']:.6f} "
                      f"(MSE: {train_metrics['mse_loss']:.6f}, "
                      f"Reg: {train_metrics['reg_loss']:.6f})")
                print(f"  验证损失: {val_metrics['val_loss']:.6f}")
                print(f"  验证指标: MSE={val_metrics['MSE']:.6f}, "
                      f"RMSE={val_metrics['RMSE']:.6f}, "
                      f"MAE={val_metrics['MAE']:.6f}, "
                      f"Pearson-r={val_metrics['Pearson_r']:.4f}")
                print(f"  当前温度: {train_metrics['temperature']:.2f}")
                print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

                # 记录历史
                self.history['epoch'].append(epoch + 1)
                self.history['phase'].append(current_phase)
                self.history['train_loss'].append(train_metrics['total_loss'])
                self.history['train_loss_mse'].append(train_metrics['mse_loss'])
                self.history['train_loss_reg'].append(train_metrics['reg_loss'])
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['val_mse'].append(val_metrics['MSE'])
                self.history['val_rmse'].append(val_metrics['RMSE'])
                self.history['val_mae'].append(val_metrics['MAE'])
                self.history['val_pearson_r'].append(val_metrics['Pearson_r'])
                self.history['temperature'].append(train_metrics['temperature'])
                self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

                # ========== Step 4: 早停检查 ==========
                if self.config.EARLY_STOPPING:
                    should_stop = self._early_stopping_check(val_metrics['val_loss'])

                    if val_metrics['val_loss'] < self.best_val_loss + self.config.MIN_DELTA:
                        self._save_checkpoint(epoch, is_best=True)

                    if should_stop:
                        print("\n🛑 训练提前终止（早停）")
                        break
                else:
                    # 不使用早停，直接检查是否最优
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self._save_checkpoint(epoch, is_best=True)

            # ========== Step 5: 定期保存 ==========
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0 and not self.config.SAVE_BEST_ONLY:
                self._save_checkpoint(epoch, is_best=False)

            # ========== Step 6: 学习率调度 ==========
            if self.scheduler is not None:
                self.scheduler.step()

        # ========== 训练结束 ==========
        print("\n" + "=" * 80)
        print("训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print("=" * 80)

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("✅ 已恢复最佳模型权重")

    def save_training_log(self):
        """保存训练日志为 CSV"""
        log_df = pd.DataFrame(self.history)
        log_path = os.path.join(
            self.config.LOG_DIR,
            f"training_log_{self.config.TARGET_SCENIC_NAME}.csv"
        )
        log_df.to_csv(log_path, index=False, encoding='utf-8')
        print(f"✅ 训练日志已保存: {log_path}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.config.PLOT_TRAINING_CURVES:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # RMSE 曲线
        axes[0, 1].plot(self.history['epoch'], self.history['val_rmse'], label='Val RMSE', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Validation RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Pearson-r 曲线
        axes[1, 0].plot(self.history['epoch'], self.history['val_pearson_r'], label='Pearson-r', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Pearson-r')
        axes[1, 0].set_title('Validation Pearson Correlation')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 温度退火曲线
        axes[1, 1].plot(self.history['epoch'], self.history['temperature'], label='Temperature', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Temperature')
        axes[1, 1].set_title('Temperature Annealing')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        # 保存图片
        plot_path = os.path.join(
            self.config.RESULT_DIR,
            f"training_curves_{self.config.TARGET_SCENIC_NAME}.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ 训练曲线已保存: {plot_path}")
        plt.close()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主训练流程"""

    print("\n" + "=" * 80)
    print("动态融合预测模型 - 三阶段渐进式训练")
    print("=" * 80)

    # ========== Step 1: 设置随机种子 ==========
    set_seed(TrainConfig.SEED)

    # ========== Step 2: 验证配置 ==========
    TrainConfig.validate_config()
    TrainConfig.print_config()
    TrainConfig.create_directories()

    # ========== Step 3: 加载数据 ==========
    print("\n" + "=" * 80)
    print("Step 1: 加载数据")
    print("=" * 80)

    data_loader = DataLoader_M8(TrainConfig)
    dataloaders = data_loader.create_dataloaders()

    # ========== Step 4: 创建模型配置 ==========
    print("\n" + "=" * 80)
    print("Step 2: 创建模型")
    print("=" * 80)

    # 创建 ModelConfig 并加载特征映射
    model_config = ModelConfig()
    model_config.load_feature_map(data_loader.feature_map)

    # ========== Step 5: 实例化模型 ==========
    model = DynamicFusionPredictor(model_config)

    # 打印模型信息
    trainable_params = model.get_trainable_params()
    print(f"\n模型参数统计:")
    print(f"  权重生成网络: {trainable_params['weight_net']:,} 参数")
    print(f"  骨干网络: {trainable_params['backbone']:,} 参数")
    print(f"  总参数: {trainable_params['total']:,} 参数")

    # ========== Step 6: 创建训练器 ==========
    print("\n" + "=" * 80)
    print("Step 3: 创建训练器")
    print("=" * 80)

    trainer = ThreePhaseTrainer(model, dataloaders, TrainConfig)

    # ========== Step 7: 开始训练 ==========
    trainer.train()

    # ========== Step 8: 保存训练日志 ==========
    print("\n" + "=" * 80)
    print("Step 4: 保存训练结果")
    print("=" * 80)

    trainer.save_training_log()
    trainer.plot_training_curves()

    # ========== Step 9: 测试集评估（可选）==========
    print("\n" + "=" * 80)
    print("Step 5: 测试集评估")
    print("=" * 80)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloaders['test']:
            x = x.to(TrainConfig.DEVICE)
            y = y.to(TrainConfig.DEVICE)

            preds, _ = model(x, temperature=TrainConfig.TEMP_END)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    test_metrics = compute_metrics(all_preds, all_targets)

    print(f"\n【测试集最终结果】")
    print(f"  MSE: {test_metrics['MSE']:.6f}")
    print(f"  RMSE: {test_metrics['RMSE']:.6f}")
    print(f"  MAE: {test_metrics['MAE']:.6f}")
    print(f"  Pearson-r: {test_metrics['Pearson_r']:.4f}")

    # 保存测试结果
    test_results = {
        'scenic_name': TrainConfig.TARGET_SCENIC_NAME,
        **test_metrics
    }

    result_path = os.path.join(
        TrainConfig.RESULT_DIR,
        f"test_results_{TrainConfig.TARGET_SCENIC_NAME}.json"
    )

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 测试结果已保存: {result_path}")

    print("\n" + "=" * 80)
    print("🎉 训练流程全部完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()