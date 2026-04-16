"""
模块 4：基线模型微调脚本
使用HuggingFace Transformers训练多维情感回归模型（离散等级输出）
直接在[-2, 2]范围内回归，使用MSE+Pearson组合损失
"""

import os
import sys
import json
import logging
import warnings
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ==================== HuggingFace 国内镜像配置 ====================
# 使用 hf-mirror.com 镜像源，无需 VPN 即可下载模型
# 注意：此设置必须在 import transformers 之前执行
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# 导入配置
sys.path.append(str(Path(__file__).parent.parent))
from config.m4_config import BaselineConfig as Config

warnings.filterwarnings('ignore')


class _OptunaPrunedException(Exception):
    """Optuna 剪枝异常，用于 HPO 流程中信号剪枝"""
    pass


# ==================== 组合损失函数 ====================
class CombinedLoss(nn.Module):
    """
    组合损失函数：MSE + Pearson相关系数损失

    Loss = α * MSE + β * (1 - Pearson_r)

    Args:
        mse_weight: MSE损失的权重
        pearson_weight: Pearson损失的权重
        eps: 数值稳定性常数
    """

    def __init__(self, mse_weight: float = 1.0, pearson_weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        self.eps = eps
        self.mse_loss = nn.MSELoss()

    def pearson_correlation_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算Pearson相关系数损失（1 - r）

        Pearson r = Cov(X,Y) / (Std(X) * Std(Y))
        Loss = 1 - r (最小化此值等价于最大化相关系数)
        """
        # 对每个维度分别计算，然后取平均
        batch_size, num_dims = y_pred.shape
        pearson_losses = []

        for dim in range(num_dims):
            pred_dim = y_pred[:, dim]
            true_dim = y_true[:, dim]

            # 中心化
            pred_mean = torch.mean(pred_dim)
            true_mean = torch.mean(true_dim)
            pred_centered = pred_dim - pred_mean
            true_centered = true_dim - true_mean

            # 计算协方差
            covariance = torch.mean(pred_centered * true_centered)

            # 计算标准差
            pred_std = torch.sqrt(torch.mean(pred_centered ** 2) + self.eps)
            true_std = torch.sqrt(torch.mean(true_centered ** 2) + self.eps)

            # Pearson相关系数
            pearson_r = covariance / (pred_std * true_std + self.eps)

            # 损失：1 - r (范围[0, 2]，r=1时损失为0)
            pearson_loss = 1.0 - pearson_r
            pearson_losses.append(pearson_loss)

        # 返回所有维度的平均损失
        return torch.mean(torch.stack(pearson_losses))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算组合损失

        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失的字典（用于日志）
        """
        # MSE损失
        mse_loss = self.mse_loss(y_pred, y_true)

        # Pearson损失
        pearson_loss = self.pearson_correlation_loss(y_pred, y_true)

        # 组合损失
        total_loss = self.mse_weight * mse_loss + self.pearson_weight * pearson_loss

        # 返回损失字典用于日志
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'pearson': pearson_loss.item()
        }

        return total_loss, loss_dict


# ==================== 离散化工具函数 ====================
def discretize_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    将连续预测值离散化到最近的等级 {-2, -1, 0, 1, 2}

    Args:
        predictions: 连续预测值（理论范围 [-2, 2]）

    Returns:
        离散化后的预测值
    """
    discretized = np.round(predictions).astype(np.int32)
    discretized = np.clip(discretized, -2, 2)
    return discretized


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算离散化后的准确率（完全匹配）

    Args:
        y_true: 原始标签 (-2到2)，整数
        y_pred: 连续预测值（理论范围 [-2, 2]）

    Returns:
        准确率
    """
    y_pred_discrete = discretize_predictions(y_pred)
    return np.mean(y_true == y_pred_discrete)


class SentimentDataset(Dataset):
    """情感分析数据集"""

    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        # 直接使用原始标签 [-2, -1, 0, 1, 2]
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


class SentimentRegressor(nn.Module):
    """多维情感回归模型（直接输出 [-2, 2] 范围）"""

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        predictions = self.regressor(pooled_output)

        return predictions


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 3, mode: str = 'max', delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True

        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class BaselineTrainer:
    """基线模型训练器"""

    def __init__(self, training_overrides: Dict = None,
                 trial=None, trial_metric: str = None, prune_enabled: bool = False):
        """
        初始化训练器

        Args:
            training_overrides: 超参数覆盖字典，键为 Config 属性名，值为要覆盖的值
            trial: Optuna trial 对象（用于 HPO 剪枝上报）
            trial_metric: HPO 中间指标名称（如 'avg_pearsonr'）
            prune_enabled: 是否启用 Optuna 剪枝
        """
        self.config = Config
        self.trial = trial
        self.trial_metric = trial_metric
        self.prune_enabled = prune_enabled

        # 应用超参数覆盖
        self._original_config_values = {}
        if training_overrides:
            self._apply_overrides(training_overrides)

        self._setup_logging()
        self._set_seed()
        self._setup_device()
        self._handle_force_restart()
        Config.create_dirs()

        self.logger.info("=" * 80)
        self.logger.info("模块 4：基线模型微调（MSE+Pearson组合损失）")
        self.logger.info("=" * 80)
        Config.print_config()

        # 打印损失函数配置
        self.logger.info("\n损失函数配置:")
        self.logger.info(f"  - MSE权重: {Config.MSE_WEIGHT}")
        self.logger.info(f"  - Pearson权重: {Config.PEARSON_WEIGHT}")
        self.logger.info(f"  - 组合损失 = {Config.MSE_WEIGHT} * MSE + {Config.PEARSON_WEIGHT} * (1 - Pearson_r)")

    def _apply_overrides(self, overrides: Dict):
        """将超参数覆盖应用到 Config 类属性"""
        for key, value in overrides.items():
            if hasattr(Config, key):
                self._original_config_values[key] = getattr(Config, key)
                setattr(Config, key, value)

    def restore_config(self):
        """恢复 Config 到覆盖前的原始值"""
        for key, value in self._original_config_values.items():
            setattr(Config, key, value)
        self._original_config_values.clear()

    def _setup_logging(self):
        """设置日志"""
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _set_seed(self):
        """设置随机种子"""
        import random
        random.seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        torch.manual_seed(Config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.RANDOM_SEED)
        self.logger.info(f"✓ 随机种子: {Config.RANDOM_SEED}")

    def _setup_device(self):
        """设置计算设备"""
        if Config.DEVICE == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"✓ 使用设备: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            self.device = torch.device("cpu")
            self.logger.info("✓ 使用设备: CPU")

    def _handle_force_restart(self):
        """处理强制重启选项"""
        if Config.FORCE_RESTART and Config.CHECKPOINT_DIR.exists():
            self.logger.warning(f"⚠ 强制重启模式：删除所有检查点")
            shutil.rmtree(Config.CHECKPOINT_DIR)
            self.logger.info(f"✓ 已删除: {Config.CHECKPOINT_DIR}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载数据集"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("加载数据集")
        self.logger.info("=" * 80)

        train_df = pd.read_csv(Config.TRAIN_FILE)
        val_df = pd.read_csv(Config.VAL_FILE)
        test_df = pd.read_csv(Config.TEST_FILE)

        self.logger.info(f"✓ 训练集: {len(train_df)} 条")
        self.logger.info(f"✓ 验证集: {len(val_df)} 条")
        self.logger.info(f"✓ 测试集: {len(test_df)} 条")

        required_cols = ['id', 'text'] + Config.DIMENSIONS
        for df_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            missing = set(required_cols) - set(df.columns)
            if missing:
                raise ValueError(f"{df_name}.csv 缺少字段: {missing}")

        # 验证标签值是否在有效范围内
        for df_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            for dim in Config.DIMENSIONS:
                unique_vals = df[dim].unique()
                invalid = set(unique_vals) - {-2, -1, 0, 1, 2}
                if invalid:
                    raise ValueError(f"{df_name}.csv 的 {dim} 列包含无效值: {invalid}，应为 -2,-1,0,1,2")

        self.logger.info("✓ 标签验证通过：所有维度均为 {-2, -1, 0, 1, 2}")

        return train_df, val_df, test_df

    def create_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        self.logger.info("\n创建数据加载器...")

        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.logger.info(f"✓ Tokenizer: {Config.MODEL_NAME}")

        train_texts = train_df['text'].tolist()
        val_texts = val_df['text'].tolist()
        test_texts = test_df['text'].tolist()

        # 直接使用原始标签（-2到2）
        train_labels = train_df[Config.DIMENSIONS].values
        val_labels = val_df[Config.DIMENSIONS].values
        test_labels = test_df[Config.DIMENSIONS].values

        # 保存原始标签供评估使用
        self.train_labels_original = train_labels
        self.val_labels_original = val_labels
        self.test_labels_original = test_labels

        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, Config.MAX_LENGTH)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, Config.MAX_LENGTH)
        test_dataset = SentimentDataset(test_texts, test_labels, self.tokenizer, Config.MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

        self.logger.info(f"✓ 批次大小: {Config.BATCH_SIZE}")
        self.logger.info(f"✓ 训练批次数: {len(train_loader)}")
        self.logger.info(f"✓ 标签范围: [-2, 2]（直接回归，无归一化）")

        return train_loader, val_loader, test_loader

    def create_model(self) -> SentimentRegressor:
        """创建模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("创建模型")
        self.logger.info("=" * 80)

        model = SentimentRegressor(
            model_name=Config.MODEL_NAME,
            num_labels=Config.NUM_LABELS,
            dropout=Config.HIDDEN_DROPOUT
        )
        model = model.to(self.device)

        self.logger.info(f"✓ 预训练模型: {Config.MODEL_NAME}")
        self.logger.info(f"✓ 输出维度: {Config.NUM_LABELS}")
        self.logger.info(f"✓ Dropout: {Config.HIDDEN_DROPOUT}")
        self.logger.info(f"✓ 输出层: 线性层（无激活函数，直接回归）")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"✓ 总参数: {total_params:,}")
        self.logger.info(f"✓ 可训练参数: {trainable_params:,}")

        return model

    def create_optimizer_scheduler(self, model: nn.Module, num_training_steps: int) -> Tuple[AdamW, any]:
        """创建优化器和学习率调度器"""
        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            eps=Config.ADAM_EPSILON,
            weight_decay=Config.WEIGHT_DECAY
        )

        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        self.logger.info(f"✓ 学习率: {Config.LEARNING_RATE}")
        self.logger.info(f"✓ 预热步数: {num_warmup_steps}")
        self.logger.info(f"✓ 总训练步数: {num_training_steps}")

        return optimizer, scheduler

    def save_checkpoint(self, model: nn.Module, optimizer: AdamW, scheduler: any,
                        epoch: int, metrics: Dict[str, float], global_best_pearson: float,
                        is_best: bool = False):
        """
        保存训练检查点
        :param global_best_pearson: 历史全局最佳 Pearsonr (确保断点重续时知道真正的最佳是多少)
        """
        if not Config.SAVE_CHECKPOINT_EVERY_EPOCH and not is_best:
            return

        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': model.encoder.state_dict(),
            'regressor_state_dict': model.regressor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'model_name': Config.MODEL_NAME,
            'num_labels': Config.NUM_LABELS,
            'dropout': Config.HIDDEN_DROPOUT,

            # [修改关键点] 这里保存传入的全局最佳，而不是当前 metrics 中的值
            'best_pearsonr': global_best_pearson,
            'best_accuracy': metrics.get('avg_accuracy', 0)
        }

        # 保存最新检查点
        if Config.SAVE_CHECKPOINT_EVERY_EPOCH:
            latest_path = Config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, latest_path)
            self.logger.info(f"✓ 检查点已保存: epoch_{epoch}.pt")

            # 创建或更新latest链接
            latest_link = Config.CHECKPOINT_DIR / 'latest_checkpoint.pt'
            if latest_link.exists():
                latest_link.unlink()
            torch.save(checkpoint, latest_link)

        # 保存最佳模型
        if is_best:
            best_path = Config.CHECKPOINT_DIR / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(
                f"✓ 最佳检查点已保存 (Pearsonr: {global_best_pearson:.3f}, "
                f"Accuracy: {metrics.get('avg_accuracy', 0):.3f})"
            )

        # 清理旧检查点
        if Config.KEEP_CHECKPOINT_MAX is not None:
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件"""
        checkpoint_files = sorted(
            [f for f in Config.CHECKPOINT_DIR.glob('checkpoint_epoch_*.pt')],
            key=lambda x: int(x.stem.split('_')[-1])
        )

        if len(checkpoint_files) > Config.KEEP_CHECKPOINT_MAX:
            for old_file in checkpoint_files[:-Config.KEEP_CHECKPOINT_MAX]:
                old_file.unlink()
                self.logger.info(f"✓ 清理旧检查点: {old_file.name}")

    def load_checkpoint(self, model: nn.Module, optimizer: AdamW = None,
                        scheduler: any = None) -> Dict:
        """加载训练检查点"""
        if not Config.RESUME_TRAINING:
            self.logger.info("未启用断点续传，从头开始训练")
            return {'epoch': 0, 'metrics': None}

        # 确定要加载的检查点
        checkpoint_path = None

        if Config.CHECKPOINT_TYPE == 'specific' and Config.RESUME_FROM_EPOCH is not None:
            checkpoint_path = Config.CHECKPOINT_DIR / f'checkpoint_epoch_{Config.RESUME_FROM_EPOCH}.pt'
        elif Config.CHECKPOINT_TYPE == 'best':
            checkpoint_path = Config.CHECKPOINT_DIR / 'best_checkpoint.pt'
        else:  # latest
            checkpoint_path = Config.CHECKPOINT_DIR / 'latest_checkpoint.pt'

        if not checkpoint_path.exists():
            self.logger.info(f"未找到检查点: {checkpoint_path.name}，从头开始训练")
            return {'epoch': 0, 'metrics': None}

        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"加载检查点: {checkpoint_path.name}")
        self.logger.info(f"{'=' * 80}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型权重
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.regressor.load_state_dict(checkpoint['regressor_state_dict'])

        # 加载优化器和调度器状态
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("✓ 优化器状态已恢复")

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("✓ 学习率调度器已恢复")

        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']

        # [修改关键点] 获取保存的历史最佳分数
        best_pearsonr = checkpoint.get('best_pearsonr', -1)

        self.logger.info(f"✓ 从 Epoch {epoch} 恢复训练")
        self.logger.info(f"✓ 历史最佳 Pearsonr: {best_pearsonr:.3f}")

        if metrics:
            self.logger.info(f"  - 上次保存时 Pearsonr: {metrics.get('avg_pearsonr', 'N/A'):.3f}")

        return {
            'epoch': epoch,
            'metrics': metrics,
            'best_pearsonr': best_pearsonr,
            'best_accuracy': checkpoint.get('best_accuracy', 0)
        }

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: AdamW, scheduler: any, epoch: int,
                   loss_fn: CombinedLoss) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0
        total_mse_loss = 0
        total_pearson_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)

            # 使用组合损失函数
            loss, loss_dict = loss_fn(predictions, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss_dict['total']
            total_mse_loss += loss_dict['mse']
            total_pearson_loss += loss_dict['pearson']
            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            if (batch_idx + 1) % Config.LOG_INTERVAL == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_mse = total_mse_loss / (batch_idx + 1)
                avg_pearson = total_pearson_loss / (batch_idx + 1)
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, Pearson: {avg_pearson:.4f})"
                )

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # 计算指标
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(train_loader)
        metrics['mse_loss'] = total_mse_loss / len(train_loader)
        metrics['pearson_loss'] = total_pearson_loss / len(train_loader)

        return metrics

    def evaluate(self, model: nn.Module, data_loader: DataLoader,
                loss_fn: CombinedLoss = None) -> Dict[str, float]:
        """评估模型"""
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        total_mse_loss = 0
        total_pearson_loss = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                predictions = model(input_ids, attention_mask)

                # 如果提供了损失函数，计算损失
                if loss_fn is not None:
                    _, loss_dict = loss_fn(predictions, labels)
                    total_loss += loss_dict['total']
                    total_mse_loss += loss_dict['mse']
                    total_pearson_loss += loss_dict['pearson']

                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        metrics = self._calculate_metrics(all_labels, all_preds)

        if loss_fn is not None:
            metrics['loss'] = total_loss / len(data_loader)
            metrics['mse_loss'] = total_mse_loss / len(data_loader)
            metrics['pearson_loss'] = total_pearson_loss / len(data_loader)

        return metrics

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            y_true: 真实标签 ([-2, 2] 范围的整数)
            y_pred: 模型预测值 (连续值)

        Returns:
            评估指标字典
        """
        metrics = {}

        for i, dim in enumerate(Config.DIMENSIONS):
            # 使用连续值计算相关系数
            pearson, _ = pearsonr(y_true[:, i], y_pred[:, i])
            metrics[f'{dim}_pearsonr'] = float(pearson)

            # 使用连续值计算RMSE
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            metrics[f'{dim}_rmse'] = float(rmse)

            # 使用连续值计算MAE
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            metrics[f'{dim}_mae'] = float(mae)

            # 使用离散化后的值计算准确率
            accuracy = calculate_accuracy(y_true[:, i], y_pred[:, i])
            metrics[f'{dim}_accuracy'] = float(accuracy)

        # 平均指标
        metrics['avg_pearsonr'] = float(np.mean([metrics[f'{dim}_pearsonr'] for dim in Config.DIMENSIONS]))
        metrics['avg_rmse'] = float(np.mean([metrics[f'{dim}_rmse'] for dim in Config.DIMENSIONS]))
        metrics['avg_mae'] = float(np.mean([metrics[f'{dim}_mae'] for dim in Config.DIMENSIONS]))
        metrics['avg_accuracy'] = float(np.mean([metrics[f'{dim}_accuracy'] for dim in Config.DIMENSIONS]))

        return metrics

    def print_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """打印指标"""
        self.logger.info(f"\n{prefix}评估指标:")

        # 如果有损失信息，先打印损失
        if 'loss' in metrics:
            self.logger.info(f"损失: Total={metrics['loss']:.4f}, "
                           f"MSE={metrics.get('mse_loss', 0):.4f}, "
                           f"Pearson={metrics.get('pearson_loss', 0):.4f}")

        self.logger.info("-" * 100)
        self.logger.info(f"{'维度':<8} {'Pearsonr':>10} {'RMSE':>10} {'MAE':>10} {'Accuracy':>10}")
        self.logger.info("-" * 100)

        for dim in Config.DIMENSIONS:
            pearson = metrics[f'{dim}_pearsonr']
            rmse = metrics[f'{dim}_rmse']
            mae = metrics[f'{dim}_mae']
            accuracy = metrics[f'{dim}_accuracy']
            self.logger.info(
                f"{dim:<8} {pearson:>10.3f} {rmse:>10.3f} {mae:>10.3f} {accuracy:>10.3f}"
            )

        self.logger.info("-" * 100)
        self.logger.info(
            f"{'平均':<8} {metrics['avg_pearsonr']:>10.3f} "
            f"{metrics['avg_rmse']:>10.3f} {metrics['avg_mae']:>10.3f} "
            f"{metrics['avg_accuracy']:>10.3f}"
        )
        self.logger.info("-" * 100)

        if metrics['avg_pearsonr'] >= Config.QUALIFIED_THRESHOLD:
            self.logger.info(f"✓ 模型合格 (Pearsonr ≥ {Config.QUALIFIED_THRESHOLD})")
        else:
            self.logger.warning(
                f"⚠ 模型未达标 (Pearsonr: {metrics['avg_pearsonr']:.3f} < {Config.QUALIFIED_THRESHOLD})"
            )

    def plot_predictions(self, model: nn.Module, test_loader: DataLoader, save_dir: Path):
        """绘制预测值-真实值散点图"""
        self.logger.info("\n生成预测散点图...")
        save_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                predictions = model(input_ids, attention_mask)

                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        for i, dim in enumerate(Config.DIMENSIONS):
            plt.figure(figsize=Config.FIGURE_SIZE, dpi=Config.FIGURE_DPI)

            # 直接使用连续预测值绘图
            plt.scatter(
                all_labels[:, i],
                all_preds[:, i],
                alpha=Config.SCATTER_ALPHA,
                s=Config.SCATTER_SIZE,
                c='steelblue',
                edgecolors='white',
                linewidth=0.5
            )

            # 绘制理想对角线
            plt.plot([-2, 2], [-2, 2], 'r--', lw=2, alpha=0.7, label='理想预测')

            # 计算离散化后的准确率
            accuracy = calculate_accuracy(all_labels[:, i], all_preds[:, i])

            # 计算连续值的pearson相关系数
            pearson, _ = pearsonr(all_labels[:, i], all_preds[:, i])

            # 计算RMSE和MAE
            rmse = np.sqrt(mean_squared_error(all_labels[:, i], all_preds[:, i]))
            mae = mean_absolute_error(all_labels[:, i], all_preds[:, i])

            plt.xlabel('真实值', fontsize=12)
            plt.ylabel('预测值', fontsize=12)
            plt.title(
                f'{dim}\nPearson r={pearson:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, Acc={accuracy:.3f}',
                fontsize=12
            )
            plt.xticks([-2, -1, 0, 1, 2])
            plt.yticks(np.arange(-3, 4, 1))
            plt.xlim(-2.5, 2.5)
            plt.ylim(-3, 3)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            save_path = save_dir / f'{dim}_scatter.png'
            plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✓ {dim} 散点图已保存: {save_path}")

    @staticmethod
    def _cleanup_gpu(*objects):
        """
        显式清理 GPU 资源，防止 trial 之间内存泄漏

        Args:
            *objects: 需要删除的 GPU 对象（model, optimizer 等）
        """
        for obj in objects:
            if obj is not None:
                del obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def train(self):
        """完整训练流程"""
        model = None
        best_model = None
        optimizer = None
        scheduler = None
        loss_fn = None

        try:
            train_df, val_df, test_df = self.load_data()
            train_loader, val_loader, test_loader = self.create_dataloaders(train_df, val_df, test_df)

            model = self.create_model()

            # 创建组合损失函数
            loss_fn = CombinedLoss(
                mse_weight=Config.MSE_WEIGHT,
                pearson_weight=Config.PEARSON_WEIGHT
            )

            num_training_steps = len(train_loader) * Config.NUM_EPOCHS
            optimizer, scheduler = self.create_optimizer_scheduler(model, num_training_steps)

            # 尝试加载检查点
            checkpoint_info = self.load_checkpoint(model, optimizer, scheduler)
            start_epoch = checkpoint_info['epoch'] + 1
            best_val_pearson = checkpoint_info.get('best_pearsonr', -1)

            early_stopping = EarlyStopping(
                patience=Config.EARLY_STOPPING_PATIENCE,
                mode=Config.EARLY_STOPPING_MODE
            )

            # 如果从检查点恢复，更新早停状态
            if checkpoint_info['epoch'] > 0:
                # 修改点：直接使用加载出的历史最优分数，而不是断点那一轮的分数
                early_stopping.best_score = best_val_pearson
                early_stopping.best_epoch = checkpoint_info['epoch']
                self.logger.info(f"✓ 早停机制已恢复 (基准分: {early_stopping.best_score:.3f})")

            self.logger.info("\n" + "=" * 80)
            self.logger.info("开始训练")
            self.logger.info("=" * 80)

            for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
                self.logger.info(f"\n{'=' * 80}")
                self.logger.info(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
                self.logger.info(f"{'=' * 80}")

                train_metrics = self.train_epoch(model, train_loader, optimizer, scheduler, epoch, loss_fn)
                val_metrics = self.evaluate(model, val_loader, loss_fn)

                self.logger.info(f"\n[训练集]")
                self.print_metrics(train_metrics, "")

                self.logger.info(f"\n[验证集]")
                self.print_metrics(val_metrics, "")

                val_pearson = val_metrics['avg_pearsonr']
                is_best = val_pearson > best_val_pearson

                if is_best:
                    best_val_pearson = val_pearson
                    self.logger.info(
                        f"✓ 新的最佳模型 (Pearsonr: {val_pearson:.3f}, "
                        f"Accuracy: {val_metrics['avg_accuracy']:.3f})"
                    )

                # 保存检查点
                # 修改点：传入 global_best_pearson 参数，确保保存文件时写入的是历史最优，而非当前轮次分数
                self.save_checkpoint(
                        model, optimizer, scheduler, epoch,
                        val_metrics,
                        global_best_pearson=best_val_pearson,
                        is_best=is_best
                    )

                # Optuna 中间指标上报与剪枝
                if self.trial is not None and self.trial_metric is not None:
                    report_value = val_metrics.get(self.trial_metric, val_pearson)
                    self.trial.report(report_value, epoch)
                    if self.prune_enabled and self.trial.should_prune():
                        self.logger.info(f"✂️ Optuna 剪枝触发 (epoch {epoch})")
                        raise _OptunaPrunedException(
                            f"Trial pruned at epoch {epoch}, "
                            f"{self.trial_metric}={report_value:.4f}"
                        )

                improved = early_stopping(val_pearson, epoch)
                if early_stopping.early_stop:
                    self.logger.info(
                        f"\n早停触发！最佳epoch: {early_stopping.best_epoch}, "
                        f"最佳Pearsonr: {early_stopping.best_score:.3f}"
                    )
                    break

            self.logger.info("\n" + "=" * 80)
            self.logger.info("测试集评估")
            self.logger.info("=" * 80)

            # 加载最佳模型进行测试
            best_model = self.create_model()
            best_checkpoint_path = Config.CHECKPOINT_DIR / 'best_checkpoint.pt'
            if best_checkpoint_path.exists():
                checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
                best_model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                best_model.regressor.load_state_dict(checkpoint['regressor_state_dict'])
                self.logger.info(f"✓ 加载最佳模型: Epoch {checkpoint['epoch']}")
            else:
                self.logger.warning("⚠ 未找到最佳模型，使用当前模型")
                best_model = model

            test_metrics = self.evaluate(best_model, test_loader, loss_fn)

            self.print_metrics(test_metrics, "[测试集] ")

            # 保存测试指标时加入损失函数配置信息
            test_metrics['loss_config'] = {
                'mse_weight': Config.MSE_WEIGHT,
                'pearson_weight': Config.PEARSON_WEIGHT
            }

            with open(Config.METRICS_FILE, 'w', encoding='utf-8') as f:
                json.dump(test_metrics, f, indent=2, ensure_ascii=False)
            self.logger.info(f"\n✓ 指标已保存: {Config.METRICS_FILE}")

            self.plot_predictions(best_model, test_loader, Config.FIGURES_DIR)

            self.logger.info("\n" + "=" * 80)
            self.logger.info("训练完成")
            self.logger.info("=" * 80)
            self.logger.info(f"✓ 最佳模型: {Config.CHECKPOINT_DIR}")
            self.logger.info(f"✓ 评估指标: {Config.METRICS_FILE}")
            self.logger.info(f"✓ 散点图: {Config.FIGURES_DIR}")
            self.logger.info(f"✓ 日志文件: {Config.LOG_FILE}")

            # 构建完整的结果字典（兼容 HPO 汇总）
            result = dict(test_metrics)
            result['best_val_pearsonr'] = best_val_pearson
            result['best_epoch'] = early_stopping.best_epoch
            result['epochs_completed'] = epoch

            return result

        except _OptunaPrunedException:
            # 剪枝异常需要直接向上传播给 Optuna
            raise
        except Exception as e:
            self.logger.error(f"\n训练失败: {str(e)}", exc_info=True)
            raise
        finally:
            # 无论成功/失败/剪枝，都必须清理 GPU 资源
            self._cleanup_gpu(model, best_model, optimizer, loss_fn)


def main(training_overrides: Dict = None,
         restore_after_run: bool = False,
         trial=None, trial_metric: str = None,
         prune_enabled: bool = False) -> Dict:
    """
    主函数（支持 HPO 调用）

    Args:
        training_overrides: 超参数覆盖字典
        restore_after_run: 运行结束后是否恢复 Config 原始值
        trial: Optuna trial 对象
        trial_metric: HPO 上报指标名
        prune_enabled: 是否启用 Optuna 剪枝

    Returns:
        测试集评估指标字典
    """
    trainer = BaselineTrainer(
        training_overrides=training_overrides,
        trial=trial,
        trial_metric=trial_metric,
        prune_enabled=prune_enabled
    )
    try:
        result = trainer.train()
        return result
    finally:
        if restore_after_run:
            trainer.restore_config()


if __name__ == "__main__":
    main()