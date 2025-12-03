# scripts/m4_train_baseline.py
"""
模块 4：基线模型微调脚本
使用HuggingFace Transformers训练多维情感回归模型（离散等级输出）
"""

import sys
import json
import logging
import warnings
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

import config.m4_config

matplotlib.use('Agg')

# 导入配置
sys.path.append(str(Path(__file__).parent.parent))
from config.m4_config import BaselineConfig as Config

warnings.filterwarnings('ignore')


# ==================== 新增：离散化工具函数 ====================
def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """
    将离散标签归一化到[0,1]区间
    -2 → 0.0, -1 → 0.25, 0 → 0.5, 1 → 0.75, 2 → 1.0
    """
    mapping = {-2: 0.0, -1: 0.25, 0: 0.5, 1: 0.75, 2: 1.0}
    normalized = np.zeros_like(labels, dtype=np.float32)
    for orig, norm in mapping.items():
        normalized[labels == orig] = norm
    return normalized


def denormalize_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    将[0,1]区间的预测值反归一化到[-2,2]区间
    0.0 → -2, 0.25 → -1, 0.5 → 0, 0.75 → 1, 1.0 → 2
    """
    return predictions * 4 - 2


def discretize_predictions(predictions: np.ndarray) -> np.ndarray:
    """
    将连续预测值离散化到最近的等级 {-2, -1, 0, 1, 2}
    """
    denormalized = denormalize_predictions(predictions)
    discretized = np.round(denormalized).astype(np.int32)
    discretized = np.clip(discretized, -2, 2)  # 确保在有效范围内
    return discretized


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算离散化后的准确率（完全匹配）
    y_true: 原始标签 (-2到2)
    y_pred: 连续预测值 (0到1)
    """
    y_pred_discrete = discretize_predictions(y_pred)
    return np.mean(y_true == y_pred_discrete)


class SentimentDataset(Dataset):
    """情感分析数据集"""

    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = texts
        # 归一化标签到[0,1]
        self.labels = normalize_labels(labels)
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
    """多维情感回归模型"""

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        predictions = self.sigmoid(logits)

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

    def __init__(self):
        self.config = Config
        self._setup_logging()
        self._set_seed()
        self._setup_device()
        self._handle_force_restart()
        Config.create_dirs()

        self.logger.info("=" * 80)
        self.logger.info("模块 4：基线模型微调（离散等级输出）")
        self.logger.info("=" * 80)
        Config.print_config()

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

        # 原始标签（-2到2）
        train_labels = train_df[Config.DIMENSIONS].values
        val_labels = val_df[Config.DIMENSIONS].values
        test_labels = test_df[Config.DIMENSIONS].values

        # 保存原始标签供评估使用
        self.train_labels_original = train_labels
        self.val_labels_original = val_labels
        self.test_labels_original = test_labels

        # Dataset内部会自动归一化
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, Config.MAX_LENGTH)
        val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, Config.MAX_LENGTH)
        test_dataset = SentimentDataset(test_texts, test_labels, self.tokenizer, Config.MAX_LENGTH)

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)

        self.logger.info(f"✓ 批次大小: {Config.BATCH_SIZE}")
        self.logger.info(f"✓ 训练批次数: {len(train_loader)}")
        self.logger.info(f"✓ 标签归一化: -2,-1,0,1,2 → 0,0.25,0.5,0.75,1.0")

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
                        epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存训练检查点"""
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
            'best_pearsonr': metrics.get('avg_pearsonr', -1),
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
                f"✓ 最佳检查点已保存 (Pearsonr: {metrics.get('avg_pearsonr', -1):.3f}, "
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

        self.logger.info(f"✓ 从 Epoch {epoch} 恢复训练")
        if metrics:
            self.logger.info(f"  - 验证集 Pearsonr: {metrics.get('avg_pearsonr', 'N/A'):.3f}")
            self.logger.info(f"  - 验证集 Accuracy: {metrics.get('avg_accuracy', 'N/A'):.3f}")
            self.logger.info(f"  - 验证集 RMSE: {metrics.get('avg_rmse', 'N/A'):.3f}")
            self.logger.info(f"  - 验证集 MAE: {metrics.get('avg_mae', 'N/A'):.3f}")

        return {
            'epoch': epoch,
            'metrics': metrics,
            'best_pearsonr': checkpoint.get('best_pearsonr', -1),
            'best_accuracy': checkpoint.get('best_accuracy', 0)
        }

    def train_epoch(self, model: nn.Module, train_loader: DataLoader, optimizer: AdamW, scheduler: any, epoch: int) -> \
    Dict[str, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)

            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            if (batch_idx + 1) % Config.LOG_INTERVAL == 0:
                avg_loss = total_loss / (batch_idx + 1)
                self.logger.info(f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {avg_loss:.4f}")

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # 计算指标（使用归一化后的连续值）
        metrics = self._calculate_metrics(all_labels, all_preds, self.train_labels_original)
        metrics['loss'] = total_loss / len(train_loader)

        return metrics

    def evaluate(self, model: nn.Module, data_loader: DataLoader, original_labels: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                predictions = model(input_ids, attention_mask)

                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        metrics = self._calculate_metrics(all_labels, all_preds, original_labels)
        return metrics

    def _calculate_metrics(self, y_true_norm: np.ndarray, y_pred_norm: np.ndarray,
                           y_true_original: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标
        y_true_norm: 归一化后的真实标签 (0-1)
        y_pred_norm: 模型预测值 (0-1)
        y_true_original: 原始标签 (-2到2)
        """
        metrics = {}

        for i, dim in enumerate(Config.DIMENSIONS):
            # 使用归一化后的连续值计算相关系数
            pearson, _ = pearsonr(y_true_norm[:, i], y_pred_norm[:, i])
            metrics[f'{dim}_pearsonr'] = float(pearson)

            # 使用归一化后的连续值计算RMSE
            rmse = np.sqrt(mean_squared_error(y_true_norm[:, i], y_pred_norm[:, i]))
            metrics[f'{dim}_rmse'] = float(rmse)

            # 使用归一化后的连续值计算MAE
            mae = mean_absolute_error(y_true_norm[:, i], y_pred_norm[:, i])
            metrics[f'{dim}_mae'] = float(mae)

            # 使用离散化后的值计算准确率
            accuracy = calculate_accuracy(y_true_original[:, i], y_pred_norm[:, i])
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

    def plot_predictions(self, model: nn.Module, test_loader: DataLoader, save_dir: Path,):
        """绘制预测值-真实值散点图（使用连续值）"""
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

        # 将归一化的预测值映射回[-2,2]区间
        all_preds_continuous = denormalize_predictions(all_preds)
        all_labels_discrete = self.test_labels_original  # 原始标签本来就是离散的[-2,2]

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        for i, dim in enumerate(Config.DIMENSIONS):
            plt.figure(figsize=Config.FIGURE_SIZE, dpi=Config.FIGURE_DPI)

            # 使用连续预测值绘图
            plt.scatter(
                all_labels_discrete[:, i],
                all_preds_continuous[:, i],
                alpha=Config.SCATTER_ALPHA,
                s=Config.SCATTER_SIZE,
                c='steelblue',
                edgecolors='white',
                linewidth=0.5
            )

            # 绘制理想对角线
            plt.plot([-2, 2], [-2, 2], 'r--', lw=2, alpha=0.7)

            # 计算离散化后的准确率(仍然保留这个指标)
            accuracy = calculate_accuracy(all_labels_discrete[:, i], all_preds[:, i])

            # 计算连续值的pearson相关系数
            pearson, _ = pearsonr(all_labels[:, i], all_preds[:, i])

            plt.xlabel('真实值', fontsize=12)
            plt.ylabel('预测值', fontsize=12)
            plt.title(f'{dim} - Accuracy = {accuracy:.3f}, Pearson r = {pearson:.3f}', fontsize=14)
            plt.xticks([-2, -1, 0, 1, 2])
            plt.yticks([-2, -1, 0, 1, 2])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            # 从MODEL_NAME中提取模型简称
            model_short_name = Config.MODEL_NAME.split('/')[-1].replace('chinese-', '')  # 移除'chinese-'前缀

            save_path = save_dir / f'{model_short_name}_{dim}_scatter.png'
            plt.savefig(save_path, dpi=Config.FIGURE_DPI, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✓ {dim} 散点图已保存: {save_path}")

    def train(self):
        """完整训练流程"""
        try:
            train_df, val_df, test_df = self.load_data()
            train_loader, val_loader, test_loader = self.create_dataloaders(train_df, val_df, test_df)

            model = self.create_model()

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
            if checkpoint_info['epoch'] > 0 and checkpoint_info['metrics']:
                early_stopping.best_score = checkpoint_info['metrics'].get('avg_pearsonr', None)
                early_stopping.best_epoch = checkpoint_info['epoch']

            self.logger.info("\n" + "=" * 80)
            self.logger.info("开始训练")
            self.logger.info("=" * 80)

            for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
                self.logger.info(f"\n{'=' * 80}")
                self.logger.info(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
                self.logger.info(f"{'=' * 80}")

                train_metrics = self.train_epoch(model, train_loader, optimizer, scheduler, epoch)
                val_metrics = self.evaluate(model, val_loader, self.val_labels_original)

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
                self.save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, is_best)

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

            test_metrics = self.evaluate(best_model, test_loader, self.test_labels_original)

            self.print_metrics(test_metrics, "[测试集] ")

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

            return test_metrics

        except Exception as e:
            self.logger.error(f"\n训练失败: {str(e)}", exc_info=True)
            raise


def main():
    """主函数"""
    trainer = BaselineTrainer()
    trainer.train()
    # print (config.m4_config.BaselineConfig.DATA_DIR)

if __name__ == "__main__":
    main()