"""
M4 情感分析模型 - 单句评论测试脚本
功能：加载训练好的模型，对单句评论进行五维度情感分析评分
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import json
import glob
from transformers import AutoTokenizer, AutoModel


# ==================== 配置 ====================
class Config:
    """所有配置集中管理"""

    # ===== 路径配置 =====
    CHECKPOINT_DIR = Path(r'E:\2024liushaoxuan\learn\shiyan1\data\输出\m4_macbert\checkpoints\best_model')
    RESULTS_DIR = Path(r'E:\2024liushaoxuan\learn\shiyan1\results')
    LOG_DIR = Path(r'E:\2024liushaoxuan\learn\shiyan1\logs')

    # ===== 模型配置 =====
    # 预训练模型名称（需要与训练时一致）
    MODEL_NAME = 'bert-base-chinese'  # 根据你的训练配置修改
    # 输出维度数（5个情感维度）
    NUM_LABELS = 5
    # Dropout比率
    HIDDEN_DROPOUT = 0.1

    # ===== Tokenizer配置 =====
    MAX_LENGTH = 128  # 最大序列长度

    # ===== 评分配置 =====
    # 五个评分维度
    SCORE_DIMENSIONS = [
        '景色',        # 效价（积极-消极）
        '交通',        # 唤醒度
        '美食',      # 支配度
        '门票',    # 喜爱度
        '服务'       # 强度
    ]

    # 评分等级：-2(很差), -1(较差), 0(一般), 1(较好), 2(很好)
    SCORE_LEVELS = [-2, -1, 0, 1, 2]
    SCORE_LABELS = {
        -2: '很差',
        -1: '较差',
        0: '一般',
        1: '较好',
        2: '很好'
    }

    # 归一化映射（训练时使用的归一化）
    NORMALIZE_MAPPING = {-2: 0.0, -1: 0.25, 0: 0.5, 1: 0.75, 2: 1.0}

    # ===== 推理配置 =====
    # 是否使用最新模型（True）还是指定模型（False）
    USE_LATEST_MODEL = True
    # 如果USE_LATEST_MODEL=False，指定模型路径
    SPECIFIC_MODEL_PATH = None  # 例如: 'checkpoints/checkpoint_epoch_50.pt'

    # ===== 设备配置 =====
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ===== 输出配置 =====
    # 是否保存详细结果
    SAVE_DETAILED_RESULTS = True
    # 是否打印调试信息
    VERBOSE = True


# ==================== 模型定义 ====================
class SentimentRegressor(nn.Module):
    """多维情感回归模型（与训练时完全一致）"""

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


# ==================== 工具函数 ====================
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
    discretized = np.clip(discretized, -2, 2)
    return discretized


# ==================== 情感分析器 ====================
class SentimentAnalyzer:
    """评论情感分析器"""

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.logger = self._setup_logger()
        self.model = None
        self.tokenizer = None
        self.model_info = {}

        # 创建必要的目录
        Config.RESULTS_DIR.mkdir(exist_ok=True)
        Config.LOG_DIR.mkdir(exist_ok=True)

        self.logger.info(f"使用设备: {self.device}")

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('SentimentAnalyzer')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 文件处理器
        Config.LOG_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            Config.LOG_DIR / f'sentiment_analysis_{timestamp}.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def _find_latest_checkpoint(self) -> Path:
        """查找最新的检查点文件"""
        checkpoint_pattern = str(Config.CHECKPOINT_DIR / '*.pt')
        checkpoints = glob.glob(checkpoint_pattern)

        if not checkpoints:
            raise FileNotFoundError(f"未找到任何检查点文件: {Config.CHECKPOINT_DIR}")

        # 排除latest_checkpoint.pt，优先使用best_checkpoint.pt
        best_checkpoint = Config.CHECKPOINT_DIR / 'best_checkpoint.pt'
        if best_checkpoint.exists():
            return best_checkpoint

        # 如果没有best，使用latest
        latest_checkpoint = Config.CHECKPOINT_DIR / 'latest_checkpoint.pt'
        if latest_checkpoint.exists():
            return latest_checkpoint

        # 否则按修改时间排序
        checkpoints = [p for p in checkpoints if 'latest' not in Path(p).name]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: Path(x).stat().st_mtime)
            return Path(latest_checkpoint)

        raise FileNotFoundError("未找到有效的检查点文件")

    def load_model(self):
        """加载模型"""
        self.logger.info("=" * 80)
        self.logger.info("加载模型...")
        self.logger.info("=" * 80)

        # 确定检查点路径
        if Config.USE_LATEST_MODEL:
            checkpoint_path = self._find_latest_checkpoint()
            self.logger.info(f"使用模型: {checkpoint_path.name}")
        else:
            if Config.SPECIFIC_MODEL_PATH is None:
                raise ValueError("请在Config中设置SPECIFIC_MODEL_PATH")
            checkpoint_path = Path(Config.SPECIFIC_MODEL_PATH)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"指定的模型不存在: {checkpoint_path}")
            self.logger.info(f"使用指定模型: {checkpoint_path}")

        # 加载检查点
        self.logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 从检查点读取模型配置
        model_name = checkpoint.get('model_name', Config.MODEL_NAME)
        num_labels = checkpoint.get('num_labels', Config.NUM_LABELS)
        dropout = checkpoint.get('dropout', Config.HIDDEN_DROPOUT)

        self.logger.info(f"模型配置:")
        self.logger.info(f"  - 预训练模型: {model_name}")
        self.logger.info(f"  - 输出维度: {num_labels}")
        self.logger.info(f"  - Dropout: {dropout}")

        # 创建模型
        model = SentimentRegressor(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout
        ).to(self.device)

        # 加载权重
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.regressor.load_state_dict(checkpoint['regressor_state_dict'])
        model.eval()

        self.model = model

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger.info(f"✓ Tokenizer加载完成")

        # 保存模型信息
        self.model_info = {
            'checkpoint_path': str(checkpoint_path),
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'model_name': model_name,
            'num_labels': num_labels,
        }

        if 'metrics' in checkpoint and checkpoint['metrics']:
            self.model_info['metrics'] = checkpoint['metrics']

        self.logger.info("✓ 模型加载成功")
        self.logger.info(f"  训练轮次: {self.model_info['epoch']}")

        if self.model_info.get('metrics'):
            metrics = self.model_info['metrics']
            self.logger.info("  验证集指标:")
            self.logger.info(f"    平均 Pearsonr: {metrics.get('avg_pearsonr', 'N/A'):.4f}")
            self.logger.info(f"    平均 Accuracy: {metrics.get('avg_accuracy', 'N/A'):.4f}")
            self.logger.info(f"    平均 RMSE: {metrics.get('avg_rmse', 'N/A'):.4f}")
            self.logger.info(f"    平均 MAE: {metrics.get('avg_mae', 'N/A'):.4f}")

        return model

    @torch.no_grad()
    def analyze(self, comment: str) -> dict:
        """
        分析单句评论

        Args:
            comment: 输入的评论文本

        Returns:
            分析结果字典
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("请先调用load_model()加载模型")

        self.logger.info("=" * 80)
        self.logger.info("开始分析评论...")
        self.logger.info("=" * 80)
        self.logger.info(f"评论内容: {comment}")

        # Tokenize输入文本
        encoding = self.tokenizer(
            comment,
            add_special_tokens=True,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 模型预测
        self.model.eval()
        predictions = self.model(input_ids, attention_mask)  # [1, num_labels]
        predictions = predictions.cpu().numpy().flatten()  # [num_labels]

        # 离散化分数
        discrete_scores = discretize_predictions(predictions)

        # 构建结果
        results = {
            'comment': comment,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': self.model_info,
            'continuous_scores': {},
            'normalized_scores': {},
            'discrete_scores': {},
            'score_labels': {}
        }

        # 填充各维度分数
        for i, dimension in enumerate(Config.SCORE_DIMENSIONS):
            # 归一化分数（0-1）
            results['normalized_scores'][dimension] = float(predictions[i])
            # 反归一化分数（-2到2的连续值）
            results['continuous_scores'][dimension] = float(denormalize_predictions(predictions)[i])
            # 离散分数（-2,-1,0,1,2）
            results['discrete_scores'][dimension] = int(discrete_scores[i])
            # 文字标签
            results['score_labels'][dimension] = Config.SCORE_LABELS[discrete_scores[i]]

        # 计算平均分
        results['average_normalized'] = float(np.mean(predictions))
        results['average_continuous'] = float(np.mean(denormalize_predictions(predictions)))
        results['average_discrete'] = float(np.mean(discrete_scores))

        # 打印结果
        self._print_results(results)

        # 保存结果
        if Config.SAVE_DETAILED_RESULTS:
            self._save_results(results)

        return results

    def _print_results(self, results: dict):
        """打印分析结果"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("分析结果")
        self.logger.info("=" * 80)

        self.logger.info(f"\n评论: {results['comment']}")
        self.logger.info(f"时间: {results['timestamp']}")

        self.logger.info("\n五维度评分:")
        self.logger.info("-" * 100)
        self.logger.info(f"{'维度':<15} {'归一化[0-1]':<15} {'连续分数[-2,2]':<18} {'离散分数':<12} {'评价':<10}")
        self.logger.info("-" * 100)

        for dimension in Config.SCORE_DIMENSIONS:
            norm_score = results['normalized_scores'][dimension]
            cont_score = results['continuous_scores'][dimension]
            disc_score = results['discrete_scores'][dimension]
            label = results['score_labels'][dimension]

            self.logger.info(
                f"{dimension:<15} {norm_score:>13.4f}  "
                f"{cont_score:>16.4f}  "
                f"{disc_score:>10}    {label:<10}"
            )

        self.logger.info("-" * 100)
        self.logger.info(
            f"{'平均分':<15} {results['average_normalized']:>13.4f}  "
            f"{results['average_continuous']:>16.4f}  "
            f"{results['average_discrete']:>10.2f}"
        )
        self.logger.info("=" * 80)

    def _save_results(self, results: dict):
        """保存分析结果到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存JSON格式
        json_file = Config.RESULTS_DIR / f'sentiment_result_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        self.logger.info(f"\n✓ 结果已保存到: {json_file}")

        # 保存文本格式
        txt_file = Config.RESULTS_DIR / f'sentiment_result_{timestamp}.txt'
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("评论情感分析结果\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"评论内容: {results['comment']}\n")
            f.write(f"分析时间: {results['timestamp']}\n")
            f.write(f"使用模型: {results['model_info']['checkpoint_path']}\n")
            f.write(f"训练轮次: {results['model_info']['epoch']}\n")
            f.write(f"预训练模型: {results['model_info']['model_name']}\n\n")

            f.write("=" * 80 + "\n")
            f.write("五维度评分\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'维度':<15} {'归一化[0-1]':<15} {'连续分数[-2,2]':<18} {'离散分数':<12} {'评价':<10}\n")
            f.write("-" * 100 + "\n")

            for dimension in Config.SCORE_DIMENSIONS:
                norm_score = results['normalized_scores'][dimension]
                cont_score = results['continuous_scores'][dimension]
                disc_score = results['discrete_scores'][dimension]
                label = results['score_labels'][dimension]

                f.write(
                    f"{dimension:<15} {norm_score:>13.4f}  "
                    f"{cont_score:>16.4f}  "
                    f"{disc_score:>10}    {label:<10}\n"
                )

            f.write("-" * 100 + "\n")
            f.write(
                f"{'平均分':<15} {results['average_normalized']:>13.4f}  "
                f"{results['average_continuous']:>16.4f}  "
                f"{results['average_discrete']:>10.2f}\n"
            )
            f.write("=" * 80 + "\n")

        self.logger.info(f"✓ 文本报告已保存到: {txt_file}")


# ==================== 主函数 ====================
def main():
    """主函数"""

    print("=" * 80)
    print("情感分析模型 - 评论分析工具")
    print("=" * 80)
    print()

    # 创建分析器
    analyzer = SentimentAnalyzer()

    # 加载模型
    analyzer.load_model()

    print("\n" + "=" * 80)
    print("请输入评论内容（输入'quit'退出）:")
    print("=" * 80)

    # 交互式输入评论
    while True:
        print()
        comment = input(">>> ").strip()

        if not comment:
            print("评论不能为空，请重新输入")
            continue

        if comment.lower() in ['quit', 'exit', 'q']:
            print("\n感谢使用！再见！")
            break

        try:
            # 分析评论
            results = analyzer.analyze(comment)

            print("\n" + "-" * 80)
            print("是否继续分析？（输入新评论或'quit'退出）")
            print("-" * 80)

        except Exception as e:
            print(f"\n错误: {str(e)}")
            import traceback
            traceback.print_exc()
            print("请重新输入评论")


if __name__ == '__main__':
    main()