"""
M4-b 全量推理脚本 (日期修复版 v2)
功能：使用M4训练好的模型对M1清洗后的全部评论进行批量情感分析
输出：带5维情感分数、日期和年月日字段的JSONL文件
"""

import sys
import json
import logging
import warnings
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 导入配置
sys.path.append(str(Path(__file__).parent.parent))
from config.m4_b_config import M4bConfig as Config

warnings.filterwarnings('ignore')


# ==================== 模型定义 ====================
class SentimentRegressor(nn.Module):
    """多维情感回归模型（与M4训练时完全一致）"""

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


# ==================== 自定义collate函数 ====================
def custom_collate_fn(batch):
    """
    自定义collate函数，正确处理字符串字段
    """
    # 分离tensor和非tensor字段
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    # 字符串字段保持为列表
    ids = [item['id'] for item in batch]
    texts = [item['text'] for item in batch]
    publish_times = [item['publish_time'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'id': ids,
        'text': texts,
        'publish_time': publish_times
    }


# ==================== 数据集定义 ====================
class InferenceDataset(Dataset):
    """推理数据集"""

    def __init__(self, comments: List[Dict], tokenizer, max_length: int):
        """
        Args:
            comments: 评论列表，每个元素是包含id、text、metadata的字典
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = self.comments[idx]
        text = str(comment.get('text', ''))

        # 从metadata中提取publish_time
        metadata = comment.get('metadata', {})
        publish_time = metadata.get('publish_time', '')

        # Tokenize
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
            'id': comment.get('id', ''),
            'text': text,
            'publish_time': publish_time  # 字符串类型
        }


# ==================== 工具函数 ====================
def denormalize_and_discretize(predictions: np.ndarray) -> np.ndarray:
    """
    将[0,1]区间的预测值反归一化并离散化到[-2,2]的整数

    Args:
        predictions: [batch_size, num_labels] 范围[0,1]的浮点数

    Returns:
        离散化后的整数分数，范围[-2, -1, 0, 1, 2]
    """
    # 线性映射: [0,1] -> [-2,2]
    denormalized = predictions * 4 - 2

    # 四舍五入到整数
    discretized = np.round(denormalized).astype(np.int32)

    # 裁剪到有效范围
    discretized = np.clip(discretized, Config.MIN_SCORE, Config.MAX_SCORE)

    return discretized


def format_memory(bytes_val: int) -> str:
    """格式化内存大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def get_gpu_memory() -> Tuple[float, float]:
    """
    获取GPU显存使用情况

    Returns:
        (当前使用的显存GB, 峰值显存GB)
    """
    if torch.cuda.is_available():
        current = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return current, peak
    return 0.0, 0.0


def parse_publish_time(publish_time: str) -> Tuple[str, int, int, int]:
    """
    解析publish_time字段

    Args:
        publish_time: 时间字符串，例如 "2021-06-15 00:00:00"

    Returns:
        (日期字符串 YYYY-MM-DD, 年, 月, 日)
        如果解析失败返回 ('', 0, 0, 0)
    """
    if not publish_time or not isinstance(publish_time, str):
        return '', 0, 0, 0

    publish_time = publish_time.strip()

    if not publish_time:
        return '', 0, 0, 0

    try:
        dt = None

        # 格式1: "2021-06-15 00:00:00" (最常见)
        if ' ' in publish_time:
            try:
                dt = datetime.strptime(publish_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
            except:
                pass

        # 格式2: "2021-06-15"
        if not dt and len(publish_time) >= 10:
            try:
                dt = datetime.strptime(publish_time[:10], '%Y-%m-%d')
            except:
                pass

        # 格式3: "2021/06/15"
        if not dt and len(publish_time) >= 10:
            try:
                dt = datetime.strptime(publish_time[:10], '%Y/%m/%d')
            except:
                pass

        # 如果成功解析
        if dt:
            date_str = dt.strftime('%Y-%m-%d')
            return date_str, dt.year, dt.month, dt.day

        # 最后尝试：直接解析前10个字符
        if len(publish_time) >= 10:
            date_part = publish_time[:10]
            if date_part[4] in ['-', '/'] and date_part[7] in ['-', '/']:
                parts = date_part.replace('/', '-').split('-')
                if len(parts) == 3:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        # 验证日期有效性
                        datetime(year, month, day)
                        return date_part.replace('/', '-'), year, month, day
                    except:
                        pass

        return '', 0, 0, 0

    except Exception:
        return '', 0, 0, 0


# ==================== 推理器 ====================
class SentimentInference:
    """情感分析批量推理器"""

    def __init__(self):
        self.config = Config
        self._setup_logging()
        self._set_seed()
        self._setup_device()
        Config.create_dirs()

        self.model = None
        self.tokenizer = None

        # 统计信息
        self.stats = {
            'total_count': 0,
            'success_count': 0,
            'error_count': 0,
            'date_parse_success': 0,
            'date_parse_fail': 0,
            'start_time': None,
            'end_time': None,
            'peak_memory_gb': 0.0
        }

    def _setup_logging(self):
        """设置日志"""
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        file_handler = logging.FileHandler(Config.LOG_FILE, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))

        handlers = [file_handler]
        if Config.LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(log_format, date_format))
            handlers.append(console_handler)

        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format=log_format,
            datefmt=date_format,
            handlers=handlers,
            force=True
        )

        self.logger = logging.getLogger(__name__)

    def _set_seed(self):
        """设置随机种子"""
        random.seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        torch.manual_seed(Config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.RANDOM_SEED)
        self.logger.info(f"✓ 随机种子: {Config.RANDOM_SEED}")

    def _setup_device(self):
        """设置计算设备"""
        if Config.USE_GPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"✓ 使用设备: CUDA ({gpu_name}, {total_memory:.1f} GB)")
        else:
            self.device = torch.device("cpu")
            self.logger.info("✓ 使用设备: CPU")

    def load_model(self):
        """加载训练好的模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("加载模型")
        self.logger.info("=" * 80)

        if not Config.BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"模型文件不存在: {Config.BEST_MODEL_PATH}")

        self.logger.info(f"✓ 模型路径: {Config.BEST_MODEL_PATH}")

        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        self.logger.info(f"✓ Checkpoint包含的keys: {list(checkpoint.keys())}")

        model_name = checkpoint.get('model_name', Config.MODEL_NAME)
        num_labels = checkpoint.get('num_labels', Config.NUM_LABELS)
        dropout = checkpoint.get('dropout', Config.HIDDEN_DROPOUT)

        self.logger.info(f"✓ 预训练模型: {model_name}")
        self.logger.info(f"✓ 输出维度: {num_labels}")
        self.logger.info(f"✓ 训练轮次: {checkpoint.get('epoch', 'Unknown')}")

        if 'metrics' in checkpoint and checkpoint['metrics']:
            metrics = checkpoint['metrics']
            self.logger.info("✓ 训练时验证集性能:")
            self.logger.info(f"  - 平均 Pearsonr: {metrics.get('avg_pearsonr', 'N/A'):.4f}")
            self.logger.info(f"  - 平均 Accuracy: {metrics.get('avg_accuracy', 'N/A'):.4f}")
            self.logger.info(f"  - 平均 RMSE: {metrics.get('avg_rmse', 'N/A'):.4f}")

        model = SentimentRegressor(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout
        ).to(self.device)

        try:
            if 'encoder_state_dict' in checkpoint and 'regressor_state_dict' in checkpoint:
                model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                model.regressor.load_state_dict(checkpoint['regressor_state_dict'])
                self.logger.info("✓ 使用encoder_state_dict和regressor_state_dict加载")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info("✓ 使用model_state_dict加载")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                self.logger.info("✓ 使用state_dict加载")
            else:
                model.load_state_dict(checkpoint)
                self.logger.info("✓ 直接加载checkpoint")
        except Exception as e:
            self.logger.error(f"权重加载失败: {str(e)}")
            raise

        model.eval()
        self.model = model
        self.logger.info("✓ 模型权重加载成功")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger.info("✓ Tokenizer加载成功")

        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"✓ 总参数量: {total_params:,}")

    def load_comments(self) -> List[Dict]:
        """加载评论数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("加载评论数据")
        self.logger.info("=" * 80)

        if not Config.INPUT_FILE.exists():
            raise FileNotFoundError(f"输入文件不存在: {Config.INPUT_FILE}")

        comments = []
        error_count = 0

        self.logger.info(f"✓ 读取文件: {Config.INPUT_FILE}")

        with open(Config.INPUT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    comment = json.loads(line)

                    if 'id' not in comment or 'text' not in comment:
                        self.logger.warning(f"行 {line_num}: 缺少必需字段 (id 或 text)")
                        error_count += 1
                        continue

                    comments.append(comment)

                except json.JSONDecodeError as e:
                    self.logger.warning(f"行 {line_num}: JSON解析失败 - {e}")
                    error_count += 1
                    continue

        self.logger.info(f"✓ 成功加载: {len(comments):,} 条评论")
        if error_count > 0:
            self.logger.warning(f"⚠ 跳过错误行: {error_count} 条")

        # 显示第一条数据的示例
        if len(comments) > 0:
            sample = comments[0]
            self.logger.info(f"\n数据样例:")
            self.logger.info(f"  ID: {sample.get('id', 'N/A')}")
            self.logger.info(f"  文本: {sample.get('text', 'N/A')[:50]}...")
            if 'metadata' in sample and 'publish_time' in sample['metadata']:
                self.logger.info(f"  发布时间: {sample['metadata']['publish_time']}")

        return comments

    @torch.no_grad()
    def batch_inference(self, comments: List[Dict]) -> List[Dict]:
        """
        批量推理

        Args:
            comments: 评论列表

        Returns:
            推理结果列表
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("开始批量推理")
        self.logger.info("=" * 80)

        dataset = InferenceDataset(comments, self.tokenizer, Config.MAX_LENGTH)

        num_workers = 0 if sys.platform == 'win32' else Config.NUM_WORKERS

        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if Config.USE_GPU else False,
            collate_fn=custom_collate_fn  # 使用自定义collate函数
        )

        self.logger.info(f"✓ 总评论数: {len(comments):,}")
        self.logger.info(f"✓ 批次大小: {Config.BATCH_SIZE}")
        self.logger.info(f"✓ 总批次数: {len(dataloader):,}")
        self.logger.info(f"✓ 工作线程: {num_workers}")

        results = []
        self.model.eval()

        self.stats['start_time'] = time.time()
        self.stats['total_count'] = len(comments)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(
            dataloader,
            desc="推理进度",
            disable=not Config.SHOW_PROGRESS,
            ncols=100,
            file=sys.stdout
        )

        for batch_idx, batch in enumerate(pbar):
            try:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                predictions = self.model(input_ids, attention_mask)
                predictions = predictions.cpu().numpy()

                discrete_scores = denormalize_and_discretize(predictions)

                # batch中的字符串字段已经是列表了
                batch_size = predictions.shape[0]
                for i in range(batch_size):
                    # 获取publish_time（已经是字符串）
                    publish_time = batch['publish_time'][i]

                    # 解析日期
                    date_str, year, month, day = parse_publish_time(publish_time)

                    # 统计日期解析成功率
                    if date_str:
                        self.stats['date_parse_success'] += 1
                    else:
                        self.stats['date_parse_fail'] += 1
                        # 调试信息：记录解析失败的publish_time
                        if self.stats['date_parse_fail'] <= 5:  # 只记录前5个
                            self.logger.warning(f"日期解析失败示例: '{publish_time}'")

                    # 构建结果字典
                    result = {
                        'id': batch['id'][i],
                        'text': batch['text'][i],
                        'date': date_str,
                        'year': year,
                        'month': month,
                        'day': day
                    }

                    # 添加5维情感分数
                    for dim_idx, dim_name in enumerate(Config.DIMENSIONS):
                        result[dim_name] = int(discrete_scores[i, dim_idx])

                    # 如果需要保存原始分数
                    if Config.SAVE_RAW_SCORES:
                        result['raw_scores'] = {}
                        for dim_idx, dim_name in enumerate(Config.DIMENSIONS):
                            result['raw_scores'][dim_name] = float(predictions[i, dim_idx])

                    results.append(result)
                    self.stats['success_count'] += 1

                if torch.cuda.is_available():
                    _, peak_mem = get_gpu_memory()
                    self.stats['peak_memory_gb'] = max(self.stats['peak_memory_gb'], peak_mem)

                if Config.SHOW_PROGRESS:
                    current_mem, peak_mem = get_gpu_memory()
                    pbar.set_postfix({
                        '成功': self.stats['success_count'],
                        '日期解析': f"{self.stats['date_parse_success']}/{self.stats['success_count']}",
                        '显存': f'{current_mem:.2f}GB'
                    })

            except Exception as e:
                self.logger.error(f"批次 {batch_idx} 推理失败: {str(e)}")
                self.logger.error(f"详细错误信息:", exc_info=True)

                batch_size = len(batch['id']) if 'id' in batch else Config.BATCH_SIZE
                self.stats['error_count'] += batch_size

                if not Config.CONTINUE_ON_ERROR:
                    raise

                with open(Config.ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"批次 {batch_idx} 错误: {str(e)}\n")
                    f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    if 'id' in batch:
                        f.write(f"受影响的ID: {batch['id']}\n")

        self.stats['end_time'] = time.time()

        self.logger.info(f"\n✓ 推理完成: {len(results):,} 条")
        self.logger.info(f"✓ 日期解析成功: {self.stats['date_parse_success']:,} 条 ({self.stats['date_parse_success']/len(results)*100:.1f}%)")
        if self.stats['date_parse_fail'] > 0:
            self.logger.warning(f"⚠ 日期解析失败: {self.stats['date_parse_fail']} 条 ({self.stats['date_parse_fail']/len(results)*100:.1f}%)")
        if self.stats['error_count'] > 0:
            self.logger.warning(f"⚠ 推理失败: {self.stats['error_count']} 条")
            self.logger.warning(f"⚠ 详细错误请查看: {Config.ERROR_LOG_FILE}")

        return results

    def save_results(self, results: List[Dict]):
        """保存推理结果"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("保存结果")
        self.logger.info("=" * 80)

        with open(Config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        self.logger.info(f"✓ 结果已保存: {Config.OUTPUT_FILE}")
        self.logger.info(f"  总条数: {len(results):,}")

        # 统计日期分布
        date_counts = defaultdict(int)
        year_counts = defaultdict(int)
        for result in results:
            if result['date']:
                date_counts[result['date']] += 1
            if result['year']:
                year_counts[result['year']] += 1

        if date_counts:
            self.logger.info(f"  日期范围: {min(date_counts.keys())} 至 {max(date_counts.keys())}")
            self.logger.info(f"  不同日期数: {len(date_counts)}")

        if year_counts:
            self.logger.info(f"  年份分布: {dict(sorted(year_counts.items()))}")

    def save_log(self):
        """保存推理日志"""
        total_time = self.stats['end_time'] - self.stats['start_time']
        avg_speed = self.stats['success_count'] / total_time if total_time > 0 else 0

        log_content = f"""
{'=' * 80}
M4-b 全量推理日志
{'=' * 80}

【推理时间】
开始时间: {datetime.fromtimestamp(self.stats['start_time']).strftime('%Y-%m-%d %H:%M:%S')}
结束时间: {datetime.fromtimestamp(self.stats['end_time']).strftime('%Y-%m-%d %H:%M:%S')}
总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)

【推理统计】
总条数: {self.stats['total_count']:,}
成功条数: {self.stats['success_count']:,}
失败条数: {self.stats['error_count']:,}
成功率: {self.stats['success_count']/self.stats['total_count']*100:.2f}%

【日期解析】
成功解析: {self.stats['date_parse_success']:,}
解析失败: {self.stats['date_parse_fail']:,}
解析成功率: {self.stats['date_parse_success']/self.stats['total_count']*100:.2f}%

【推理速度】
平均速度: {avg_speed:.2f} 条/秒
平均速度: {avg_speed*60:.2f} 条/分钟

【显存使用】
峰值显存: {self.stats['peak_memory_gb']:.2f} GB

【配置信息】
模型路径: {Config.BEST_MODEL_PATH}
预训练模型: {Config.MODEL_NAME}
批次大小: {Config.BATCH_SIZE}
最大长度: {Config.MAX_LENGTH}
工作线程: {'0 (Windows)' if sys.platform == 'win32' else Config.NUM_WORKERS}
使用设备: {'GPU' if Config.USE_GPU and torch.cuda.is_available() else 'CPU'}

【输出文件】
结果文件: {Config.OUTPUT_FILE}
日志文件: {Config.LOG_FILE}

{'=' * 80}
"""

        with open(Config.LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_content)

        print(log_content)

    def display_samples(self, results: List[Dict], count: int = None):
        """随机显示样本结果"""
        if count is None:
            count = Config.SAMPLE_DISPLAY_COUNT

        if len(results) == 0:
            self.logger.warning("没有结果可显示")
            return

        sample_count = min(count, len(results))
        samples = random.sample(results, sample_count)

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"随机抽样展示（共 {sample_count} 条）")
        self.logger.info("=" * 80)

        for idx, sample in enumerate(samples, 1):
            self.logger.info(f"\n【样本 {idx}】")
            self.logger.info(f"ID: {sample['id']}")
            self.logger.info(f"日期: {sample['date']} (年:{sample['year']} 月:{sample['month']} 日:{sample['day']})")
            self.logger.info(f"评论: {sample['text'][:100]}...")

            score_str = "情感分数: "
            for dim in Config.DIMENSIONS:
                score = sample[dim]
                score_str += f"{dim}={score:+2d}  "
            self.logger.info(score_str)

            if Config.SAVE_RAW_SCORES and 'raw_scores' in sample:
                raw_str = "原始分数: "
                for dim in Config.DIMENSIONS:
                    raw_score = sample['raw_scores'][dim]
                    raw_str += f"{dim}={raw_score:.3f}  "
                self.logger.info(raw_str)

        self.logger.info("\n" + "=" * 80)

    def run(self):
        """执行完整推理流程"""
        try:
            Config.print_config()
            self.load_model()
            comments = self.load_comments()

            if len(comments) == 0:
                self.logger.error("没有评论数据，退出")
                return

            results = self.batch_inference(comments)

            if len(results) > 0:
                self.save_results(results)
                self.save_log()
                self.display_samples(results)
            else:
                self.logger.error("没有成功的推理结果")

            self.logger.info("\n" + "=" * 80)
            if self.stats['success_count'] > 0:
                self.logger.info("全量推理完成！")
            else:
                self.logger.error("全量推理失败！请检查错误日志")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"\n推理失败: {str(e)}", exc_info=True)
            raise


def main():
    """主函数"""
    try:
        inference = SentimentInference()
        inference.run()
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n程序异常退出: {str(e)}")
        raise


if __name__ == "__main__":
    main()