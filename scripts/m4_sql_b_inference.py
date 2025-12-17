#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M4-b 全量推理脚本 (修复版)
功能：加载分体式保存权重的模型，对数据库评论进行情感分析。
修复：DataLoader collate 函数处理原始数据字段的问题
"""

import sys
import logging
import warnings
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime, date

import pymysql
from pymysql.cursors import DictCursor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 导入配置
sys.path.append(str(Path(__file__).parent.parent))
from config.m4_sql_b_config import M4bConfig as Config

warnings.filterwarnings('ignore')


# ==========================================
# 1. 模型定义
# ==========================================
class SentimentRegressor(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        predictions = self.regressor(pooled_output)
        return predictions


# ==========================================
# 2. 数据集定义
# ==========================================
class InferenceDataset(Dataset):
    """推理数据集"""

    def __init__(self, comments: List[Dict], tokenizer, max_length):
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.content_field = Config.DATA_SOURCE_CONFIG['content_field']

        # 定义需要处理为数字的字段
        self.numeric_fields = {
            'isvalid', 'source_type', 'highly_relevant',
            'negative_probability', 'sentiment'
        }

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        item = self.comments[idx]

        # === 数据清洗与类型转换 ===
        safe_item = {}
        for k, v in item.items():
            if v is None:
                if k in self.numeric_fields:
                    safe_item[k] = 0
                else:
                    safe_item[k] = ""
            elif isinstance(v, (datetime, date)):
                safe_item[k] = v.strftime('%Y-%m-%d %H:%M:%S')
            else:
                safe_item[k] = v

        text = str(safe_item.get(self.content_field, ""))

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'original_data': safe_item  # 保持字典格式
        }


# ==========================================
# 3. 自定义 Collate 函数
# ==========================================
def custom_collate_fn(batch):
    """
    自定义批处理函数
    - 模型输入字段（tensor）正常堆叠
    - 原始数据字段保持为列表
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    original_data = [item['original_data'] for item in batch]  # 保持为列表

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'original_data': original_data
    }


# ==========================================
# 4. 推理引擎核心类
# ==========================================
class InferenceEngine:
    def __init__(self):
        self._setup_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

        self.stats = {
            'total_count': 0,
            'success_count': 0,
            'start_time': None,
            'end_time': None
        }
        self.logger.info(f"初始化推理引擎，使用设备: {self.device}")

    def _setup_logger(self):
        Config.create_dirs()
        self.logger = logging.getLogger('InferenceEngine')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        fh = logging.FileHandler(Config.LOG_FILE, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        self.logger.addHandler(ch)

    def load_model(self):
        """加载模型权重"""
        self.logger.info("正在加载模型和分词器...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
            self.model = SentimentRegressor(Config.MODEL_NAME, Config.NUM_LABELS)

            if not Config.BEST_MODEL_PATH.exists():
                raise FileNotFoundError(f"模型权重文件不存在: {Config.BEST_MODEL_PATH}")

            checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)

            if 'encoder_state_dict' in checkpoint and 'regressor_state_dict' in checkpoint:
                self.logger.info("检测到分体式权重，正在加载...")
                self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                self.model.regressor.load_state_dict(checkpoint['regressor_state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.logger.warning("尝试直接加载 checkpoint...")
                self.model.load_state_dict(checkpoint, strict=False)

            self.model.to(self.device)
            self.model.eval()
            self.logger.info("✓ 模型权重加载成功")
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise

    def _fetch_data_from_db(self) -> List[Dict]:
        """从数据库读取"""
        db_conf = Config.DATABASE_CONFIG
        source_conf = Config.DATA_SOURCE_CONFIG

        self.logger.info(f"正在连接数据库: {db_conf['host']}")
        try:
            conn = pymysql.connect(
                host=db_conf['host'],
                port=db_conf['port'],
                user=db_conf['user'],
                password=db_conf['password'],
                database=db_conf['db'],
                charset=db_conf['charset'],
                cursorclass=DictCursor
            )
            with conn:
                with conn.cursor() as cursor:
                    fields_str = ', '.join([f"`{f}`" for f in source_conf['all_fields']])
                    table = source_conf['input_table']
                    sql = f"SELECT {fields_str} FROM `{table}`"
                    if source_conf.get('filter_condition'):
                        sql += f" WHERE {source_conf['filter_condition']}"

                    self.logger.info(f"执行SQL查询...")
                    cursor.execute(sql)
                    data = list(cursor.fetchall())
                    self.logger.info(f"成功读取 {len(data)} 条数据")
                    return data
        except pymysql.MySQLError as e:
            self.logger.error(f"数据库读取异常: {e}")
            raise

    def batch_inference(self, comments: List[Dict]) -> List[Dict]:
        """执行批量推理"""
        dataset = InferenceDataset(comments, self.tokenizer, Config.MAX_LENGTH)
        num_workers = 0 if sys.platform == 'win32' else 4

        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn  # ← 关键：使用自定义 collate 函数
        )

        results = []
        self.logger.info(f"开始推理 {len(comments)} 条数据...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="推理中"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)

                outputs = self.model(input_ids, attention_mask, token_type_ids)
                predictions = outputs.cpu().numpy()

                original_data_list = batch['original_data']  # 已经是列表

                for i, item in enumerate(original_data_list):
                    raw_preds = predictions[i]
                    scores = {}
                    for idx, dim in enumerate(Config.DIMENSIONS):
                        score_int = int(round(raw_preds[idx]))
                        score_int = max(min(score_int, 2), -2)
                        scores[dim] = score_int

                    item['scores'] = scores
                    results.append(item)

        self.stats['success_count'] = len(results)
        self.logger.info(f"推理完成，共处理 {len(results)} 条数据")
        return results

    def _create_output_table(self, cursor, original_fields, score_fields):
        """创建结果表"""
        defs = []
        for f in original_fields:
            if f == 'id':
                defs.append(f"`{f}` VARCHAR(100) PRIMARY KEY")
            elif f in ['release_date', 'date']:
                defs.append(f"`{f}` DATETIME")
            elif f in ['isvalid', 'source_type', 'highly_relevant']:
                defs.append(f"`{f}` INT")
            elif f in ['negative_probability', 'sentiment']:
                defs.append(f"`{f}` FLOAT")
            elif f in ['format_content', 'scores_comment', 'keywords', 'feature_words']:
                defs.append(f"`{f}` LONGTEXT")
            else:
                defs.append(f"`{f}` TEXT")

        for f in score_fields:
            defs.append(f"`{f}` INT DEFAULT 0")  # 五维情感值为整数 -2~2

        table = Config.DATA_OUTPUT_CONFIG['output_table']
        sql = f"CREATE TABLE IF NOT EXISTS `{table}` ({', '.join(defs)}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
        cursor.execute(sql)
        self.logger.info(f"表 `{table}` 已准备就绪")

    def save_results(self, results: List[Dict]):
        """保存入库"""
        if not results:
            self.logger.warning("没有结果需要保存")
            return

        db_conf = Config.DATABASE_CONFIG
        out_conf = Config.DATA_OUTPUT_CONFIG
        orig_fields = Config.DATA_SOURCE_CONFIG['all_fields']
        score_fields = Config.SENTIMENT_SCORE_FIELDS

        self.logger.info(f"正在保存 {len(results)} 条结果到数据库...")

        try:
            conn = pymysql.connect(
                host=db_conf['host'],
                port=db_conf['port'],
                user=db_conf['user'],
                password=db_conf['password'],
                database=db_conf['db'],
                charset=db_conf['charset']
            )
            with conn:
                with conn.cursor() as cursor:
                    if out_conf['create_table_if_not_exists']:
                        self._create_output_table(cursor, orig_fields, score_fields)

                    cols = orig_fields + score_fields
                    placeholders = ', '.join(['%s'] * len(cols))
                    col_names = ','.join([f'`{c}`' for c in cols])
                    sql = f"REPLACE INTO `{out_conf['output_table']}` ({col_names}) VALUES ({placeholders})"

                    batch_data = []
                    batch_size = out_conf['batch_size']

                    for item in tqdm(results, desc="写入数据库"):
                        row = []

                        # 原始字段
                        for f in orig_fields:
                            val = item.get(f)
                            if hasattr(val, 'item'):
                                val = val.item()
                            row.append(val)

                        # 五维情感评分
                        scores = item.get('scores', {})
                        for dim in Config.DIMENSIONS:
                            row.append(scores.get(dim, 0))

                        batch_data.append(tuple(row))

                        if len(batch_data) >= batch_size:
                            cursor.executemany(sql, batch_data)
                            conn.commit()
                            batch_data = []

                    if batch_data:
                        cursor.executemany(sql, batch_data)
                        conn.commit()

            self.logger.info("✓ 数据保存完成")
        except Exception as e:
            self.logger.error(f"保存失败: {e}", exc_info=True)
            raise

    def run(self):
        self.stats['start_time'] = time.time()
        try:
            Config.print_config()
            self.load_model()

            raw_data = self._fetch_data_from_db()
            if not raw_data:
                self.logger.error("数据库无数据，退出")
                return

            if Config.DATA_SOURCE_CONFIG['content_field'] not in raw_data[0]:
                self.logger.error(f"数据缺少内容字段: {Config.DATA_SOURCE_CONFIG['content_field']}")
                return

            self.stats['total_count'] = len(raw_data)
            results = self.batch_inference(raw_data)
            self.save_results(results)

            self.logger.info(f"✓ 任务完成！共处理 {self.stats['total_count']} 条，成功 {self.stats['success_count']} 条")

        except Exception as e:
            self.logger.error(f"主流程异常: {e}", exc_info=True)
        finally:
            dur = time.time() - self.stats['start_time']
            self.logger.info(f"总耗时: {dur:.2f}s ({dur / 60:.2f}分钟)")


if __name__ == "__main__":
    engine = InferenceEngine()
    engine.run()