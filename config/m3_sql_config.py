#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块3：数据集构建与划分配置文件
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# ========================================
# 数据库配置 (新增：用于读取评论原文)
# ========================================
DATABASE_CONFIG = {
    'host': 'localhost',       # 数据库地址
    'port': 3306,              # 端口
    'user': 'root',            # 用户名
    'password': 'password', # 密码 (请修改这里)
    'db': 'yq_data',     # 数据库名 (请修改这里)
    'table': 'yq_clean_all',     # 表名 (请修改这里)
    'charset': 'utf8mb4'
}

# ========================================
# 输入文件配置
# ========================================
INPUT_FILES = {
    # 'cleaned_comments': 已移除，改为从数据库读取
    'sentiment_scores': PROJECT_ROOT / r'data/output/m2_sql_ai_results.json',  # 模块2的输出
}

# ========================================
# 输出文件配置
# ========================================
out_path = PROJECT_ROOT / r'data/output/dataset_v1'
OUTPUT_FILES = {
    'merged_dataset': out_path / r'dataset_v1.jsonl',
    'train_csv': out_path / r'train.csv',
    'val_csv': out_path / r'val.csv',
    'test_csv': out_path/ r'test.csv',
    'data_card': out_path / r'data_card.md',
    'archive': out_path / r'dataset_v1.zip',
}

# ========================================
# 情感维度配置
# ========================================
# 注意：这里需要与M2保持一致，M2中已修改为"消费"
DIMENSIONS = ['景色', '交通', '美食', '消费', '服务']

DIMENSION_MAPPING = {
    'scenery': '景色',
    'transportation': '交通',
    'food': '美食',
    'consumption': '消费', # 注意：Key已更新
    'service': '服务',
}

# ========================================
# 数据集划分配置
# ========================================
SPLIT_CONFIG = {
    'random_seed': 42,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'shuffle': True,
}

# ========================================
# 数据质量检查
# ========================================
QUALITY_CHECK = {
    'check_missing_values': True,
    'check_score_range': True,
    'valid_score_range': (-2, 2),
    'remove_invalid': True,
}

# ========================================
# 分布检查配置
# ========================================
DISTRIBUTION_CHECK = {
    'check_kl_divergence': True,
    'kl_threshold': 0.05,
    'smoothing': 1e-10,
}

# ========================================
# 压缩包配置
# ========================================
ARCHIVE_CONFIG = {
    'create_archive': True,
    'compression': 'deflated',
    'include_files': [
        'dataset_v1.jsonl',
        'train.csv',
        'val.csv',
        'test.csv',
        'data_card.md',
    ],
}

# ========================================
# 数据卡配置
# ========================================
DATA_CARD_CONFIG = {
    'dataset_name': 'Tourism Review Multi-dimensional Sentiment Dataset V1',
    'dataset_version': 'v1.0',
    'creation_date': True,
    'data_source': '旅游评论数据 (MySQL)',
    'processing_steps': [
        '模块1：文本清洗与标准化',
        '模块2：多维情感评分（GPT-4o-mini）',
        '模块3：数据库对齐、合并与划分',
    ],
    'known_biases': [
        '数据来源于爬虫采集，可能存在偏差',
        '情感评分由GPT模型生成，可能存在模型偏差',
    ],
    'intended_use': '旅游评论多维情感分析模型训练',
    'limitations': [
        '仅适用于中文旅游领域评论',
        '评分范围为-2到2，粒度有限',
    ],
}

# ========================================
# 日志配置
# ========================================
LOGGING = {
    'level': 'INFO',
    'verbose': True,
    'save_log': True,
    'log_file': 'logs/m3_dataset_build.log',
}

# ========================================
# 统计报告配置
# ========================================
STATS_REPORT = {
    'print_summary': True,
    'print_distribution': True,
    'print_kl_divergence': True,
    'decimal_places': 3,
}

# ========================================
# 辅助函数：获取完整配置
# ========================================
def get_config():
    """获取完整配置字典"""
    return {
        'database': DATABASE_CONFIG, # 新增数据库配置
        'input_files': INPUT_FILES,
        'output_files': OUTPUT_FILES,
        'dimensions': DIMENSIONS,
        'dimension_mapping': DIMENSION_MAPPING,
        'split_config': SPLIT_CONFIG,
        'quality_check': QUALITY_CHECK,
        'distribution_check': DISTRIBUTION_CHECK,
        'archive_config': ARCHIVE_CONFIG,
        'data_card_config': DATA_CARD_CONFIG,
        'logging': LOGGING,
        'stats_report': STATS_REPORT,
    }

# ========================================
# 验证配置
# ========================================
def validate_config():
    split_sum = (SPLIT_CONFIG['train_ratio'] + SPLIT_CONFIG['val_ratio'] + SPLIT_CONFIG['test_ratio'])
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和必须为1.0，当前为{split_sum}")
    if len(DIMENSIONS) == 0:
        raise ValueError("DIMENSIONS不能为空")
    if len(DIMENSION_MAPPING) != len(DIMENSIONS):
        raise ValueError("DIMENSION_MAPPING的维度数量与DIMENSIONS不一致")
    return True

validate_config()