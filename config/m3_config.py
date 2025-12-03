#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块3：数据集构建与划分配置文件
"""
from pathlib import Path
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
# ========================================
# 输入文件配置
# ========================================
INPUT_FILES = {
    'cleaned_comments': r'data\输出\m1_cleaned_comments.jsonl',  # 模块1的输出
    'sentiment_scores': r'data\输出\m2_sentiment_scores.json',  # 模块2的输出
}

# ========================================
# 输出文件配置
# ========================================
OUTPUT_FILES = {
    'merged_dataset': r'data\输出\dataset_v1/dataset_v1.jsonl',  # 合并后的完整数据集
    'train_csv': r'data\输出\dataset_v1/train.csv',
    'val_csv': r'data\输出\dataset_v1/val.csv',
    'test_csv': r'data\输出\dataset_v1/test.csv',
    'data_card': r'data\输出\data_card.md',
    'archive': r'data\输出\dataset_v1.zip',  # 压缩包
}

# ========================================
# 情感维度配置（必须与模块2一致）
# ========================================
DIMENSIONS = ['景色', '交通', '美食', '门票', '服务']

# 如果模块2用的是英文key，需要映射到中文
DIMENSION_MAPPING = {
    'scenery': '景色',
    'transportation': '交通',
    'food': '美食',
    'ticket': '门票',
    'service': '服务',
}

# ========================================
# 数据集划分配置
# ========================================
SPLIT_CONFIG = {
    'random_seed': 42,  # 随机种子，保证可复现
    'train_ratio': 0.8,  # 训练集比例
    'val_ratio': 0.1,  # 验证集比例
    'test_ratio': 0.1,  # 测试集比例
    'shuffle': True,  # 是否打乱数据
}

# ========================================
# 数据质量检查
# ========================================
QUALITY_CHECK = {
    'check_missing_values': True,  # 检查缺失值
    'check_score_range': True,  # 检查分数范围是否合法
    'valid_score_range': (-2, 2),  # 合法分数范围
    'remove_invalid': True,  # 是否移除不合法的数据
}

# ========================================
# 分布检查配置
# ========================================
DISTRIBUTION_CHECK = {
    'check_kl_divergence': True,  # 是否检查KL散度
    'kl_threshold': 0.05,  # KL散度阈值（<0.05算合格）
    'smoothing': 1e-10,  # 平滑参数（避免除零）
}

# ========================================
# 压缩包配置
# ========================================
ARCHIVE_CONFIG = {
    'create_archive': True,  # 是否创建压缩包
    'compression': 'deflated',  # 压缩方法：stored, deflated, bzip2, lzma
    'include_files': [  # 要打包的文件
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
    'creation_date': None,  # 自动填充为当前日期
    'data_source': '携程旅游评论数据',
    'processing_steps': [
        '模块1：文本清洗与标准化',
        '模块2：多维情感评分（GPT-4o-mini）',
        '模块3：数据对齐、合并与划分',
    ],
    'known_biases': [
        '数据来源于携程平台，可能存在平台偏差',
        '情感评分由GPT模型生成，可能存在模型偏差',
        '评论可能更多来自对景点有强烈情感的用户',
    ],
    'intended_use': '旅游评论多维情感分析模型训练',
    'limitations': [
        '仅适用于中文旅游领域评论',
        '情感维度固定为5维，可能无法完全覆盖所有情感方面',
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
    'print_summary': True,  # 是否打印统计摘要
    'print_distribution': True,  # 是否打印分布信息
    'print_kl_divergence': True,  # 是否打印KL散度
    'decimal_places': 3,  # 小数位数
}


# ========================================
# 辅助函数：获取完整配置
# ========================================
def get_config():
    """获取完整配置字典"""
    return {
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
    """验证配置的合法性"""
    # 检查划分比例是否为1
    split_sum = (SPLIT_CONFIG['train_ratio'] +
                 SPLIT_CONFIG['val_ratio'] +
                 SPLIT_CONFIG['test_ratio'])

    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(
            f"划分比例之和必须为1.0，当前为{split_sum}。"
            f"train={SPLIT_CONFIG['train_ratio']}, "
            f"val={SPLIT_CONFIG['val_ratio']}, "
            f"test={SPLIT_CONFIG['test_ratio']}"
        )

    # 检查维度数量
    if len(DIMENSIONS) == 0:
        raise ValueError("DIMENSIONS不能为空")

    # 检查映射是否完整
    if len(DIMENSION_MAPPING) != len(DIMENSIONS):
        raise ValueError("DIMENSION_MAPPING的维度数量与DIMENSIONS不一致")

    return True


# 配置加载时自动验证
validate_config()