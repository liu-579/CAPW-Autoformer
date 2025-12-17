#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块2：多维情感评分配置文件
"""
from pathlib import Path

# ========================================
# OpenAI API 配置
# ========================================
OPENAI_CONFIG = {
    'api_key': 'BhjZ7X6tdfoU6XrgPWj3Ttq0uPkkxz0HmTx3TG8ehqo',  # 请替换为你的 Key
    'base_url': "https://api.poe.com/v1",
    'model': 'deepseek-v3.2',
    'temperature': 0.0,
    'max_tokens': 200,
}

# ========================================
# 数据库配置 (新增)
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
# 输入输出文件配置
# ========================================
PROJECT_ROOT = Path(__file__).parent.parent
# 输入文件配置已被数据库替代，这里保留输出路径
OUTPUT_FILE = PROJECT_ROOT / r'data\output\m2_sql_ai_results.json'

# ========================================
# 情感评分维度配置 (已修改：门票 -> 消费)
# ========================================
DIMENSIONS = [
    {
        'name': '景色',
        'key': 'scenery',
        'description': '景点的自然风光、建筑美感、视觉体验'
    },
    {
        'name': '交通',
        'key': 'transportation',
        'description': '到达景点的交通便利性、路况、停车等'
    },
    {
        'name': '美食',
        'key': 'food',
        'description': '景区内外的餐饮质量、价格、特色美食'
    },
    {
        'name': '消费',  # 修改点：名称变更
        'key': 'consumption', # 修改点：Key变更
        'description': '门票价格、景区内物价及整体性价比' # 修改点：描述变更
    },
    {
        'name': '服务',
        'key': 'service',
        'description': '景区工作人员态度、设施维护、游客服务'
    },
]

# ========================================
# 评分标准
# ========================================
SCORE_LEVELS = {
    -2: '极差',
    -1: '差',
    0: '中性或未提及',
    1: '好',
    2: '极好',
}

# ========================================
# 系统提示词
# ========================================
SYSTEM_PROMPT = """你是一个专业的旅游评论情感分析专家。你的任务是分析游客对景点的评论，并对多个维度进行情感打分。

评分标准：
-2（极差）：明确表达强烈不满、严重问题、极度失望
-1（差）：表达不满、有明显问题、体验不佳
 0（中性或未提及）：没有提及该维度，或者评价中性、无明显倾向
 1（好）：表达满意、体验良好、有正面评价
 2（极好）：表达强烈满意、印象深刻、极力推荐

注意事项：
1. 如果评论中完全没有提到某个维度，该维度打0分
2. 要基于评论的实际内容，不要过度推测
3. 注意识别反讽和委婉表达
4. 返回必须是严格的JSON格式，不要有任何额外文字"""

# ========================================
# 用户提示词模板 (已同步更新字段名)
# ========================================
USER_PROMPT_TEMPLATE = """请分析以下游客评论，并对每个维度进行情感打分。

评论内容：
{review}

需要评分的维度：
{dimensions}

请以JSON格式返回结果，格式如下：
{{
  "scenery": 0,
  "transportation": 0,
  "food": 0,
  "consumption": 0,
  "service": 0
}}

只返回JSON，不要有任何其他文字说明。"""

# ========================================
# 并发控制
# ========================================
CONCURRENCY = {
    'max_workers': 10,
    'rate_limit': 50,
    'batch_size': 100,
}

# ========================================
# 重试配置
# ========================================
RETRY = {
    'max_attempts': 3,
    'initial_delay': 1,
    'max_delay': 10,
    'exponential_base': 2,
}

# ========================================
# 失败处理
# ========================================
FAILURE_HANDLING = {
    'default_score': 0,
    'save_failed_records': True,
    'failed_records_file': 'data/sentiment/failed_records.json',
}

# ========================================
# 成本估算
# ========================================
COST_ESTIMATION = {
    'input_price': 0.15,
    'output_price': 0.60,
}

# ========================================
# 日志和调试
# ========================================
LOGGING = {
    'level': 'INFO',
    'verbose': True,
    'save_log': True,
    'log_file': 'logs/m2_sentiment.log',
}

# ========================================
# 测试模式
# ========================================
TEST_MODE = {
    'enabled': False,
    'sample_size': 5,
}

# ========================================
# 辅助函数：获取完整配置
# ========================================
def get_config():
    """获取完整配置字典"""
    return {
        'openai': OPENAI_CONFIG,
        'database': DATABASE_CONFIG, # 添加数据库配置到返回结果
        'output_file': OUTPUT_FILE,
        'dimensions': DIMENSIONS,
        'score_levels': SCORE_LEVELS,
        'system_prompt': SYSTEM_PROMPT,
        'user_prompt_template': USER_PROMPT_TEMPLATE,
        'concurrency': CONCURRENCY,
        'retry': RETRY,
        'failure_handling': FAILURE_HANDLING,
        'cost_estimation': COST_ESTIMATION,
        'logging': LOGGING,
        'test_mode': TEST_MODE,
    }