#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块2：多维情感评分配置文件
"""

# ========================================
# OpenAI API 配置
# ========================================
OPENAI_CONFIG = {
    'api_key': 'DAyTT70oAEiOd76JNDrhu4zLxmtRAmZk_xZ6X4cTHQM',  # 你的 OpenAI API Key
    'base_url': "https://api.poe.com/v1",  # 如果使用代理，填写代理地址，如 "https://api.openai-proxy.com/v1"
    'model': 'assistant',  # 使用的模型，推荐 gpt-4o-mini（便宜快速）
    'temperature': 0.0,  # 温度参数，0表示确定性输出
    'max_tokens': 200,  # 最大返回token数
}

# ========================================
# 输入输出文件配置
# ========================================
INPUT_FILE = r'D:\learning\shiyan1\data\输出\m1_cleaned_comments.jsonl'
OUTPUT_FILE = r'D:\learning\shiyan1\data\输出\m2_sentiment_scores.json'

# ========================================
# 情感评分维度配置（核心配置，修改这里可以改变维度）
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
        'name': '门票',
        'key': 'ticket',
        'description': '门票价格的合理性、性价比、购票便利性'
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
# 用户提示词模板（使用{review}和{dimensions}作为占位符）
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
  "ticket": 0,
  "service": 0
}}

只返回JSON，不要有任何其他文字说明。"""

# ========================================
# 并发控制
# ========================================
CONCURRENCY = {
    'max_workers': 10,  # 最大并发数
    'rate_limit': 50,  # 每分钟最多请求数
    'batch_size': 100,  # 每批处理的数据量（用于分批保存）
}

# ========================================
# 重试配置
# ========================================
RETRY = {
    'max_attempts': 3,  # 最大重试次数
    'initial_delay': 1,  # 初始延迟（秒）
    'max_delay': 10,  # 最大延迟（秒）
    'exponential_base': 2,  # 指数退避基数
}

# ========================================
# 失败处理
# ========================================
FAILURE_HANDLING = {
    'default_score': 0,  # 失败后的默认分数
    'save_failed_records': True,  # 是否保存失败记录
    'failed_records_file': 'data/sentiment/failed_records.json',
}

# ========================================
# 成本估算（基于OpenAI定价，单位：美元/1M tokens）
# ========================================
COST_ESTIMATION = {
    'input_price': 0.15,  # gpt-4o-mini 输入价格
    'output_price': 0.60,  # gpt-4o-mini 输出价格
}

# ========================================
# 日志和调试
# ========================================
LOGGING = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'verbose': True,  # 是否显示详细进度
    'save_log': True,  # 是否保存日志
    'log_file': 'logs/m2_sentiment.log',
}

# ========================================
# 测试模式
# ========================================
TEST_MODE = {
    'enabled': True,  # 是否启用测试模式
    'sample_size': 5,  # 测试模式下处理的样本数量
}


# ========================================
# 辅助函数：获取完整配置
# ========================================
def get_config():
    """获取完整配置字典"""
    return {
        'openai': OPENAI_CONFIG,
        'input_file': INPUT_FILE,
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


# ========================================
# 维度修改示例
# ========================================
"""
如何修改为6维评分：

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
        'name': '门票',
        'key': 'ticket',
        'description': '门票价格的合理性、性价比、购票便利性'
    },
    {
        'name': '服务',
        'key': 'service',
        'description': '景区工作人员态度、设施维护、游客服务'
    },
    {
        'name': '住宿',  # 新增第6维
        'key': 'accommodation',
        'description': '景区附近酒店民宿的质量、价格、便利性'
    },
]

如何修改为8维评分（情绪维度示例）：

DIMENSIONS = [
    {'name': '愉悦度', 'key': 'joy', 'description': '游客的快乐、兴奋程度'},
    {'name': '放松度', 'key': 'relaxation', 'description': '游客的舒适、放松感受'},
    {'name': '惊喜度', 'key': 'surprise', 'description': '超出预期的惊喜体验'},
    {'name': '失望度', 'key': 'disappointment', 'description': '未达预期的失望程度'},
    {'name': '愤怒度', 'key': 'anger', 'description': '不满、生气的情绪'},
    {'name': '焦虑度', 'key': 'anxiety', 'description': '担心、紧张的情绪'},
    {'name': '感动度', 'key': 'touched', 'description': '被感动、温暖的体验'},
    {'name': '自豪度', 'key': 'pride', 'description': '自豪、骄傲的感受'},
]

修改后记得同步更新 USER_PROMPT_TEMPLATE 中的示例JSON格式！
"""