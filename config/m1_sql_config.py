# -*- coding: utf-8 -*-
"""
配置文件：数据库连接与详细清洗规则
"""

# ==================== 1. 数据库连接配置 ====================
DB_CONFIG = {
    'host': '127.0.0.1',       # 数据库地址
    'port': 3306,              # 端口
    'user': 'root',            # 用户名
    'password': 'password', # 密码 (请修改这里)
    'database': 'yq_data'      # 数据库名
}

# ==================== 2. 表结构配置 ====================
# 需要读取的源表列表 (请确保这三张表结构一致)
SOURCE_TABLES = [
    'yq_24_2',
    'yq_23_xsw_2',
    'yq_23_eedscy_2'
]

# 最终输出的新表名称
TARGET_TABLE = 'yq_clean_all_3'

# 需要进行清洗的特定字段名
CLEAN_TARGET_COLUMN = 'format_content'

# ==================== 3. 文本长度筛选规则 ====================
# 清洗后，如果文本长度小于此值，将被视为无效数据（置为 None 或空字符串）
MIN_TEXT_LENGTH = 2  

# 清洗后，如果文本长度大于此值，将被截断或视为无效（防止过长）
# 注意：feature_words 通常较短，如果设得太大可能会保留垃圾信息
MAX_TEXT_LENGTH = 5000 

# ==================== 4. 清洗逻辑开关 ====================

# 是否进行 URL 解码 (解决 %E5%B0%8F%E7%B1%B3 这类乱码)
# 【强烈建议开启】你提供的坏案例必须开这个才能修好
ENABLE_URL_DECODING = True 

# 是否移除所有 HTML 标签 (如 <a>, <div>, <br>) 并保留标签内的文字
REMOVE_HTML_TAGS = True 

# 是否移除 URL 链接 (http/https 开头的文本)
REMOVE_URLS = True

# 是否移除 Emoji 表情和特殊图形符号
REMOVE_SPECIAL_CHARS = True 

# 是否移除 @用户名
REMOVE_MENTIONS = True

# 是否移除 #话题#
REMOVE_HASHTAGS = False 

# ==================== 5. 系统运行配置 ====================
# 数据库写入批次大小
BATCH_SIZE = 1000