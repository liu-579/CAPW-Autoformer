#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import re
import html
import json
import logging
import sys
from urllib.parse import unquote
from sqlalchemy import create_engine
# 引入 MySQL 长文本类型，防止报错
from sqlalchemy.dialects.mysql import LONGTEXT

# 导入配置
try:
    from config import m1_sql_config as config
except ImportError:
    print("❌ 错误：未找到 config.py 文件。")
    sys.exit(1)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureCleaner:
    """
    增强版清洗器：支持 URL 解码、深度 HTML 清理
    """

    def parse_list_text(self, text) -> str:
        """解析列表格式，如果是列表则合并为字符串"""
        if pd.isna(text):
            return ""

        # 如果已经是 list 类型
        if isinstance(text, list):
            return ' '.join(str(item) for item in text if item)

        # 如果是字符串，尝试解析 JSON
        if isinstance(text, str):
            text = text.strip()
            if text.startswith('[') and text.endswith(']'):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return ' '.join(str(item) for item in parsed if item)
                except (json.JSONDecodeError, ValueError):
                    pass  # 解析失败则按原样处理

        return str(text) if text else ""

    def clean_text(self, text: str) -> str:
        """执行配置中定义的所有清洗步骤"""
        if not isinstance(text, str):
            text = str(text) if text else ""

        if not text.strip():
            return None  # 空内容直接返回 None

        # 1. URL 解码 (处理 %23%E5 这类字符)
        # 这一步必须在去 HTML 之前做，否则 tag 里的编码可能解出来后变成 tag
        if config.ENABLE_URL_DECODING:
            try:
                text = unquote(text)
            except Exception:
                pass

        # 2. HTML 实体解码 (&nbsp; -> 空格)
        text = html.unescape(text)

        # 3. 移除 HTML 标签
        if config.REMOVE_HTML_TAGS:
            # 先把 <br> 换成空格，防止文字粘连
            text = re.sub(r'<(br|p|div)[^>]*>', ' ', text, flags=re.IGNORECASE)
            # 移除所有 <...> 格式的标签，保留标签内的文字
            text = re.sub(r'<[^>]+>', '', text)

        # 4. 移除 URL 链接
        if config.REMOVE_URLS:
            text = re.sub(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                '', text
            )

        # 5. 移除 Emoji 和特殊符号
        if config.REMOVE_SPECIAL_CHARS:
            # 扩展的 unicode 范围，覆盖更多表情
            emoji_pattern = re.compile(
                "["
                "\U0001F000-\U0001F9FF"  # 大部分 Emoji
                "\u2600-\u26FF"  # 杂项符号
                "\u2700-\u27BF"  # 装饰符号
                "]+", flags=re.UNICODE
            )
            text = emoji_pattern.sub('', text)

        # 6. 移除 @用户
        if config.REMOVE_MENTIONS:
            text = re.sub(r'@[\w\u4e00-\u9fa5]+', '', text)

        # 7. 移除 #话题#
        if config.REMOVE_HASHTAGS:
            text = re.sub(r'#[^#]+#', '', text)

        # 8. 清理不可见字符和多余空格
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # 9. 最终长度筛选 (依据 Config 里的配置)
        if len(text) < config.MIN_TEXT_LENGTH:
            return None  # 太短的数据直接丢弃

        if len(text) > config.MAX_TEXT_LENGTH:
            # 如果配置要求截断，也可以在这里切片 text[:config.MAX_TEXT_LENGTH]
            # 这里如果不符合最大长度，我们选择保留（或者你可以选择丢弃）
            pass

        return text


def get_db_engine():
    c = config.DB_CONFIG
    conn_str = f"mysql+pymysql://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['database']}"
    return create_engine(conn_str)


def main():
    logger.info("🚀 开始执行强力清洗任务...")
    logger.info(f"配置检查 -> 最小长度: {config.MIN_TEXT_LENGTH}, 移除HTML: {config.REMOVE_HTML_TAGS}")

    engine = get_db_engine()
    cleaner = FeatureCleaner()
    all_dfs = []

    # 1. 遍历读取
    for table_name in config.SOURCE_TABLES:
        try:
            logger.info(f"📄 处理表: {table_name}")
            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

            if df.empty:
                continue

            target_col = config.CLEAN_TARGET_COLUMN
            if target_col in df.columns:
                logger.info(f"   正在清洗 {target_col} (行数: {len(df)})...")
                # 步骤 A: 解析列表
                df[target_col] = df[target_col].apply(cleaner.parse_list_text)
                # 步骤 B: 深度清洗
                df[target_col] = df[target_col].apply(cleaner.clean_text)

                # 1. 保留原 ID 但重命名，防止与新的自增 ID 冲突
                if 'id' in df.columns:
                    df.rename(columns={'id': 'original_id'}, inplace=True)

                # 2. (强烈建议) 增加来源表字段。
                # 因为合并了多个表，不同表可能有相同的 ID (例如 yq_24 和 yq_23 都有 ID=1)
                # 只有配合表名，original_id 才能唯一回溯到原始数据
                df['source_table'] = table_name

            all_dfs.append(df)

        except Exception as e:
            logger.error(f"表 {table_name} 出错: {e}")

    # 2. 合并写入
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)

        # ID 重构
        final_df.reset_index(drop=True, inplace=True)
        final_df.index = final_df.index + 1
        final_df.index.name = 'id'

        logger.info(f"💾 写入数据库: {config.TARGET_TABLE} (总行数: {len(final_df)})")

        # 强制指定字段类型为 LONGTEXT，避免 "Data too long"
        dtype_mapping = {
            'format_content': LONGTEXT,
            'title': LONGTEXT,
            'feature_words': LONGTEXT
        }

        try:
            final_df.to_sql(
                name=config.TARGET_TABLE,
                con=engine,
                if_exists='replace',
                index=True,
                dtype=dtype_mapping,
                chunksize=config.BATCH_SIZE
            )
            logger.info("✅ 任务圆满完成！")
        except Exception as e:
            logger.error(f"写入失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning("没有数据可写。")


if __name__ == "__main__":
    main()