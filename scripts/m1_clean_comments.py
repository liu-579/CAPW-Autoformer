#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评论数据清洗脚本
从Excel文件读取数据，清洗后输出为JSONL格式
"""

import re
import json
import html
import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入配置
from config.m1_config import *


class CommentCleaner:
    """评论数据清洗器"""

    def __init__(self):
        """初始化清洗器"""
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('CommentCleaner')
        logger.setLevel(logging.INFO if VERBOSE else logging.WARNING)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def parse_list_text(self, text) -> str:
        """解析可能是列表格式的文本"""
        if pd.isna(text):
            return ""

        if isinstance(text, list):
            return ' '.join(str(item) for item in text if item)

        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return ' '.join(str(item) for item in parsed if item)
                return str(parsed)
            except (json.JSONDecodeError, ValueError):
                return text

        return str(text) if text else ""

    def clean_text(self, text: str) -> str:
        """文本清洗 - 温和模式"""
        if not isinstance(text, str):
            text = str(text) if text else ""

        if not text:
            return ""

        # 1. HTML解码
        text = html.unescape(text)

        # 2. 移除HTML标签（如果配置要求）
        if REMOVE_HTML_TAGS:
            # 移除<a>标签及其内容
            text = re.sub(r'<a\s+[^>]*>.*?</a>', '', text, flags=re.DOTALL | re.IGNORECASE)
            # 移除script和style标签
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            # 移除所有其他HTML标签（保留内容）
            text = re.sub(r'<[^>]+>', '', text)

        # 3. 移除HTML实体编码
        text = re.sub(r'&#[xX]?[0-9a-fA-F]+;?', '', text)
        text = re.sub(r'&[a-zA-Z]{2,};', ' ', text)

        # 4. 移除URL（如果配置要求）
        if REMOVE_URLS:
            text = re.sub(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                '', text
            )

        # 5. 移除emoji（如果配置要求）
        if REMOVE_SPECIAL_CHARS:
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # 表情符号
                "\U0001F300-\U0001F5FF"  # 符号和图标
                "\U0001F680-\U0001F6FF"  # 交通和地图
                "\U0001F1E0-\U0001F1FF"  # 旗帜
                "\U00002500-\U00002BEF"  # 中文符号
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "\U0001f926-\U0001f937"
                "\U00010000-\U0010ffff"
                "\u2640-\u2642"
                "\u2600-\u2B55"
                "\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
                "]+", flags=re.UNICODE
            )
            text = emoji_pattern.sub('', text)

        # 6. 移除@用户名（如果配置要求）
        if REMOVE_MENTIONS:
            text = re.sub(r'@[\w\u4e00-\u9fa5]+', '', text)

        # 7. 移除话题标签（如果配置要求）
        if REMOVE_HASHTAGS:
            text = re.sub(r'#[^#]+#', '', text)

        # 8. 移除一些明显的垃圾字符（但保留正常标点）
        # 只移除控制字符和一些特殊符号
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # 9. 处理连续的标点符号（3个以上才算异常）
        text = re.sub(r'([，。！？、；：]){3,}', r'\1', text)

        # 10. 合并多个空格为一个
        text = re.sub(r'\s+', ' ', text)

        # 11. 去除首尾空格
        text = text.strip()

        return text

    def is_valid_text(self, text: str) -> bool:
        """检查文本是否有效"""
        if not text or not isinstance(text, str):
            return False

        # 检查长度
        text_len = len(text)
        if text_len < MIN_TEXT_LENGTH:
            return False

        if text_len > MAX_TEXT_LENGTH:
            return False

        # 检查中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        if chinese_chars == 0:
            return False

        # 检查中文占比
        chinese_ratio = chinese_chars / text_len if text_len > 0 else 0
        if chinese_ratio < MIN_CHINESE_RATIO:
            return False

        return True

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据框"""
        self.logger.info(f"开始处理数据，原始数据量: {len(df)}")

        # 检查文本列
        if TEXT_COLUMN not in df.columns:
            self.logger.error(f"错误：未找到文本列 '{TEXT_COLUMN}'")
            self.logger.error(f"可用的列: {df.columns.tolist()}")
            raise ValueError(f"未找到文本列: {TEXT_COLUMN}")

        # 显示原始样例
        if VERBOSE:
            self.logger.info("\n原始文本样例（前3条）：")
            for i, text in enumerate(df[TEXT_COLUMN].head(3)):
                preview = str(text)[:200]
                self.logger.info(f"  [{i + 1}] {preview}...")

        # 第一步：解析列表格式
        self.logger.info("\n正在解析文本格式...")
        df['parsed_text'] = df[TEXT_COLUMN].apply(self.parse_list_text)

        # 显示解析后样例
        if VERBOSE:
            self.logger.info("\n解析后文本样例（前3条）：")
            for i, text in enumerate(df['parsed_text'].head(3)):
                self.logger.info(f"  [{i + 1}] [长度:{len(text)}] {text[:200]}...")

        # 第二步：清洗文本
        self.logger.info("\n正在清洗文本...")
        df['cleaned_text'] = df['parsed_text'].apply(self.clean_text)

        # 显示清洗后样例
        if VERBOSE:
            self.logger.info("\n清洗后文本样例（前3条）：")
            for i, text in enumerate(df['cleaned_text'].head(3)):
                self.logger.info(f"  [{i + 1}] [长度:{len(text)}] {text[:200]}...")

        # 第三步：过滤有效数据
        self.logger.info("\n正在过滤有效数据...")
        invalid_reasons = {
            'empty': 0,
            'too_short': 0,
            'too_long': 0,
            'no_chinese': 0,
            'low_chinese': 0
        }

        valid_rows = []
        for idx, row in df.iterrows():
            text = row['cleaned_text']

            if not text:
                invalid_reasons['empty'] += 1
                continue

            text_len = len(text)
            if text_len < MIN_TEXT_LENGTH:
                invalid_reasons['too_short'] += 1
                continue

            if text_len > MAX_TEXT_LENGTH:
                invalid_reasons['too_long'] += 1
                continue

            chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
            if chinese_chars == 0:
                invalid_reasons['no_chinese'] += 1
                continue

            chinese_ratio = chinese_chars / text_len
            if chinese_ratio < MIN_CHINESE_RATIO:
                invalid_reasons['low_chinese'] += 1
                continue

            valid_rows.append(row)

        # 显示统计
        valid_count = len(valid_rows)
        total_count = len(df)

        self.logger.info(f"\n有效数据: {valid_count} / {total_count} ({valid_count / total_count * 100:.1f}%)")

        if invalid_reasons['empty'] > 0:
            self.logger.info(f"  空文本: {invalid_reasons['empty']}")
        if invalid_reasons['too_short'] > 0:
            self.logger.info(f"  太短 (<{MIN_TEXT_LENGTH}字符): {invalid_reasons['too_short']}")
        if invalid_reasons['too_long'] > 0:
            self.logger.info(f"  太长 (>{MAX_TEXT_LENGTH}字符): {invalid_reasons['too_long']}")
        if invalid_reasons['no_chinese'] > 0:
            self.logger.info(f"  无中文: {invalid_reasons['no_chinese']}")
        if invalid_reasons['low_chinese'] > 0:
            self.logger.info(f"  中文占比低 (<{MIN_CHINESE_RATIO * 100}%): {invalid_reasons['low_chinese']}")

        if not valid_rows:
            self.logger.warning("\n没有有效数据！")
            return pd.DataFrame()

        valid_df = pd.DataFrame(valid_rows)

        # 去重
        if REMOVE_DUPLICATES:
            before = len(valid_df)
            valid_df = valid_df.drop_duplicates(subset=['cleaned_text'], keep='first')
            removed = before - len(valid_df)
            if removed > 0:
                self.logger.info(f"\n去重: 删除 {removed} 条重复数据")

        # 生成最终数据
        valid_df = valid_df.reset_index(drop=True)
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        final_data = []
        for i, row in valid_df.iterrows():
            item = {
                'id': f"review_{str(i).zfill(6)}",
                'text': row['cleaned_text']
            }

            # 添加元数据
            if KEEP_METADATA:
                metadata = {
                    'original_index': int(row.name) if hasattr(row, 'name') else int(i),
                    'cleaned_at': current_time,
                    'original_length': len(row['parsed_text']),
                    'cleaned_length': len(row['cleaned_text'])
                }

                # 添加配置中的字段
                for meta_key, col_name in METADATA_FIELDS.items():
                    if col_name in row.index and pd.notna(row[col_name]):
                        metadata[meta_key] = str(row[col_name])

                item['metadata'] = metadata

            final_data.append(item)

        result_df = pd.DataFrame(final_data)
        self.logger.info("\n数据处理完成！")

        return result_df

    def print_stats(self, df: pd.DataFrame):
        """打印统计信息"""
        if df is None or len(df) == 0:
            self.logger.warning("没有数据可统计！")
            return

        df = df.copy()
        df['text_length'] = df['text'].apply(len)

        self.logger.info("\n" + "=" * 50)
        self.logger.info("数据统计报告")
        self.logger.info("=" * 50)
        self.logger.info(f"总数据量: {len(df)}")
        self.logger.info(f"平均长度: {df['text_length'].mean():.2f} 字符")
        self.logger.info(f"最短文本: {df['text_length'].min()} 字符")
        self.logger.info(f"最长文本: {df['text_length'].max()} 字符")

        # 长度分布
        self.logger.info("\n长度分布:")
        bins = [0, 50, 100, 200, 500, float('inf')]
        labels = ['<50', '50-100', '100-200', '200-500', '>500']
        df['length_cat'] = pd.cut(df['text_length'], bins=bins, labels=labels)

        for label in labels:
            count = (df['length_cat'] == label).sum()
            pct = count / len(df) * 100
            self.logger.info(f"  {label:>10}: {count:>6} ({pct:>5.2f}%)")

        self.logger.info("=" * 50)

        # 保存统计报告
        if SAVE_STATS_REPORT:
            self._save_stats_report(df)

    def _save_stats_report(self, df: pd.DataFrame):
        """保存统计报告到文件"""
        try:
            with open(STATS_OUTPUT, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("数据清洗统计报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"输入文件: {INPUT_FILE}\n")
                f.write(f"输出文件: {OUTPUT_FILE}\n\n")

                f.write(f"总数据量: {len(df)}\n")
                f.write(f"平均长度: {df['text_length'].mean():.2f} 字符\n")
                f.write(f"最短文本: {df['text_length'].min()} 字符\n")
                f.write(f"最长文本: {df['text_length'].max()} 字符\n\n")

                f.write("长度分布:\n")
                for label in ['<50', '50-100', '100-200', '200-500', '>500']:
                    count = (df['length_cat'] == label).sum()
                    pct = count / len(df) * 100
                    f.write(f"  {label:>10}: {count:>6} ({pct:>5.2f}%)\n")

                f.write("=" * 50 + "\n")

            self.logger.info(f"✓ 统计报告已保存: {STATS_OUTPUT}")
        except Exception as e:
            self.logger.warning(f"保存统计报告失败: {e}")


def main():
    """主函数"""

    cleaner = CommentCleaner()

    try:
        cleaner.logger.info("=" * 50)
        cleaner.logger.info("数据清洗程序")
        cleaner.logger.info("=" * 50)
        cleaner.logger.info(f"输入文件: {INPUT_FILE}")
        cleaner.logger.info(f"输出文件: {OUTPUT_FILE}")
        cleaner.logger.info(f"文本列名: {TEXT_COLUMN}")
        cleaner.logger.info(f"预览模式: {'是' if DRY_RUN else '否'}")
        cleaner.logger.info(f"\n清洗配置:")
        cleaner.logger.info(f"  最小长度: {MIN_TEXT_LENGTH} 字符")
        cleaner.logger.info(f"  最大长度: {MAX_TEXT_LENGTH} 字符")
        cleaner.logger.info(f"  中文占比: ≥{MIN_CHINESE_RATIO * 100}%")
        cleaner.logger.info(f"  移除URL: {'是' if REMOVE_URLS else '否'}")
        cleaner.logger.info(f"  移除特殊字符: {'是' if REMOVE_SPECIAL_CHARS else '否'}")
        cleaner.logger.info("=" * 50)

        # 检查文件
        if not Path(INPUT_FILE).exists():
            cleaner.logger.error(f"\n输入文件不存在: {INPUT_FILE}")
            return

        # 读取Excel
        cleaner.logger.info(f"\n读取Excel文件...")
        df = pd.read_excel(INPUT_FILE)
        cleaner.logger.info(f"读取成功！原始数据量: {len(df)}")

        # 预览模式
        if DRY_RUN:
            df = df.head(PREVIEW_COUNT)
            cleaner.logger.info(f"⚠️  预览模式：只处理前 {PREVIEW_COUNT} 条数据")

        # 清洗数据
        cleaned_df = cleaner.process_dataframe(df)

        if cleaned_df is None or len(cleaned_df) == 0:
            cleaner.logger.error("\n✗ 清洗后没有有效数据！")
            return

        # 保存结果
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_json(OUTPUT_FILE, orient='records', lines=True, force_ascii=False)

        # 打印统计
        cleaner.print_stats(cleaned_df)

        cleaner.logger.info(f"\n✓ 清洗完成！")
        cleaner.logger.info(f"✓ 输出文件: {OUTPUT_FILE}")

    except Exception as e:
        cleaner.logger.error(f"\n✗ 处理失败: {str(e)}")
        import traceback
        cleaner.logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()