#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模块3：数据集构建与划分主程序
功能：合并清洗后的评论和情感评分，划分数据集，生成数据卡
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
import zipfile


class DatasetBuilder:
    """数据集构建器"""

    def __init__(self):
        """初始化数据集构建器"""
        # 导入配置
        from config.m3_config import get_config
        self.config = get_config()

        # 设置日志
        self.logger = self._setup_logger()

        # 初始化统计信息
        self.stats = {
            'total': 0,
            'aligned': 0,
            'train_size': 0,
            'val_size': 0,
            'test_size': 0
        }

        self.logger.info("数据集构建器初始化完成")

    def _setup_logger(self):
        """设置日志"""
        log_config = self.config['logging']

        # 创建日志目录
        log_file = Path(log_config['log_file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # 配置日志
        logger = logging.getLogger('DatasetBuilder')
        logger.setLevel(getattr(logging, log_config['level']))

        # 清除现有handlers
        logger.handlers.clear()

        # 文件handler
        if log_config['save_log']:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(fh)

        # 控制台handler
        if log_config['verbose']:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(
                '%(levelname)s - %(message)s'
            ))
            logger.addHandler(ch)

        return logger

    def build(self):
        """执行数据集构建流程"""
        self.logger.info("=" * 60)
        self.logger.info("开始构建数据集")
        self.logger.info("=" * 60)

        # 1. 加载数据
        self.logger.info("\n步骤1: 加载数据...")
        comments_data, scores_data = self._load_data()

        # 2. 对齐数据
        self.logger.info("\n步骤2: 对齐评论与情感分数...")
        aligned_data = self._align_data(comments_data, scores_data)

        # 3. 质量检查
        self.logger.info("\n步骤3: 数据质量检查...")
        aligned_data = self._quality_check(aligned_data)

        # 4. 划分数据集
        self.logger.info("\n步骤4: 划分训练集、验证集、测试集...")
        train_data, val_data, test_data = self._split_dataset(aligned_data)

        # 5. 保存数据
        self.logger.info("\n步骤5: 保存数据集...")
        self._save_datasets(aligned_data, train_data, val_data, test_data)

        # 6. 统计分析
        self.logger.info("\n步骤6: 计算统计信息...")
        stats, kl_divergences = self._compute_statistics(train_data, test_data)

        # 7. 生成数据卡
        self.logger.info("\n步骤7: 生成数据卡...")
        self._generate_data_card(stats, kl_divergences)

        # 8. 创建压缩包
        if self.config['archive_config']['create_archive']:
            self.logger.info("\n步骤8: 创建压缩包...")
            self._create_archive()

        # 9. 打印总结
        self._print_summary(stats, kl_divergences)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("数据集构建完成！")
        self.logger.info("=" * 60)

    def _load_data(self) -> Tuple[List[Dict], Dict]:
        """加载清洗后的评论和情感评分"""
        # 加载清洗后的评论
        comments_file = Path(self.config['input_files']['cleaned_comments'])
        comments_data = []
        with open(comments_file, 'r', encoding='utf-8') as f:
            for line in f:
                comments_data.append(json.loads(line))

        # 加载情感评分
        scores_file = Path(self.config['input_files']['sentiment_scores'])
        with open(scores_file, 'r', encoding='utf-8') as f:
            raw_scores = json.load(f)

        # 将评分数据转换为字典（以id为key）
        scores_data = {}

        if isinstance(raw_scores, list):
            # 列表格式: [{"id": "xxx", "scores": {...}}, ...]
            self.logger.info("评分数据格式: 列表")
            for item in raw_scores:
                if isinstance(item, dict) and 'id' in item:
                    item_id = str(item['id'])
                    scores_data[item_id] = item
                else:
                    self.logger.warning(f"跳过无效的评分项: {item}")

        elif isinstance(raw_scores, dict):
            # 字典格式
            if 'results' in raw_scores and isinstance(raw_scores['results'], list):
                # 格式: {"results": [{"id": "xxx", ...}, ...]}
                self.logger.info("评分数据格式: 字典(包含results列表)")
                for item in raw_scores['results']:
                    if isinstance(item, dict) and 'id' in item:
                        item_id = str(item['id'])
                        scores_data[item_id] = item
            else:
                # 格式: {"id1": {...}, "id2": {...}}
                self.logger.info("评分数据格式: 字典(id为key)")
                scores_data = {str(k): v for k, v in raw_scores.items()}

        else:
            raise ValueError(f"不支持的评分数据格式: {type(raw_scores)}")

        self.stats['total'] = len(comments_data)
        self.logger.info(f"✓ 加载评论数: {len(comments_data)}")
        self.logger.info(f"✓ 加载评分数: {len(scores_data)}")

        # 打印示例数据（调试用）
        if comments_data and scores_data:
            sample_comment_id = str(comments_data[0]['id'])
            sample_score_id = list(scores_data.keys())[0]
            self.logger.info(f"评论示例ID: {sample_comment_id}")
            self.logger.info(f"评分示例ID: {sample_score_id}")

            # 检查评分结构
            sample_score = scores_data[sample_score_id]
            if 'scores' in sample_score:
                self.logger.info(f"评分包含 'scores' 子对象")

        return comments_data, scores_data

    def _align_data(self, comments_data: List[Dict], scores_data: Dict) -> List[Dict]:
        """对齐评论和评分数据"""
        aligned_data = []
        dimension_mapping = self.config['dimension_mapping']
        not_found_count = 0
        missing_dim_count = 0

        for comment in comments_data:
            comment_id = str(comment['id'])

            if comment_id not in scores_data:
                not_found_count += 1
                if not_found_count <= 5:
                    self.logger.debug(f"未找到评分: {comment_id}")
                continue

            score_item = scores_data[comment_id]

            # 检查是否有 scores 子对象（M2格式）
            if 'scores' in score_item:
                scores = score_item['scores']
            else:
                scores = score_item

            # 构建对齐后的数据
            aligned_item = {
                'id': comment_id,
                'text': comment['text']
            }

            # 添加情感分数（支持英文key和中文key）
            has_all_dims = True
            missing_dims = []

            for eng_key, chi_key in dimension_mapping.items():
                # 尝试英文key
                if eng_key in scores:
                    aligned_item[chi_key] = scores[eng_key]
                # 尝试中文key
                elif chi_key in scores:
                    aligned_item[chi_key] = scores[chi_key]
                else:
                    has_all_dims = False
                    missing_dims.append(f"{chi_key}({eng_key})")

            if has_all_dims:
                aligned_data.append(aligned_item)
            else:
                missing_dim_count += 1
                if missing_dim_count <= 5:
                    self.logger.warning(
                        f"评论 {comment_id} 缺少维度: {', '.join(missing_dims)}"
                    )

        self.stats['aligned'] = len(aligned_data)

        # 输出对齐统计
        self.logger.info(f"✓ 成功对齐数据: {len(aligned_data)} 条")
        if not_found_count > 0:
            self.logger.warning(f"  未找到匹配评分: {not_found_count} 条")
        if missing_dim_count > 0:
            self.logger.warning(f"  缺少维度信息: {missing_dim_count} 条")

        # 如果完全失败，打印详细诊断
        if len(aligned_data) == 0:
            self.logger.error("=" * 60)
            self.logger.error("对齐完全失败！诊断信息:")

            if comments_data:
                sample_comment = comments_data[0]
                self.logger.error(f"\n评论示例:")
                self.logger.error(f"  ID: {sample_comment['id']}")

            if scores_data:
                sample_score_id = list(scores_data.keys())[0]
                sample_score = scores_data[sample_score_id]
                self.logger.error(f"\n评分示例:")
                self.logger.error(f"  ID: {sample_score_id}")
                self.logger.error(f"  完整结构: {sample_score}")

            self.logger.error(f"\n配置: {dimension_mapping}")
            self.logger.error("=" * 60)

        return aligned_data

    def _quality_check(self, data: List[Dict]) -> List[Dict]:
        """数据质量检查"""
        qc_config = self.config['quality_check']

        if not qc_config['check_missing_values'] and not qc_config['check_score_range']:
            return data

        clean_data = []
        min_score, max_score = qc_config['valid_score_range']

        for item in data:
            is_valid = True

            # 检查缺失值
            if qc_config['check_missing_values']:
                for dim in self.config['dimensions']:
                    if dim not in item or item[dim] is None:
                        self.logger.warning(f"ID {item['id']}: 维度 {dim} 缺失")
                        is_valid = False
                        break

            # 检查分数范围
            if is_valid and qc_config['check_score_range']:
                for dim in self.config['dimensions']:
                    score = item[dim]
                    if not (min_score <= score <= max_score):
                        self.logger.warning(
                            f"ID {item['id']}: 维度 {dim} 分数 {score} 超出范围 [{min_score}, {max_score}]"
                        )
                        is_valid = False
                        break

            if is_valid or not qc_config['remove_invalid']:
                clean_data.append(item)

        removed = len(data) - len(clean_data)
        if removed > 0:
            self.logger.info(f"✓ 质量检查完成，移除 {removed} 条不合法数据")
        else:
            self.logger.info(f"✓ 质量检查完成，所有数据合法")

        return clean_data

    def _split_dataset(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """划分数据集"""
        split_config = self.config['split_config']

        # 第一次划分：分出测试集
        train_val_data, test_data = train_test_split(
            data,
            test_size=split_config['test_ratio'],
            random_state=split_config['random_seed'],
            shuffle=split_config['shuffle']
        )

        # 第二次划分：从训练+验证中分出验证集
        val_ratio_adjusted = split_config['val_ratio'] / (1 - split_config['test_ratio'])
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio_adjusted,
            random_state=split_config['random_seed'],
            shuffle=split_config['shuffle']
        )

        self.stats['train_size'] = len(train_data)
        self.stats['val_size'] = len(val_data)
        self.stats['test_size'] = len(test_data)

        self.logger.info(f"✓ 训练集: {len(train_data)} 条")
        self.logger.info(f"✓ 验证集: {len(val_data)} 条")
        self.logger.info(f"✓ 测试集: {len(test_data)} 条")

        return train_data, val_data, test_data

    def _save_datasets(self, full_data: List[Dict], train_data: List[Dict],
                       val_data: List[Dict], test_data: List[Dict]):
        """保存数据集"""
        output_dir = Path(self.config['output_files']['merged_dataset']).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整数据集（JSONL格式）
        full_file = Path(self.config['output_files']['merged_dataset'])
        with open(full_file, 'w', encoding='utf-8') as f:
            for item in full_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        self.logger.info(f"✓ 完整数据集已保存: {full_file}")

        # 保存CSV格式
        self._save_csv(train_data, self.config['output_files']['train_csv'], 'train')
        self._save_csv(val_data, self.config['output_files']['val_csv'], 'val')
        self._save_csv(test_data, self.config['output_files']['test_csv'], 'test')

    def _save_csv(self, data: List[Dict], filepath: str, name: str):
        """保存为CSV格式"""
        df = pd.DataFrame(data)

        # 调整列顺序
        columns = ['id', 'text'] + self.config['dimensions']
        df = df[columns]

        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        self.logger.info(f"✓ {name}.csv 已保存: {filepath}")

    def _compute_statistics(self, train_data: List[Dict], test_data: List[Dict]) -> Tuple[Dict, Dict]:
        """计算统计信息"""
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        stats = {}
        kl_divergences = {}

        for dim in self.config['dimensions']:
            # 计算训练集统计
            stats[dim] = {
                'mean': train_df[dim].mean(),
                'std': train_df[dim].std(),
                'min': train_df[dim].min(),
                'max': train_df[dim].max()
            }

            # 计算KL散度
            if self.config['distribution_check']['check_kl_divergence']:
                kl = self._compute_kl_divergence(
                    train_df[dim].values,
                    test_df[dim].values
                )
                kl_divergences[dim] = kl

        return stats, kl_divergences

    def _compute_kl_divergence(self, train_scores: np.ndarray, test_scores: np.ndarray) -> float:
        """计算KL散度"""
        smoothing = self.config['distribution_check']['smoothing']

        # 计算分布
        bins = np.arange(-2.5, 3.5, 1)  # -2, -1, 0, 1, 2
        train_hist, _ = np.histogram(train_scores, bins=bins, density=True)
        test_hist, _ = np.histogram(test_scores, bins=bins, density=True)

        # 平滑处理
        train_hist = train_hist + smoothing
        test_hist = test_hist + smoothing

        # 归一化
        train_hist = train_hist / train_hist.sum()
        test_hist = test_hist / test_hist.sum()

        # 计算KL散度
        kl = entropy(test_hist, train_hist)

        return kl

    def _generate_data_card(self, stats: Dict, kl_divergences: Dict):
        """生成数据卡"""
        card_config = self.config['data_card_config']

        # 当前日期
        creation_date = datetime.now().strftime('%Y-%m-%d')

        # 构建Markdown内容
        content = f"# {card_config['dataset_name']}\n\n"
        content += "## 基本信息\n\n"
        content += f"- **版本**: {card_config['dataset_version']}\n"
        content += f"- **创建日期**: {creation_date}\n"
        content += f"- **数据来源**: {card_config['data_source']}\n"
        content += "- **语言**: 中文\n"
        content += "- **领域**: 旅游评论\n\n"

        content += "## 数据规模\n\n"
        content += "| 集合 | 样本数 | 占比 |\n"
        content += "|------|--------|------|\n"
        content += f"| 训练集 | {self.stats['train_size']} | {self.stats['train_size'] / self.stats['aligned'] * 100:.1f}% |\n"
        content += f"| 验证集 | {self.stats['val_size']} | {self.stats['val_size'] / self.stats['aligned'] * 100:.1f}% |\n"
        content += f"| 测试集 | {self.stats['test_size']} | {self.stats['test_size'] / self.stats['aligned'] * 100:.1f}% |\n"
        content += f"| **总计** | **{self.stats['aligned']}** | **100%** |\n\n"

        content += "## 数据字段\n\n"
        content += "| 字段名 | 类型 | 说明 | 取值范围 |\n"
        content += "|--------|------|------|----------|\n"
        content += "| id | string | 评论唯一标识符 | - |\n"
        content += "| text | string | 清洗后的评论文本 | - |\n"

        for dim in self.config['dimensions']:
            content += f"| {dim} | int | {dim}维度情感分数 | -2 到 2 |\n"

        content += "\n## 情感评分说明\n\n"
        content += "评分采用5级量表：\n\n"
        content += "- **-2**: 极差（明确表达强烈不满）\n"
        content += "- **-1**: 差（表达不满、体验不佳）\n"
        content += "- **0**: 中性或未提及\n"
        content += "- **1**: 好（表达满意、体验良好）\n"
        content += "- **2**: 极好（表达强烈满意、极力推荐）\n\n"

        content += "## 数据分布统计\n\n"
        content += "### 五维情感均值与标准差\n\n"
        content += "| 维度 | 均值 | 标准差 |\n"
        content += "|------|------|--------|\n"

        for dim in self.config['dimensions']:
            content += f"| {dim} | {stats[dim]['mean']:.3f} | {stats[dim]['std']:.3f} |\n"

        if kl_divergences:
            content += "\n### 测试集与训练集分布差异（KL散度）\n\n"
            content += "| 维度 | KL散度 | 状态 |\n"
            content += "|------|--------|------|\n"

            threshold = self.config['distribution_check']['kl_threshold']
            for dim in self.config['dimensions']:
                kl = kl_divergences[dim]
                status = "✓ 合格" if kl < threshold else "✗ 超标"
                content += f"| {dim} | {kl:.4f} | {status} |\n"

            content += f"\n> 阈值: < {threshold}（KL散度越小，说明测试集与训练集分布越接近）\n\n"

        content += "## 数据处理流程\n\n"
        for i, step in enumerate(card_config['processing_steps'], 1):
            content += f"{i}. {step}\n"

        content += "\n## 划分方法\n\n"
        split_config = self.config['split_config']
        content += f"- **随机种子**: {split_config['random_seed']}（保证可复现）\n"
        content += f"- **训练集比例**: {split_config['train_ratio'] * 100:.0f}%\n"
        content += f"- **验证集比例**: {split_config['val_ratio'] * 100:.0f}%\n"
        content += f"- **测试集比例**: {split_config['test_ratio'] * 100:.0f}%\n"
        content += f"- **打乱数据**: {'是' if split_config['shuffle'] else '否'}\n\n"

        content += "## 已知偏差与局限性\n\n"
        content += "### 可能偏差\n\n"
        for bias in card_config['known_biases']:
            content += f"- {bias}\n"

        content += "\n### 局限性\n\n"
        for limitation in card_config['limitations']:
            content += f"- {limitation}\n"

        content += f"\n## 使用建议\n\n"
        content += f"- **适用场景**: {card_config['intended_use']}\n"
        content += "- **不适用场景**: 非旅游领域文本、非中文文本、其他情感维度需求\n\n"

        content += "## 引用格式\n\n"
        content += "如果使用本数据集，请引用：\n\n"
        content += "```\n"
        content += "Tourism Review Multi-dimensional Sentiment Dataset V1\n"
        content += f"Created: {creation_date}\n"
        content += "Processing: Text Cleaning + GPT-4o-mini Multi-dimensional Scoring\n"
        content += "```\n\n"

        content += "## 文件清单\n\n"
        content += "```\n"
        content += "dataset_v1/\n"
        content += "├── dataset_v1.jsonl    # 完整数据集（JSONL格式）\n"
        content += "├── train.csv           # 训练集（CSV格式）\n"
        content += "├── val.csv             # 验证集（CSV格式）\n"
        content += "├── test.csv            # 测试集（CSV格式）\n"
        content += "└── data_card.md        # 数据卡（本文件）\n"
        content += "```\n\n"

        content += "## 版本历史\n\n"
        content += f"- **v1.0** ({creation_date}): 初始版本\n\n"
        content += "---\n\n"
        content += "*本数据集仅供学术研究使用，请勿用于商业用途。*\n"

        # 保存数据卡
        card_file = Path(self.config['output_files']['data_card'])
        with open(card_file, 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.info(f"✓ 数据卡已生成: {card_file}")

    def _create_archive(self):
        """创建压缩包"""
        archive_config = self.config['archive_config']
        archive_file = Path(self.config['output_files']['archive'])
        dataset_dir = Path(self.config['output_files']['merged_dataset']).parent

        with zipfile.ZipFile(archive_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename in archive_config['include_files']:
                file_path = dataset_dir / filename
                if file_path.exists():
                    zf.write(file_path, arcname=filename)
                else:
                    self.logger.warning(f"文件不存在，跳过: {file_path}")

        self.logger.info(f"✓ 压缩包已创建: {archive_file}")

    def _print_summary(self, stats: Dict, kl_divergences: Dict):
        """打印统计摘要"""
        if not self.config['stats_report']['print_summary']:
            return

        print("\n" + "=" * 60)
        print("数据集构建统计摘要")
        print("=" * 60)

        print(f"\n总样本数: {self.stats['total']}")
        print(f"对齐成功: {self.stats['aligned']}")
        print(f"训练集: {self.stats['train_size']} ({self.stats['train_size'] / self.stats['aligned'] * 100:.1f}%)")
        print(f"验证集: {self.stats['val_size']} ({self.stats['val_size'] / self.stats['aligned'] * 100:.1f}%)")
        print(f"测试集: {self.stats['test_size']} ({self.stats['test_size'] / self.stats['aligned'] * 100:.1f}%)")

        if self.config['stats_report']['print_distribution']:
            print("\n五维情感统计:")
            print("-" * 60)
            for dim in self.config['dimensions']:
                print(f"{dim:6s}: 均值={stats[dim]['mean']:6.3f}, 标准差={stats[dim]['std']:6.3f}")

        if self.config['stats_report']['print_kl_divergence'] and kl_divergences:
            print("\nKL散度（测试集 vs 训练集）:")
            print("-" * 60)
            threshold = self.config['distribution_check']['kl_threshold']
            for dim in self.config['dimensions']:
                kl = kl_divergences[dim]
                status = "✓" if kl < threshold else "✗"
                print(f"{dim:6s}: {kl:.4f} {status}")

        print("\n" + "=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("开始构建数据集...")
    print("=" * 60)

    # 创建数据集构建器并运行
    builder = DatasetBuilder()
    builder.build()

    print("\n" + "=" * 60)
    print("数据集构建完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()