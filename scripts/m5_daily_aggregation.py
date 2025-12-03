"""
M5 日情感聚合脚本
功能：将按条的情感分析结果聚合为按天统计，支持缺失值填充
输入：sentiment_by_comment_int.jsonl (M4-b输出)
输出：sentiment_daily_int.jsonl (按天聚合的情感均值)
作者：AI Assistant
日期：2025-12-01
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
import statistics

# 导入配置
sys.path.append(str(Path(__file__).parent.parent))
from config.m5_config import M5Config


# ==================== 日期工具函数 ====================
def parse_date(date_str: str) -> datetime:
    """
    解析日期字符串为datetime对象

    Args:
        date_str: 日期字符串，格式 'YYYY-MM-DD'

    Returns:
        datetime对象
    """
    return datetime.strptime(date_str, '%Y-%m-%d')


def format_date(dt: datetime) -> str:
    """
    格式化datetime对象为日期字符串

    Args:
        dt: datetime对象

    Returns:
        日期字符串，格式 'YYYY-MM-DD'
    """
    return dt.strftime('%Y-%m-%d')


def date_range(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    生成日期范围（包含首尾）

    Args:
        start_date: 起始日期
        end_date: 结束日期

    Returns:
        日期列表
    """
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


# ==================== 聚合器 ====================
class SentimentAggregator:
    """情感数据日聚合器"""

    def __init__(self):
        """初始化聚合器"""
        self.config = M5Config
        self._setup_logging()
        M5Config.create_dirs()

        # 统计信息
        self.stats = {
            'total_comments': 0,  # 总评论数
            'valid_comments': 0,  # 有效评论数
            'invalid_comments': 0,  # 无效评论数
            'total_days': 0,  # 总天数
            'days_with_comments': 0,  # 有评论的天数
            'days_without_comments': 0,  # 无评论的天数
            'filled_by_moving_avg': 0,  # 移动平均填充的天数
            'filled_by_global_mean': 0,  # 全局均值填充的天数
            'global_means': {},  # 各维度全局均值
            'date_range': None  # 日期范围 (开始, 结束)
        }

    def _setup_logging(self):
        """设置日志系统"""
        M5Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        # 文件处理器
        file_handler = logging.FileHandler(
            M5Config.LOG_FILE,
            encoding='utf-8',
            mode='w'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))

        handlers = [file_handler]

        # 控制台处理器
        if M5Config.LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(log_format, date_format))
            handlers.append(console_handler)

        logging.basicConfig(
            level=getattr(logging, M5Config.LOG_LEVEL),
            format=log_format,
            datefmt=date_format,
            handlers=handlers,
            force=True
        )

        self.logger = logging.getLogger(__name__)

    def load_comments(self) -> Dict[str, List[Dict]]:
        """
        加载评论数据并按日期分组

        Returns:
            {日期字符串: [评论列表]}
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤1：加载评论数据")
        self.logger.info("=" * 80)

        if not M5Config.INPUT_FILE.exists():
            raise FileNotFoundError(f"输入文件不存在: {M5Config.INPUT_FILE}")

        comments_by_date = defaultdict(list)

        self.logger.info(f"✓ 读取文件: {M5Config.INPUT_FILE}")

        with open(M5Config.INPUT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    comment = json.loads(line)
                    self.stats['total_comments'] += 1

                    # 验证必需字段：date
                    if 'date' not in comment or not comment['date']:
                        self.logger.warning(f"行 {line_num}: 缺少或为空的date字段")
                        self.stats['invalid_comments'] += 1
                        continue

                    # 验证情感维度字段
                    missing_dims = []
                    for dim in M5Config.DIMENSIONS:
                        if dim not in comment:
                            missing_dims.append(dim)

                    if missing_dims:
                        self.logger.warning(
                            f"行 {line_num}: 缺少情感维度 {missing_dims}"
                        )
                        self.stats['invalid_comments'] += 1
                        continue

                    # 按日期分组
                    date_str = comment['date']
                    comments_by_date[date_str].append(comment)
                    self.stats['valid_comments'] += 1

                except json.JSONDecodeError as e:
                    self.logger.warning(f"行 {line_num}: JSON解析失败 - {e}")
                    self.stats['invalid_comments'] += 1
                    continue
                except Exception as e:
                    self.logger.warning(f"行 {line_num}: 处理失败 - {e}")
                    self.stats['invalid_comments'] += 1
                    continue

        self.logger.info(f"✓ 总评论数: {self.stats['total_comments']:,}")
        self.logger.info(f"✓ 有效评论: {self.stats['valid_comments']:,}")
        if self.stats['invalid_comments'] > 0:
            self.logger.warning(f"⚠ 无效评论: {self.stats['invalid_comments']:,}")
        self.logger.info(f"✓ 有评论的日期数: {len(comments_by_date):,}")

        # 显示日期范围
        if comments_by_date:
            dates = sorted(comments_by_date.keys())
            self.logger.info(f"✓ 日期范围: {dates[0]} 至 {dates[-1]}")

        return dict(comments_by_date)

    def calculate_daily_means(
            self,
            comments_by_date: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """
        计算每天的情感均值

        Args:
            comments_by_date: {日期: [评论列表]}

        Returns:
            {日期: {维度: 均值, 'n_comments': 数量}}
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤2：计算每日情感均值")
        self.logger.info("=" * 80)

        daily_means = {}

        for date_str, comments in sorted(comments_by_date.items()):
            day_data = {'n_comments': len(comments)}

            # 计算每个维度的均值
            for dim in M5Config.DIMENSIONS:
                scores = [c[dim] for c in comments if dim in c]
                if scores:
                    mean_score = statistics.mean(scores)
                    day_data[dim] = round(mean_score, M5Config.DECIMAL_PLACES)
                else:
                    day_data[dim] = None

            daily_means[date_str] = day_data

        self.logger.info(f"✓ 计算完成: {len(daily_means)} 天")

        # 显示样例
        sample_dates = sorted(daily_means.keys())[:3]
        self.logger.info("✓ 样例数据:")
        for date_str in sample_dates:
            data = daily_means[date_str]
            self.logger.info(
                f"  {date_str}: n={data['n_comments']:3d}, "
                f"景色={data.get('景色', 'N/A'):+6.3f}, "
                f"服务={data.get('服务', 'N/A'):+6.3f}"
            )

        return daily_means

    def calculate_global_means(
            self,
            comments_by_date: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """
        计算全局均值（所有评论的情感均值）

        Args:
            comments_by_date: {日期: [评论列表]}

        Returns:
            {维度: 全局均值}
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤3：计算全局均值")
        self.logger.info("=" * 80)

        all_scores = defaultdict(list)

        # 收集所有评论的各维度分数
        for comments in comments_by_date.values():
            for comment in comments:
                for dim in M5Config.DIMENSIONS:
                    if dim in comment:
                        all_scores[dim].append(comment[dim])

        # 计算均值
        global_means = {}
        for dim in M5Config.DIMENSIONS:
            if all_scores[dim]:
                mean_val = statistics.mean(all_scores[dim])
                global_means[dim] = round(mean_val, M5Config.DECIMAL_PLACES)
            else:
                global_means[dim] = 0.0

        self.stats['global_means'] = global_means

        self.logger.info("✓ 全局均值:")
        for dim in M5Config.DIMENSIONS:
            mean_val = global_means[dim]
            count = len(all_scores[dim])
            self.logger.info(f"  {dim}: {mean_val:+6.3f} (基于 {count:,} 条评论)")

        return global_means

    def fill_missing_days(
            self,
            daily_means: Dict[str, Dict],
            global_means: Dict[str, float]
    ) -> Dict[str, Dict]:
        """
        填充缺失的日期，确保日期连续

        Args:
            daily_means: 已有的每日均值
            global_means: 全局均值

        Returns:
            填充后的每日数据
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤4：填充缺失日期")
        self.logger.info("=" * 80)

        if not daily_means:
            self.logger.warning("没有数据可填充")
            return {}

        # 确定日期范围
        all_dates = sorted(daily_means.keys())
        start_date = parse_date(all_dates[0])
        end_date = parse_date(all_dates[-1])

        self.stats['date_range'] = (format_date(start_date), format_date(end_date))

        self.logger.info(f"✓ 日期范围: {format_date(start_date)} 至 {format_date(end_date)}")

        # 生成完整日期序列
        full_dates = date_range(start_date, end_date)
        self.stats['total_days'] = len(full_dates)

        self.logger.info(f"✓ 总天数: {self.stats['total_days']}")

        # 填充
        filled_daily = {}

        for dt in full_dates:
            date_str = format_date(dt)

            if date_str in daily_means:
                # 有数据的日期
                filled_daily[date_str] = daily_means[date_str]
                self.stats['days_with_comments'] += 1
            else:
                # 缺失的日期
                self.stats['days_without_comments'] += 1

                # 尝试前N天移动平均
                filled_data = self._fill_with_moving_average(
                    dt, filled_daily, M5Config.MOVING_AVERAGE_WINDOW
                )

                if filled_data:
                    filled_data['n_comments'] = 0
                    filled_daily[date_str] = filled_data
                    self.stats['filled_by_moving_avg'] += 1
                elif M5Config.USE_GLOBAL_MEAN_FALLBACK:
                    # 使用全局均值
                    filled_data = global_means.copy()
                    filled_data['n_comments'] = 0
                    filled_daily[date_str] = filled_data
                    self.stats['filled_by_global_mean'] += 1
                else:
                    # 不填充，跳过
                    self.logger.warning(f"⚠ 日期 {date_str} 无法填充（移动平均不足且未启用全局均值）")

        self.logger.info(f"✓ 有评论天数: {self.stats['days_with_comments']:,}")
        self.logger.info(f"✓ 无评论天数: {self.stats['days_without_comments']:,}")
        if self.stats['filled_by_moving_avg'] > 0:
            self.logger.info(f"  - 移动平均填充: {self.stats['filled_by_moving_avg']:,} 天")
        if self.stats['filled_by_global_mean'] > 0:
            self.logger.info(f"  - 全局均值填充: {self.stats['filled_by_global_mean']:,} 天")

        return filled_daily

    def _fill_with_moving_average(
            self,
            target_date: datetime,
            filled_data: Dict[str, Dict],
            window: int
    ) -> Optional[Dict[str, float]]:
        """
        使用前N天的移动平均填充

        Args:
            target_date: 目标日期
            filled_data: 已填充的数据
            window: 窗口大小（天数）

        Returns:
            填充的数据字典，如果无法填充返回None
        """
        # 获取前window天的数据（只使用有实际评论的天数）
        past_data = []
        for i in range(1, window + 1):
            past_date = target_date - timedelta(days=i)
            past_date_str = format_date(past_date)

            if past_date_str in filled_data:
                day_data = filled_data[past_date_str]
                # 只使用有实际评论的数据
                if day_data.get('n_comments', 0) > 0:
                    past_data.append(day_data)

        # 至少需要1天数据
        if not past_data:
            return None

        # 计算均值
        filled = {}
        for dim in M5Config.DIMENSIONS:
            values = [d[dim] for d in past_data if dim in d and d[dim] is not None]
            if values:
                filled[dim] = round(statistics.mean(values), M5Config.DECIMAL_PLACES)
            else:
                # 某个维度无数据，放弃填充
                return None

        return filled

    def save_results(self, daily_data: Dict[str, Dict]):
        """
        保存结果到文件

        Args:
            daily_data: 每日聚合数据
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤5：保存结果")
        self.logger.info("=" * 80)

        # 按日期排序
        sorted_dates = sorted(daily_data.keys())

        with open(M5Config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for date_str in sorted_dates:
                data = daily_data[date_str]

                # 构建输出记录（保持字段顺序）
                record = {'date': date_str}

                # 添加情感维度（按配置顺序）
                for dim in M5Config.DIMENSIONS:
                    record[dim] = data[dim]

                # 添加评论数
                record['n_comments'] = data['n_comments']

                # 写入JSON行
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        self.logger.info(f"✓ 结果已保存: {M5Config.OUTPUT_FILE}")
        self.logger.info(f"  总天数: {len(sorted_dates):,}")

        # 显示样例
        self.logger.info("\n样例数据（前3天）:")
        for date_str in sorted_dates[:3]:
            data = daily_data[date_str]
            record = {'date': date_str}
            for dim in M5Config.DIMENSIONS:
                record[dim] = data[dim]
            record['n_comments'] = data['n_comments']
            self.logger.info(f"  {json.dumps(record, ensure_ascii=False)}")

    def save_summary_log(self):
        """保存汇总日志"""
        summary = f"""
{'=' * 80}
M5 日情感聚合汇总报告
{'=' * 80}

【输入统计】
总评论数: {self.stats['total_comments']:,}
有效评论: {self.stats['valid_comments']:,}
无效评论: {self.stats['invalid_comments']:,}
有效率: {self.stats['valid_comments'] / self.stats['total_comments'] * 100:.2f}%

【日期统计】
日期范围: {self.stats['date_range'][0]} 至 {self.stats['date_range'][1]}
总天数: {self.stats['total_days']:,}
有评论天数: {self.stats['days_with_comments']:,} ({self.stats['days_with_comments'] / self.stats['total_days'] * 100:.2f}%)
无评论天数: {self.stats['days_without_comments']:,} ({self.stats['days_without_comments'] / self.stats['total_days'] * 100:.2f}%)

【填充统计】
移动平均填充: {self.stats['filled_by_moving_avg']:,} 天
全局均值填充: {self.stats['filled_by_global_mean']:,} 天
填充总计: {self.stats['filled_by_moving_avg'] + self.stats['filled_by_global_mean']:,} 天

【全局均值】（基于所有 {self.stats['valid_comments']:,} 条有效评论）
"""
        for dim in M5Config.DIMENSIONS:
            mean_val = self.stats['global_means'].get(dim, 0)
            summary += f"{dim}: {mean_val:+7.3f}\n"

        summary += f"""
【配置信息】
输入文件: {M5Config.INPUT_FILE}
输出文件: {M5Config.OUTPUT_FILE}
情感维度: {', '.join(M5Config.DIMENSIONS)}
移动平均窗口: {M5Config.MOVING_AVERAGE_WINDOW} 天
全局均值填充: {'启用' if M5Config.USE_GLOBAL_MEAN_FALLBACK else '禁用'}
保留小数位: {M5Config.DECIMAL_PLACES}

{'=' * 80}
"""

        # 写入日志文件
        with open(M5Config.LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(summary)

        # 打印到控制台
        print(summary)

    def run(self):
        """执行完整聚合流程"""
        try:
            # 打印配置
            M5Config.print_config()

            # 1. 加载数据
            comments_by_date = self.load_comments()

            if not comments_by_date:
                self.logger.error("没有有效数据，退出")
                return

            # 2. 计算每日均值
            daily_means = self.calculate_daily_means(comments_by_date)

            # 3. 计算全局均值
            global_means = self.calculate_global_means(comments_by_date)

            # 4. 填充缺失日期
            filled_daily = self.fill_missing_days(daily_means, global_means)

            # 5. 保存结果
            if filled_daily:
                self.save_results(filled_daily)
                self.save_summary_log()
            else:
                self.logger.error("没有数据可保存")

            # 6. 最终提示
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✓ M5 日情感聚合完成！")
            self.logger.info("=" * 80)
            self.logger.info(f"输出文件: {M5Config.OUTPUT_FILE}")
            self.logger.info(f"日志文件: {M5Config.LOG_FILE}")

        except Exception as e:
            self.logger.error(f"\n✗ 聚合失败: {str(e)}", exc_info=True)
            raise


def main():
    """主函数"""
    try:
        aggregator = SentimentAggregator()
        aggregator.run()
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n程序异常退出: {str(e)}")
        raise


if __name__ == "__main__":
    main()