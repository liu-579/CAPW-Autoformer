"""
M5 日情感聚合脚本（MySQL版） - 贝叶斯平滑升级版 (非零过滤)
功能：从MySQL读取情感分析结果，应用分维度动态贝叶斯平滑算法进行聚合
      * 修正逻辑：只有分数 != 0 才视为有效提及
日期：2025-12-24
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
import statistics
import math

# 引入数据科学库
import pandas as pd
import numpy as np
import pymysql
from pymysql.cursors import DictCursor

# 导入配置
sys.path.append(str(Path(__file__).parent.parent))
from config.m5_sql_config import M5Config


# ==================== 日期工具函数 (保持不变) ====================
def parse_date(date_obj) -> datetime:
    """解析日期对象为datetime"""
    if isinstance(date_obj, datetime):
        return date_obj
    elif isinstance(date_obj, str):
        for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S']:
            try:
                return datetime.strptime(date_obj, fmt)
            except ValueError:
                continue
        raise ValueError(f"无法解析日期: {date_obj}")
    else:
        raise TypeError(f"不支持的日期类型: {type(date_obj)}")


def format_date(dt: datetime) -> str:
    """格式化datetime对象为日期字符串"""
    return dt.strftime('%Y-%m-%d')


def date_range(start_date: datetime, end_date: datetime) -> List[datetime]:
    """生成日期范围（包含首尾）"""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


# ==================== 聚合器 ====================
class SentimentAggregator:
    """情感数据日聚合器（MySQL版 - 贝叶斯平滑）"""

    def __init__(self):
        """初始化聚合器"""
        self.config = M5Config
        self._setup_logging()
        M5Config.create_dirs()

        # 统计信息
        self.stats = {
            scenic: {
                'total_records': 0,
                'valid_records': 0,
                'invalid_records': 0,
                'total_days': 0,
                'days_with_data': 0,
                'days_without_data': 0,
                'filled_by_moving_avg': 0,
                'filled_by_global_mean': 0,
                'global_means': {},
                'date_range': None,
                'k_values': {}
            }
            for scenic in M5Config.SCENIC_NAMES
        }

    def _setup_logging(self):
        """设置日志系统"""
        M5Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        file_handler = logging.FileHandler(M5Config.LOG_FILE, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))

        handlers = [file_handler]
        if M5Config.LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(log_format, date_format))
            handlers.append(console_handler)

        logging.basicConfig(level=getattr(logging, M5Config.LOG_LEVEL), format=log_format, datefmt=date_format, handlers=handlers, force=True)
        self.logger = logging.getLogger(__name__)

    def load_data_from_db(self) -> Dict[str, Dict[str, List[Dict]]]:
        """从数据库加载数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤1：从MySQL加载数据")
        self.logger.info("=" * 80)

        db_conf = M5Config.DATABASE_CONFIG
        try:
            conn = pymysql.connect(host=db_conf['host'], port=db_conf['port'], user=db_conf['user'], password=db_conf['password'], database=db_conf['db'], charset=db_conf['charset'], cursorclass=DictCursor)
            with conn:
                with conn.cursor() as cursor:
                    fields = ', '.join([f"`{f}`" for f in M5Config.SOURCE_FIELDS])
                    scenic_filter = ', '.join([f"'{s}'" for s in M5Config.SCENIC_NAMES])
                    sql = f"""
                        SELECT {fields} FROM `{M5Config.INPUT_TABLE}`
                        WHERE `{M5Config.SCENIC_FIELD}` IN ({scenic_filter})
                        AND `{M5Config.DATE_FIELD}` IS NOT NULL
                        ORDER BY `{M5Config.SCENIC_FIELD}`, `{M5Config.DATE_FIELD}`
                    """
                    self.logger.info(f"执行SQL查询...")
                    cursor.execute(sql)
                    raw_data = list(cursor.fetchall())
                    self.logger.info(f"✓ 读取记录数: {len(raw_data):,}")
        except pymysql.MySQLError as e:
            self.logger.error(f"数据库读取失败: {e}")
            raise

        data_by_scenic = {scenic: defaultdict(list) for scenic in M5Config.SCENIC_NAMES}

        for record in raw_data:
            scenic = record.get(M5Config.SCENIC_FIELD)
            if scenic not in M5Config.SCENIC_NAMES: continue
            self.stats[scenic]['total_records'] += 1

            try:
                date_obj = record.get(M5Config.DATE_FIELD)
                if not date_obj:
                    self.stats[scenic]['invalid_records'] += 1
                    continue

                date_str = format_date(parse_date(date_obj))

                # 提取分数
                scores = {}
                valid = True
                for dim, field in M5Config.SCORE_FIELD_MAPPING.items():
                    value = record.get(field)
                    if value is None:
                        valid = False
                        break
                    scores[dim] = float(value)

                if not valid:
                    self.stats[scenic]['invalid_records'] += 1
                    continue

                data_by_scenic[scenic][date_str].append(scores)
                self.stats[scenic]['valid_records'] += 1

            except Exception as e:
                self.stats[scenic]['invalid_records'] += 1
                continue

        return data_by_scenic

    def calculate_daily_means(
            self,
            data_by_date: Dict[str, List[Dict]],
            scenic_name: str
    ) -> Dict[str, Dict]:
        """
        计算每天的情感得分 —— 使用动态贝叶斯平滑
        修正逻辑：'没提到' = 0，因此计算提及次数和均值时，必须剔除 0 值。
        """
        if not data_by_date:
            return {}

        # 数据预处理
        all_records = []
        for date_str, records in data_by_date.items():
            for r in records:
                r_flat = r.copy()
                r_flat['date'] = date_str
                all_records.append(r_flat)

        df = pd.DataFrame(all_records)

        # ---------------------------------------------------------
        # 步骤 1：计算动态 K 值 (基于非0值的分布)
        # ---------------------------------------------------------
        k_dict = {}
        self.logger.info(f"  正在计算【{scenic_name}】的动态平滑系数 K (分位数: {M5Config.BAYESIAN_QUANTILE})...")

        for dim in M5Config.DIMENSIONS:
            if dim not in df.columns:
                k_dict[dim] = 1
                continue

            # === 修改核心：只统计分数不为0的记录 ===
            # 原逻辑：df[df[dim].notna()]
            # 新逻辑：df[df[dim] != 0]
            valid_mentions = df[df[dim] != 0]

            daily_counts = valid_mentions.groupby('date').size()

            if daily_counts.empty:
                k_dict[dim] = 1
                continue

            q_val = daily_counts.quantile(M5Config.BAYESIAN_QUANTILE)
            k_val = max(1, int(np.floor(q_val)))
            k_dict[dim] = k_val

        self.logger.info(f"  > 动态 K 值字典 (已剔除0分): {k_dict}")
        self.stats[scenic_name]['k_values'] = k_dict

        # ---------------------------------------------------------
        # 步骤 2 & 3：执行聚合与应用平滑公式
        # ---------------------------------------------------------
        daily_results = {}
        grouped = df.groupby('date')

        for date_str, group in grouped:
            day_stats = {'n_records': len(group)} # 这是总评论数，仅作参考

            for dim in M5Config.DIMENSIONS:
                if dim not in group.columns:
                    day_stats[dim] = 0.0
                    continue

                # === 修改核心：只获取非0的有效分数 ===
                # 如果某天全是0分，valid_scores 为空
                valid_scores = group[group[dim] != 0][dim]

                sum_score = valid_scores.sum()
                count = len(valid_scores) # 有效提及次数

                k = k_dict.get(dim, 1)

                # 贝叶斯平滑公式：Score = Sum / (Count + K)
                # 这里的 Count 也是剔除了0分后的数量
                smoothed_score = sum_score / (count + k)
                day_stats[dim] = round(smoothed_score, M5Config.DECIMAL_PLACES)

            daily_results[date_str] = day_stats

        return daily_results

    def calculate_global_means(self, data_by_date: Dict[str, List[Dict]]) -> Dict[str, float]:
        """计算全局均值 (修正：同样需要剔除0值)"""
        all_scores = defaultdict(list)
        for records in data_by_date.values():
            for record in records:
                for dim in M5Config.DIMENSIONS:
                    # 只有当维度存在且不为0时才计入全局均值
                    if dim in record and record[dim] != 0:
                        all_scores[dim].append(record[dim])

        global_means = {}
        for dim in M5Config.DIMENSIONS:
            if all_scores[dim]:
                mean_val = statistics.mean(all_scores[dim])
                global_means[dim] = round(mean_val, M5Config.DECIMAL_PLACES)
            else:
                global_means[dim] = 0.0
        return global_means

    def fill_missing_days(self, daily_means: Dict[str, Dict], global_means: Dict[str, float], scenic_name: str) -> Dict[str, Dict]:
        """填充缺失的日期 (保持原逻辑)"""
        if not daily_means: return {}

        dates = sorted(daily_means.keys())
        start_date, end_date = parse_date(dates[0]), parse_date(dates[-1])
        self.stats[scenic_name]['date_range'] = (format_date(start_date), format_date(end_date))

        full_dates = date_range(start_date, end_date)
        self.stats[scenic_name]['total_days'] = len(full_dates)

        filled_daily = {}

        for dt in full_dates:
            date_str = format_date(dt)
            if date_str in daily_means:
                filled_daily[date_str] = daily_means[date_str]
                self.stats[scenic_name]['days_with_data'] += 1
            else:
                self.stats[scenic_name]['days_without_data'] += 1
                filled_data = self._fill_with_moving_average(dt, filled_daily, M5Config.MOVING_AVERAGE_WINDOW)

                if filled_data:
                    filled_data['n_records'] = 0
                    filled_daily[date_str] = filled_data
                    self.stats[scenic_name]['filled_by_moving_avg'] += 1
                elif M5Config.USE_GLOBAL_MEAN_FALLBACK:
                    filled_data = global_means.copy()
                    filled_data['n_records'] = 0
                    filled_daily[date_str] = filled_data
                    self.stats[scenic_name]['filled_by_global_mean'] += 1

        return filled_daily

    def _fill_with_moving_average(self, target_date: datetime, filled_data: Dict[str, Dict], window: int) -> Optional[Dict[str, float]]:
        """使用前N天的移动平均填充 (保持原逻辑)"""
        past_data = []
        for i in range(1, window + 1):
            past_date_str = format_date(target_date - timedelta(days=i))
            if past_date_str in filled_data:
                day_data = filled_data[past_date_str]
                # 只有当该天是真实数据(n_records>0)时才用于计算均值
                if day_data.get('n_records', 0) > 0:
                    past_data.append(day_data)

        if not past_data: return None

        filled = {}
        for dim in M5Config.DIMENSIONS:
            # 移动平均计算时，也应只考虑非None值（这里已经是聚合后的分数，通常不为0，除非算出来正好是0）
            values = [d[dim] for d in past_data if dim in d and d[dim] is not None]
            if values:
                filled[dim] = round(statistics.mean(values), M5Config.DECIMAL_PLACES)
            else:
                return None
        return filled

    def create_output_table(self):
        """创建输出表 (保持原逻辑)"""
        self.logger.info("\n创建输出表...")
        db_conf = M5Config.DATABASE_CONFIG
        try:
            conn = pymysql.connect(host=db_conf['host'], port=db_conf['port'], user=db_conf['user'], password=db_conf['password'], database=db_conf['db'], charset=db_conf['charset'])
            with conn:
                with conn.cursor() as cursor:
                    sql = f"""
                        CREATE TABLE IF NOT EXISTS `{M5Config.OUTPUT_TABLE}` (
                            `id` INT AUTO_INCREMENT PRIMARY KEY,
                            `scenic_name` VARCHAR(100) NOT NULL,
                            `date` DATE NOT NULL,
                            `scenery_score` FLOAT(6,3) DEFAULT 0.000,
                            `transportation_score` FLOAT(6,3) DEFAULT 0.000,
                            `food_score` FLOAT(6,3) DEFAULT 0.000,
                            `consumption_score` FLOAT(6,3) DEFAULT 0.000,
                            `service_score` FLOAT(6,3) DEFAULT 0.000,
                            `n_records` INT DEFAULT 0,
                            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE KEY `unique_scenic_date` (`scenic_name`, `date`),
                            INDEX `idx_scenic` (`scenic_name`),
                            INDEX `idx_date` (`date`)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='日度情感聚合数据(贝叶斯平滑)'
                    """
                    cursor.execute(sql)
                    conn.commit()
                    self.logger.info(f"✓ 表 `{M5Config.OUTPUT_TABLE}` 已准备就绪")
        except pymysql.MySQLError as e:
            self.logger.error(f"创建表失败: {e}")
            raise

    def save_to_database(self, scenic_name: str, daily_data: Dict[str, Dict]):
        """保存结果到数据库 (保持原逻辑)"""
        if not daily_data: return
        self.logger.info(f"保存【{scenic_name}】数据到数据库...")
        db_conf = M5Config.DATABASE_CONFIG
        try:
            conn = pymysql.connect(host=db_conf['host'], port=db_conf['port'], user=db_conf['user'], password=db_conf['password'], database=db_conf['db'], charset=db_conf['charset'])
            with conn:
                with conn.cursor() as cursor:
                    sql = f"""
                        REPLACE INTO `{M5Config.OUTPUT_TABLE}`
                        (scenic_name, date, scenery_score, transportation_score, food_score, consumption_score, service_score, n_records)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    batch_data = []
                    for date_str in sorted(daily_data.keys()):
                        data = daily_data[date_str]
                        row = (scenic_name, date_str, data.get('景色', 0), data.get('交通', 0), data.get('美食', 0), data.get('消费', 0), data.get('服务', 0), data.get('n_records', 0))
                        batch_data.append(row)
                    cursor.executemany(sql, batch_data)
                    conn.commit()
                    self.logger.info(f"✓ 【{scenic_name}】保存完成: {len(batch_data):,} 条记录")
        except pymysql.MySQLError as e:
            self.logger.error(f"保存失败: {e}")
            raise

    def process_scenic(self, scenic_name: str, data_by_date: Dict[str, List[Dict]]):
        """处理单个景区的数据"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"处理景区: 【{scenic_name}】")
        self.logger.info("=" * 80)

        if not data_by_date:
            self.logger.warning(f"【{scenic_name}】无数据")
            return

        # 1. 计算每日均值 (贝叶斯平滑)
        self.logger.info("步骤2: 执行动态贝叶斯平滑聚合...")
        daily_means = self.calculate_daily_means(data_by_date, scenic_name)
        self.logger.info(f"✓ 聚合完成: {len(daily_means)} 天")

        # 2. 计算全局均值 (同时修正了全局均值计算，也剔除了0分)
        self.logger.info("步骤3: 计算全局均值 (用于填充)...")
        global_means = self.calculate_global_means(data_by_date)
        self.stats[scenic_name]['global_means'] = global_means

        # 3. 填充缺失日期
        self.logger.info("步骤4: 填充缺失日期...")
        filled_daily = self.fill_missing_days(daily_means, global_means, scenic_name)

        stats = self.stats[scenic_name]
        self.logger.info(f"✓ 填充完成: 总天数 {stats['total_days']:,}")

        # 4. 保存到数据库
        self.save_to_database(scenic_name, filled_daily)

    def save_k_values_json(self):
        """保存所有景区的K值到JSON"""
        k_data = {
            scenic: self.stats[scenic].get('k_values', {})
            for scenic in M5Config.SCENIC_NAMES
        }
        try:
            with open(M5Config.K_VALUES_FILE, 'w', encoding='utf-8') as f:
                json.dump(k_data, f, ensure_ascii=False, indent=4)
            self.logger.info(f"\n✓ K值计算结果已保存: {M5Config.K_VALUES_FILE}")
        except Exception as e:
            self.logger.error(f"保存K值JSON失败: {e}")

    def save_summary_log(self):
        """保存汇总日志"""
        summary = f"\n{'=' * 80}\nM5 日情感聚合汇总报告（贝叶斯平滑版）\n{'=' * 80}\n\n"
        for scenic in M5Config.SCENIC_NAMES:
            stats = self.stats[scenic]
            summary += f"【{scenic}】\n"
            summary += f"  K值参数: {stats.get('k_values', 'N/A')}\n"
            summary += f"  记录统计: 总 {stats['total_records']:,} | 有效 {stats['valid_records']:,}\n"
            summary += f"  日期覆盖: {stats['days_with_data']:,} 天有数据 / {stats['total_days']:,} 总天数\n\n"

        with open(M5Config.LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(summary)
        print(summary)

    def run(self):
        """执行完整聚合流程"""
        try:
            M5Config.print_config()
            M5Config.validate()
            data_by_scenic = self.load_data_from_db()
            self.create_output_table()
            for scenic_name in M5Config.SCENIC_NAMES:
                self.process_scenic(scenic_name, data_by_scenic[scenic_name])

            self.save_k_values_json()
            self.save_summary_log()
            self.logger.info("\n✓ M5 日情感聚合完成！")
        except Exception as e:
            self.logger.error(f"\n✗ 聚合失败: {str(e)}", exc_info=True)
            raise


def main():
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