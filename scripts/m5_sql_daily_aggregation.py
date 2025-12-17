"""
M5 日情感聚合脚本（MySQL版）
功能：从MySQL读取情感分析结果，按景区和日期聚合，支持缺失值填充
日期：2025-12-15
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
import statistics

import pymysql
from pymysql.cursors import DictCursor
from tqdm import tqdm

# 导入配置
sys.path.append(str(Path(__file__).parent.parent))
from config.m5_sql_config import M5Config


# ==================== 日期工具函数 ====================
def parse_date(date_obj) -> datetime:
    """
    解析日期对象为datetime

    Args:
        date_obj: datetime对象或日期字符串

    Returns:
        datetime对象
    """
    if isinstance(date_obj, datetime):
        return date_obj
    elif isinstance(date_obj, str):
        # 尝试多种格式
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
    """情感数据日聚合器（MySQL版）"""

    def __init__(self):
        """初始化聚合器"""
        self.config = M5Config
        self._setup_logging()
        M5Config.create_dirs()

        # 统计信息（按景区）
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
                'date_range': None
            }
            for scenic in M5Config.SCENIC_NAMES
        }

    def _setup_logging(self):
        """设置日志系统"""
        M5Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        file_handler = logging.FileHandler(
            M5Config.LOG_FILE,
            encoding='utf-8',
            mode='w'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))

        handlers = [file_handler]

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

    def load_data_from_db(self) -> Dict[str, Dict[str, List[Dict]]]:
        """
        从数据库加载数据并按景区和日期分组

        Returns:
            {景区名: {日期字符串: [记录列表]}}
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤1：从MySQL加载数据")
        self.logger.info("=" * 80)

        db_conf = M5Config.DATABASE_CONFIG

        try:
            conn = pymysql.connect(
                host=db_conf['host'],
                port=db_conf['port'],
                user=db_conf['user'],
                password=db_conf['password'],
                database=db_conf['db'],
                charset=db_conf['charset'],
                cursorclass=DictCursor
            )

            with conn:
                with conn.cursor() as cursor:
                    # 构建SQL查询
                    fields = ', '.join([f"`{f}`" for f in M5Config.SOURCE_FIELDS])
                    scenic_filter = ', '.join([f"'{s}'" for s in M5Config.SCENIC_NAMES])

                    sql = f"""
                        SELECT {fields}
                        FROM `{M5Config.INPUT_TABLE}`
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

        # 按景区和日期分组
        data_by_scenic = {scenic: defaultdict(list) for scenic in M5Config.SCENIC_NAMES}

        for record in raw_data:
            scenic = record.get(M5Config.SCENIC_FIELD)

            if scenic not in M5Config.SCENIC_NAMES:
                self.logger.warning(f"未知景区: {scenic}")
                continue

            self.stats[scenic]['total_records'] += 1

            # 验证日期
            try:
                date_obj = record.get(M5Config.DATE_FIELD)
                if not date_obj:
                    self.stats[scenic]['invalid_records'] += 1
                    continue

                date_dt = parse_date(date_obj)
                date_str = format_date(date_dt)

                # 验证情感分数
                scores = {}
                valid = True
                for dim, field in M5Config.SCORE_FIELD_MAPPING.items():
                    value = record.get(field)
                    if value is None:
                        self.logger.warning(f"记录缺少字段 {field}: {record.get('id')}")
                        valid = False
                        break
                    scores[dim] = float(value)

                if not valid:
                    self.stats[scenic]['invalid_records'] += 1
                    continue

                # 添加到分组
                data_by_scenic[scenic][date_str].append(scores)
                self.stats[scenic]['valid_records'] += 1

            except Exception as e:
                self.logger.warning(f"记录处理失败: {e}")
                self.stats[scenic]['invalid_records'] += 1
                continue

        # 输出统计
        self.logger.info("\n✓ 数据加载完成:")
        for scenic in M5Config.SCENIC_NAMES:
            stats = self.stats[scenic]
            self.logger.info(f"\n  【{scenic}】")
            self.logger.info(f"    总记录: {stats['total_records']:,}")
            self.logger.info(f"    有效记录: {stats['valid_records']:,}")
            if stats['invalid_records'] > 0:
                self.logger.info(f"    无效记录: {stats['invalid_records']:,}")

            scenic_data = data_by_scenic[scenic]
            if scenic_data:
                dates = sorted(scenic_data.keys())
                self.logger.info(f"    有数据天数: {len(dates):,}")
                self.logger.info(f"    日期范围: {dates[0]} 至 {dates[-1]}")

        return data_by_scenic

    def calculate_daily_means(
            self,
            data_by_date: Dict[str, List[Dict]]
    ) -> Dict[str, Dict]:
        """
        计算每天的情感均值

        Args:
            data_by_date: {日期: [评分字典列表]}

        Returns:
            {日期: {维度: 均值, 'n_records': 数量}}
        """
        daily_means = {}

        for date_str, records in sorted(data_by_date.items()):
            day_data = {'n_records': len(records)}

            for dim in M5Config.DIMENSIONS:
                scores = [r[dim] for r in records if dim in r]
                if scores:
                    mean_score = statistics.mean(scores)
                    day_data[dim] = round(mean_score, M5Config.DECIMAL_PLACES)
                else:
                    day_data[dim] = None

            daily_means[date_str] = day_data

        return daily_means

    def calculate_global_means(
            self,
            data_by_date: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """
        计算全局均值

        Args:
            data_by_date: {日期: [评分字典列表]}

        Returns:
            {维度: 全局均值}
        """
        all_scores = defaultdict(list)

        for records in data_by_date.values():
            for record in records:
                for dim in M5Config.DIMENSIONS:
                    if dim in record:
                        all_scores[dim].append(record[dim])

        global_means = {}
        for dim in M5Config.DIMENSIONS:
            if all_scores[dim]:
                mean_val = statistics.mean(all_scores[dim])
                global_means[dim] = round(mean_val, M5Config.DECIMAL_PLACES)
            else:
                global_means[dim] = 0.0

        return global_means

    def fill_missing_days(
            self,
            daily_means: Dict[str, Dict],
            global_means: Dict[str, float],
            scenic_name: str
    ) -> Dict[str, Dict]:
        """
        填充缺失的日期

        Args:
            daily_means: 已有的每日均值
            global_means: 全局均值
            scenic_name: 景区名称（用于统计）

        Returns:
            填充后的每日数据
        """
        if not daily_means:
            return {}

        # 确定日期范围
        all_dates = sorted(daily_means.keys())
        start_date = parse_date(all_dates[0])
        end_date = parse_date(all_dates[-1])

        self.stats[scenic_name]['date_range'] = (format_date(start_date), format_date(end_date))

        # 生成完整日期序列
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

                # 尝试移动平均
                filled_data = self._fill_with_moving_average(
                    dt, filled_daily, M5Config.MOVING_AVERAGE_WINDOW
                )

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

    def _fill_with_moving_average(
            self,
            target_date: datetime,
            filled_data: Dict[str, Dict],
            window: int
    ) -> Optional[Dict[str, float]]:
        """使用前N天的移动平均填充"""
        past_data = []

        for i in range(1, window + 1):
            past_date = target_date - timedelta(days=i)
            past_date_str = format_date(past_date)

            if past_date_str in filled_data:
                day_data = filled_data[past_date_str]
                if day_data.get('n_records', 0) > 0:
                    past_data.append(day_data)

        if not past_data:
            return None

        filled = {}
        for dim in M5Config.DIMENSIONS:
            values = [d[dim] for d in past_data if dim in d and d[dim] is not None]
            if values:
                filled[dim] = round(statistics.mean(values), M5Config.DECIMAL_PLACES)
            else:
                return None

        return filled

    def create_output_table(self):
        """创建输出表"""
        self.logger.info("\n创建输出表...")

        db_conf = M5Config.DATABASE_CONFIG

        try:
            conn = pymysql.connect(
                host=db_conf['host'],
                port=db_conf['port'],
                user=db_conf['user'],
                password=db_conf['password'],
                database=db_conf['db'],
                charset=db_conf['charset']
            )

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
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='日度情感聚合数据'
                    """
                    cursor.execute(sql)
                    conn.commit()
                    self.logger.info(f"✓ 表 `{M5Config.OUTPUT_TABLE}` 已准备就绪")

        except pymysql.MySQLError as e:
            self.logger.error(f"创建表失败: {e}")
            raise

    def save_to_database(self, scenic_name: str, daily_data: Dict[str, Dict]):
        """
        保存结果到数据库

        Args:
            scenic_name: 景区名称
            daily_data: 每日聚合数据
        """
        if not daily_data:
            self.logger.warning(f"【{scenic_name}】没有数据需要保存")
            return

        self.logger.info(f"\n保存【{scenic_name}】数据到数据库...")

        db_conf = M5Config.DATABASE_CONFIG

        try:
            conn = pymysql.connect(
                host=db_conf['host'],
                port=db_conf['port'],
                user=db_conf['user'],
                password=db_conf['password'],
                database=db_conf['db'],
                charset=db_conf['charset']
            )

            with conn:
                with conn.cursor() as cursor:
                    sql = f"""
                        REPLACE INTO `{M5Config.OUTPUT_TABLE}`
                        (scenic_name, date, scenery_score, transportation_score,
                         food_score, consumption_score, service_score, n_records)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """

                    batch_data = []
                    for date_str in sorted(daily_data.keys()):
                        data = daily_data[date_str]
                        row = (
                            scenic_name,
                            date_str,
                            data.get('景色', 0),
                            data.get('交通', 0),
                            data.get('美食', 0),
                            data.get('消费', 0),
                            data.get('服务', 0),
                            data.get('n_records', 0)
                        )
                        batch_data.append(row)

                    # 批量插入
                    cursor.executemany(sql, batch_data)
                    conn.commit()

                    self.logger.info(f"✓ 【{scenic_name}】保存完成: {len(batch_data):,} 条记录")

        except pymysql.MySQLError as e:
            self.logger.error(f"保存失败: {e}")
            raise

    def process_scenic(self, scenic_name: str, data_by_date: Dict[str, List[Dict]]):
        """
        处理单个景区的数据

        Args:
            scenic_name: 景区名称
            data_by_date: 该景区的按日期分组数据
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"处理景区: 【{scenic_name}】")
        self.logger.info("=" * 80)

        if not data_by_date:
            self.logger.warning(f"【{scenic_name}】无数据")
            return

        # 1. 计算每日均值
        self.logger.info("步骤2: 计算每日均值...")
        daily_means = self.calculate_daily_means(data_by_date)
        self.logger.info(f"✓ 计算完成: {len(daily_means)} 天")

        # 2. 计算全局均值
        self.logger.info("步骤3: 计算全局均值...")
        global_means = self.calculate_global_means(data_by_date)
        self.stats[scenic_name]['global_means'] = global_means

        self.logger.info("✓ 全局均值:")
        for dim in M5Config.DIMENSIONS:
            self.logger.info(f"  {dim}: {global_means[dim]:+7.3f}")

        # 3. 填充缺失日期
        self.logger.info("步骤4: 填充缺失日期...")
        filled_daily = self.fill_missing_days(daily_means, global_means, scenic_name)

        stats = self.stats[scenic_name]
        self.logger.info(f"✓ 填充完成:")
        self.logger.info(f"  总天数: {stats['total_days']:,}")
        self.logger.info(f"  有数据天数: {stats['days_with_data']:,}")
        self.logger.info(f"  无数据天数: {stats['days_without_data']:,}")
        if stats['filled_by_moving_avg'] > 0:
            self.logger.info(f"  移动平均填充: {stats['filled_by_moving_avg']:,} 天")
        if stats['filled_by_global_mean'] > 0:
            self.logger.info(f"  全局均值填充: {stats['filled_by_global_mean']:,} 天")

        # 4. 保存到数据库
        self.save_to_database(scenic_name, filled_daily)

    def save_summary_log(self):
        """保存汇总日志"""
        summary = f"""
{'=' * 80}
M5 日情感聚合汇总报告（MySQL版）
{'=' * 80}

"""
        for scenic in M5Config.SCENIC_NAMES:
            stats = self.stats[scenic]
            summary += f"""
【{scenic}】

输入统计:
  总记录数: {stats['total_records']:,}
  有效记录: {stats['valid_records']:,}
  无效记录: {stats['invalid_records']:,}
  有效率: {stats['valid_records'] / max(stats['total_records'], 1) * 100:.2f}%

日期统计:
  日期范围: {stats['date_range'][0] if stats['date_range'] else 'N/A'} 至 {stats['date_range'][1] if stats['date_range'] else 'N/A'}
  总天数: {stats['total_days']:,}
  有数据天数: {stats['days_with_data']:,} ({stats['days_with_data'] / max(stats['total_days'], 1) * 100:.2f}%)
  无数据天数: {stats['days_without_data']:,} ({stats['days_without_data'] / max(stats['total_days'], 1) * 100:.2f}%)

填充统计:
  移动平均填充: {stats['filled_by_moving_avg']:,} 天
  全局均值填充: {stats['filled_by_global_mean']:,} 天

全局均值:
"""
            for dim in M5Config.DIMENSIONS:
                mean_val = stats['global_means'].get(dim, 0)
                summary += f"  {dim}: {mean_val:+7.3f}\n"

        summary += f"""
{'=' * 80}
配置信息:
  数据库: {M5Config.DATABASE_CONFIG['host']}:{M5Config.DATABASE_CONFIG['port']}/{M5Config.DATABASE_CONFIG['db']}
  输入表: {M5Config.INPUT_TABLE}
  输出表: {M5Config.OUTPUT_TABLE}
  景区: {', '.join(M5Config.SCENIC_NAMES)}
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
            M5Config.validate()

            # 1. 加载数据
            data_by_scenic = self.load_data_from_db()

            # 2. 创建输出表
            self.create_output_table()

            # 3. 按景区处理
            for scenic_name in M5Config.SCENIC_NAMES:
                self.process_scenic(scenic_name, data_by_scenic[scenic_name])

            # 4. 保存汇总日志
            self.save_summary_log()

            # 5. 最终提示
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✓ M5 日情感聚合完成！")
            self.logger.info("=" * 80)
            self.logger.info(f"输出表: {M5Config.OUTPUT_TABLE}")
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