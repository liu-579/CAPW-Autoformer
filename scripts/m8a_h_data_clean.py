# -*- coding: utf-8 -*-
"""
M8A 数据清洗主程序（小时尺度版本）
ETL Engineer: Chinese Text Data Cleaning Pipeline
功能: 处理中文文本脏数据,执行强力清洗,输出可读数值化宽表
修改: 添加淡旺季特征支持（从m7c季节分析结果读取）
"""

import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
import datetime
import json
# 新增: 引入中国节假日库
import chinese_calendar

# 导入配置文件
import config.m8a_h_config as config

warnings.filterwarnings('ignore')


class ChineseDataCleaner:
    """中文数据清洗器 - 专门处理包含中文文本的脏数据"""

    def __init__(self):
        self.df = None
        self.scenic_name = config.TARGET_SCENIC_NAME
        self.encoding_mappings = {}  # 存储编码映射关系
        self.season_config = None    # 淡旺季配置
        self.week_to_season = {}     # 周序号到季节的映射

    def connect_mysql(self):
        """建立 MySQL 连接"""
        print(f"\n{'=' * 60}")
        print("Step 0: 建立数据库连接")
        print(f"{'=' * 60}")

        try:
            connection_string = (
                f"mysql+pymysql://{config.DB_CONFIG['user']}:"
                f"{config.DB_CONFIG['password']}@"
                f"{config.DB_CONFIG['host']}:"
                f"{config.DB_CONFIG['port']}/"
                f"{config.DB_CONFIG['database']}?"
                f"charset={config.DB_CONFIG['charset']}"
            )
            self.engine = create_engine(connection_string)
            print(f"✓ 数据库连接成功: {config.DB_CONFIG['database']}")
            return True
        except Exception as e:
            print(f"✗ 数据库连接失败: {e}")
            return False

    def load_season_config(self):
        """加载淡旺季配置文件（由 m7c_season_analysis.py 生成）"""
        if not config.ENABLE_SEASON_FEATURE:
            print("淡旺季特征已禁用，跳过加载配置")
            return
        
        print(f"\n--- 加载淡旺季配置文件 ---")
        print(f"配置文件路径: {config.SEASON_CONFIG_PATH}")
        
        try:
            with open(config.SEASON_CONFIG_PATH, 'r', encoding='utf-8') as f:
                self.season_config = json.load(f)
            
            # 构建周序号到季节代码的映射字典
            self.week_to_season = {}
            for week_info in self.season_config['weeks']:
                week_num = week_info['week']
                self.week_to_season[week_num] = {
                    'season_code': week_info['season_code'],
                    'season_name': week_info['season_name'],
                    'season_cn': week_info['season_cn'],
                    'date_range': week_info['date_range']
                }
            
            print(f"✓ 淡旺季配置加载成功，共 {len(self.week_to_season)} 周配置")
            
            # 统计季节分布
            season_counts = {0: 0, 1: 0, 2: 0}
            for week_info in self.season_config['weeks']:
                season_counts[week_info['season_code']] += 1
            
            print(f"  淡季: {season_counts[0]} 周, 平季: {season_counts[1]} 周, 旺季: {season_counts[2]} 周")
            
        except FileNotFoundError:
            print(f"⚠ 警告：淡旺季配置文件不存在: {config.SEASON_CONFIG_PATH}")
            print("请先运行 m7c_season_analysis.py 生成配置文件")
            self.season_config = None
        except Exception as e:
            print(f"✗ 淡旺季配置加载失败：{e}")
            self.season_config = None

    def calculate_fiscal_week(self, dt):
        """
        计算给定日期时间的财年周序号
        
        规则（与 m7c 保持一致）：
        - 以10月1日为财年第一天
        - 每7天为一周
        - 第52周包含剩余所有天数（8-9天）
        
        参数:
            dt: datetime 对象
        
        返回:
            int: 财年周序号 (1-52)
        """
        # 确定财年起始日期
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        
        year = dt.year
        month = dt.month
        
        # 判断属于哪个财年
        if month >= config.FISCAL_YEAR_START_MONTH:
            # 当前年的财年（如 2024年10月 属于 FY2024-2025）
            fiscal_year_start = datetime.datetime(year, config.FISCAL_YEAR_START_MONTH, config.FISCAL_YEAR_START_DAY)
        else:
            # 上一年的财年（如 2025年3月 属于 FY2024-2025）
            fiscal_year_start = datetime.datetime(year - 1, config.FISCAL_YEAR_START_MONTH, config.FISCAL_YEAR_START_DAY)
        
        # 计算距离财年起始的天数
        days_since_start = (dt - fiscal_year_start).days
        
        # 计算周序号
        if days_since_start < 0:
            return 1
        elif days_since_start < 51 * 7:  # 前51周
            week_num = days_since_start // 7 + 1
        else:  # 第52周
            week_num = 52
        
        return week_num

    def load_data(self):
        """Step 1: 定向读取与类型修复"""
        print(f"\n{'=' * 60}")
        print("Step 1: 定向读取与类型修复")
        print(f"{'=' * 60}")

        try:
            # 读取指定景区的数据
            query = f"""
            SELECT * FROM {config.SOURCE_TABLE}
            WHERE scenic_name = '{self.scenic_name}'
            """

            self.df = pd.read_sql(query, self.engine)
            print(f"✓ 读取数据成功: {len(self.df)} 行")

            # --- 新增: 确保 date 列是 datetime 类型 ---
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
                print(f"✓ Date列已转换为 datetime 类型")
            else:
                raise ValueError("数据中缺少 'date' 列，无法进行时间特征工程")

            # 确保 datetime 列是 datetime 类型
            if config.DATETIME_FIELD in self.df.columns:
                self.df[config.DATETIME_FIELD] = pd.to_datetime(self.df[config.DATETIME_FIELD])
                print(f"✓ Datetime列已转换为 datetime 类型")

            # 强制转换 passenger_count 为 float
            if 'passenger_count' in self.df.columns:
                print("\n--- 修复 passenger_count 类型 (text → float) ---")

                # 简单处理非数值，强制转换
                self.df['passenger_count'] = pd.to_numeric(
                    self.df['passenger_count'],
                    errors='coerce'
                )
                print(f"✓ 转换后类型: {self.df['passenger_count'].dtype}")
                print(f"✓ 有效数值: {self.df['passenger_count'].notna().sum()} 行")

        except Exception as e:
            print(f"✗ 数据读取失败: {e}")
            raise

    def encode_chinese_features(self):
        """Step 2: 特征工程 (周期性编码与节假日处理)"""
        print(f"\n{'=' * 60}")
        print("Step 2: 高级特征工程 (周期性时间 & 中国节假日)")
        print(f"{'=' * 60}")

        # ==========================================
        # 任务 1: 月份特征 (周期性编码)
        # ==========================================
        print("\n--- 2.1 月份周期性编码 (Month Cyclic) ---")
        # 提取月份 (1-12)
        months = self.df['date'].dt.month

        # 周期性编码公式: sin(2 * pi * month / 12)
        self.df['month_sin'] = np.sin(2 * np.pi * months / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * months / 12)

        print(f"✓ 生成列: month_sin, month_cos")
        print(self.df[['date', 'month_sin', 'month_cos']].head(3))

        # ==========================================
        # 任务 2: 星期特征 (周期性编码)
        # ==========================================
        print("\n--- 2.2 星期周期性编码 (Weekday Cyclic) ---")
        # 提取星期 (0=周一, 6=周日)
        weekdays = self.df['date'].dt.dayofweek

        # 周期性编码公式: sin(2 * pi * weekday / 7)
        self.df['day_sin'] = np.sin(2 * np.pi * weekdays / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * weekdays / 7)

        print(f"✓ 生成列: day_sin, day_cos")
        print(self.df[['date', 'day_sin', 'day_cos']].head(3))

        # ==========================================
        # 任务 3: 小时特征 (周期性编码)
        # ==========================================
        print("\n--- 2.3 小时周期性编码 (Hour Cyclic) ---")
        # 提取小时 (0-23)
        if 'hour' in self.df.columns:
            hours = self.df['hour']
        elif config.DATETIME_FIELD in self.df.columns:
            # 如果没有单独的 hour 列，从 datetime 字段提取
            self.df['hour'] = pd.to_datetime(self.df[config.DATETIME_FIELD]).dt.hour
            hours = self.df['hour']
        else:
            print("⚠ 警告: 未找到 hour 列或 datetime 字段，跳过小时周期性编码")
            hours = None

        if hours is not None:
            # 周期性编码公式: sin(2 * pi * hour / 24)
            self.df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

            print(f"✓ 生成列: hour_sin, hour_cos")
            print(self.df[['hour', 'hour_sin', 'hour_cos']].head(3))

        # ==========================================
        # 任务 4: 是否休息日 (基于 chinese_calendar)
        # ==========================================
        print("\n--- 2.4 智能节假日判断 (Chinese Calendar) ---")
        print("正在计算复杂的调休逻辑，请稍候...")

        # 定义判断函数
        def check_is_day_off(date_val):
            # chinese_calendar.is_workday(date) 返回 True 表示是工作日（含补班）
            # 我们需要 is_day_off: 1=休息(含节假日), 0=上班(含补班)
            # 所以需要取反
            is_work = chinese_calendar.is_workday(date_val)
            return 0 if is_work else 1

        self.df['is_day_off'] = self.df['date'].apply(lambda x: check_is_day_off(x))

        count_off = self.df['is_day_off'].sum()
        count_work = len(self.df) - count_off
        print(f"✓ 生成列: is_day_off (处理了调休)")
        print(f"  - 休息状态 (1): {count_off} 条")
        print(f"  - 工作状态 (0): {count_work} 条")

        self.encoding_mappings['is_day_off'] = {0: '工作/补班', 1: '周末/节假日'}

        # ==========================================
        # 任务 5: 淡旺季特征 (基于 m7c 分析结果)
        # ==========================================
        self.add_season_features()

        # ==========================================
        # 原有的 LabelEncoder (天气与风向)
        # ==========================================
        # 2.6 天气等级分类映射 (好/中/坏)
        if 'weather_cond' in self.df.columns:
            self.add_weather_grade_feature()

        # 2.7 天气状况 LabelEncoder
        if 'weather_cond' in self.df.columns:
            print("\n--- 2.7 天气状况编码 (LabelEncoder) ---")
            self._encode_with_label_encoder('weather_cond', 'weather_cond_int')

        # 2.8 风向 LabelEncoder
        if 'wind_dir' in self.df.columns:
            print("\n--- 2.8 风向编码 (LabelEncoder) ---")
            self._encode_with_label_encoder('wind_dir', 'wind_dir_int')

    def add_season_features(self):
        """
        任务 5: 添加淡旺季特征（基于 m7c 分析结果）
        """
        print("\n--- 2.5 淡旺季特征 (Season Features from M7C) ---")
        
        if not config.ENABLE_SEASON_FEATURE or self.season_config is None:
            print("⚠ 淡旺季特征未启用或配置未加载，跳过")
            return
        
        # 确定使用哪个时间字段
        if config.DATETIME_FIELD in self.df.columns:
            time_col = config.DATETIME_FIELD
        else:
            time_col = 'date'
        
        print(f"使用时间字段: {time_col}")
        
        # 计算每行的财年周序号
        self.df[config.FISCAL_WEEK_FIELD] = self.df[time_col].apply(
            lambda x: self.calculate_fiscal_week(x)
        )
        
        # 根据周序号映射季节代码
        self.df[config.SEASON_CODE_FIELD] = self.df[config.FISCAL_WEEK_FIELD].apply(
            lambda w: self.week_to_season.get(w, {}).get('season_code', 1)  # 默认平季
        )
        
        # 添加季节名称
        self.df[config.SEASON_NAME_FIELD] = self.df[config.FISCAL_WEEK_FIELD].apply(
            lambda w: self.week_to_season.get(w, {}).get('season_name', 'normal')
        )
        
        # 添加 One-Hot 编码（如果启用）
        if config.ENABLE_SEASON_ONEHOT:
            for code, col_name in config.SEASON_ONEHOT_COLS.items():
                self.df[col_name] = (self.df[config.SEASON_CODE_FIELD] == code).astype(int)
            print(f"✓ 生成 One-Hot 列: {list(config.SEASON_ONEHOT_COLS.values())}")
        
        # 统计季节分布
        season_dist = self.df[config.SEASON_CODE_FIELD].value_counts().sort_index()
        print(f"✓ 生成列: {config.SEASON_CODE_FIELD}, {config.SEASON_NAME_FIELD}, {config.FISCAL_WEEK_FIELD}")
        print(f"  数据季节分布:")
        print(f"    - 淡季 (0): {season_dist.get(0, 0)} 条")
        print(f"    - 平季 (1): {season_dist.get(1, 0)} 条")
        print(f"    - 旺季 (2): {season_dist.get(2, 0)} 条")
        
        # 记录编码映射
        self.encoding_mappings['season_code'] = {0: '淡季', 1: '平季', 2: '旺季'}

    def add_weather_grade_feature(self):
        """
        任务 2.6: 天气等级分类映射
        将 weather_cond 中文文本映射为好天气(2)/中等天气(1)/坏天气(0)
        """
        print("\n--- 2.6 天气等级分类映射 (Weather Grade) ---")

        if not getattr(config, 'ENABLE_WEATHER_GRADE', False):
            print("⚠ 天气等级分类已禁用，跳过")
            return

        grade_field = config.WEATHER_GRADE_FIELD
        mapping = config.WEATHER_GRADE_MAPPING
        default = config.WEATHER_GRADE_DEFAULT
        labels = config.WEATHER_GRADE_LABELS

        # 对 weather_cond 列执行映射，未匹配到的填充为默认值
        raw_weather = self.df['weather_cond'].fillna('未知').str.strip()
        self.df[grade_field] = raw_weather.map(mapping).fillna(default).astype(int)

        # 检查是否有未匹配的天气类型
        unmapped = set(raw_weather.unique()) - set(mapping.keys())
        if unmapped:
            print(f"  ⚠ 以下天气类型未在映射表中定义，已归入「{labels[default]}」(={default}): {unmapped}")

        # 打印分布统计
        grade_dist = self.df[grade_field].value_counts().sort_index()
        print(f"✓ 生成列: {grade_field}")
        print(f"  天气等级分布:")
        for code in sorted(labels.keys(), reverse=True):
            count = grade_dist.get(code, 0)
            pct = count / len(self.df) * 100
            print(f"    - {labels[code]} ({code}): {count} 条 ({pct:.1f}%)")

        # 记录编码映射
        self.encoding_mappings[grade_field] = labels

    def _encode_with_label_encoder(self, col_name, new_col_name):
        """使用 LabelEncoder 编码,并打印映射关系"""
        # 处理空值: 填充为 "未知"
        temp_col = self.df[col_name].fillna('未知')

        # 执行编码
        le = LabelEncoder()
        self.df[new_col_name] = le.fit_transform(temp_col)

        # 记录映射关系
        mapping = {}
        for idx, label in enumerate(le.classes_):
            mapping[idx] = label

        self.encoding_mappings[col_name] = mapping
        print(f"✓ {col_name} 编码完成，类别数: {len(le.classes_)}")

    def strong_cleaning(self):
        """Step 3: 强力数据清洗"""
        print(f"\n{'=' * 60}")
        print("Step 3: 强力数据清洗")
        print(f"{'=' * 60}")

        initial_shape = self.df.shape
        print(f"清洗前: {initial_shape[0]} 行 × {initial_shape[1]} 列")

        # 3.1 删除全空列
        if config.DROP_EMPTY_COLS:
            self.df = self.df.dropna(axis=1, how='all')

        # 3.2 删除空行
        if config.DROP_EMPTY_ROWS:
            self.df = self.df.dropna()

        # 3.3 删除常数列 (排除保留列)
        if config.DROP_CONSTANT_COLS:
            print("\n--- 检查常数列 ---")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 排除保留列和新生成的特征
            preserve_cols = set(config.PRESERVE_COLS +
                                ['is_day_off', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                                 'hour_sin', 'hour_cos'])

            # 添加天气等级列到保留列表
            if getattr(config, 'ENABLE_WEATHER_GRADE', False):
                preserve_cols.add(config.WEATHER_GRADE_FIELD)

            # 添加淡旺季相关列到保留列表
            if config.ENABLE_SEASON_FEATURE:
                preserve_cols.add(config.SEASON_CODE_FIELD)
                preserve_cols.add(config.FISCAL_WEEK_FIELD)
                if config.ENABLE_SEASON_ONEHOT:
                    preserve_cols.update(config.SEASON_ONEHOT_COLS.values())

            check_cols = [col for col in numeric_cols if col not in preserve_cols]

            cols_to_drop = []
            for col in check_cols:
                if self.df[col].std() == 0:  # std=0 等同于方差=0
                    cols_to_drop.append(col)

            if cols_to_drop:
                self.df = self.df.drop(columns=cols_to_drop)
                print(f"✓ 已删除常数列: {cols_to_drop}")

        # 3.4 删除原始中文列
        print("\n--- 删除原始中文列 ---")
        # 配置文件中定义的 TEXT_COLS 里的列，如果在 df 中存在，则删除
        text_cols_to_drop = [col for col in config.TEXT_COLS.keys() if col in self.df.columns]
        if text_cols_to_drop:
            self.df = self.df.drop(columns=text_cols_to_drop)
            print(f"✓ 已删除原始文本列: {text_cols_to_drop}")
        
        # 3.5 删除淡旺季的文本列（保留数值列）
        if config.ENABLE_SEASON_FEATURE and config.SEASON_NAME_FIELD in self.df.columns:
            self.df = self.df.drop(columns=[config.SEASON_NAME_FIELD])
            print(f"✓ 已删除季节文本列: {config.SEASON_NAME_FIELD}")

        final_shape = self.df.shape
        print(f"\n清洗后: {final_shape[0]} 行 × {final_shape[1]} 列")

    def output_results(self):
        """Step 4: 结果输出"""
        print(f"\n{'=' * 60}")
        print("Step 4: 结果输出")
        print(f"{'=' * 60}")

        # 4.1 保存到 MySQL
        table_name = f"{config.OUTPUT_TABLE_PREFIX}_{self.scenic_name}"
        try:
            self.df.to_sql(
                table_name,
                self.engine,
                if_exists='replace',
                index=False,
                chunksize=1000
            )
            print(f"✓ MySQL 保存成功: {table_name}")
        except Exception as e:
            print(f"✗ MySQL 保存失败: {e}")

        # 4.2 保存到 CSV
        csv_filename = f"{config.CSV_OUTPUT_PREFIX}_{self.scenic_name}.csv"
        try:
            self.df.to_csv(
                csv_filename,
                index=False,
                encoding=config.CSV_ENCODING
            )
            print(f"✓ CSV 保存成功: {csv_filename}")
        except Exception as e:
            print(f"✗ CSV 保存失败: {e}")

        # 4.3 打印预览
        print(f"\n数据列预览:")
        print(self.df.columns.tolist())
        print(f"\n前5行数据:")
        print(self.df.head())
        
        # 4.4 显示淡旺季特征信息
        if config.ENABLE_SEASON_FEATURE and config.SEASON_CODE_FIELD in self.df.columns:
            print(f"\n淡旺季特征已添加:")
            print(f"  - {config.SEASON_CODE_FIELD}: 季节代码 (0=淡季, 1=平季, 2=旺季)")
            print(f"  - {config.FISCAL_WEEK_FIELD}: 财年周序号 (1-52)")
            if config.ENABLE_SEASON_ONEHOT:
                print(f"  - One-Hot编码: {list(config.SEASON_ONEHOT_COLS.values())}")

    def run(self):
        """执行完整的清洗流程"""
        print(f"\n{'#' * 60}")
        print(f"# M8A 数据清洗流程启动 (Enhanced Feature Engineering)")
        print(f"# 目标景区: {self.scenic_name}")
        print(f"# 淡旺季特征: {'启用' if config.ENABLE_SEASON_FEATURE else '禁用'}")
        print(f"{'#' * 60}")

        if not self.connect_mysql():
            return False

        try:
            # 加载淡旺季配置（如果启用）
            self.load_season_config()
            
            self.load_data()
            self.encode_chinese_features()
            self.strong_cleaning()
            self.output_results()

            print(f"\n{'#' * 60}")
            print(f"# 清洗流程完成!")
            print(f"{'#' * 60}\n")
            return True

        except Exception as e:
            print(f"\n✗ 清洗流程出错: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    cleaner = ChineseDataCleaner()
    cleaner.run()
