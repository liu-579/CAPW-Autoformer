"""
数据仓库ETL核心处理模块
功能：从MySQL读取混合粒度的客流数据，按景区分离并生成标准化日度特征表
"""

import pandas as pd
import numpy as np
import json
import warnings
from sqlalchemy import create_engine, text
from datetime import datetime
import sys

# 导入配置文件
import config.m6_config as config


class ScenicSpotDataProcessor:
    """景区数据处理器 - 负责单个景区的数据清洗与特征工程"""

    def __init__(self, engine, fence_id):
        """
        初始化处理器

        参数:
            engine: SQLAlchemy 数据库引擎
            fence_id: 景区ID
        """
        self.engine = engine
        self.fence_id = fence_id
        self.raw_data = None
        self.daily_data = None
        self.weather_data = None
        self.final_data = None

    def log(self, message, level='INFO'):
        """打印日志信息"""
        if config.VERBOSE_LOGGING:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] [{level}] [景区{self.fence_id}] {message}")

    def load_raw_data(self):
        """
        步骤1: 从数据库加载原始数据并进行景区隔离
        """
        self.log("开始加载原始数据...")

        query = f"""
        SELECT * FROM {config.SOURCE_TABLE}
        WHERE {config.SOURCE_FIELDS['fence_id']} = '{self.fence_id}'
        """

        self.raw_data = pd.read_sql(query, self.engine)
        self.log(f"数据加载完成，共 {len(self.raw_data)} 条记录")

        if config.ENABLE_DATA_QUALITY_CHECK and len(self.raw_data) < config.MIN_RECORDS_THRESHOLD:
            self.log(f"警告：记录数少于阈值 {config.MIN_RECORDS_THRESHOLD}", level='WARNING')

        return self

    def process_daily_base_flow(self):
        """
        步骤2: 处理主表流（天粒度数据）
        提取日度客流基础信息
        """
        self.log("开始处理主表流（天粒度）...")

        # 筛选天粒度数据
        daily_raw = self.raw_data[
            self.raw_data[config.SOURCE_FIELDS['particle_size']] == config.GRANULARITY['daily']
        ].copy()

        self.log(f"天粒度记录数: {len(daily_raw)}")

        # 提取关键字段
        self.daily_data = daily_raw[[
            config.SOURCE_FIELDS['date'],
            config.SOURCE_FIELDS['passenger_count'],
            config.SOURCE_FIELDS['day_of_week'],
            config.SOURCE_FIELDS['holiday']
        ]].copy()

        # 日期转换：将 "20231005" 转换为 datetime
        self.daily_data['date_parsed'] = pd.to_datetime(
            self.daily_data[config.SOURCE_FIELDS['date']],
            format=config.DATE_FORMAT
        )

        # 节假日空值填充
        self.daily_data[config.SOURCE_FIELDS['holiday']] = (
            self.daily_data[config.SOURCE_FIELDS['holiday']]
            .fillna(config.DEFAULT_HOLIDAY_VALUE)
            .replace('', config.DEFAULT_HOLIDAY_VALUE)
        )

        self.log(f"主表流处理完成，共 {len(self.daily_data)} 条有效记录")

        return self

    def parse_mongo_json(self, json_str):
        """
        辅助函数：解析 MongoDB 风格的 JSON 字符串

        参数:
            json_str: MongoDB JSON 字符串

        返回:
            dict: 解析后的字典，失败则返回 None
        """
        if pd.isna(json_str) or json_str == '':
            return None

        try:
            # 解析 JSON
            data = json.loads(json_str)

            # 处理 MongoDB 特殊格式（如 {"$numberInt": "11"}）
            result = {}
            for key, value in data.items():
                if isinstance(value, dict) and '$numberInt' in value:
                    # 提取数值并转换为整数
                    result[key] = int(value['$numberInt'])
                else:
                    result[key] = value

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.log(f"JSON解析失败: {json_str[:50]}... 错误: {e}", level='WARNING')
            return None

    def safe_mode(self, series):
        """
        安全的众数计算函数

        参数:
            series: pandas Series

        返回:
            众数值，如果没有则返回 None
        """
        try:
            mode_result = series.mode()
            if len(mode_result) > 0:
                return mode_result.iloc[0]
            else:
                return None
        except:
            return None

    def safe_mean(self, series):
        """
        安全的平均值计算函数（自动转换数据类型）

        参数:
            series: pandas Series

        返回:
            平均值，如果无法计算则返回 None
        """
        try:
            # 尝试转换为数值类型
            numeric_series = pd.to_numeric(series, errors='coerce')
            # 过滤掉 NaN 值
            numeric_series = numeric_series.dropna()
            if len(numeric_series) > 0:
                return numeric_series.mean()
            else:
                return None
        except:
            return None

    def process_weather_flow(self):
        """
        步骤3: 处理天气流（小时粒度数据）
        解析 JSON 并聚合到日度
        """
        self.log("开始处理天气流（小时粒度）...")

        # 筛选小时粒度数据
        hourly_raw = self.raw_data[
            self.raw_data[config.SOURCE_FIELDS['particle_size']] == config.GRANULARITY['hourly']
        ].copy()

        self.log(f"小时粒度记录数: {len(hourly_raw)}")

        if len(hourly_raw) == 0:
            self.log("警告：没有小时粒度数据，天气特征将全部为空", level='WARNING')
            self.weather_data = pd.DataFrame()
            return self

        # 解析 weather JSON 字段
        self.log("开始解析天气 JSON 数据...")
        weather_parsed = hourly_raw[config.SOURCE_FIELDS['weather']].apply(self.parse_mongo_json)

        # 将解析结果转换为 DataFrame
        weather_df = pd.DataFrame(weather_parsed.tolist())

        # 合并日期字段
        weather_df[config.SOURCE_FIELDS['date']] = hourly_raw[config.SOURCE_FIELDS['date']].values

        # 提取并重命名字段
        weather_extract = pd.DataFrame()
        weather_extract[config.SOURCE_FIELDS['date']] = weather_df[config.SOURCE_FIELDS['date']]

        # 根据配置映射提取字段，并确保数值类型正确
        if 'weather' in weather_df.columns:
            weather_extract['weather_cond'] = weather_df['weather']

        if 'temperature' in weather_df.columns:
            # 确保温度为数值类型
            weather_extract['temperature'] = pd.to_numeric(weather_df['temperature'], errors='coerce')

        if 'winddirection' in weather_df.columns:
            weather_extract['wind_dir'] = weather_df['winddirection']

        if 'windpower' in weather_df.columns:
            # 确保风力为数值类型
            weather_extract['wind_power'] = pd.to_numeric(weather_df['windpower'], errors='coerce')

        if 'humidity' in weather_df.columns:
            # 确保湿度为数值类型
            weather_extract['humidity'] = pd.to_numeric(weather_df['humidity'], errors='coerce')

        self.log(f"JSON 解析完成，提取字段: {weather_extract.columns.tolist()}")

        # 检查数据类型
        self.log(f"数据类型: {weather_extract.dtypes.to_dict()}")

        # 日度聚合 - 使用自定义聚合函数
        self.log("开始日度聚合...")

        agg_dict = {}

        # 天气状况 - 众数
        if 'weather_cond' in weather_extract.columns:
            agg_dict['weather_cond'] = self.safe_mode

        # 温度 - 需要分别计算最高和最低
        if 'temperature' in weather_extract.columns:
            agg_dict['temp_high'] = lambda x: x.max() if len(x.dropna()) > 0 else None
            agg_dict['temp_low'] = lambda x: x.min() if len(x.dropna()) > 0 else None

        # 风向 - 众数
        if 'wind_dir' in weather_extract.columns:
            agg_dict['wind_dir'] = self.safe_mode

        # 风力 - 平均值
        if 'wind_power' in weather_extract.columns:
            agg_dict['wind_power'] = self.safe_mean

        # 湿度 - 平均值
        if 'humidity' in weather_extract.columns:
            agg_dict['humidity'] = self.safe_mean

        # 准备聚合数据
        if 'temperature' in weather_extract.columns:
            # 为温度创建两列用于聚合
            agg_data = weather_extract.copy()

            # 执行分组聚合
            grouped = agg_data.groupby(config.SOURCE_FIELDS['date'])

            result_dict = {config.SOURCE_FIELDS['date']: []}

            for col in ['weather_cond', 'temp_high', 'temp_low', 'wind_dir', 'wind_power', 'humidity']:
                result_dict[col] = []

            for date, group in grouped:
                result_dict[config.SOURCE_FIELDS['date']].append(date)

                # 天气状况 - 众数
                if 'weather_cond' in group.columns:
                    result_dict['weather_cond'].append(self.safe_mode(group['weather_cond']))

                # 温度 - 最高和最低
                if 'temperature' in group.columns:
                    temps = group['temperature'].dropna()
                    result_dict['temp_high'].append(temps.max() if len(temps) > 0 else None)
                    result_dict['temp_low'].append(temps.min() if len(temps) > 0 else None)

                # 风向 - 众数
                if 'wind_dir' in group.columns:
                    result_dict['wind_dir'].append(self.safe_mode(group['wind_dir']))

                # 风力 - 平均值
                if 'wind_power' in group.columns:
                    result_dict['wind_power'].append(self.safe_mean(group['wind_power']))

                # 湿度 - 平均值
                if 'humidity' in group.columns:
                    result_dict['humidity'].append(self.safe_mean(group['humidity']))

            self.weather_data = pd.DataFrame(result_dict)
        else:
            # 没有温度数据的情况
            self.weather_data = weather_extract.groupby(config.SOURCE_FIELDS['date']).agg(agg_dict).reset_index()

        self.log(f"天气流聚合完成，日度记录数: {len(self.weather_data)}")
        self.log(f"聚合后数据类型: {self.weather_data.dtypes.to_dict()}")

        return self

    def merge_and_finalize(self):
        """
        步骤4: 合并主表流和天气流，生成最终特征表
        """
        self.log("开始合并数据流...")

        # Left Join：以主表流（天粒度客流数据）为准
        self.final_data = self.daily_data.merge(
            self.weather_data,
            on=config.SOURCE_FIELDS['date'],
            how='left'
        )

        # 检查是否有缺失天气数据的记录
        if config.WARN_ON_MISSING_WEATHER and 'weather_cond' in self.final_data.columns:
            missing_weather = self.final_data['weather_cond'].isna().sum()
            if missing_weather > 0:
                self.log(
                    f"警告：有 {missing_weather} 天的客流记录缺少对应的天气数据",
                    level='WARNING'
                )
                # 输出具体日期
                missing_dates = self.final_data[
                    self.final_data['weather_cond'].isna()
                ][config.SOURCE_FIELDS['date']].tolist()
                self.log(f"缺失天气的日期: {missing_dates[:10]}..." if len(missing_dates) > 10
                        else f"缺失天气的日期: {missing_dates}", level='WARNING')

        # 重命名列以符合业务标准
        column_mapping = {
            config.SOURCE_FIELDS['date']: 'date_str',
            'date_parsed': 'date',
            config.SOURCE_FIELDS['passenger_count']: 'passenger_count',
            config.SOURCE_FIELDS['day_of_week']: 'day_of_week',
            config.SOURCE_FIELDS['holiday']: 'holiday'
        }

        self.final_data.rename(columns=column_mapping, inplace=True)

        # 调整列顺序
        final_columns = [
            'date', 'date_str', 'passenger_count', 'day_of_week', 'holiday',
            'weather_cond', 'temp_high', 'temp_low', 'wind_dir', 'wind_power', 'humidity'
        ]

        # 只选择存在的列
        existing_columns = [col for col in final_columns if col in self.final_data.columns]
        self.final_data = self.final_data[existing_columns]

        self.log(f"数据合并完成，最终记录数: {len(self.final_data)}")

        return self

    def save_to_database(self):
        """
        步骤5: 将最终数据保存到数据库
        """
        output_table = config.OUTPUT_TABLE_TEMPLATE.format(fence_id=self.fence_id)
        self.log(f"开始保存数据到表: {output_table}")

        try:
            # 保存到数据库
            self.final_data.to_sql(
                name=output_table,
                con=self.engine,
                if_exists=config.DB_WRITE_MODE,
                index=False
            )

            self.log(f"数据保存成功！表名: {output_table}, 记录数: {len(self.final_data)}")

        except Exception as e:
            self.log(f"数据保存失败: {e}", level='ERROR')
            raise

        return self


def process_scenic_spot_data(fence_id):
    """
    主处理函数：处理单个景区的数据清洗任务

    参数:
        fence_id: 景区ID（字符串）
    """
    # 创建数据库引擎
    engine = create_engine(config.get_database_url())

    try:
        # 创建处理器实例
        processor = ScenicSpotDataProcessor(engine, fence_id)

        # 执行处理流程（链式调用）
        processor \
            .load_raw_data() \
            .process_daily_base_flow() \
            .process_weather_flow() \
            .merge_and_finalize() \
            .save_to_database()

        print(f"\n{'='*60}")
        print(f"景区 {fence_id} 数据处理完成！")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"景区 {fence_id} 处理失败: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        raise

    finally:
        engine.dispose()


def main():
    """
    主函数：批量处理所有景区
    """
    print("\n" + "="*60)
    print("数据仓库 ETL 任务启动")
    print(f"处理景区数量: {len(config.SCENIC_SPOT_IDS)}")
    print(f"源数据表: {config.SOURCE_TABLE}")
    print("="*60 + "\n")

    success_count = 0
    fail_count = 0

    for fence_id in config.SCENIC_SPOT_IDS:
        try:
            process_scenic_spot_data(fence_id)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"错误详情: {e}\n")

    # 输出汇总信息
    print("\n" + "="*60)
    print("ETL 任务执行完成")
    print(f"成功: {success_count} 个景区")
    print(f"失败: {fail_count} 个景区")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()