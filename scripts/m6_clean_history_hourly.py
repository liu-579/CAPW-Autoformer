"""
数据仓库ETL核心处理模块（小时粒度版本）
功能：从MySQL读取小时粒度的客流数据，按景区分离并生成标准化小时度特征表
特点：处理缺失数据，使用前后区间均值填充
"""

import pandas as pd
import numpy as np
import json
import warnings
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import sys

# 导入配置文件
import config.m6_h_config as config


class ScenicSpotHourlyDataProcessor:
    """景区小时数据处理器 - 负责单个景区的小时粒度数据清洗与特征工程"""

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
        self.hourly_data = None
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

    def process_hourly_flow(self):
        """
        步骤2: 处理小时粒度数据
        提取小时客流基础信息和天气信息
        """
        self.log("开始处理小时粒度数据...")

        # 筛选小时粒度数据
        hourly_raw = self.raw_data[
            self.raw_data[config.SOURCE_FIELDS['particle_size']] == config.GRANULARITY['hourly']
        ].copy()

        self.log(f"小时粒度记录数: {len(hourly_raw)}")

        if len(hourly_raw) == 0:
            self.log("错误：没有小时粒度数据", level='ERROR')
            raise ValueError("没有可处理的小时粒度数据")

        # 重置索引以确保后续操作的索引对齐
        hourly_raw = hourly_raw.reset_index(drop=True)

        # 解析天气 JSON 字段以获取小时信息
        self.log("开始解析天气 JSON 数据...")
        weather_parsed = hourly_raw[config.SOURCE_FIELDS['weather']].apply(self.parse_mongo_json)

        # 将解析结果转换为 DataFrame
        weather_df = pd.DataFrame(weather_parsed.tolist())

        # 提取关键字段（使用.values确保索引对齐）
        self.hourly_data = pd.DataFrame()
        self.hourly_data[config.SOURCE_FIELDS['date']] = hourly_raw[config.SOURCE_FIELDS['date']].values
        self.hourly_data[config.SOURCE_FIELDS['passenger_count']] = hourly_raw[config.SOURCE_FIELDS['passenger_count']].values
        self.hourly_data[config.SOURCE_FIELDS['day_of_week']] = hourly_raw[config.SOURCE_FIELDS['day_of_week']].values
        self.hourly_data[config.SOURCE_FIELDS['holiday']] = hourly_raw[config.SOURCE_FIELDS['holiday']].values

        # 直接从数据库的 hourNumber 字段提取小时信息（0-23）
        # 注意：数据库字段为varchar，需要先去除空格，再转换为数值
        hour_raw = hourly_raw[config.SOURCE_FIELDS['hour']].astype(str).str.strip()
        
        # 将空字符串、'None'、'nan'等转换为NaN
        hour_raw = hour_raw.replace(['', 'None', 'nan', 'NaN', 'null', 'NULL'], np.nan)
        
        # 转换为数值类型（文本格式的数字如"1"会被正确转换）
        hour_numeric = pd.to_numeric(hour_raw, errors='coerce')
        self.hourly_data['hour'] = hour_numeric.values
        
        # 检查并删除小时字段无效的记录
        invalid_hours = self.hourly_data['hour'].isna().sum()
        if invalid_hours > 0:
            self.log(f"警告：发现 {invalid_hours} 条记录的小时字段无效（null或非数字），将被删除", level='WARNING')
            # 记录被删除的原始值（用于调试）
            invalid_mask = hour_numeric.isna()
            invalid_values = hour_raw[invalid_mask].unique()
            if len(invalid_values) > 0 and len(invalid_values) <= 10:
                self.log(f"无效的小时值示例: {invalid_values.tolist()}", level='WARNING')
            
            # 同步删除天气数据中对应的行
            valid_mask = ~self.hourly_data['hour'].isna()
            self.hourly_data = self.hourly_data[valid_mask].reset_index(drop=True)
            weather_df = weather_df[valid_mask.values].reset_index(drop=True)
            self.log(f"删除无效记录后，剩余 {len(self.hourly_data)} 条有效记录")
        
        # 检查小时范围
        if len(self.hourly_data) > 0:
            min_hour = self.hourly_data['hour'].min()
            max_hour = self.hourly_data['hour'].max()
            if min_hour < 0 or max_hour > 23:
                self.log(f"警告：小时字段超出正常范围 [0-23]，实际范围: [{min_hour}, {max_hour}]", level='WARNING')
            else:
                self.log(f"小时字段范围正常: [{int(min_hour)}, {int(max_hour)}]")

        # 提取天气字段（使用.values确保索引对齐）
        if 'weather' in weather_df.columns:
            self.hourly_data['weather_cond'] = weather_df['weather'].values

        if 'temperature' in weather_df.columns:
            self.hourly_data['temperature'] = pd.to_numeric(weather_df['temperature'], errors='coerce').values

        if 'winddirection' in weather_df.columns:
            self.hourly_data['wind_dir'] = weather_df['winddirection'].values

        if 'windpower' in weather_df.columns:
            # 1. 转换为字符串
            wp_series = weather_df['windpower'].astype(str)
            # 2. 清洗特殊符号：将 "≤3" 替换为 "3"，同时也处理 "<3" 的情况
            wp_series = wp_series.str.replace('≤', '', regex=False).str.replace('<', '', regex=False)
            # 3. 转换为数值，此时 "3" 会被正确转为 3.0，而不是 NaN
            self.hourly_data['wind_power'] = pd.to_numeric(wp_series, errors='coerce').values

        if 'humidity' in weather_df.columns:
            self.hourly_data['humidity'] = pd.to_numeric(weather_df['humidity'], errors='coerce').values

        # 创建完整的日期时间字段
        self.hourly_data['datetime'] = pd.to_datetime(
            self.hourly_data[config.SOURCE_FIELDS['date']].astype(str) +
            self.hourly_data['hour'].astype(int).astype(str).str.zfill(2),
            format='%Y%m%d%H',
            errors='coerce'
        )

        # 删除无效的datetime记录
        invalid_count = self.hourly_data['datetime'].isna().sum()
        if invalid_count > 0:
            self.log(f"警告：发现 {invalid_count} 条无效的时间记录，将被删除", level='WARNING')
            self.hourly_data = self.hourly_data.dropna(subset=['datetime']).reset_index(drop=True)

        # 节假日空值填充
        self.hourly_data[config.SOURCE_FIELDS['holiday']] = (
            self.hourly_data[config.SOURCE_FIELDS['holiday']]
            .fillna(config.DEFAULT_HOLIDAY_VALUE)
            .replace('', config.DEFAULT_HOLIDAY_VALUE)
        )

        # 按时间排序
        self.hourly_data = self.hourly_data.sort_values('datetime').reset_index(drop=True)

        self.log(f"小时数据处理完成，共 {len(self.hourly_data)} 条有效记录")

        return self

    def check_time_continuity(self):
        """
        步骤3: 检查时间连续性并识别缺失的时间点
        """
        self.log("开始检查时间连续性...")

        if len(self.hourly_data) == 0:
            return self

        # 获取时间范围
        min_time = self.hourly_data['datetime'].min()
        max_time = self.hourly_data['datetime'].max()

        # 生成完整的小时序列
        full_time_range = pd.date_range(start=min_time, end=max_time, freq='h')

        # 找出缺失的时间点
        existing_times = set(self.hourly_data['datetime'])
        missing_times = [t for t in full_time_range if t not in existing_times]

        if len(missing_times) > 0:
            self.log(f"发现 {len(missing_times)} 个缺失的小时数据点", level='WARNING')
            
            # 检查连续缺失
            if config.CHECK_TIME_CONTINUITY:
                consecutive_missing = self._find_consecutive_missing(missing_times)
                if consecutive_missing > config.MAX_CONSECUTIVE_MISSING_HOURS:
                    self.log(
                        f"警告：发现连续缺失 {consecutive_missing} 小时，超过阈值 {config.MAX_CONSECUTIVE_MISSING_HOURS}",
                        level='WARNING'
                    )
        else:
            self.log("时间序列完整，无缺失数据")

        return self

    def _find_consecutive_missing(self, missing_times):
        """
        查找最大连续缺失小时数

        参数:
            missing_times: 缺失时间点列表

        返回:
            int: 最大连续缺失小时数
        """
        if len(missing_times) == 0:
            return 0

        missing_times_sorted = sorted(missing_times)
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(missing_times_sorted)):
            if (missing_times_sorted[i] - missing_times_sorted[i-1]).total_seconds() == 3600:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        return max_consecutive

    def fill_missing_data(self):
        """
        步骤4: 填充缺失数据
        使用前后区间的均值进行插值填充
        """
        self.log("开始填充缺失数据...")

        if len(self.hourly_data) == 0:
            return self

        # 获取时间范围
        min_time = self.hourly_data['datetime'].min()
        max_time = self.hourly_data['datetime'].max()

        # 生成完整的小时序列
        full_time_range = pd.date_range(start=min_time, end=max_time, freq='h')
        full_df = pd.DataFrame({'datetime': full_time_range})

        # 合并数据，保留所有时间点
        merged_df = full_df.merge(self.hourly_data, on='datetime', how='left')

        # 记录缺失数据数量
        missing_count = merged_df[config.SOURCE_FIELDS['passenger_count']].isna().sum()
        if missing_count > 0:
            self.log(f"需要填充 {missing_count} 个缺失的客流数据点")

        # 填充日期和小时字段
        merged_df[config.SOURCE_FIELDS['date']] = merged_df['datetime'].dt.strftime('%Y%m%d').astype(int)
        merged_df['hour'] = merged_df['datetime'].dt.hour
        
        # 创建包含小时的日期时间字符串（格式：2023100101，确保小时补零）
        merged_df['datetime_str'] = merged_df['datetime'].dt.strftime('%Y%m%d%H')

        # 填充星期字段
        merged_df[config.SOURCE_FIELDS['day_of_week']] = merged_df['datetime'].dt.dayofweek + 1

        # 填充节假日字段（前向填充）
        merged_df[config.SOURCE_FIELDS['holiday']] = merged_df[config.SOURCE_FIELDS['holiday']].ffill()
        merged_df[config.SOURCE_FIELDS['holiday']] = merged_df[config.SOURCE_FIELDS['holiday']].bfill()
        merged_df[config.SOURCE_FIELDS['holiday']] = merged_df[config.SOURCE_FIELDS['holiday']].fillna(config.DEFAULT_HOLIDAY_VALUE)

        # 对数值字段进行插值填充
        numeric_columns = [config.SOURCE_FIELDS['passenger_count'], 'temperature', 'wind_power', 'humidity']
        
        for col in numeric_columns:
            if col in merged_df.columns:
                # 确保数据类型正确后再插值
                if merged_df[col].dtype == 'object':
                    merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
                # 使用线性插值
                merged_df[col] = merged_df[col].interpolate(method='linear', limit_direction='both')
                
                # 如果首尾仍有缺失，使用最近的有效值填充
                if config.ALLOW_EXTRAPOLATION:
                    merged_df[col] = merged_df[col].ffill().bfill()

        # 对分类字段进行前向/后向填充
        categorical_columns = ['weather_cond', 'wind_dir']
        
        for col in categorical_columns:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].ffill().bfill()

        self.hourly_data = merged_df
        self.log(f"数据填充完成，最终记录数: {len(self.hourly_data)}")

        return self

    def finalize_data(self):
        """
        步骤5: 最终数据整理
        """
        self.log("开始最终数据整理...")

        # 重命名列以符合业务标准
        column_mapping = {
            'datetime_str': 'date_str',
            'datetime': 'datetime',
            config.SOURCE_FIELDS['passenger_count']: 'passenger_count',
            config.SOURCE_FIELDS['day_of_week']: 'day_of_week',
            config.SOURCE_FIELDS['holiday']: 'holiday'
        }

        self.final_data = self.hourly_data.copy()
        
        # 只重命名存在的列
        existing_mappings = {k: v for k, v in column_mapping.items() if k in self.final_data.columns}
        self.final_data.rename(columns=existing_mappings, inplace=True)

        # 调整列顺序
        final_columns = [
            'datetime', 'date_str', 'hour', 'passenger_count', 'day_of_week', 'holiday',
            'weather_cond', 'temperature', 'wind_dir', 'wind_power', 'humidity'
        ]

        # 只选择存在的列
        existing_columns = [col for col in final_columns if col in self.final_data.columns]
        self.final_data = self.final_data[existing_columns]

        self.log(f"数据整理完成，最终记录数: {len(self.final_data)}")

        return self

    def save_to_database(self):
        """
        步骤6: 将最终数据保存到数据库
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


def process_scenic_spot_hourly_data(fence_id):
    """
    主处理函数：处理单个景区的小时粒度数据清洗任务

    参数:
        fence_id: 景区ID（字符串）
    """
    # 创建数据库引擎
    engine = create_engine(config.get_database_url())

    try:
        # 创建处理器实例
        processor = ScenicSpotHourlyDataProcessor(engine, fence_id)

        # 执行处理流程（链式调用）
        processor \
            .load_raw_data() \
            .process_hourly_flow() \
            .check_time_continuity() \
            .fill_missing_data() \
            .finalize_data() \
            .save_to_database()

        print(f"\n{'='*60}")
        print(f"景区 {fence_id} 小时粒度数据处理完成！")
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
    print("数据仓库 ETL 任务启动（小时粒度版本）")
    print(f"处理景区数量: {len(config.SCENIC_SPOT_IDS)}")
    print(f"源数据表: {config.SOURCE_TABLE}")
    print("="*60 + "\n")

    success_count = 0
    fail_count = 0

    for fence_id in config.SCENIC_SPOT_IDS:
        try:
            process_scenic_spot_hourly_data(fence_id)
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
