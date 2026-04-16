# -*- coding: utf-8 -*-
"""
M7C 淡旺季分析脚本
Season Analysis using Jenks Natural Breaks + GVF Validation + Two-Year Comparison

功能:
1. 从数据库读取客流数据（支持小时级或日度数据）
2. 按自定义财年周（从10月1日开始）聚合，每年固定52周
3. 使用 Jenks Natural Breaks 算法分类淡/平/旺季
4. GVF 验证分类效果
5. 两年数据比对，输出待审核列表
6. 生成最终52周季节配置（不一致周取平均后重新分类）

数据范围: 2023.10.01 - 2025.09.30
财年定义: 10月1日 - 次年9月30日
周定义: 每7天为一周，第52周包含剩余所有天数（8-9天）
"""

import pandas as pd 
import numpy as np
import jenkspy
from sqlalchemy import create_engine
import json
import os
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入配置
import config.m7c_config as config

warnings.filterwarnings('ignore')


class SeasonAnalyzer:
    """淡旺季分析器 - 使用 Jenks Natural Breaks 算法"""

    def __init__(self):
        self.engine = None
        self.raw_data = None
        self.weekly_data = {}  # {fiscal_year: DataFrame}
        self.jenks_results = {}  # {fiscal_year: {'breaks': [], 'labels': [], 'gvf': float}}
        self.comparison = None
        self.final_config = None

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def connect_database(self):
        """建立数据库连接"""
        print(f"\n{'=' * 70}")
        print("Step 1: 建立数据库连接")
        print(f"{'=' * 70}")

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

    def load_data(self):
        """加载数据（支持小时级或日度数据）"""
        print(f"\n{'=' * 70}")
        data_type_cn = "小时级" if config.DATA_SOURCE_TYPE == 'hourly' else "日度"
        print(f"Step 2: 加载{data_type_cn}数据")
        print(f"{'=' * 70}")

        query = f"""
        SELECT {config.DATETIME_COLUMN}, {config.TARGET_COLUMN}
        FROM {config.INPUT_TABLE}
        ORDER BY {config.DATETIME_COLUMN}
        """

        self.raw_data = pd.read_sql(query, self.engine)
        self.raw_data[config.DATETIME_COLUMN] = pd.to_datetime(self.raw_data[config.DATETIME_COLUMN])
        
        # 对于小时级数据，提取日期部分用于后续处理
        if config.DATA_SOURCE_TYPE == 'hourly':
            self.raw_data['date'] = self.raw_data[config.DATETIME_COLUMN].dt.date
        else:
            # 日度数据，确保datetime列只包含日期
            self.raw_data['date'] = self.raw_data[config.DATETIME_COLUMN].dt.date

        print(f"✓ 加载数据: {len(self.raw_data)} 行 (数据类型: {data_type_cn})")
        print(f"  时间范围: {self.raw_data[config.DATETIME_COLUMN].min()} ~ {self.raw_data[config.DATETIME_COLUMN].max()}")
        print(f"  客流范围: {self.raw_data[config.TARGET_COLUMN].min()} ~ {self.raw_data[config.TARGET_COLUMN].max()}")

    def calculate_fiscal_week(self, date, fiscal_year_info):
        """
        计算财年周序号
        
        规则:
        - 以10月1日为第一天
        - 每7天为一周
        - 第52周包含剩余所有天数（8-9天）
        
        Args:
            date: 日期（可以是 datetime 或 date 对象）
            fiscal_year_info: 财年信息字典
        
        Returns:
            week_num: 周序号 (1-52)
        """
        fiscal_start = pd.Timestamp(fiscal_year_info['start'])
        
        # 确保 date 是 Timestamp 类型
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        
        # 只使用日期部分计算
        days_since_start = (date.normalize() - fiscal_start.normalize()).days

        # 计算周序号：前51周每周7天，第52周包含剩余天数
        # 第1-51周: 天数 0-356 (357天)
        # 第52周: 天数 357-364/365 (8-9天)
        if days_since_start < 0:
            return 1
        elif days_since_start < 51 * 7:  # 前51周
            week_num = days_since_start // 7 + 1
        else:  # 第52周（包含剩余所有天数）
            week_num = 52
        
        return week_num

    def aggregate_to_weekly(self):
        """将数据聚合为周数据（支持小时级和日度数据）"""
        print(f"\n{'=' * 70}")
        print("Step 3: 聚合为周数据 (自定义财年周)")
        print(f"{'=' * 70}")
        
        data_type_cn = "小时级" if config.DATA_SOURCE_TYPE == 'hourly' else "日度"
        print(f"  数据源类型: {data_type_cn}")

        for fy_info in config.FISCAL_YEARS:
            fy_year = fy_info['year']
            fy_start = pd.Timestamp(fy_info['start'])
            fy_end = pd.Timestamp(fy_info['end'])
            fy_label = fy_info['label']

            print(f"\n--- 处理财年 {fy_label} ---")
            print(f"  范围: {fy_start.date()} ~ {(fy_end - timedelta(days=1)).date()}")

            # 筛选该财年的数据（使用 < end 而非 <= end，避免边界问题）
            mask = (self.raw_data[config.DATETIME_COLUMN] >= fy_start) & \
                   (self.raw_data[config.DATETIME_COLUMN] < fy_end)
            fy_data = self.raw_data[mask].copy()

            if len(fy_data) == 0:
                print(f"  ⚠ 该财年无数据，跳过")
                continue

            print(f"  数据量: {len(fy_data)} 条 ({data_type_cn})")

            # 根据数据类型进行不同的聚合处理
            if config.DATA_SOURCE_TYPE == 'hourly':
                # 小时级数据：先聚合到日度，再聚合到周度
                # 计算每天的日均客流
                fy_data['date'] = fy_data[config.DATETIME_COLUMN].dt.date
                daily_data = fy_data.groupby('date')[config.TARGET_COLUMN].sum().reset_index()
                daily_data.columns = ['date', 'daily_passenger']
                daily_data['date'] = pd.to_datetime(daily_data['date'])
                
                # 计算财年周
                daily_data['fiscal_week'] = daily_data['date'].apply(
                    lambda x: self.calculate_fiscal_week(x, fy_info)
                )
                
                # 按周聚合（计算周内日均客流）
                if config.AGGREGATION_METHOD == 'mean':
                    weekly = daily_data.groupby('fiscal_week')['daily_passenger'].mean()
                elif config.AGGREGATION_METHOD == 'sum':
                    weekly = daily_data.groupby('fiscal_week')['daily_passenger'].sum()
                else:
                    weekly = daily_data.groupby('fiscal_week')['daily_passenger'].median()
            else:
                # 日度数据：直接聚合到周度
                fy_data['fiscal_week'] = fy_data[config.DATETIME_COLUMN].apply(
                    lambda x: self.calculate_fiscal_week(x, fy_info)
                )
                
                # 按周聚合
                if config.AGGREGATION_METHOD == 'mean':
                    weekly = fy_data.groupby('fiscal_week')[config.TARGET_COLUMN].mean()
                elif config.AGGREGATION_METHOD == 'sum':
                    weekly = fy_data.groupby('fiscal_week')[config.TARGET_COLUMN].sum()
                else:
                    weekly = fy_data.groupby('fiscal_week')[config.TARGET_COLUMN].median()

            # 转换为 DataFrame
            weekly_df = weekly.reset_index()
            weekly_df.columns = ['fiscal_week', 'avg_passenger']
            weekly_df['fiscal_year'] = fy_year
            weekly_df['fiscal_label'] = fy_label

            # 添加周的日期范围信息
            def get_week_date_range(week_num):
                """获取周的日期范围"""
                week_start = fy_start + timedelta(days=(week_num - 1) * 7)
                if week_num < 52:
                    week_end = fy_start + timedelta(days=week_num * 7 - 1)
                else:
                    # 第52周结束于财年最后一天
                    week_end = fy_end - timedelta(days=1)
                return week_start.strftime('%m-%d'), week_end.strftime('%m-%d')
            
            weekly_df['week_start'] = weekly_df['fiscal_week'].apply(
                lambda w: get_week_date_range(w)[0]
            )
            weekly_df['week_end'] = weekly_df['fiscal_week'].apply(
                lambda w: get_week_date_range(w)[1]
            )
            
            # 计算每周的实际天数
            weekly_df['days_in_week'] = weekly_df['fiscal_week'].apply(
                lambda w: 7 if w < 52 else (366 if fy_year % 4 == 0 else 365) - 51 * 7
            )

            self.weekly_data[fy_year] = weekly_df

            print(f"  ✓ 聚合完成: {len(weekly_df)} 周")
            print(f"    周均客流范围: {weekly_df['avg_passenger'].min():.0f} ~ {weekly_df['avg_passenger'].max():.0f}")
            print(f"    周均客流均值: {weekly_df['avg_passenger'].mean():.0f}")
            print(f"    第52周天数: {weekly_df[weekly_df['fiscal_week']==52]['days_in_week'].values[0] if 52 in weekly_df['fiscal_week'].values else 'N/A'}")

        # 保存周聚合数据
        all_weekly = pd.concat(self.weekly_data.values(), ignore_index=True)
        output_path = config.OUTPUT_DIR / config.OUTPUT_FILES['weekly_data']
        all_weekly.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 周聚合数据已保存: {output_path}")

    def calculate_gvf(self, data, breaks):
        """
        计算 GVF (Goodness of Variance Fit)
        
        GVF = 1 - SDCM / SDAM
        SDCM: Sum of Squared Deviations from Class Means
        SDAM: Sum of Squared Deviations from Array Mean
        
        修复说明：
        - 原代码存在重复计算最后一个区间的漏洞
        - 现在正确处理每个区间，最后一个区间包含上界
        """
        array_mean = np.mean(data)
        sdam = np.sum((data - array_mean) ** 2)

        sdcm = 0
        n_intervals = len(breaks) - 1  # 区间数量
        
        for i in range(n_intervals):
            lower = breaks[i]
            upper = breaks[i + 1]
            
            # 最后一个区间包含上界，其他区间不包含上界
            if i == n_intervals - 1:
                # 最后一个区间: [lower, upper]
                class_data = data[(data >= lower) & (data <= upper)]
            else:
                # 其他区间: [lower, upper)
                class_data = data[(data >= lower) & (data < upper)]
            
            if len(class_data) > 0:
                class_mean = np.mean(class_data)
                sdcm += np.sum((class_data - class_mean) ** 2)

        gvf = 1 - (sdcm / sdam) if sdam > 0 else 0
        return gvf

    def jenks_classify(self):
        """使用 Jenks Natural Breaks 分类"""
        print(f"\n{'=' * 70}")
        print("Step 4: Jenks Natural Breaks 分类")
        print(f"{'=' * 70}")

        for fy_year, weekly_df in self.weekly_data.items():
            fy_label = weekly_df['fiscal_label'].iloc[0]
            print(f"\n--- 分类财年 {fy_label} ---")

            data = weekly_df['avg_passenger'].values

            # Jenks 分类
            breaks = jenkspy.jenks_breaks(data, n_classes=config.N_CLASSES)

            # 计算 GVF
            gvf = self.calculate_gvf(data, breaks)

            # 分配标签
            labels = []
            for value in data:
                if value <= breaks[1]:
                    labels.append(0)  # 淡季
                elif value <= breaks[2]:
                    labels.append(1)  # 平季
                else:
                    labels.append(2)  # 旺季

            # 保存结果
            self.jenks_results[fy_year] = {
                'breaks': breaks,
                'labels': labels,
                'gvf': gvf
            }

            # 更新 DataFrame
            weekly_df['season_code'] = labels
            weekly_df['season_name'] = weekly_df['season_code'].map(
                lambda x: config.SEASON_LABELS[x]['name']
            )
            weekly_df['season_cn'] = weekly_df['season_code'].map(
                lambda x: config.SEASON_LABELS[x]['cn']
            )

            # 打印结果
            print(f"  断点 (Breaks): {[f'{b:.0f}' for b in breaks]}")
            print(f"  GVF: {gvf:.4f} {'✓ 良好' if gvf >= config.GVF_THRESHOLD else '⚠ 较低'}")

            # 统计各季节周数
            for code, info in config.SEASON_LABELS.items():
                count = labels.count(code)
                pct = count / len(labels) * 100
                print(f"    {info['cn']}: {count} 周 ({pct:.1f}%)")

        # 保存分类结果
        result = {
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fiscal_years': {}
        }
        for fy_year, res in self.jenks_results.items():
            result['fiscal_years'][str(fy_year)] = {
                'breaks': [float(b) for b in res['breaks']],
                'gvf': float(res['gvf']),
                'season_distribution': {
                    config.SEASON_LABELS[code]['cn']: res['labels'].count(code)
                    for code in range(config.N_CLASSES)
                }
            }

        output_path = config.OUTPUT_DIR / config.OUTPUT_FILES['season_result']
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 分类结果已保存: {output_path}")

    def compare_two_years(self):
        """比对两年数据"""
        print(f"\n{'=' * 70}")
        print("Step 5: 两年数据比对")
        print(f"{'=' * 70}")

        if len(self.weekly_data) < 2:
            print("⚠ 数据不足两年，跳过比对")
            return

        years = sorted(self.weekly_data.keys())
        year1, year2 = years[0], years[1]

        df1 = self.weekly_data[year1][['fiscal_week', 'avg_passenger', 'season_code', 'season_cn', 'week_start', 'week_end']].copy()
        df2 = self.weekly_data[year2][['fiscal_week', 'avg_passenger', 'season_code', 'season_cn']].copy()

        df1.columns = ['fiscal_week', f'passenger_{year1}', f'season_{year1}', f'season_cn_{year1}', 'week_start', 'week_end']
        df2.columns = ['fiscal_week', f'passenger_{year2}', f'season_{year2}', f'season_cn_{year2}']

        # 合并
        self.comparison = pd.merge(df1, df2, on='fiscal_week', how='outer')

        # 标记一致性
        self.comparison['is_consistent'] = self.comparison[f'season_{year1}'] == self.comparison[f'season_{year2}']

        # 统计
        consistent_count = self.comparison['is_consistent'].sum()
        inconsistent_count = len(self.comparison) - consistent_count

        print(f"\n比对结果:")
        print(f"  一致周数: {consistent_count} ({consistent_count / len(self.comparison) * 100:.1f}%)")
        print(f"  不一致周数: {inconsistent_count} ({inconsistent_count / len(self.comparison) * 100:.1f}%)")

        # 保存比对结果
        output_path = config.OUTPUT_DIR / config.OUTPUT_FILES['comparison']
        self.comparison.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 比对结果已保存: {output_path}")

        # 输出待审核列表
        if inconsistent_count > 0:
            review_df = self.comparison[~self.comparison['is_consistent']].copy()
            review_df['review_note'] = ''

            print(f"\n{'=' * 70}")
            print("待审核周列表:")
            print(f"{'=' * 70}")

            for _, row in review_df.iterrows():
                week = row['fiscal_week']
                print(f"  第{week:2d}周 ({row['week_start']}~{row['week_end']}): "
                      f"{year1}年={row[f'season_cn_{year1}']}({row[f'passenger_{year1}']:.0f}人), "
                      f"{year2}年={row[f'season_cn_{year2}']}({row[f'passenger_{year2}']:.0f}人)")

            output_path = config.OUTPUT_DIR / config.OUTPUT_FILES['review_list']
            review_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ 待审核列表已保存: {output_path}")

    def generate_final_config(self):
        """
        生成最终52周配置
        
        决策逻辑：
        - 对于两年分类一致的周：直接使用该分类
        - 对于两年分类不一致的周：取两年客流平均值，使用合并后的Jenks断点重新分类
        """
        print(f"\n{'=' * 70}")
        print("Step 6: 生成最终52周配置")
        print(f"{'=' * 70}")

        if self.comparison is None:
            print("⚠ 无比对数据，使用第一年数据作为配置")
            first_year = list(self.weekly_data.keys())[0]
            df = self.weekly_data[first_year]
            final_seasons = df['season_code'].tolist()
            avg_passengers = df['avg_passenger'].tolist()
        else:
            years = sorted(self.weekly_data.keys())
            year1, year2 = years[0], years[1]

            # 计算两年的平均客流
            self.comparison['avg_passenger'] = (
                self.comparison[f'passenger_{year1}'] + self.comparison[f'passenger_{year2}']
            ) / 2
            
            # 使用平均客流重新进行Jenks分类
            avg_data = self.comparison['avg_passenger'].values
            
            # 对平均数据进行Jenks分类
            avg_breaks = jenkspy.jenks_breaks(avg_data, n_classes=config.N_CLASSES)
            print(f"\n  两年平均客流 Jenks 断点: {[f'{b:.0f}' for b in avg_breaks]}")
            
            # 根据平均断点对每周进行分类
            final_seasons = []
            avg_passengers = []
            
            for _, row in self.comparison.iterrows():
                avg_val = row['avg_passenger']
                avg_passengers.append(avg_val)
                
                if row['is_consistent']:
                    # 一致的周：直接使用原分类
                    final_seasons.append(int(row[f'season_{year1}']))
                else:
                    # 不一致的周：使用平均值和新断点重新分类
                    if avg_val <= avg_breaks[1]:
                        season = 0  # 淡季
                    elif avg_val <= avg_breaks[2]:
                        season = 1  # 平季
                    else:
                        season = 2  # 旺季
                    final_seasons.append(season)
                    
                    # 打印不一致周的决策过程
                    week = int(row['fiscal_week'])
                    old_season_1 = config.SEASON_LABELS[int(row[f'season_{year1}'])]['cn']
                    old_season_2 = config.SEASON_LABELS[int(row[f'season_{year2}'])]['cn']
                    new_season = config.SEASON_LABELS[season]['cn']
                    print(f"    第{week:2d}周: {year1}年={old_season_1}, {year2}年={old_season_2} "
                          f"-> 平均={avg_val:.0f} -> {new_season}")

        # 生成配置
        self.final_config = {
            'description': '52周淡旺季配置 (基于 Jenks Natural Breaks 分析，不一致周取平均后重新分类)',
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source_type': config.DATA_SOURCE_TYPE,
            'fiscal_year_start': f"{config.FISCAL_YEAR_START_MONTH:02d}-{config.FISCAL_YEAR_START_DAY:02d}",
            'season_labels': {
                str(k): v for k, v in config.SEASON_LABELS.items()
            },
            'weeks': []
        }

        # 计算每周的日期范围（以第一个财年为基准）
        first_fy = config.FISCAL_YEARS[0]
        fy_start = pd.Timestamp(first_fy['start'])
        fy_end = pd.Timestamp(first_fy['end'])

        for week_num in range(1, config.WEEKS_PER_YEAR + 1):
            week_start = fy_start + timedelta(days=(week_num - 1) * 7)
            
            # 第52周结束于财年最后一天
            if week_num < 52:
                week_end = fy_start + timedelta(days=week_num * 7 - 1)
            else:
                week_end = fy_end - timedelta(days=1)

            season_code = final_seasons[week_num - 1] if week_num <= len(final_seasons) else 1
            avg_pass = avg_passengers[week_num - 1] if week_num <= len(avg_passengers) else 0

            self.final_config['weeks'].append({
                'week': week_num,
                'date_range': f"{week_start.strftime('%m-%d')} ~ {week_end.strftime('%m-%d')}",
                'avg_passenger': round(avg_pass, 2),
                'season_code': season_code,
                'season_name': config.SEASON_LABELS[season_code]['name'],
                'season_cn': config.SEASON_LABELS[season_code]['cn']
            })

        # 统计
        season_counts = {0: 0, 1: 0, 2: 0}
        for week in self.final_config['weeks']:
            season_counts[week['season_code']] += 1

        print(f"\n最终配置统计:")
        for code, count in season_counts.items():
            info = config.SEASON_LABELS[code]
            print(f"  {info['cn']}: {count} 周 ({count / 52 * 100:.1f}%)")

        # 保存配置
        output_path = config.OUTPUT_DIR / config.OUTPUT_FILES['final_config']
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.final_config, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 最终配置已保存: {output_path}")

    def visualize(self):
        """可视化分析结果"""
        print(f"\n{'=' * 70}")
        print("Step 7: 生成可视化图表")
        print(f"{'=' * 70}")

        fig, axes = plt.subplots(2, 2, figsize=config.FIGURE_SIZE)

        # 图1: 两年周客流对比
        ax1 = axes[0, 0]
        for fy_year, weekly_df in self.weekly_data.items():
            ax1.plot(weekly_df['fiscal_week'], weekly_df['avg_passenger'],
                     marker='o', markersize=3, label=f'FY{fy_year}-{fy_year + 1}')
        ax1.set_xlabel('财年周序号')
        ax1.set_ylabel('周均客流量')
        ax1.set_title('两年周客流量对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2: Jenks 断点可视化
        ax2 = axes[0, 1]
        colors = [config.SEASON_LABELS[i]['color'] for i in range(config.N_CLASSES)]

        for fy_year, weekly_df in self.weekly_data.items():
            data = weekly_df['avg_passenger'].values
            breaks = self.jenks_results[fy_year]['breaks']

            # 绘制直方图
            ax2.hist(data, bins=20, alpha=0.5, label=f'FY{fy_year}')

            # 绘制断点线
            for i, b in enumerate(breaks[1:-1]):
                ax2.axvline(x=b, color=colors[i + 1], linestyle='--', linewidth=2)

        ax2.set_xlabel('周均客流量')
        ax2.set_ylabel('频数')
        ax2.set_title('客流分布与 Jenks 断点')
        ax2.legend()

        # 图3: 季节分布热力图
        ax3 = axes[1, 0]
        if self.final_config:
            weeks = list(range(1, 53))
            seasons = [w['season_code'] for w in self.final_config['weeks']]

            # 创建热力图数据
            heatmap_data = np.array(seasons).reshape(4, 13)  # 4行13列
            im = ax3.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')

            ax3.set_xlabel('周 (每行13周)')
            ax3.set_ylabel('季度')
            ax3.set_title('52周季节分布热力图')

            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax3, ticks=[0, 1, 2])
            cbar.ax.set_yticklabels(['淡季', '平季', '旺季'])

        # 图4: 季节时间线
        ax4 = axes[1, 1]
        if self.final_config:
            weeks = list(range(1, 53))
            seasons = [w['season_code'] for w in self.final_config['weeks']]

            # 绘制条形图
            for i, (week, season) in enumerate(zip(weeks, seasons)):
                color = config.SEASON_LABELS[season]['color']
                ax4.bar(week, 1, color=color, width=1, edgecolor='none')

            ax4.set_xlabel('财年周序号')
            ax4.set_ylabel('')
            ax4.set_title('52周季节时间线')
            ax4.set_yticks([])

            # 添加图例
            patches = [mpatches.Patch(color=info['color'], label=info['cn'])
                       for info in config.SEASON_LABELS.values()]
            ax4.legend(handles=patches, loc='upper right')

            # 添加月份标记
            month_labels = ['10月', '11月', '12月', '1月', '2月', '3月',
                            '4月', '5月', '6月', '7月', '8月', '9月']
            month_positions = [1, 5, 9, 14, 18, 22, 27, 31, 35, 40, 44, 48]
            for pos, label in zip(month_positions, month_labels):
                ax4.axvline(x=pos, color='gray', linestyle=':', alpha=0.5)
                ax4.text(pos, 1.1, label, ha='center', fontsize=8)

        plt.tight_layout()

        output_path = config.OUTPUT_DIR / config.OUTPUT_FILES['visualization']
        plt.savefig(output_path, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        print(f"✓ 可视化图表已保存: {output_path}")

    def run(self):
        """执行完整分析流程"""
        print(f"\n{'#' * 70}")
        print("# M7C 淡旺季分析 (Jenks Natural Breaks)")
        print(f"# 数据范围: 2023.10.01 - 2025.09.30")
        print(f"# 财年定义: 10月1日 - 次年9月30日")
        print(f"# 周定义: 每7天为一周，第52周包含剩余天数(8-9天)")
        print(f"# 数据源类型: {config.DATA_SOURCE_TYPE}")
        print(f"{'#' * 70}")

        if not self.connect_database():
            return False

        try:
            self.load_data()
            self.aggregate_to_weekly()
            self.jenks_classify()
            self.compare_two_years()
            self.generate_final_config()
            self.visualize()

            print(f"\n{'=' * 70}")
            print("✓ 分析完成!")
            print(f"  输出目录: {config.OUTPUT_DIR}")
            print(f"{'=' * 70}")

            # 打印最终配置摘要
            if self.final_config:
                print(f"\n最终52周配置摘要:")
                current_season = None
                segment_start = 1

                for week_info in self.final_config['weeks']:
                    week = week_info['week']
                    season = week_info['season_cn']

                    if current_season is None:
                        current_season = season
                    elif season != current_season or week == 52:
                        # 输出上一个分段
                        if week == 52 and season == current_season:
                            print(f"  第{segment_start:2d}周 ~ 第{week:2d}周: {current_season}")
                        else:
                            print(f"  第{segment_start:2d}周 ~ 第{week - 1:2d}周: {current_season}")
                            if week == 52:
                                print(f"  第{week:2d}周 ~ 第{week:2d}周: {season}")
                        segment_start = week
                        current_season = season

            return True

        except Exception as e:
            print(f"\n✗ 分析出错: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    analyzer = SeasonAnalyzer()
    analyzer.run()
