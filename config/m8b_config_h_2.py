# -*- coding: utf-8 -*-
"""
张量构建配置文件 V2 (小时级数据粒度)
Data Pipeline Engineer: Tensor Builder Configuration (Hourly) V2
用途: 定义数据集划分、窗口参数、归一化配置等
改进: 
  1. 添加类别特征配置（One-Hot编码）
  2. 添加冗余滞后特征移除配置
"""

# ==================== 数据库连接配置 ====================
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'password',  # 请修改为实际密码
    'database': 'yq_data',  # 请修改为实际数据库名
    'charset': 'utf8mb4'
}

# ==================== 输入表配置 ====================
INPUT_TABLE = "m8a_cleaned_feature_store_hourly_2_响沙湾"
TARGET_COLUMN = "passenger_count"

# ==================== 排除列配置 ====================
# 这些列不参与特征工程（元数据列）
EXCLUDE_COLS = [
    'id',
    'scenic_name',
    'date',
    'datetime',
    'hour',
    'created_at',
    'date_str',
    'source_table',
    'wind_dir_int',
    'fiscal_week',
    'season_code',
    'weather_cond_int'
]

# ==================== [新增] 类别特征配置 ====================
# 这些列是类别编码，需要使用 One-Hot 编码而非数值缩放
CATEGORICAL_COLS = [
    'weather_grade',  # 天气条件编码
    # 'wind_dir_int'       # 风向编码
]

# ==================== [新增] 冗余滞后特征移除配置 ====================
# 滑动窗口本身已提供历史信息，这些手工构造的滞后特征是冗余的
# 设置为 True 时移除所有滞后特征，False 时保留
REMOVE_LAG_FEATURES = False

# 滞后特征的识别模式（正则表达式后缀）
LAG_FEATURE_PATTERNS = [
    # '_lag1h',
    # '_lag2h', 
    # '_lag3h',
    '_lag24h',
    '_lag48h',
    '_lag168h',
    '_roll168h'
]

# ==================== 时间序列窗口配置 ====================
# 小时级数据：使用过去168小时(7天)的数据，预测未来24小时(1天)
SEQ_LEN = 96      # 输入序列长度（使用过去168小时的数据，约7天）
PRED_LEN = 24      # 预测长度（预测未来24小时，约1天）

# ==================== 数据集划分配置 ====================
TRAIN_RATIO = 0.7    # 训练集比例
VAL_RATIO = 0.2     # 验证集比例
TEST_RATIO = 0.1    # 测试集比例（自动计算: 1 - TRAIN - VAL）

# 验证比例总和
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "数据集划分比例总和必须为1.0"

# ==================== 归一化配置 (混合策略) ====================
# 注意：代码已升级为 Hybrid Strategy，以下参数仅对"其他连续数值列"生效
NORMALIZATION_METHOD = 'minmax'  # 默认归一化方法
NORM_RANGE = (0, 1)              # MinMaxScaler 的范围

# ==================== [新增] Target 列归一化配置 ====================
# Target 列 (passenger_count) 的归一化方法
# 可选值: 'minmax' 或 'standard'
# - 'minmax': Log1p + MinMaxScaler(0,1)，输出范围 [0, 1]
# - 'standard': Log1p + StandardScaler，输出范围约 [-2, 2]，高值区误差放大更可控
TARGET_SCALER_TYPE = 'minmax'  # 'minmax' 或 'standard'

# ==================== 特征分类关键词 ====================
# [关键修改] 情感维度分组关键词
# 注意：列表顺序至关重要，必须与 M9 模型中 Grouped Broadcasting 的权重顺序一致
SENTIMENT_KEYWORDS = [
    'scenery',        # 景色 (Group 0)
    'transportation', # 交通 (Group 1)
    'food',           # 美食 (Group 2)
    'consumption',    # 消费 (Group 3)
    'service'         # 服务 (Group 4)
]

# ==================== [新增] 客流衍生特征配置 ====================
# 客流衍生特征的列名前缀（滞后项和移动平均项）
# 这些特征与 passenger_count 同源，需要使用相同的 Log1p + Scaler 处理
PASSENGER_DERIVED_PREFIX = 'passenger_count_'

# ==================== [新增] 时间范围筛选配置 ====================
# 设置为 None 时使用全部数据（保持原有行为）
# 设置具体日期字符串时，仅使用该时间范围内的数据构建数据集
# 格式: 'YYYY-MM-DD'  例如: '2023-10-01'
# 注意: 生成的数据集仍然与原版模型完全兼容（特征结构不变）
DATE_RANGE_START = None   # 起始日期（含），例如 '2023-10-01'
DATE_RANGE_END = None     # 结束日期（含），例如 '2023-10-07'

# ==================== 输出文件配置 ====================
from pathlib import Path
# 项目根目录
BASE_DIR = Path(__file__).parent.parent

OUTPUT_DIR = BASE_DIR / "data/output/dataset/m8b_xsw_hourly_time_96"   # 输出目录（小时级 V2）
SCALERS_FILENAME = "scalers.pkl"   # [修改] 保存所有 Scaler 的字典
FEATURE_MAP_FILENAME = "feature_map.json"  # 特征映射保存文件名
ONEHOT_ENCODER_FILENAME = "onehot_encoder.pkl"  # [新增] One-Hot 编码器保存文件名
DATETIME_FILENAME = "datetime_index.npy"         # [新增] 保存原始时间戳供推理使用

# 张量文件命名
TENSOR_FILES = {
    'train_x': 'train_x.npy',
    'train_y': 'train_y.npy',
    'val_x': 'val_x.npy',
    'val_y': 'val_y.npy',
    'test_x': 'test_x.npy',
    'test_y': 'test_y.npy'
}

# ==================== 数据类型配置 ====================
DTYPE = 'float32'  # 张量数据类型

# ==================== 日志配置 ====================
LOG_LEVEL = 'INFO'
SHOW_SHAPE_INFO = True      # 是否显示形状信息
SHOW_SAMPLE_DATA = True     # 是否显示样本数据
