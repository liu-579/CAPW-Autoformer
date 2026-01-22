# -*- coding: utf-8 -*-
"""
张量构建配置文件 (小时级数据粒度)
Data Pipeline Engineer: Tensor Builder Configuration (Hourly)
用途: 定义数据集划分、窗口参数、归一化配置等
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
INPUT_TABLE = "cleaned_feature_store_hourly_响沙湾"
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
    'source_table'
]

# ==================== 时间序列窗口配置 ====================
# 小时级数据：使用过去720小时(30天)的数据，预测未来168小时(7天)
SEQ_LEN = 720      # 输入序列长度（使用过去720小时的数据，约30天）
PRED_LEN = 168     # 预测长度（预测未来168小时，约7天）

# ==================== 数据集划分配置 ====================
TRAIN_RATIO = 0.7    # 训练集比例
VAL_RATIO = 0.15     # 验证集比例
TEST_RATIO = 0.15    # 测试集比例（自动计算: 1 - TRAIN - VAL）

# 验证比例总和
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "数据集划分比例总和必须为1.0"

# ==================== 归一化配置 (混合策略) ====================
# 注意：代码已升级为 Hybrid Strategy，以下参数仅对"其他连续数值列"生效
NORMALIZATION_METHOD = 'minmax'  # 默认归一化方法
NORM_RANGE = (0, 1)              # MinMaxScaler 的范围

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

# ==================== 输出文件配置 ====================
from pathlib import Path
# 项目根目录
BASE_DIR = Path(__file__).parent.parent

OUTPUT_DIR = BASE_DIR / "data/output/m8b_xsw_hourly"   # 输出目录（小时级）
SCALERS_FILENAME = "scalers.pkl"   # [修改] 保存所有 Scaler 的字典
FEATURE_MAP_FILENAME = "feature_map.json"  # 特征映射保存文件名

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