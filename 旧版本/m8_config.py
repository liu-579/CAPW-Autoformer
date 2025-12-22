"""
时序数据集构建配置文件
集中管理所有参数，便于统一修改和维护
"""

import os
from datetime import datetime

# ============================================================================
# 1. 数据库配置
# ============================================================================
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'password',
    'database': 'yq_data',
    'charset': 'utf8mb4'
}

# ============================================================================
# 2. 数据源配置
# ============================================================================
# 输入数据表名
INPUT_TABLE = 'fused_training_data'

# 目标景区名称（用于筛选数据）
TARGET_SCENIC_NAME = '鄂尔多斯草原'

# 目标预测列（需要预测的指标）
TARGET_COL = 'passenger_count'

# ============================================================================
# 3. 输出路径配置（新增）
# ============================================================================
from pathlib import Path
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
# 输出文件根目录
OUTPUT_ROOT_DIR = PROJECT_ROOT/'data/output/m8_eedscy_new'

# 数据集输出目录
DATASET_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'datasets')

# 模型文件输出目录（用于保存 scaler 和 feature_map）
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'models')

# 日志输出目录（可选，用于保存运行日志）
LOG_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'logs')

# 是否自动创建输出目录
AUTO_CREATE_DIRS = True

# ============================================================================
# 4. 时间序列参数
# ============================================================================
# 输入序列长度（使用过去 N 天的数据）
SEQ_LEN = 30

# 预测序列长度（预测未来 N 天）
PRED_LEN = 7

# ============================================================================
# 5. 数据集切分比例
# ============================================================================
# 训练集比例
TRAIN_RATIO = 0.7

# 验证集比例
VAL_RATIO = 0.15

# 测试集比例（自动计算：1 - TRAIN_RATIO - VAL_RATIO）
TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO

# ============================================================================
# 6. 特征工程配置
# ============================================================================
# 需要排除的列（不参与模型训练）
EXCLUDE_COLS = [
    'date',           # 日期字段
    'scenic_name',    # 景区名称（已通过筛选固定）
    'id'              # 主键ID（如果有）
]

# 类别型特征（需要进行 LabelEncoding）
CATEGORICAL_COLS = [
    'weather',        # 天气类型
    'wind_direction', # 风向
    'is_weekend',     # 是否周末
    'is_holiday'      # 是否节假日
]

# 情感特征识别关键词（用于区分情感特征和环境特征）
SENTIMENT_FEATURE_KEYWORD = 'score'  # 列名包含 'score' 的认为是情感特征

# ============================================================================
# 7. 归一化配置
# ============================================================================
# MinMaxScaler 的归一化范围
SCALER_FEATURE_RANGE = (0, 1)

# ============================================================================
# 8. 文件命名配置
# ============================================================================
# 数据集文件名前缀（会自动加上景区名称和时间戳）
DATASET_PREFIX = 'tourism_seq'

# 是否在文件名中包含时间戳
INCLUDE_TIMESTAMP = False

# 是否在文件名中包含景区名称
INCLUDE_SCENIC_NAME = True

# ============================================================================
# 9. 日志配置
# ============================================================================
# 是否显示详细日志
VERBOSE_LOGGING = True

# 是否显示数据统计信息
SHOW_DATA_STATS = True

# 是否显示完整特征列表
SHOW_FEATURE_LIST = False

# 是否保存日志到文件
SAVE_LOG_TO_FILE = False

# ============================================================================
# 辅助函数
# ============================================================================

def get_database_url():
    """
    生成数据库连接 URL

    返回:
        str: SQLAlchemy 数据库连接字符串
    """
    return (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        f"?charset={DB_CONFIG['charset']}"
    )


def create_output_directories():
    """
    创建所有输出目录（如果不存在）
    """
    if AUTO_CREATE_DIRS:
        dirs_to_create = [
            OUTPUT_ROOT_DIR,
            DATASET_OUTPUT_DIR,
            MODEL_OUTPUT_DIR,
            LOG_OUTPUT_DIR
        ]

        for dir_path in dirs_to_create:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"✅ 已创建目录：{dir_path}")


def get_filename_prefix():
    """
    生成文件名前缀

    返回:
        str: 文件名前缀（包含景区名称和时间戳）
    """
    parts = [DATASET_PREFIX]

    # 添加景区名称
    if INCLUDE_SCENIC_NAME:
        # 清理景区名称中的特殊字符
        scenic_name_clean = TARGET_SCENIC_NAME.replace(' ', '_')
        parts.append(scenic_name_clean)

    # 添加时间戳
    if INCLUDE_TIMESTAMP:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        parts.append(timestamp)

    return '_'.join(parts)


def get_dataset_filenames():
    """
    生成所有数据集文件的完整路径

    返回:
        dict: 包含所有数据集文件路径的字典
    """
    prefix = get_filename_prefix()

    return {
        'train_x': os.path.join(DATASET_OUTPUT_DIR, f'{prefix}_train_x.npy'),
        'train_y': os.path.join(DATASET_OUTPUT_DIR, f'{prefix}_train_y.npy'),
        'val_x': os.path.join(DATASET_OUTPUT_DIR, f'{prefix}_val_x.npy'),
        'val_y': os.path.join(DATASET_OUTPUT_DIR, f'{prefix}_val_y.npy'),
        'test_x': os.path.join(DATASET_OUTPUT_DIR, f'{prefix}_test_x.npy'),
        'test_y': os.path.join(DATASET_OUTPUT_DIR, f'{prefix}_test_y.npy')
    }


def get_scaler_filename():
    """
    生成 Scaler 对象的保存路径

    返回:
        str: Scaler 文件完整路径
    """
    prefix = get_filename_prefix()
    return os.path.join(MODEL_OUTPUT_DIR, f'{prefix}_scaler.pkl')


def get_feature_map_filename():
    """
    生成特征映射表的保存路径

    返回:
        str: 特征映射表文件完整路径
    """
    prefix = get_filename_prefix()
    return os.path.join(MODEL_OUTPUT_DIR, f'{prefix}_feature_map.json')


def get_log_filename():
    """
    生成日志文件的保存路径

    返回:
        str: 日志文件完整路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(LOG_OUTPUT_DIR, f'build_log_{timestamp}.txt')


def validate_config():
    """
    验证配置参数的合法性

    抛出:
        ValueError: 如果配置参数不合法
    """
    # 验证数据集切分比例
    if not (0 < TRAIN_RATIO < 1):
        raise ValueError(f"TRAIN_RATIO 必须在 (0, 1) 范围内，当前值：{TRAIN_RATIO}")

    if not (0 < VAL_RATIO < 1):
        raise ValueError(f"VAL_RATIO 必须在 (0, 1) 范围内，当前值：{VAL_RATIO}")

    if TRAIN_RATIO + VAL_RATIO >= 1:
        raise ValueError(
            f"TRAIN_RATIO + VAL_RATIO 必须小于 1，"
            f"当前值：{TRAIN_RATIO} + {VAL_RATIO} = {TRAIN_RATIO + VAL_RATIO}"
        )

    # 验证序列长度
    if SEQ_LEN <= 0:
        raise ValueError(f"SEQ_LEN 必须大于 0，当前值：{SEQ_LEN}")

    if PRED_LEN <= 0:
        raise ValueError(f"PRED_LEN 必须大于 0，当前值：{PRED_LEN}")

    # 验证归一化范围
    if len(SCALER_FEATURE_RANGE) != 2:
        raise ValueError(f"SCALER_FEATURE_RANGE 必须是一个包含两个元素的元组")

    if SCALER_FEATURE_RANGE[0] >= SCALER_FEATURE_RANGE[1]:
        raise ValueError(
            f"SCALER_FEATURE_RANGE 的第一个值必须小于第二个值，"
            f"当前值：{SCALER_FEATURE_RANGE}"
        )

    # 验证目标列
    if not TARGET_COL:
        raise ValueError("TARGET_COL 不能为空")

    # 验证景区名称
    if not TARGET_SCENIC_NAME:
        raise ValueError("TARGET_SCENIC_NAME 不能为空")

    # 创建输出目录
    create_output_directories()

    print("✅ 配置验证通过")


def print_config_summary():
    """
    打印配置摘要信息
    """
    print("\n" + "="*60)
    print("配置摘要")
    print("="*60)
    print(f"数据库：{DB_CONFIG['database']}")
    print(f"输入表：{INPUT_TABLE}")
    print(f"目标景区：{TARGET_SCENIC_NAME}")
    print(f"目标列：{TARGET_COL}")
    print(f"\n输出路径：")
    print(f"  根目录：{OUTPUT_ROOT_DIR}")
    print(f"  数据集目录：{DATASET_OUTPUT_DIR}")
    print(f"  模型文件目录：{MODEL_OUTPUT_DIR}")
    print(f"  日志目录：{LOG_OUTPUT_DIR}")
    print(f"\n时序参数：")
    print(f"  输入序列长度：{SEQ_LEN}")
    print(f"  预测序列长度：{PRED_LEN}")
    print(f"\n数据集切分：")
    print(f"  训练集：{TRAIN_RATIO*100:.1f}%")
    print(f"  验证集：{VAL_RATIO*100:.1f}%")
    print(f"  测试集：{TEST_RATIO*100:.1f}%")
    print("="*60 + "\n")


# ============================================================================
# 配置初始化（导入时自动执行）
# ============================================================================
if __name__ != "__main__":
    # 当作为模块导入时，自动创建输出目录
    create_output_directories()