"""
M1数据清洗配置文件
在PyCharm中直接修改这个文件，然后运行主程序即可
"""

from pathlib import Path

# ==================== 项目路径配置 ====================
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
OUTPUT_DIR = DATA_DIR / '输出'

# 确保目录存在
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 文件路径配置 ====================
# 输入文件路径（请修改为你的实际路径）
INPUT_FILE = r"D:\learning\shiyan1\data\raw\乐山大佛、黄鹤楼评价数据.xlsx"

# 输出文件路径
OUTPUT_FILE = OUTPUT_DIR / "m1_cleaned_comments.jsonl"

# 统计报告输出路径（可选）
STATS_OUTPUT = OUTPUT_DIR / "m1_cleaning_report.txt"


# ==================== 数据字段配置 ====================
# 文本列名（根据你的Excel文件调整）
TEXT_COLUMN = "正文"

# 去重使用的列（可以是多个列的组合）
DEDUP_COLUMNS = ["作者", "正文"]

# 其他可能用到的列名
AUTHOR_COLUMN = "作者"          # 作者列
SCENIC_SPOT_COLUMN = "景区"     # 景区列
PUBLISH_TIME_COLUMN = "发布时间"  # 发布时间列
SOURCE_TYPE_COLUMN = "source_type"  # 来源类型列
KEYWORDS_COLUMN = "keywords"    # 关键词列


# ==================== 清洗规则配置 ====================
# 最小文本长度（字符数）
MIN_TEXT_LENGTH = 10

# 最大文本长度（字符数）
MAX_TEXT_LENGTH = 5000

# 中文字符最小占比
MIN_CHINESE_RATIO = 0.3

# 是否移除URL链接
REMOVE_URLS = True

# 是否移除emoji和特殊符号
REMOVE_SPECIAL_CHARS = False

# 是否移除HTML标签
REMOVE_HTML_TAGS = True

# 是否移除@用户名
REMOVE_MENTIONS = True

# 是否移除话题标签（#话题#）
REMOVE_HASHTAGS = False  # Excel数据一般不需要

# 是否去重
REMOVE_DUPLICATES = True


# ==================== 运行模式配置 ====================
# 是否为预览模式（True=只看前10条，False=正式运行）
DRY_RUN = False

# 预览模式下显示的条数
PREVIEW_COUNT = 10

# 是否显示详细日志
VERBOSE = True

# 是否生成统计报告文件
SAVE_STATS_REPORT = True


# ==================== 高级配置 ====================
# 是否保留元数据
KEEP_METADATA = True

# 元数据字段映射（可以添加更多字段）
METADATA_FIELDS = {
    'author': AUTHOR_COLUMN,
    'scenic_spot': SCENIC_SPOT_COLUMN,
    'publish_time': PUBLISH_TIME_COLUMN,
    'source_type': SOURCE_TYPE_COLUMN,
    'keywords': KEYWORDS_COLUMN
}


# ==================== 使用说明 ====================
"""
使用方法：

1. 修改输入文件路径：
   INPUT_FILE = r"D:\learning\shiyan1\data\raw\你的文件.xlsx"

2. 检查字段名称是否匹配：
   TEXT_COLUMN = "正文"  # 修改为你Excel中的文本列名
   AUTHOR_COLUMN = "作者"  # 修改为你Excel中的作者列名
   等等...

3. 调整清洗规则：
   MIN_TEXT_LENGTH = 10  # 最小文本长度
   MIN_CHINESE_RATIO = 0.3  # 中文占比
   REMOVE_URLS = True  # 是否移除链接

4. 运行脚本：
   python scripts/m1_clean_comments.py

5. 查看结果：
   输出文件：data/输出/cleaned_comments.jsonl
   统计报告：data/输出/cleaning_report.txt
"""