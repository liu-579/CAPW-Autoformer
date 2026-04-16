import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# =====================================================================
# 配置区域 (CONFIG)
# 所有需要更改的设置（数据、路径、外观等）都集中在此，请根据需要调整
# =====================================================================
class Config:
    # ---------------------------------------------------------
    # 1. 数据与路径设置
    # ---------------------------------------------------------
    # 数据文件绝对路径
    DATA_PATH = r"e:/2024liushaoxuan/learn/shiyan1/data/output/m11/m11_LSTM/inference_eedscy_lstm_all.csv"

    # 图片输出目录 (如果不存在会自动创建)
    OUTPUT_DIR = r"e:/2024liushaoxuan/learn/shiyan1/data/output/画图"
    # 输出图片名称前缀
    OUTPUT_FILENAME = "动态权重网络"
    # 输出格式列表 (支持多种格式如 'png', 'svg', 'pdf')
    OUTPUT_FORMATS = ['png']
    # 导出图片的清晰度 (DPI)，通常 300 适合打印和高清展示
    DPI = 300

    # ---------------------------------------------------------
    # 2. 数据筛选与字段设置
    # ---------------------------------------------------------
    # 时间列的表头名称
    TIME_COLUMN = "datetime"

    # 需要绘制折线图的 5 组权重数据列名
    WEIGHT_COLUMNS = ["w_scenery", "w_transportation", "w_food", "w_consumption", "w_service"]

    # 筛选的时间范围 (格式: YYYY-MM-DD)
    START_DATE = "2025-06-30"
    END_DATE = "2025-09-30"

    # ---------------------------------------------------------
    # 3. 背景染色 (工作日/非工作日) 开关与设置
    # ---------------------------------------------------------
    # 【开关】是否开启非工作日的背景染色 (True: 开启染色, False: 关闭染色)
    USE_DAY_OFF_BACKGROUND = True
    # 表示是否为休息日的二值列名 (1: 非工作日, 0: 工作日)
    DAY_OFF_COLUMN = "is_day_off"
    # 背景条纹的填充颜色
    DAY_OFF_COLOR = "lightgray"
    # 背景颜色的透明度 (0.0 完全透明 ~ 1.0 完全不透明)
    DAY_OFF_ALPHA = 0.5
    # 图例中背景颜色的名称
    DAY_OFF_LABEL = "非工作日 (Day Off)"

    # ---------------------------------------------------------
    # 4. 图表整体尺寸、标题与坐标轴设置
    # ---------------------------------------------------------
    # 图表尺寸 (宽度, 高度) 单位: 英寸
    FIG_SIZE = (14, 6)

    # 图表标题及字体大小
    TITLE = "动态权重变化趋势 (2025.06.30 - 2025.09.30)"
    TITLE_FONTSIZE = 16

    # Y轴显示范围 (上下限)
    Y_MIN = -0.1
    Y_MAX = 2.1
    # X轴与Y轴的标签名字和大小
    Y_LABEL = "权重值"
    X_LABEL = "日期"
    LABEL_FONTSIZE = 12
    # 坐标轴刻度字体大小
    TICK_FONTSIZE = 10

    # ---------------------------------------------------------
    # 5. 折线样式与图例设置
    # ---------------------------------------------------------
    # 折线颜色列表 (顺序对应 WEIGHT_COLUMNS 的 5 个变量)
    LINE_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]
    # 折线对应的图例显示名称 (顺序对应 WEIGHT_COLUMNS 的 5 个变量)
    LEGEND_LABELS = ["景点 (w_scenery)", "交通 (w_transportation)",
                     "餐饮 (w_food)", "消费 (w_consumption)", "服务 (w_service)"]
    # 折线的线条粗细
    LINE_WIDTH = 1.5

    # 图例字体大小
    LEGEND_FONTSIZE = 10

    # ---------------------------------------------------------
    # 6. 基准线 (横向标尺) 设置
    # ---------------------------------------------------------
    # 【开关】是否绘制基准线
    DRAW_BASELINE = True
    # 基准线在 Y 轴的位置
    BASELINE_Y = 1.0
    # 基准线颜色
    BASELINE_COLOR = "red"
    # 基准线样式 ('--' 为虚线, '-' 为实线, ':' 为点线)
    BASELINE_STYLE = "--"
    # 基准线线条粗细
    BASELINE_WIDTH = 1.5
    # 基准线图例名称
    BASELINE_LABEL = "基准线 (y=1.0)"

    # ---------------------------------------------------------
    # 7. 系统中文字体设置 (解决中文变成方块的问题)
    # ---------------------------------------------------------
    # Windows 推荐: 'SimHei' (黑体), 'Microsoft YaHei' (微软雅黑)
    # macOS 推荐: 'Arial Unicode MS', 'PingFang SC'
    FONT_FAMILY = "SimHei"

# =====================================================================
# 代码执行逻辑 (通常不需要修改此处代码)
# =====================================================================
def plot_dynamic_weights():
    # 1. 配置 matplotlib 显示中文和负号
    plt.rcParams['font.sans-serif'] = [Config.FONT_FAMILY]
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 读取 CSV 数据
    print(f"正在读取数据: {Config.DATA_PATH}")
    try:
        df = pd.read_csv(Config.DATA_PATH)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {Config.DATA_PATH}，请检查路径是否正确。")
        return

    # 将时间列转换为 Pandas 的 datetime 格式，并设置为数据框的索引，方便做时间筛选
    df[Config.TIME_COLUMN] = pd.to_datetime(df[Config.TIME_COLUMN])
    df = df.set_index(Config.TIME_COLUMN)
    df = df.sort_index()

    # 3. 按时间段截取数据
    print(f"正在筛选时间段: {Config.START_DATE} 至 {Config.END_DATE}")
    df_filtered = df.loc[Config.START_DATE:Config.END_DATE]

    if df_filtered.empty:
        print("警告: 筛选后数据为空！请检查时间范围是否正确或数据中是否包含该时间段。")
        return

    # 4. 创建画板与图表
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    # 5. 【核心】绘制 5 条权重随时间变化的折线
    for col, color, label in zip(Config.WEIGHT_COLUMNS, Config.LINE_COLORS, Config.LEGEND_LABELS):
        if col in df_filtered.columns:
            ax.plot(df_filtered.index, df_filtered[col],
                    color=color, label=label, linewidth=Config.LINE_WIDTH)
        else:
            print(f"警告: 数据中不存在列名 '{col}'，已跳过该列绘制。")

    # 6. 【核心】绘制 y=1.0 的横向基准线
    if Config.DRAW_BASELINE:
        ax.axhline(y=Config.BASELINE_Y, color=Config.BASELINE_COLOR,
                   linestyle=Config.BASELINE_STYLE, linewidth=Config.BASELINE_WIDTH,
                   label=Config.BASELINE_LABEL)

    # 7. 【核心】绘制背景条纹 (根据 is_day_off 标识非工作日)
    if Config.USE_DAY_OFF_BACKGROUND and Config.DAY_OFF_COLUMN in df_filtered.columns:
        # 获取一个布尔值序列：当 is_day_off 为 1 时为 True
        is_day_off = df_filtered[Config.DAY_OFF_COLUMN] == 1

        # 填充背景色：在 x 轴为时间，在布尔序列为 True 的区域填充从 y=0 到 y=1 的图表高度
        # transform=ax.get_xaxis_transform() 确保填充的高度充满整个图表
        ax.fill_between(df_filtered.index, 0, 1,
                        where=is_day_off, color=Config.DAY_OFF_COLOR,
                        alpha=Config.DAY_OFF_ALPHA, transform=ax.get_xaxis_transform(),
                        label=Config.DAY_OFF_LABEL, step='mid')
    elif Config.USE_DAY_OFF_BACKGROUND:
        print(f"警告: 数据中不存在标识工作日的列 '{Config.DAY_OFF_COLUMN}'，无法绘制背景染色。")

    # 8. 图表排版与细节设置
    # 设置Y轴上下限
    ax.set_ylim(Config.Y_MIN, Config.Y_MAX)
    # 设置标题和标签
    ax.set_title(Config.TITLE, fontsize=Config.TITLE_FONTSIZE, pad=15)
    ax.set_xlabel(Config.X_LABEL, fontsize=Config.LABEL_FONTSIZE)
    ax.set_ylabel(Config.Y_LABEL, fontsize=Config.LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=Config.TICK_FONTSIZE)

    # 设置X轴日期的显示格式 (如 '07-01')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    # 日期标签倾斜，避免文字过长互相重叠
    plt.xticks(rotation=45)

    # 9. 图例设置：将图例放在图表外部的右侧，避免遮挡折线和背景
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=Config.LEGEND_FONTSIZE)

    # 自动紧凑布局，防止图例外溢被裁切
    plt.tight_layout()

    # 10. 保存与导出
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    for fmt in Config.OUTPUT_FORMATS:
        out_path = os.path.join(Config.OUTPUT_DIR, f"{Config.OUTPUT_FILENAME}.{fmt}")
        plt.savefig(out_path, format=fmt, dpi=Config.DPI, bbox_inches='tight')
        print(f"图表已成功保存: {out_path}")

    # 如果想在代码执行时实时预览图表，可以取消下方代码的注释
    # plt.show()

if __name__ == "__main__":
    plot_dynamic_weights()
