import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# =====================================================================
# 配置区域 (CONFIG)
# =====================================================================
class Config:
    # ---------------------------------------------------------
    # 0. 语言选项 (LANGUAGE SETTINGS)
    # ---------------------------------------------------------
    # 可选 'EN' (英文, 适用于SCI发表) 或 'ZH' (中文)
    LANGUAGE = 'EN'

    # ---------------------------------------------------------
    # 1. 数据与路径设置
    # ---------------------------------------------------------
    DATA_PATH = r"e:/2024liushaoxuan/learn/shiyan1/data/output/m11/m11_LSTM/inference_eedscy_lstm_all.csv"
    OUTPUT_DIR = r"e:/2024liushaoxuan/learn/shiyan1/data/output/画图"
    OUTPUT_FORMATS = ['pdf', 'png'] # 建议为SCI期刊添加 pdf 矢量图格式
    DPI = 300

    # 根据语言动态设置输出文件名
    OUTPUT_FILENAME = "Dynamic_Weights_Network" if LANGUAGE == 'EN' else "动态权重网络"

    # ---------------------------------------------------------
    # 2. 数据筛选与字段设置
    # ---------------------------------------------------------
    TIME_COLUMN = "datetime"
    WEIGHT_COLUMNS = ["w_scenery", "w_transportation", "w_food", "w_consumption", "w_service"]
    START_DATE = "2025-06-30"
    END_DATE = "2025-09-30"

    # ---------------------------------------------------------
    # 3. 背景染色 (工作日/非工作日) 设置
    # ---------------------------------------------------------
    USE_DAY_OFF_BACKGROUND = True
    DAY_OFF_COLUMN = "is_day_off"
    DAY_OFF_COLOR = "lightgray"
    DAY_OFF_ALPHA = 0.5

    # ---------------------------------------------------------
    # 4. 图表整体尺寸与坐标轴范围设置
    # ---------------------------------------------------------
    FIG_SIZE = (14, 6)
    TITLE_FONTSIZE = 16
    Y_MIN = -0.1
    Y_MAX = 2.1
    LABEL_FONTSIZE = 12
    TICK_FONTSIZE = 10

    # ---------------------------------------------------------
    # 5. 折线样式与设置
    # ---------------------------------------------------------
    LINE_COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]
    LINE_WIDTH = 1.5
    LEGEND_FONTSIZE = 10

    # ---------------------------------------------------------
    # 6. 基准线 (横向标尺) 设置
    # ---------------------------------------------------------
    DRAW_BASELINE = True
    BASELINE_Y = 1.0
    BASELINE_COLOR = "red"
    BASELINE_STYLE = "--"
    BASELINE_WIDTH = 1.5

    # ---------------------------------------------------------
    # 7. 语言相关的动态文本配置
    # ---------------------------------------------------------
    if LANGUAGE == 'EN':
        TITLE = "Trends of Dynamic Weights (2025.06.30 - 2025.09.30)"
        Y_LABEL = "Weight Value"
        X_LABEL = "Date"
        LEGEND_LABELS = ["Scenery", "Transportation", "Food", "Consumption", "Service"]
        DAY_OFF_LABEL = "Non-working Days"
        BASELINE_LABEL = "Baseline (y=1.0)"
        FONT_FAMILY = "Times New Roman"  # SCI 推荐字体
    else:
        TITLE = "动态权重变化趋势 (2025.06.30 - 2025.09.30)"
        Y_LABEL = "权重值"
        X_LABEL = "日期"
        LEGEND_LABELS = ["景点 (w_scenery)", "交通 (w_transportation)",
                         "餐饮 (w_food)", "消费 (w_consumption)", "服务 (w_service)"]
        DAY_OFF_LABEL = "非工作日 (Day Off)"
        BASELINE_LABEL = "基准线 (y=1.0)"
        FONT_FAMILY = "SimHei"  # 中文显示字体


# =====================================================================
# 代码执行逻辑
# =====================================================================
def plot_dynamic_weights():
    # 1. 字体与显示配置
    plt.rcParams['font.family'] = Config.FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 读取 CSV 数据
    print(f"Loading data from: {Config.DATA_PATH}")
    try:
        df = pd.read_csv(Config.DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {Config.DATA_PATH}")
        return

    df[Config.TIME_COLUMN] = pd.to_datetime(df[Config.TIME_COLUMN])
    df = df.set_index(Config.TIME_COLUMN)
    df = df.sort_index()

    # 3. 按时间段截取数据
    df_filtered = df.loc[Config.START_DATE:Config.END_DATE]
    if df_filtered.empty:
        print("Warning: The filtered DataFrame is empty. Please check the date range.")
        return

    # 4. 创建画板与图表
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    # 5. 绘制 5 条权重折线
    for col, color, label in zip(Config.WEIGHT_COLUMNS, Config.LINE_COLORS, Config.LEGEND_LABELS):
        if col in df_filtered.columns:
            ax.plot(df_filtered.index, df_filtered[col],
                    color=color, label=label, linewidth=Config.LINE_WIDTH)

    # 6. 绘制基准线
    if Config.DRAW_BASELINE:
        ax.axhline(y=Config.BASELINE_Y, color=Config.BASELINE_COLOR,
                   linestyle=Config.BASELINE_STYLE, linewidth=Config.BASELINE_WIDTH,
                   label=Config.BASELINE_LABEL)

    # 7. 绘制背景条纹
    if Config.USE_DAY_OFF_BACKGROUND and Config.DAY_OFF_COLUMN in df_filtered.columns:
        is_day_off = df_filtered[Config.DAY_OFF_COLUMN] == 1
        ax.fill_between(df_filtered.index, 0, 1,
                        where=is_day_off, color=Config.DAY_OFF_COLOR,
                        alpha=Config.DAY_OFF_ALPHA, transform=ax.get_xaxis_transform(),
                        label=Config.DAY_OFF_LABEL, step='mid')

    # 8. 图表排版与细节设置
    ax.set_ylim(Config.Y_MIN, Config.Y_MAX)
    ax.set_title(Config.TITLE, fontsize=Config.TITLE_FONTSIZE, pad=15)
    ax.set_xlabel(Config.X_LABEL, fontsize=Config.LABEL_FONTSIZE)
    ax.set_ylabel(Config.Y_LABEL, fontsize=Config.LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=Config.TICK_FONTSIZE)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=45)

    # 9. 图例设置
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=Config.LEGEND_FONTSIZE)
    plt.tight_layout()

    # 10. 保存与导出
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    for fmt in Config.OUTPUT_FORMATS:
        out_path = os.path.join(Config.OUTPUT_DIR, f"{Config.OUTPUT_FILENAME}.{fmt}")
        plt.savefig(out_path, format=fmt, dpi=Config.DPI, bbox_inches='tight')
        print(f"Figure saved successfully: {out_path}")

if __name__ == "__main__":
    plot_dynamic_weights()