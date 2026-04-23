import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# =====================================================================
# 配置区域 (CONFIG) - 客流预测对比图 (SCI学术级排版)
# =====================================================================
class Config:
    # ---------------------------------------------------------
    # 1. 输出路径与格式设置
    # ---------------------------------------------------------
    OUTPUT_DIR = r"e:/2024liushaoxuan/learn/shiyan1/data/output/画图"
    OUTPUT_FILENAME = "基线模型对比"
    # SCI期刊通常要求高分辨率(如600 DPI)和矢量图(SVG/EPS/PDF)或高质量TIFF/PNG
    OUTPUT_FORMATS = ['png']
    DPI = 600

    # ---------------------------------------------------------
    # 2. 数据筛选设置
    # ---------------------------------------------------------
    TIME_COLUMN = "datetime"
    REAL_FLOW_COLUMN = "real_flow"
    PRED_FLOW_COLUMN = "pred_flow"

    # 筛选的时间范围 (格式: YYYY-MM-DD)
    START_DATE = "2025-06-30"
    END_DATE = "2025-09-30"

    # ---------------------------------------------------------
    # 3. 核心数据源配置 (支持无限添加对比模型)
    # ---------------------------------------------------------
    # [真实值] 的线条样式 (通常真实值用黑色实线加粗，以示基准)
    REAL_FLOW_STYLE = {
        "label": "Ground Truth",     # 图例名称
        "color": "#000000",          # 纯黑色
        "linestyle": "-",            # 实线
        "linewidth": 1.5,            # 较粗的线条突出真实值
        "alpha": 0.6,                # 透明度
        "zorder": 10                 # 确保真实值图层在最上方，不被预测线遮挡
    }

    # [预测模型列表]
    # 你可以在这里任意添加或删除模型，只需复制字典并修改路径、颜色、名称即可。
    # 配色建议：使用了Nature/Science期刊常用的NPG配色方案，高级且对色弱友好
    MODELS = [
        {
            "name": "LSTM", # 图例中显示的名称
            "path": r"e:/2024liushaoxuan/learn/shiyan1/data/output/m11/m11_LSTM/inference_eedscy_lstm_all.csv",
            "color": "#8DA0CB",      # NPG 红色 (学术红)
            "linestyle": "-",        # 实线
            "linewidth": 1.5,
            "alpha": 0.9,
            "zorder": 1
        },
        {
            "name": "BP",  # 图例中显示的名称
            "path": r"E:\2024liushaoxuan\learn\shiyan1\data\output\m11\m11_BP\inference_eedscy_bp_all.csv",
            "color": "#66C2A5",  # NPG 红色 (学术红)
            "linestyle": "-",  # 实线
            "linewidth": 1.5,
            "alpha": 0.9,
            "zorder": 2
        },
        {
            "name": "Transformer",  # 图例中显示的名称
            "path": r"E:\2024liushaoxuan\learn\shiyan1\data\output\m11\m11_Transformer\inference_eedscy_transformer_all.csv",
            "color": "#E78AC3",  # NPG 红色 (学术红)
            "linestyle": "-",  # 实线
            "linewidth": 1.5,
            "alpha": 0.9,
            "zorder": 3
        },
        {
            "name": "autoformer(Baseline)",  # 图例中显示的名称
            "path": r"E:\2024liushaoxuan\learn\shiyan1\data\output\m11\m11_auto\inference_eedscy_autoformer_all.csv",
            "color": "#E5C494",  # NPG 红色 (学术红)
            "linestyle": "-",  # 实线
            "linewidth": 1.5,
            "alpha": 0.9,
            "zorder": 4
        },
        {
            "name": "CAPW-Autoformer",  # 图例中显示的名称
            "path": r"E:\2024liushaoxuan\learn\shiyan1\data\output\m11\m11_best\inference_eedscy_autoformer_all.csv",
            "color": "#C82423",  # NPG 红色 (学术红)
            "linestyle": "-",  # 实线
            "linewidth": 1.5,
            "alpha": 0.9,
            "zorder": 5
        },
        # {
        #     "name": "Autoformer(no_ccc)",  # 图例中显示的名称
        #     "path": r"E:\2024liushaoxuan\learn\shiyan1\data\output\m11\m11_auto_no_ccc\inference_eedscy_autoformer_all.csv",
        #     "color": "#C82423",  # NPG 红色 (学术红)
        #     "linestyle": "-",  # 实线
        #     "linewidth": 1.5,
        #     "alpha": 0.9,
        #     "zorder": 5
        # },
        # 示例：如果之后有 ARIMA 模型，可以取消注释并修改路径
        # {
        #     "name": "ARIMA Baseline",
        #     "path": r"e:/2024liushaoxuan/learn/shiyan1/data/output/m11/m11_ARIMA/inference_arima.csv",
        #     "color": "#4DBBD5",      # NPG 蓝色
        #     "linestyle": "--",       # 虚线区分
        #     "linewidth": 1.5,
        #     "alpha": 0.9,
        #     "zorder": 4
        # },
        # 示例：Transformer 模型
        # {
        #     "name": "Transformer",
        #     "path": r"e:/2024liushaoxuan/learn/shiyan1/data/output/m11/m11_Transformer/inference_transformer.csv",
        #     "color": "#00A087",      # NPG 绿色
        #     "linestyle": "-.",       # 点划线
        #     "linewidth": 1.5,
        #     "alpha": 0.9,
        #     "zorder": 4
        # }
    ]

    # ---------------------------------------------------------
    # 4. SCI 学术审美图表整体外观设置
    # ---------------------------------------------------------
    FIG_SIZE = (12, 5)  # (宽度, 高度) 适合双栏排版的单栏跨度

    # 字体设置 (SCI一般要求 Times New Roman 或 Arial)
    # 如果必须有中文，可以加入 'SimSun' (宋体) 备用
    FONT_FAMILY = "Times New Roman"

    # 图表标题与坐标轴标签
    TITLE = ""  # 学术图通常在论文排版时写Caption，代码里也可不设(留空"")
    TITLE_FONTSIZE = 14
    X_LABEL = "Date (2025)"
    Y_LABEL = "Passenger Flow"
    LABEL_FONTSIZE = 12
    TICK_FONTSIZE = 11

    # Y轴显示范围 (如果不确定可以设为 None 自动缩放)
    # Y_MIN, Y_MAX = 0, 50000
    Y_MIN, Y_MAX = None, None

    # ---------------------------------------------------------
    # 5. 图例与网格设置
    # ---------------------------------------------------------
    LEGEND_FONTSIZE = 10
    LEGEND_LOC = "upper left"
    # 图例列数，模型多时可以设为 2 或 3
    LEGEND_COLUMNS = 1
    # 是否开启边框包裹的学术网格
    SHOW_GRID = True
    GRID_STYLE = {"color": "#e0e0e0", "linestyle": "--", "linewidth": 0.5}

# =====================================================================
# 代码执行逻辑
# =====================================================================
def plot_prediction_comparison():
    # 1. 字体全局设置
    plt.rcParams['font.family'] = Config.FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    real_flow_plotted = False

    # 2. 遍历所有配置的模型进行绘制
    for model in Config.MODELS:
        print(f"正在处理模型: {model['name']}, 数据文件: {model['path']}")
        try:
            df = pd.read_csv(model['path'])
        except FileNotFoundError:
            print(f"  -> 警告: 找不到文件 {model['path']}，已跳过该模型。")
            continue

        # 处理时间索引
        df[Config.TIME_COLUMN] = pd.to_datetime(df[Config.TIME_COLUMN])
        df = df.set_index(Config.TIME_COLUMN)
        df = df.sort_index()

        # 按时间段筛选
        df_filtered = df.loc[Config.START_DATE:Config.END_DATE]
        if df_filtered.empty:
            print(f"  -> 警告: {model['name']} 筛选后数据为空！")
            continue

        # 3. 绘制真实值 (只需绘制一次即可)
        # 我们假设所有的CSV中 real_flow 都是一样的，所以取第一个成功读取的文件绘制
        if not real_flow_plotted and Config.REAL_FLOW_COLUMN in df_filtered.columns:
            ax.plot(
                df_filtered.index,
                df_filtered[Config.REAL_FLOW_COLUMN],
                label=Config.REAL_FLOW_STYLE["label"],
                color=Config.REAL_FLOW_STYLE["color"],
                linestyle=Config.REAL_FLOW_STYLE["linestyle"],
                linewidth=Config.REAL_FLOW_STYLE["linewidth"],
                alpha=Config.REAL_FLOW_STYLE["alpha"],
                zorder=Config.REAL_FLOW_STYLE["zorder"]
            )
            real_flow_plotted = True
            print("  -> 已成功加载并绘制真实客流 Ground Truth。")

        # 4. 绘制当前模型的预测值
        if Config.PRED_FLOW_COLUMN in df_filtered.columns:
            ax.plot(
                df_filtered.index,
                df_filtered[Config.PRED_FLOW_COLUMN],
                label=model["name"],
                color=model["color"],
                linestyle=model["linestyle"],
                linewidth=model["linewidth"],
                alpha=model["alpha"],
                zorder=model["zorder"]
            )
        else:
            print(f"  -> 警告: {model['name']} 找不到预测列 '{Config.PRED_FLOW_COLUMN}'。")

    if not real_flow_plotted:
        print("严重警告: 所有的文件中都找不到真实客流数据！")

    # 5. 图表坐标系及刻度学术化设置
    if Config.TITLE:
        ax.set_title(Config.TITLE, fontsize=Config.TITLE_FONTSIZE, pad=12, fontweight='bold')

    ax.set_xlabel(Config.X_LABEL, fontsize=Config.LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel(Config.Y_LABEL, fontsize=Config.LABEL_FONTSIZE, fontweight='bold')

    # 刻度朝内 (SCI 常见要求)，并且设置四个边框都可见但顶部和右侧没刻度
    ax.tick_params(direction='in', length=5, width=1, colors='black',
                   grid_alpha=0.5, labelsize=Config.TICK_FONTSIZE)
    # 使顶部和右侧的坐标系边线可见但隐藏刻度标签
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # 设置Y轴上下限
    if Config.Y_MIN is not None or Config.Y_MAX is not None:
        ax.set_ylim(bottom=Config.Y_MIN, top=Config.Y_MAX)

    # 设置X轴日期的显示格式 (如 'Jul 01')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=0) # 日期格式短，0度水平放置更学术

    # 6. 网格和图例设置
    if Config.SHOW_GRID:
        ax.grid(True, **Config.GRID_STYLE)

    # Frameon=False 可以去掉图例边框，更加简洁学术；
    ax.legend(loc=Config.LEGEND_LOC, fontsize=Config.LEGEND_FONTSIZE,
              ncol=Config.LEGEND_COLUMNS, frameon=True, edgecolor='black')

    plt.tight_layout()

    # 7. 保存输出
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    for fmt in Config.OUTPUT_FORMATS:
        out_path = os.path.join(Config.OUTPUT_DIR, f"{Config.OUTPUT_FILENAME}.{fmt}")
        plt.savefig(out_path, format=fmt, dpi=Config.DPI, bbox_inches='tight')
        print(f"SCI对比图已保存: {out_path}")

if __name__ == "__main__":
    plot_prediction_comparison()
