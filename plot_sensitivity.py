import json
import matplotlib.pyplot as plt
import os

# =====================================================================
# 配置区域 (CONFIG) - 敏感性分析图 (双Y轴, SCI学术级排版)
# =====================================================================
class Config:
    # ---------------------------------------------------------
    # 1. 输出路径与格式设置
    # ---------------------------------------------------------
    OUTPUT_DIR = r"e:/2024liushaoxuan/learn/shiyan1/data/output/画图"
    OUTPUT_FILENAME = "敏感性分析图"
    OUTPUT_FORMATS = ['png',]
    DPI = 600

    # ---------------------------------------------------------
    # 2. 数据源配置 (X轴标签与对应的 JSON 文件路径)
    # ---------------------------------------------------------
    # 请根据你实际的时间步长和文件夹名称修改以下列表。
    # 格式: {"x_label": "X轴显示的文本", "path": "对应的json文件绝对路径"}
    # 如果有的文件还没跑出来，可以先注释掉
    DATA_POINTS = [
        {
            "x_label": "48h",
            "path": r"E:/2024liushaoxuan/learn/shiyan1/data/output/实验结果/敏感性分析/m10_checkpoints_eedscy_auto_48h/result_final.json"
        },
        {
            "x_label": "72h",
            "path": r"E:/2024liushaoxuan/learn/shiyan1/data/output/实验结果/敏感性分析/m10_checkpoints_eedscy_auto_72h/result_final.json"
        },
        {
            "x_label": "96h",
            "path": r"E:/2024liushaoxuan/learn/shiyan1/data/output/实验结果/敏感性分析/m10_checkpoints_eedscy_auto_96h/result_final.json"
        },
        {
            "x_label": "120h",
            "path": r"E:/2024liushaoxuan/learn/shiyan1/data/output/实验结果/敏感性分析/m10_checkpoints_eedscy_auto_120h/result_final.json"
        },
        {
            "x_label": "168h",
            "path": r"E:/2024liushaoxuan/learn/shiyan1/data/output/实验结果/敏感性分析/m10_checkpoints_eedscy_auto_168h/result_final.json"
        },
        # {
        #     "x_label": "120h",
        #     "path": r"E:/2024liushaoxuan/learn/shiyan1/data/output/实验结果/敏感性分析/m10_checkpoints_eedscy_auto_120h/result_final.json"
        # }
    ]

    # ---------------------------------------------------------
    # 3. SCI 学术配色与指标样式设置 (NPG配色方案)
    # ---------------------------------------------------------
    # 误差指标 (左侧 Y 轴，数值大: 1000-3000) - 使用暖色系
    METRICS_LEFT = {
        "RMSE": {"name": "RMSE", "color": "#E64B35", "marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 8},  # NPG 红
        "MAE":  {"name": "MAE",  "color": "#F39B7F", "marker": "s", "linestyle": "-", "linewidth": 2, "markersize": 8}   # NPG 浅红/橙
    }

    # 相关性/准确度指标 (右侧 Y 轴，数值小: 0-1) - 使用冷色系
    METRICS_RIGHT = {
        "R2":        {"name": "$R^2$",     "color": "#4DBBD5", "marker": "^", "linestyle": "--", "linewidth": 2, "markersize": 8}, # NPG 蓝
        "Pearson_R": {"name": "Pearson R", "color": "#3C5488", "marker": "D", "linestyle": "--", "linewidth": 2, "markersize": 8}  # NPG 深蓝
    }

    # ---------------------------------------------------------
    # 4. 图表整体尺寸、字体与坐标轴设置
    # ---------------------------------------------------------
    FIG_SIZE = (8, 5.5)  # (宽度, 高度)
    FONT_FAMILY = "Times New Roman"

    TITLE = "Sensitivity Analysis of Different Time Steps"
    TITLE_FONTSIZE = 14
    X_LABEL = "Input Time Steps"

    # 左右 Y 轴的标签
    Y_LABEL_LEFT = "(RMSE / MAE)"
    Y_LABEL_RIGHT = "($R^2$ / Pearson R)"

    LABEL_FONTSIZE = 12
    TICK_FONTSIZE = 11

    # Y轴显示范围 (如果填 None 则根据数据自动缩放)
    Y_MIN_LEFT, Y_MAX_LEFT = None, None     # 示例: 1000, 3000
    Y_MIN_RIGHT, Y_MAX_RIGHT = 0.0, 1.0     # R2 和 Pearson_R 建议固定 0~1 或者 0.5~1.0

    # ---------------------------------------------------------
    # 5. 图例与网格设置
    # ---------------------------------------------------------
    LEGEND_FONTSIZE = 10
    SHOW_GRID = True
    GRID_STYLE = {"color": "#e0e0e0", "linestyle": "--", "linewidth": 0.5, "alpha": 0.7}


# =====================================================================
# 代码执行逻辑
# =====================================================================
def plot_sensitivity_analysis():
    # 1. 字体全局设置
    plt.rcParams['font.family'] = Config.FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False

    # 用于存放解析后的数据
    x_labels = []
    data_records = { "RMSE": [], "MAE": [], "R2": [], "Pearson_R": [] }

    # 2. 读取所有配置的 JSON 文件
    for item in Config.DATA_POINTS:
        label = item["x_label"]
        path = item["path"]

        if not os.path.exists(path):
            print(f"警告: 找不到文件 {path}，跳过时间步长 {label}。")
            continue

        try:
            # 这些JSON文件的格式有的是纯JSON，有的类似带有未加引号的键
            # 采用更灵活的读取方式
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # 手动解析类似 {MAE:1109.45,MAPE:38.96,Pearson_R:0.71,R2:0.44,RMSE:2121.81} 的格式
            if content.startswith('{') and content.endswith('}'):
                content = content[1:-1]
                parts = content.split(',')
                result = {}
                for part in parts:
                    if ':' in part:
                        k, v = part.split(':', 1)
                        k = k.strip().replace('"', '').replace("'", "")
                        result[k] = float(v.strip())
            else:
                result = json.loads(content)

            # 提取需要的4个指标
            data_records["RMSE"].append(result.get("RMSE", None))
            data_records["MAE"].append(result.get("MAE", None))
            data_records["R2"].append(result.get("R2", None))
            data_records["Pearson_R"].append(result.get("Pearson_R", None))
            x_labels.append(label)
            print(f"成功读取 {label} 的数据。")

        except Exception as e:
            print(f"读取或解析 {path} 时出错: {e}")

    if not x_labels:
        print("错误: 没有成功读取到任何数据，请检查 JSON 路径是否正确。")
        return

    # 3. 创建画板与双 Y 轴
    fig, ax1 = plt.subplots(figsize=Config.FIG_SIZE)
    ax2 = ax1.twinx()  # 实例化共享相同 x 轴的第二个坐标轴

    lines_left = []
    lines_right = []

    # 4. 绘制左侧 Y 轴数据 (RMSE, MAE)
    for key, style in Config.METRICS_LEFT.items():
        if key in data_records and any(v is not None for v in data_records[key]):
            line, = ax1.plot(
                x_labels, data_records[key],
                label=style["name"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"]
            )
            lines_left.append(line)

    # 5. 绘制右侧 Y 轴数据 (R2, Pearson R)
    for key, style in Config.METRICS_RIGHT.items():
        if key in data_records and any(v is not None for v in data_records[key]):
            line, = ax2.plot(
                x_labels, data_records[key],
                label=style["name"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"]
            )
            lines_right.append(line)

    # 6. 图表学术化排版与细节设置
    if Config.TITLE:
        ax1.set_title(Config.TITLE, fontsize=Config.TITLE_FONTSIZE, pad=15, fontweight='bold')

    ax1.set_xlabel(Config.X_LABEL, fontsize=Config.LABEL_FONTSIZE, fontweight='bold')

    # 设置左右 Y 轴标签
    ax1.set_ylabel(Config.Y_LABEL_LEFT, fontsize=Config.LABEL_FONTSIZE, fontweight='bold', color='black')
    ax2.set_ylabel(Config.Y_LABEL_RIGHT, fontsize=Config.LABEL_FONTSIZE, fontweight='bold', color='black')

    # 设置刻度朝内
    ax1.tick_params(axis='y', direction='in', labelsize=Config.TICK_FONTSIZE)
    ax2.tick_params(axis='y', direction='in', labelsize=Config.TICK_FONTSIZE)
    ax1.tick_params(axis='x', direction='in', labelsize=Config.TICK_FONTSIZE)

    # 设置 Y 轴范围
    if Config.Y_MIN_LEFT is not None or Config.Y_MAX_LEFT is not None:
        ax1.set_ylim(bottom=Config.Y_MIN_LEFT, top=Config.Y_MAX_LEFT)
    if Config.Y_MIN_RIGHT is not None or Config.Y_MAX_RIGHT is not None:
        ax2.set_ylim(bottom=Config.Y_MIN_RIGHT, top=Config.Y_MAX_RIGHT)

    # 显示网格 (挂在 ax1 上)
    if Config.SHOW_GRID:
        ax1.grid(True, **Config.GRID_STYLE)

    # 7. 合并双轴图例
    # 为了让两边的图例在一个框里显示，我们需要收集两条轴的 lines 和 labels
    all_lines = lines_left + lines_right
    all_labels = [l.get_label() for l in all_lines]

    # 将图例放在左中或者最佳位置
    ax1.legend(all_lines, all_labels, loc='center left', bbox_to_anchor=(1.15, 0.5),
               fontsize=Config.LEGEND_FONTSIZE, frameon=True, edgecolor='black')

    # 自动紧凑布局，防止图例或标签被裁切
    plt.tight_layout()

    # 8. 保存与导出
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    for fmt in Config.OUTPUT_FORMATS:
        out_path = os.path.join(Config.OUTPUT_DIR, f"{Config.OUTPUT_FILENAME}.{fmt}")
        plt.savefig(out_path, format=fmt, dpi=Config.DPI, bbox_inches='tight')
        print(f"敏感性分析图已成功保存: {out_path}")


if __name__ == "__main__":
    plot_sensitivity_analysis()
