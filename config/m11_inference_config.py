# -*- coding: utf-8 -*-
"""
M11 统一推理配置文件 V2
Unified Inference Configuration
用途: 定义推理时的模型类型、数据路径、输出路径、CSV/可视化参数等
改进:
  1. 支持选择推理数据集 (train/val/test/all)
  2. 支持分析时间窗口 (绘图+指标聚焦到指定日期范围)
  3. 指标分层输出 (整体/分 split/时间切片)
"""

from pathlib import Path


class InferenceConfig:
    """统一推理配置类"""

    # ==================== 项目路径 ====================
    BASE_DIR = Path(__file__).parent.parent

    # ==================== 模型选择 ====================
    # 支持: 'transformer', 'autoformer', 'bp', 'lstm', 'autoformer_moe'
    MODEL_TYPE = 'autoformer'

    # ==================== 场景配置 ====================
    scenic_name = "eedscy"
    test_description = "鄂尔多斯草原景区游客流量预测推理"

    # ==================== 数据路径 ====================
    # 数据集目录 (包含 train/val/test_x.npy, feature_map.json, scalers.pkl, dataset_info.json)
    DATA_DIR = BASE_DIR / "data/output/dataset/eedscy/m8b_eedscy_hourly_time_96"

    # ==================== 模型权重路径 ====================
    MODEL_PATH = BASE_DIR / "data/output/实验结果/消融实验/m10_checkpoints_eedscy_auto_h_4d_no_ccc/best_model_unified.pth"

    # ==================== 输出路径 ====================
    OUTPUT_DIR = BASE_DIR / "data/output/m11/m11_auto_no_ccc"

    # ==================== [新增] 推理数据集选择 ====================
    # 'train': 仅推理训练集
    # 'val':   仅推理验证集
    # 'test':  仅推理测试集（默认，与原版行为一致）
    # 'all':   全量推理（train + val + test 拼接，用于全时间段分析）
    RUN_SPLIT = 'all'

    # ==================== 推理参数 ====================
    BATCH_SIZE = 32
    DEVICE = 'cuda'
    NUM_WORKERS = 0        # Windows 下建议设为 0
    PIN_MEMORY = True
    TEMPERATURE = 0.5      # 推理温度 (用于权重网络的 softmax/tanh)
    SEED = 42

    # ==================== CSV 导出配置 ====================
    # 权重聚合方式: 'last' (取最后时间步) 或 'mean' (取时间步均值)
    WEIGHT_AGG_METHOD = 'last'

    # 环境特征聚合方式: 'last' 或 'mean'
    CONTEXT_AGG_METHOD = 'last'

    # 预测步选择: 'first' (仅 t+1), 'all' (所有步展开), 'mean' (所有步平均)
    PRED_STEP_MODE = 'first'

    # 想要导出到 CSV 的环境特征名称列表
    # 会自动在 feature_map 中查找, 找不到则跳过
    CSV_CONTEXT_FEATURES = [
        'temperature', 'wind_power', 'humidity',
        'is_day_off',
        'is_low_season', 'is_normal_season', 'is_peak_season',
    ]

    # datetime 时间戳文件名（由 m8b 生成，用于 CSV 添加时间列和绘图 x 轴）
    DATETIME_FILENAME = "datetime_index.npy"

    # ==================== [新增] 分析时间窗口配置 ====================
    # 设置后，会额外生成该时间窗口的:
    #   - 切片指标 (time_slice_metrics)
    #   - 切片对比图 (comparison_xxx_slice.png)
    #   - 切片权重图 (weights_xxx_slice.png)
    # 设置为 None 时不做时间筛选
    # 需要 datetime_index.npy 存在（由 m8b 生成）
    # 格式: 'YYYY-MM-DD'
    ANALYSIS_DATE_START = '2025-06-30'   # 例如 '2024-10-01'
    ANALYSIS_DATE_END = '2025-09-30'     # 例如 '2024-10-07'

    # ==================== 可视化配置 ====================
    FIGURE_SIZE = (15, 6)
    FIGURE_DPI = 300
    SHOW_PLOTS = False
    PLOT_SAMPLE_LIMIT = 0  # 0 = 不限制（仅对全量图生效，切片图始终不限制）

    # ==================== 输出文件路径 (由 setup_dirs 自动设置) ====================
    CSV_FILE = None
    METRICS_FILE = None
    PLOT_COMPARISON = None
    PLOT_WEIGHTS = None
    PLOT_COMPARISON_SLICE = None   # [新增] 时间切片对比图
    PLOT_WEIGHTS_SLICE = None      # [新增] 时间切片权重图
    REPORT_FILE = None

    # ==================== 辅助方法 ====================

    @classmethod
    def setup_dirs(cls):
        """创建输出目录并设置输出文件路径"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"{cls.scenic_name}_{cls.MODEL_TYPE}_{cls.RUN_SPLIT}"
        cls.CSV_FILE = cls.OUTPUT_DIR / f'inference_{tag}.csv'
        cls.METRICS_FILE = cls.OUTPUT_DIR / f'metrics_{tag}.json'
        cls.PLOT_COMPARISON = cls.OUTPUT_DIR / f'comparison_{tag}.png'
        cls.PLOT_WEIGHTS = cls.OUTPUT_DIR / f'weights_{tag}.png'
        cls.PLOT_COMPARISON_SLICE = cls.OUTPUT_DIR / f'comparison_{tag}_slice.png'
        cls.PLOT_WEIGHTS_SLICE = cls.OUTPUT_DIR / f'weights_{tag}_slice.png'
        cls.REPORT_FILE = cls.OUTPUT_DIR / f'report_{tag}.txt'
        print(f"✓ 输出目录: {cls.OUTPUT_DIR}")

    @classmethod
    def validate(cls):
        """验证配置参数"""
        supported_models = ['transformer', 'autoformer', 'bp', 'lstm', 'autoformer_moe']
        supported_splits = ['train', 'val', 'test', 'all']

        assert cls.MODEL_TYPE in supported_models, \
            f"不支持的模型类型: {cls.MODEL_TYPE}, 支持: {supported_models}"
        assert cls.RUN_SPLIT in supported_splits, \
            f"不支持的推理集: {cls.RUN_SPLIT}, 支持: {supported_splits}"
        assert cls.DATA_DIR.exists(), f"数据目录不存在: {cls.DATA_DIR}"
        assert cls.MODEL_PATH.exists(), f"模型文件不存在: {cls.MODEL_PATH}"

        # 根据 RUN_SPLIT 检查所需的 npy 文件
        if cls.RUN_SPLIT == 'all':
            for s in ['train', 'val', 'test']:
                assert (cls.DATA_DIR / f'{s}_x.npy').exists(), f"{s}_x.npy 不存在"
                assert (cls.DATA_DIR / f'{s}_y.npy').exists(), f"{s}_y.npy 不存在"
        else:
            assert (cls.DATA_DIR / f'{cls.RUN_SPLIT}_x.npy').exists(), \
                f"{cls.RUN_SPLIT}_x.npy 不存在"
            assert (cls.DATA_DIR / f'{cls.RUN_SPLIT}_y.npy').exists(), \
                f"{cls.RUN_SPLIT}_y.npy 不存在"

        assert (cls.DATA_DIR / 'feature_map.json').exists(), f"feature_map.json 不存在"
        assert (cls.DATA_DIR / 'scalers.pkl').exists(), f"scalers.pkl 不存在"
        assert cls.WEIGHT_AGG_METHOD in ['last', 'mean'], \
            f"WEIGHT_AGG_METHOD 必须为 'last' 或 'mean'"
        assert cls.CONTEXT_AGG_METHOD in ['last', 'mean'], \
            f"CONTEXT_AGG_METHOD 必须为 'last' 或 'mean'"
        assert cls.PRED_STEP_MODE in ['first', 'all', 'mean'], \
            f"PRED_STEP_MODE 必须为 'first', 'all' 或 'mean'"

        # 检查分析时间窗口依赖
        if cls.ANALYSIS_DATE_START or cls.ANALYSIS_DATE_END:
            dt_path = cls.DATA_DIR / cls.DATETIME_FILENAME
            if not dt_path.exists():
                print(f"  ⚠️  设置了分析时间窗口但 {cls.DATETIME_FILENAME} 不存在，"
                      f"时间切片功能将不可用")

        print("✓ 推理配置验证通过")

    @classmethod
    def print_config(cls):
        """打印配置摘要"""
        print(f"\n{'=' * 70}")
        print("M11 统一推理配置 V2")
        print(f"{'=' * 70}")
        print(f"  模型类型:     {cls.MODEL_TYPE}")
        print(f"  场景名称:     {cls.scenic_name}")
        print(f"  测试描述:     {cls.test_description}")
        print(f"  推理数据集:   {cls.RUN_SPLIT}")
        print(f"  数据目录:     {cls.DATA_DIR}")
        print(f"  模型路径:     {cls.MODEL_PATH}")
        print(f"  输出目录:     {cls.OUTPUT_DIR}")
        print(f"  推理温度:     {cls.TEMPERATURE}")
        print(f"  权重聚合:     {cls.WEIGHT_AGG_METHOD}")
        print(f"  特征聚合:     {cls.CONTEXT_AGG_METHOD}")
        print(f"  预测步模式:   {cls.PRED_STEP_MODE}")
        print(f"  设备:         {cls.DEVICE}")
        if cls.ANALYSIS_DATE_START or cls.ANALYSIS_DATE_END:
            start_str = cls.ANALYSIS_DATE_START or '数据起始'
            end_str = cls.ANALYSIS_DATE_END or '数据结束'
            print(f"  分析时间窗:   {start_str} ~ {end_str}")
        else:
            print(f"  分析时间窗:   未设置（使用全部推理数据）")
        print(f"{'=' * 70}\n")
