# -*- coding: utf-8 -*-
"""
M11 统一联合训练配置文件 (Unified Joint Training Configuration)
=================================================================
版本: M11 (基于 M10 优化注释版本)
用途: 定义训练超参数、损失函数策略、路径配置等

核心特性:
---------
1. CCC Loss (一致性相关系数损失): 统一解决 RMSE 与 Pearson 指标冲突
2. 自适应峰值加权 (Adaptive Peak Weighting): 解决极值削峰问题
3. 温度退火机制 (Temperature Annealing): 控制注意力权重的锐化过程
4. 早停策略 (Early Stopping): 防止过拟合

修改记录:
---------
- M11: 优化注释结构，增加详细说明和使用指南
- M10: 引入 CCC Loss 和自适应峰值加权机制
"""

from pathlib import Path


class TrainingConfig:
    """
    统一联合训练配置类
    
    该配置类包含了模型训练所需的所有超参数和路径配置。
    使用前请根据实际情况修改数据路径和景区名称。
    
    主要配置模块:
    ------------
    1. 数据路径配置: 输入数据和输出模型的路径
    2. 训练基础参数: 批次大小、学习率、训练轮数等
    3. 损失函数配置: CCC Loss 和自适应加权 MSE 的参数
    4. 温度退火策略: 控制注意力权重的锐化过程
    5. 正则化参数: L1 正则化、权重衰减等
    6. 优化器配置: AdamW 优化器及其参数
    7. 学习率调度器: Cosine Annealing 等策略
    8. 早停策略: 防止过拟合的机制
    """

    # ==================== 数据路径配置 ====================
    # 项目根目录 (自动获取)
    BASE_DIR = Path(__file__).parent.parent

    # 景区名称 (请根据实际情况修改)
    # 可选值: "eedscy" (鄂尔多斯草原), "xsw" (响沙湾) 等
    scenic_name = "eedscy"

    # M8B 数据输出目录 (作为本模块的输入)
    # 该目录应包含: train_x.npy, train_y.npy, val_x.npy, val_y.npy, 
    #              test_x.npy, test_y.npy, feature_map.json, scalers.pkl
    DATA_DIR = BASE_DIR / f"data/output/m8b_{scenic_name}_hourly"

    # M11 模型输出目录 (保存训练好的模型和日志)
    # 命名规则: m11_checkpoints_{景区名称}_h_{版本号}
    SAVE_DIR = Path(BASE_DIR / f'data/output/m11_checkpoints_{scenic_name}_h_1')

    # 训练数据文件路径
    TRAIN_X = DATA_DIR / 'train_x.npy'  # 训练集特征 (Batch, SeqLen, Features)
    TRAIN_Y = DATA_DIR / 'train_y.npy'  # 训练集标签 (Batch, PredLen, 1)
    VAL_X = DATA_DIR / 'val_x.npy'      # 验证集特征
    VAL_Y = DATA_DIR / 'val_y.npy'      # 验证集标签
    TEST_X = DATA_DIR / 'test_x.npy'    # 测试集特征
    TEST_Y = DATA_DIR / 'test_y.npy'    # 测试集标签

    # 配置文件路径
    FEATURE_MAP = DATA_DIR / 'feature_map.json'  # 特征映射表 (包含特征分组信息)
    SCALER_PATH = DATA_DIR / 'scalers.pkl'       # 归一化器 (用于反归一化)

    # ==================== 训练基础参数 ====================
    
    BATCH_SIZE = 64  
    """批次大小
    
    说明: 每次训练迭代处理的样本数量
    建议: 
    - GPU 显存充足: 64-128
    - GPU 显存有限: 32-64
    - CPU 训练: 16-32
    """

    NUM_EPOCHS = 1000  
    """总训练轮数
    
    说明: 完整遍历训练集的次数
    建议: 配合早停机制使用，设置较大值 (500-1000)，让早停自动终止
    """

    LEARNING_RATE = 1e-4  
    """初始学习率
    
    说明: 优化器的步长大小
    建议:
    - AdamW: 1e-4 到 5e-4
    - Adam: 1e-3 到 5e-3
    - 如果训练不稳定，可降低到 5e-5
    """

    WEIGHT_DECAY = 1e-3  
    """权重衰减系数 (L2 正则化)
    
    说明: 防止权重过大，提高模型泛化能力
    建议:
    - 标准值: 1e-4 到 1e-3
    - 如果过拟合严重，可增大到 1e-2
    - 如果欠拟合，可减小到 1e-5 或 0
    """

    # ==================== 温度退火策略 ====================
    
    TEMP_START = 5.0  
    """初始温度 (Temperature Start)
    
    说明: 控制注意力权重的平滑程度
    - 高温 (5.0-10.0): 权重分布平滑，模型探索多种特征组合
    - 低温 (0.01-0.1): 权重分布尖锐，模型专注于少数关键特征
    
    训练初期使用高温，让模型充分探索特征空间
    """

    TEMP_END = 0.01  
    """最终温度 (Temperature End)
    
    说明: 训练结束时的温度值
    - 较低的最终温度 (0.01-0.1) 使权重更加尖锐 (接近 One-hot)
    - 避免权重过于平均化导致的性能下降
    
    建议: 0.01 (标准值) 或 0.05 (如果需要保留一定平滑性)
    """

    TEMP_SCHEDULE = 'linear'  
    """退火策略 (Annealing Schedule)
    
    可选值:
    - 'linear': 线性退火 (推荐，稳定可控)
    - 'cosine': 余弦退火 (前期下降慢，后期下降快)
    """

    # ==================== 损失函数配置 ====================
    
    # --- CCC Loss (主损失函数) ---
    # CCC (Concordance Correlation Coefficient) 一致性相关系数
    # 公式: CCC = (2 * cov) / (var_pred + var_target + (mean_pred - mean_target)^2)
    # Loss = 1 - CCC
    # 
    # 优势:
    # 1. 同时考虑数值准确性 (均值/方差偏差) 和趋势相关性 (Pearson)
    # 2. 避免 RMSE 与 Pearson 指标的冲突
    # 3. 取值范围 [0, 2]，0 表示完美拟合
    
    LOSS_WEIGHT_MSE = 1.0  
    """MSE 损失权重 (已弃用，保留用于兼容性)
    
    说明: 在 M11 版本中，主损失函数已改为 CCC Loss
    该参数仅在旧版本中使用
    """

    LOSS_WEIGHT_PEARSON = 1.0  
    """Pearson 损失权重 (已弃用，保留用于兼容性)
    
    说明: 在 M11 版本中，主损失函数已改为 CCC Loss
    该参数仅在旧版本中使用
    """

    # --- 自适应峰值加权 (Adaptive Peak Weighting) ---
    # 目的: 解决模型对高数值样本的削峰问题
    # 原理: 对超过阈值的样本施加更高的损失权重，强迫模型关注峰值
    
    USE_ADAPTIVE_WEIGHT = True  
    """是否启用自适应峰值加权
    
    说明: 
    - True: 启用自适应加权，解决削峰问题
    - False: 仅使用 CCC Loss，不对峰值样本特殊处理
    
    建议: 如果发现模型预测的峰值偏低，设为 True
    """

    PEAK_SIGMA = 1.5  
    """峰值阈值系数 (Peak Threshold Sigma)
    
    说明: 定义"高峰"的标准
    阈值计算公式: Threshold = Mean + PEAK_SIGMA * Std
    
    建议:
    - 1.5: 标准值，约 93% 的样本低于阈值
    - 2.0: 更严格，约 97.5% 的样本低于阈值
    - 1.0: 更宽松，约 84% 的样本低于阈值
    """

    PEAK_PENALTY_WEIGHT = 10  
    """峰值样本的惩罚权重倍数
    
    说明: 超过阈值的样本，其 MSE 损失将乘以该倍数
    
    建议:
    - 5.0: 温和惩罚
    - 10.0: 标准惩罚 (推荐)
    - 20.0: 强力惩罚 (如果削峰问题严重)
    """

    LOSS_WEIGHT_PEAK_MSE = 0.5  
    """加权 MSE 在总损失中的系数
    
    说明: 总损失 = CCC_Loss + LOSS_WEIGHT_PEAK_MSE * Weighted_MSE
    
    建议:
    - 0.3-0.5: 标准值，平衡 CCC 和峰值拟合
    - 0.5-1.0: 更关注峰值拟合
    - 0.1-0.3: 更关注整体趋势
    """

    # ==================== 正则化参数 ====================
    
    L1_LAMBDA = 1e-5  
    """L1 正则化系数 (稀疏性约束)
    
    说明: 对注意力权重施加 L1 惩罚，促进权重稀疏化
    
    建议:
    - 1e-5: 轻微约束 (推荐)
    - 1e-4: 中等约束
    - 0: 不使用 L1 正则化
    
    注意: 过大的 L1 系数可能导致模型输出过于平滑
    """

    DIVERSITY_LAMBDA = 1e-5  
    """组间多样性正则化系数 (已弃用)
    
    说明: 促进不同情感组的权重分布差异化
    在当前版本中已禁用
    """

    USE_DIVERSITY_LOSS = False  
    """是否启用多样性损失 (已弃用)"""

    # ==================== 优化器配置 ====================
    
    OPTIMIZER = 'adamw'  
    """优化器类型
    
    可选值:
    - 'adamw': AdamW (推荐，带权重衰减的 Adam)
    - 'adam': Adam (标准 Adam 优化器)
    """

    BETAS = (0.9, 0.999)  
    """Adam/AdamW 的动量参数
    
    说明:
    - beta1 (0.9): 一阶矩估计的指数衰减率
    - beta2 (0.999): 二阶矩估计的指数衰减率
    
    通常不需要修改
    """

    EPS = 1e-8  
    """Adam/AdamW 的数值稳定性参数
    
    说明: 防止除零错误
    通常不需要修改
    """

    # ==================== 学习率调度器配置 ====================
    
    SCHEDULER = 'cosine'  
    """学习率调度器类型
    
    可选值:
    - 'cosine': Cosine Annealing (推荐，平滑下降)
    - 'step': Step LR (阶梯式下降)
    - 'plateau': Reduce on Plateau (根据验证集性能自适应调整)
    - 'none': 不使用调度器
    """

    # Cosine Annealing 参数
    LR_MIN = 1e-7  
    """Cosine Annealing 的最小学习率
    
    说明: 学习率下降的下界
    建议: 设为初始学习率的 1/1000 到 1/10000
    """

    # Step LR 参数 (仅在 SCHEDULER='step' 时使用)
    LR_STEP_SIZE = 20  
    """Step LR 的步长 (每隔多少轮降低学习率)"""
    
    LR_GAMMA = 0.5  
    """Step LR 的衰减系数 (每次降低的倍数)"""

    # Reduce on Plateau 参数 (仅在 SCHEDULER='plateau' 时使用)
    LR_PATIENCE = 5  
    """Plateau 的耐心值 (验证集性能多少轮不提升后降低学习率)"""
    
    LR_FACTOR = 0.5  
    """Plateau 的衰减系数"""

    # ==================== 早停策略 ====================
    
    EARLY_STOP = True  
    """是否启用早停机制
    
    说明: 当验证集性能不再提升时，提前终止训练
    建议: 设为 True，配合较大的 NUM_EPOCHS 使用
    """

    PATIENCE = 30  
    """早停的耐心值
    
    说明: 验证集性能多少轮不提升后触发早停
    建议:
    - 30-50: 标准值
    - 20: 如果训练时间有限
    - 100: 如果想充分训练
    """

    MIN_DELTA = 1e-4  
    """早停的最小改善阈值
    
    说明: 验证集性能提升小于该值时，视为没有改善
    建议: 1e-4 (标准值)
    """

    # ==================== 梯度裁剪 ====================
    
    GRAD_CLIP = True  
    """是否启用梯度裁剪
    
    说明: 防止梯度爆炸，提高训练稳定性
    建议: 设为 True
    """

    MAX_GRAD_NORM = 1.0  
    """梯度裁剪的最大范数
    
    说明: 梯度的 L2 范数超过该值时，将被缩放
    建议:
    - 1.0: 标准值
    - 0.5: 如果训练不稳定
    - 5.0: 如果梯度消失
    """

    # ==================== 设备配置 ====================
    
    DEVICE = 'cuda'  
    """训练设备
    
    可选值:
    - 'cuda': GPU 训练 (推荐)
    - 'cpu': CPU 训练 (速度较慢)
    """

    NUM_WORKERS = 0  
    """数据加载的工作进程数
    
    建议:
    - Windows: 0 (更稳定)
    - Linux: 4-8 (加速数据加载)
    """

    PIN_MEMORY = True  
    """是否将数据固定在内存中
    
    说明: 加速 CPU 到 GPU 的数据传输
    建议: GPU 训练时设为 True
    """

    # ==================== 日志与保存配置 ====================
    
    LOG_INTERVAL = 10  
    """日志打印间隔 (每隔多少个 batch 打印一次)"""

    SAVE_INTERVAL = 5  
    """模型保存间隔 (每隔多少轮保存一次)"""

    # 保存路径
    SAVE_LOG = SAVE_DIR / 'training_log.csv'  
    """训练日志保存路径 (CSV 格式)"""

    SAVE_LATEST_MODEL = SAVE_DIR / 'latest_model.pth'  
    """最新模型保存路径"""

    # 分阶段最佳模型保存路径 (保留用于兼容性)
    SAVE_BEST_MODEL_PHASE1 = SAVE_DIR / 'best_model_phase1.pth'
    SAVE_BEST_MODEL_PHASE2 = SAVE_DIR / 'best_model_phase2.pth'
    SAVE_BEST_MODEL_PHASE3 = SAVE_DIR / 'best_model_phase3.pth'

    # ==================== 可视化配置 ====================
    
    SAVE_ATTENTION_WEIGHTS = True  
    """是否保存注意力权重 (用于可视化分析)"""

    SAVE_PRED_VS_TRUE = True  
    """是否保存预测值与真实值的对比图"""

    # ==================== 随机种子 ====================
    
    SEED = 42  
    """随机种子
    
    说明: 确保实验可复现
    建议: 固定为某个值 (如 42)
    """

    # ==================== 辅助方法 ====================

    @classmethod
    def setup_dirs(cls):
        """
        创建必要的目录
        
        说明: 在训练开始前调用，自动创建模型保存目录
        """
        cls.SAVE_DIR.mkdir(exist_ok=True, parents=True)
        print(f"✓ 创建保存目录: {cls.SAVE_DIR}")

    @classmethod
    def validate(cls):
        """
        验证配置参数的合法性
        
        说明: 检查数据路径是否存在，参数设置是否合理
        
        Returns:
            bool: 验证是否通过
        
        Raises:
            AssertionError: 如果配置参数不合法
        """
        assert cls.DATA_DIR.exists(), f"数据目录不存在: {cls.DATA_DIR}"
        print("✓ 配置参数验证通过")
        return True

    @classmethod
    def print_config(cls):
        """
        打印配置信息摘要
        
        说明: 在训练开始前调用，显示关键配置参数
        """
        print("\n" + "=" * 70)
        print("M11 统一联合训练配置 (Unified Joint Training)")
        print("=" * 70)
        print(f"  - 数据来源:        {cls.DATA_DIR.name}")
        print(f"  - 模型保存:        {cls.SAVE_DIR.name}")
        print(f"  - 主损失函数:      CCC Loss (一致性相关系数)")
        print(f"  - 自适应加权:      {'启用' if cls.USE_ADAPTIVE_WEIGHT else '禁用'}")
        if cls.USE_ADAPTIVE_WEIGHT:
            print(f"    * 峰值阈值:      Mean + {cls.PEAK_SIGMA} * Std")
            print(f"    * 惩罚权重:      {cls.PEAK_PENALTY_WEIGHT}x")
        print(f"  - 批次大小:        {cls.BATCH_SIZE}")
        print(f"  - 学习率:          {cls.LEARNING_RATE}")
        print(f"  - 总训练轮数:      {cls.NUM_EPOCHS}")
        print(f"  - 早停耐心值:      {cls.PATIENCE}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # 测试配置类
    TrainingConfig.validate()
    TrainingConfig.print_config()