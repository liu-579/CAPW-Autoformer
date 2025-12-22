"""
m9_config.py - 动态融合预测模型配置文件
===================================================================
包含所有模型超参数、训练配置和数据接口配置

作者：[您的姓名]
日期：2025-12-18
版本：v3.0 - 最终整合版
"""

from typing import List, Dict, Optional
import torch
import os
import json


# ============================================================================
# 数据接口配置
# ============================================================================

class DataConfig:
    """
    数据集接口配置

    适配上一模块的 NumPy 张量输出格式
    """

    # ========== 景区标识 ==========
    TARGET_SCENIC_NAME: str = "鄂尔多斯草原"     # 目标景区名称（控制文件后缀）

    # ========== 数据路径 ==========
    DATA_DIR = r"/data/output/m8_eedscy/datasets"  # 数据文件所在目录

    # ========== 文件命名模板 ==========
    # 格式：{split}_{x/y}_{scenic_name}.npy
    TRAIN_X_TEMPLATE: str = DATA_DIR+r"tourism_seq_{scenic_name}_train_x.npy"
    TRAIN_Y_TEMPLATE: str =DATA_DIR+ r"tourism_seq_{scenic_name}_train_y.npy"
    VAL_X_TEMPLATE: str = DATA_DIR+r"tourism_seq_{scenic_name}_val_x.npy"
    VAL_Y_TEMPLATE: str =DATA_DIR+ r"tourism_seq_{scenic_name}_val_y.npy"
    TEST_X_TEMPLATE: str =DATA_DIR+ r"tourism_seq_{scenic_name}_test_x.npy"
    TEST_Y_TEMPLATE: str = DATA_DIR+r"tourism_seq_{scenic_name}_test_y.npy"

    # 特征映射表：feature_map_{scenic_name}.json
    FEATURE_MAP_TEMPLATE: str = DATA_DIR+"tourism_seq_{scenic_name}_feature_map.json"

    # 归一化器：scaler_{scenic_name}.pkl
    SCALER_TEMPLATE: str = DATA_DIR+"tourism_seq_{scenic_name}_scaler.pkl"

    # ========== 张量形状约定 ==========
    SEQ_LEN: int = 30                      # 输入序列长度（历史窗口）
    PRED_LEN: int = 7                      # 预测序列长度（预测未来天数）
    TARGET_DIM: int = 1                    # 预测目标维度（仅客流 passenger_count）

    # ========== 特征维度（动态从 feature_map 读取）==========
    SENTIMENT_DIM: Optional[int] = None    # 情感特征维度（从 JSON 读取）
    CONTEXT_DIM: Optional[int] = None      # 环境特征维度（从 JSON 读取）
    TOTAL_DIM: Optional[int] = None        # 总特征维度（从 JSON 读取）

    @classmethod
    def get_file_path(cls, template_str: str) -> str:
        """
        根据模板字符串和景区名称生成文件路径

        参数：
            template_str: 文件名模板字符串（例如 "train_x_{scenic_name}.npy"）

        返回：
            完整文件路径（字符串）
        """
        if not isinstance(template_str, str):
            raise TypeError(f"template_str 必须是字符串类型，当前类型: {type(template_str)}")

        filename = template_str.format(scenic_name=cls.TARGET_SCENIC_NAME)
        full_path = os.path.join(cls.DATA_DIR, filename)

        return full_path

    @classmethod
    def get_all_paths(cls) -> Dict[str, str]:
        """
        返回所有数据文件的路径字典

        返回：
            dict: 包含所有数据文件的路径
        """
        return {
            'train_x': cls.get_file_path(cls.TRAIN_X_TEMPLATE),
            'train_y': cls.get_file_path(cls.TRAIN_Y_TEMPLATE),
            'val_x': cls.get_file_path(cls.VAL_X_TEMPLATE),
            'val_y': cls.get_file_path(cls.VAL_Y_TEMPLATE),
            'test_x': cls.get_file_path(cls.TEST_X_TEMPLATE),
            'test_y': cls.get_file_path(cls.TEST_Y_TEMPLATE),
            'feature_map': cls.get_file_path(cls.FEATURE_MAP_TEMPLATE),
            'scaler': cls.get_file_path(cls.SCALER_TEMPLATE)
        }

    @classmethod
    def validate_paths(cls) -> Dict[str, bool]:
        """
        验证所有数据文件是否存在

        返回：
            dict: 每个文件的存在性检查结果
        """
        paths = cls.get_all_paths()
        return {key: os.path.exists(path) for key, path in paths.items()}

    @classmethod
    def create_data_dir(cls):
        """创建数据目录（如果不存在）"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        print(f"✅ 数据目录已创建/确认存在: {cls.DATA_DIR}")

    @classmethod
    def print_required_files(cls):
        """
        打印所有需要的数据文件清单
        """
        print("\n" + "="*80)
        print("📋 数据文件清单")
        print("="*80)
        print(f"\n目标景区: {cls.TARGET_SCENIC_NAME}")
        print(f"数据目录: {cls.DATA_DIR}\n")

        print("【训练集】")
        print(f"  1. {cls.get_file_path(cls.TRAIN_X_TEMPLATE)}")
        print(f"     - 形状: (N_train, {cls.SEQ_LEN}, Feature_Dim)")
        print(f"     - 说明: 训练集输入特征")

        print(f"\n  2. {cls.get_file_path(cls.TRAIN_Y_TEMPLATE)}")
        print(f"     - 形状: (N_train, {cls.PRED_LEN}, {cls.TARGET_DIM})")
        print(f"     - 说明: 训练集标签（客流量）")

        print("\n【验证集】")
        print(f"  3. {cls.get_file_path(cls.VAL_X_TEMPLATE)}")
        print(f"     - 形状: (N_val, {cls.SEQ_LEN}, Feature_Dim)")
        print(f"     - 说明: 验证集输入特征")

        print(f"\n  4. {cls.get_file_path(cls.VAL_Y_TEMPLATE)}")
        print(f"     - 形状: (N_val, {cls.PRED_LEN}, {cls.TARGET_DIM})")
        print(f"     - 说明: 验证集标签（客流量）")

        print("\n【测试集】")
        print(f"  5. {cls.get_file_path(cls.TEST_X_TEMPLATE)}")
        print(f"     - 形状: (N_test, {cls.SEQ_LEN}, Feature_Dim)")
        print(f"     - 说明: 测试集输入特征")

        print(f"\n  6. {cls.get_file_path(cls.TEST_Y_TEMPLATE)}")
        print(f"     - 形状: (N_test, {cls.PRED_LEN}, {cls.TARGET_DIM})")
        print(f"     - 说明: 测试集标签（客流量）")

        print("\n【元数据文件】")
        print(f"  7. {cls.get_file_path(cls.FEATURE_MAP_TEMPLATE)}")
        print(f"     - 格式: JSON")
        print(f"     - 说明: 特征映射表")
        print(f"     - 必须包含键: 'sentiment_idxs', 'context_idxs'")
        print(f"     - 示例: {{\n"
               f"              'sentiment_idxs': [0, 1, 2, 3, 4],\n"
               f"              'context_idxs': [5, 6, 7, 8, 9, ...]\n"
               f"            }}")

        print(f"\n  8. {cls.get_file_path(cls.SCALER_TEMPLATE)}")
        print(f"     - 格式: Pickle (.pkl)")
        print(f"     - 说明: 归一化器对象（可选）")

        print("\n" + "="*80)

    @classmethod
    def print_paths_info(cls):
        """打印所有路径信息和存在性检查"""
        print("\n" + "="*80)
        print("📂 数据文件路径信息")
        print("="*80)

        paths = cls.get_all_paths()
        validation = cls.validate_paths()

        print(f"\n目标景区: {cls.TARGET_SCENIC_NAME}")
        print(f"数据目录: {cls.DATA_DIR}\n")

        for key in paths.keys():
            status = "✅ 存在" if validation[key] else "❌ 不存在"
            print(f"{status}  {key:15s}: {paths[key]}")

        # 统计
        exist_count = sum(validation.values())
        total_count = len(validation)
        print(f"\n状态统计: {exist_count}/{total_count} 文件存在")

        if exist_count < total_count:
            print("\n⚠️  警告: 部分数据文件缺失，请检查！")
        else:
            print("\n✅ 所有数据文件完整！")

        print("="*80)


# ============================================================================
# 权重生成网络配置
# ============================================================================

class WeightNetConfig:
    """ContextAwareWeightNet 的超参数配置"""

    HIDDEN_DIM: int = 128               # 隐藏层维度
    NUM_ATTENTION_HEADS: int = 4        # 多头注意力头数
    NUM_ENCODER_LAYERS: int = 2         # Transformer Encoder 层数
    DROPOUT: float = 0.1                # Dropout 概率
    DEFAULT_TEMPERATURE: float = 1.0    # Softmax 默认温度
    ACTIVATION: str = 'gelu'            # 激活函数类型


# ============================================================================
# 时序骨干网络配置
# ============================================================================

class BackboneConfig:
    """TimeSeriesBackbone 的超参数配置"""

    D_MODEL: int = 256                  # Transformer 模型维度
    NUM_HEADS: int = 8                  # 多头注意力头数
    NUM_LAYERS: int = 3                 # Transformer Encoder 层数
    DIM_FEEDFORWARD: int = 1024         # FFN 前馈网络维度
    DROPOUT: float = 0.1                # Dropout 概率
    MAX_POSITION_EMBEDDINGS: int = 5000 # 最大位置编码长度
    ACTIVATION: str = 'gelu'            # 激活函数类型


# ============================================================================
# 训练配置
# ============================================================================

class TrainingConfig:
    """模型训练相关的超参数配置"""

    # ========== 优化器配置 ==========
    LEARNING_RATE: float = 1e-4         # 学习率
    WEIGHT_DECAY: float = 1e-5          # L2 正则化系数
    OPTIMIZER: str = 'adamw'            # 优化器类型

    # ========== 学习率调度器 ==========
    LR_SCHEDULER: str = 'cosine'        # 学习率调度器
    LR_WARMUP_EPOCHS: int = 5           # 预热轮数
    LR_MIN: float = 1e-6                # 最小学习率

    # ========== 训练参数 ==========
    BATCH_SIZE: int = 32                # 批次大小
    NUM_EPOCHS: int = 100               # 总训练轮数
    GRADIENT_CLIP: float = 1.0          # 梯度裁剪阈值

    # ========== 三阶段训练 ==========
    STAGE1_EPOCHS: int = 30             # 阶段1：预训练骨干网络
    STAGE2_EPOCHS: int = 30             # 阶段2：训练权重生成网络
    STAGE3_EPOCHS: int = 40             # 阶段3：联合微调

    # ========== 早停配置 ==========
    EARLY_STOPPING: bool = True         # 是否启用早停
    PATIENCE: int = 10                  # 早停容忍轮数
    MIN_DELTA: float = 1e-4             # 最小改进阈值

    # ========== 验证与保存 ==========
    VAL_INTERVAL: int = 1               # 验证间隔
    SAVE_INTERVAL: int = 5              # 保存间隔
    SAVE_BEST_ONLY: bool = True         # 是否只保存最优模型

    # ========== 设备配置 ==========
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS: int = 4                # DataLoader 工作进程数
    PIN_MEMORY: bool = True             # 是否使用锁页内存


# ============================================================================
# 损失函数配置
# ============================================================================

class LossConfig:
    """损失函数及其权重配置"""

    MAIN_LOSS: str = 'mse'              # 主损失函数类型

    # 正则化损失
    USE_WEIGHT_SPARSITY: bool = True    # 权重稀疏性损失
    SPARSITY_WEIGHT: float = 0.01       # 稀疏性损失权重

    USE_WEIGHT_SMOOTHNESS: bool = True  # 权重平滑性损失
    SMOOTHNESS_WEIGHT: float = 0.01     # 平滑性损失权重

    USE_WEIGHT_ENTROPY: bool = False    # 权重熵损失
    ENTROPY_WEIGHT: float = 0.001       # 熵损失权重

    HUBER_DELTA: float = 1.0            # Huber Loss 的 delta 参数


# ============================================================================
# 实验配置
# ============================================================================

class ExperimentConfig:
    """实验记录和可视化配置"""

    EXPERIMENT_NAME: str = 'DynamicFusion_v3'  # 实验名称
    RUN_ID: str = None                  # 运行ID（自动生成）

    # 保存路径
    CHECKPOINT_DIR: str = './checkpoints'       # 模型保存目录
    LOG_DIR: str = './logs'                     # 日志保存目录
    RESULT_DIR: str = './results'               # 结果保存目录

    # 日志配置
    USE_TENSORBOARD: bool = True        # 是否使用 TensorBoard
    USE_WANDB: bool = False             # 是否使用 Weights & Biases
    LOG_INTERVAL: int = 10              # 日志记录间隔

    # 可视化配置
    PLOT_WEIGHTS: bool = True           # 是否绘制权重变化图
    PLOT_PREDICTIONS: bool = True       # 是否绘制预测结果图
    SAVE_WEIGHT_HISTORY: bool = True    # 是否保存权重历史

    SEED: int = 42                      # 随机种子


# ============================================================================
# 消融实验配置
# ============================================================================

class AblationConfig:
    """消融实验配置"""

    RUN_STATIC_BASELINE: bool = True    # 是否运行静态权重基线
    ABLATE_SELF_ATTENTION: bool = False # 移除 Self-Attention
    ABLATE_TEMPERATURE: bool = False    # 固定温度为1.0
    ABLATE_SENTIMENT: bool = False      # 移除情感特征
    ABLATE_CONTEXT: bool = False        # 移除环境特征


# ============================================================================
# 主配置类
# ============================================================================

class ModelConfig:
    """
    主配置类

    整合所有子配置，提供统一的访问接口
    """

    def __init__(self):
        # 数据配置
        self.data_config = DataConfig()
        self.scenic_name = DataConfig.TARGET_SCENIC_NAME
        self.data_dir = DataConfig.DATA_DIR
        self.seq_len = DataConfig.SEQ_LEN
        self.pred_len = DataConfig.PRED_LEN
        self.target_dim = DataConfig.TARGET_DIM

        # 特征维度（运行时从 feature_map.json 动态设置）
        self.sentiment_dim = None
        self.context_dim = None
        self.total_dim = None
        self.sentiment_indices = None
        self.context_indices = None

        # 权重生成网络配置
        self.hidden_dim = WeightNetConfig.HIDDEN_DIM
        self.num_attention_heads = WeightNetConfig.NUM_ATTENTION_HEADS
        self.num_encoder_layers = WeightNetConfig.NUM_ENCODER_LAYERS
        self.weight_net_dropout = WeightNetConfig.DROPOUT
        self.default_temperature = WeightNetConfig.DEFAULT_TEMPERATURE
        self.weight_net_activation = WeightNetConfig.ACTIVATION

        # 骨干网络配置
        self.d_model = BackboneConfig.D_MODEL
        self.num_heads = BackboneConfig.NUM_HEADS
        self.num_layers = BackboneConfig.NUM_LAYERS
        self.dim_feedforward = BackboneConfig.DIM_FEEDFORWARD
        self.backbone_dropout = BackboneConfig.DROPOUT
        self.max_position_embeddings = BackboneConfig.MAX_POSITION_EMBEDDINGS
        self.backbone_activation = BackboneConfig.ACTIVATION

        # 训练配置
        self.learning_rate = TrainingConfig.LEARNING_RATE
        self.weight_decay = TrainingConfig.WEIGHT_DECAY
        self.optimizer = TrainingConfig.OPTIMIZER
        self.lr_scheduler = TrainingConfig.LR_SCHEDULER
        self.lr_warmup_epochs = TrainingConfig.LR_WARMUP_EPOCHS
        self.lr_min = TrainingConfig.LR_MIN
        self.batch_size = TrainingConfig.BATCH_SIZE
        self.num_epochs = TrainingConfig.NUM_EPOCHS
        self.gradient_clip = TrainingConfig.GRADIENT_CLIP
        self.stage1_epochs = TrainingConfig.STAGE1_EPOCHS
        self.stage2_epochs = TrainingConfig.STAGE2_EPOCHS
        self.stage3_epochs = TrainingConfig.STAGE3_EPOCHS
        self.early_stopping = TrainingConfig.EARLY_STOPPING
        self.patience = TrainingConfig.PATIENCE
        self.min_delta = TrainingConfig.MIN_DELTA
        self.val_interval = TrainingConfig.VAL_INTERVAL
        self.save_interval = TrainingConfig.SAVE_INTERVAL
        self.save_best_only = TrainingConfig.SAVE_BEST_ONLY
        self.device = TrainingConfig.DEVICE
        self.num_workers = TrainingConfig.NUM_WORKERS
        self.pin_memory = TrainingConfig.PIN_MEMORY

        # 损失配置
        self.main_loss = LossConfig.MAIN_LOSS
        self.use_weight_sparsity = LossConfig.USE_WEIGHT_SPARSITY
        self.sparsity_weight = LossConfig.SPARSITY_WEIGHT
        self.use_weight_smoothness = LossConfig.USE_WEIGHT_SMOOTHNESS
        self.smoothness_weight = LossConfig.SMOOTHNESS_WEIGHT
        self.use_weight_entropy = LossConfig.USE_WEIGHT_ENTROPY
        self.entropy_weight = LossConfig.ENTROPY_WEIGHT
        self.huber_delta = LossConfig.HUBER_DELTA

        # 实验配置
        self.experiment_name = ExperimentConfig.EXPERIMENT_NAME
        self.run_id = ExperimentConfig.RUN_ID
        self.checkpoint_dir = ExperimentConfig.CHECKPOINT_DIR
        self.log_dir = ExperimentConfig.LOG_DIR
        self.result_dir = ExperimentConfig.RESULT_DIR
        self.use_tensorboard = ExperimentConfig.USE_TENSORBOARD
        self.use_wandb = ExperimentConfig.USE_WANDB
        self.log_interval = ExperimentConfig.LOG_INTERVAL
        self.plot_weights = ExperimentConfig.PLOT_WEIGHTS
        self.plot_predictions = ExperimentConfig.PLOT_PREDICTIONS
        self.save_weight_history = ExperimentConfig.SAVE_WEIGHT_HISTORY
        self.seed = ExperimentConfig.SEED

        # 消融实验配置
        self.run_static_baseline = AblationConfig.RUN_STATIC_BASELINE
        self.ablate_self_attention = AblationConfig.ABLATE_SELF_ATTENTION
        self.ablate_temperature = AblationConfig.ABLATE_TEMPERATURE
        self.ablate_sentiment = AblationConfig.ABLATE_SENTIMENT
        self.ablate_context = AblationConfig.ABLATE_CONTEXT

    def load_feature_map(self, feature_map: Dict):
        """
        从 feature_map.json 加载特征索引

        参数：
            feature_map: dict, 包含 sentiment_idxs 和 context_idxs
        """
        self.sentiment_indices = feature_map.get('sentiment_idxs', [])
        self.context_indices = feature_map.get('context_idxs', [])

        self.sentiment_dim = len(self.sentiment_indices)
        self.context_dim = len(self.context_indices)
        self.total_dim = self.sentiment_dim + self.context_dim

        print(f"✅ 特征映射已加载:")
        print(f"   情感特征维度: {self.sentiment_dim}, 索引: {self.sentiment_indices}")
        print(f"   环境特征维度: {self.context_dim}, 索引: {self.context_indices}")
        print(f"   总特征维度: {self.total_dim}")

    def load_feature_map_from_file(self, filepath: str = None):
        """
        从 JSON 文件加载特征映射

        参数：
            filepath: JSON 文件路径（默认使用配置路径）
        """
        if filepath is None:
            filepath = self.data_config.get_file_path(DataConfig.FEATURE_MAP_TEMPLATE)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"特征映射表文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            feature_map = json.load(f)

        self.load_feature_map(feature_map)

    def __repr__(self):
        """打印配置信息"""
        config_dict = {
            '数据配置': {
                '景区名称': self.scenic_name,
                '序列长度': self.seq_len,
                '预测长度': self.pred_len,
                '特征维度': f'{self.total_dim} = {self.sentiment_dim}(情感) + {self.context_dim}(环境)' if self.total_dim else '未加载'
            },
            '权重生成网络': {
                '隐藏维度': self.hidden_dim,
                '注意力头数': self.num_attention_heads,
                'Encoder层数': self.num_encoder_layers
            },
            '骨干网络': {
                '模型维度': self.d_model,
                '注意力头数': self.num_heads,
                'Transformer层数': self.num_layers
            },
            '训练配置': {
                '学习率': self.learning_rate,
                '批次大小': self.batch_size,
                '总轮数': self.num_epochs,
                '设备': self.device
            }
        }

        output = "="*80 + "\n"
        output += "模型配置详情\n"
        output += "="*80 + "\n"

        for section, params in config_dict.items():
            output += f"\n【{section}】\n"
            for key, value in params.items():
                output += f"  {key}: {value}\n"

        output += "="*80
        return output


# ============================================================================
# 测试与工具函数
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("配置文件测试")
    print("="*80)

    # 1. 打印所有需要的数据文件清单
    DataConfig.print_required_files()

    # 2. 创建数据目录
    print("\n【创建数据目录】")
    DataConfig.create_data_dir()

    # 3. 检查文件存在性
    DataConfig.print_paths_info()

    # 4. 创建配置对象
    print("\n【配置对象创建】")
    config = ModelConfig()

    # 5. 模拟加载特征映射
    print("\n【特征映射加载测试】")
    mock_feature_map = {
        'sentiment_idxs': [0, 1, 2, 3, 4],
        'context_idxs': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    }
    config.load_feature_map(mock_feature_map)

    # 6. 打印配置
    print("\n" + str(config))

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)