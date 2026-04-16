# -*- coding: utf-8 -*-
"""
M9 模型配置文件
Dynamic Fusion Predictor Configuration
用途: 定义模型超参数、架构配置
改进: 支持从 dataset_info.json 自动加载 seq_len 和 pred_len
"""

import json
import os
from pathlib import Path


class ModelConfig:
    """
    动态融合预测器配置类
    支持三阶段训练和可解释性分析
    """

    # ==================== 模型核心架构参数 ====================

    # Transformer Backbone 参数
    d_model = 64  # 隐藏层维度 (Embedding维度)
    nhead = 2  # 多头注意力头数 (必须能被 d_model 整除)
    num_layers = 1  # Transformer Encoder 层数
    dim_feedforward = 4*d_model  # 前馈网络隐藏层维度 (通常是 d_model 的 4 倍)
    dropout = 0.3  # Dropout 比率

    # ==================== 权重网络参数 ====================

    # Context-Aware Weight Network 参数
    weight_net_hidden_dim = 64  # 权重网络隐藏层维度
    weight_net_num_heads = 4  # 权重网络的注意力头数
    weight_net_num_layers = 2  # 权重网络的 Transformer 层数
    weight_net_dropout = 0.1  # 权重网络的 Dropout

    # Temperature 参数 (用于 Softmax)
    temperature_init = 1.0  # 初始温度 (1.0 表示标准 Softmax)
    temperature_trainable = True  # 是否将温度设为可学习参数

    # ==================== 输出层参数 ====================
    seq_len = 336   # 输入序列长度（使用过去30天数据）
    output_dim = 1  # 输出维度 (单变量预测)
    pred_len = 72  # 预测长度 (未来7天)

    # ==================== 激活函数 ====================

    activation = 'gelu'  # 激活函数: 'relu', 'gelu', 'silu'

    # ==================== 位置编码参数 ====================

    max_seq_length = 5000  # 最大序列长度 (用于位置编码)
    positional_encoding_type = 'sinusoidal'  # 'sinusoidal' 或 'learnable'

    # ==================== 正则化参数 ====================

    # L1 正则化系数 (用于稀疏化权重)
    weight_l1_lambda = 0.001  # 权重 L1 正则化系数

    # LayerNorm 参数
    layer_norm_eps = 1e-5  # LayerNorm epsilon

    # ==================== 训练相关参数 ====================

    # 三阶段训练配置
    stage1_epochs = 10  # 阶段1: 仅训练骨干网络
    stage2_epochs = 10  # 阶段2: 仅训练权重网络
    stage3_epochs = 20  # 阶段3: 联合微调

    # ==================== 输入嵌入参数 ====================

    input_embedding_type = 'linear'  # 'linear' 或 'conv1d'

    # ==================== 可视化参数 ====================

    save_attention_weights = True  # 是否保存注意力权重
    save_dynamic_weights = True  # 是否保存动态权重

    # ==================== 数据集目录 (用于自动加载) ====================
    
    _dataset_dir = None  # 当前加载的数据集目录
    _loaded_from_dataset = False  # 是否已从数据集加载

    # ==================== 从数据集自动加载参数 ====================

    @classmethod
    def from_dataset(cls, dataset_dir: str, verbose: bool = True):
        """
        从数据集目录的 dataset_info.json 自动加载 seq_len 和 pred_len
        
        Args:
            dataset_dir: 数据集目录路径 (包含 dataset_info.json)
            verbose: 是否打印加载信息
            
        Returns:
            cls: 返回配置类本身，支持链式调用
            
        Example:
            >>> ModelConfig.from_dataset("data/output/m8b_xsw_hourly_1day")
            >>> model = build_model(feature_map, ModelConfig)
        """
        dataset_path = Path(dataset_dir)
        info_file = dataset_path / "dataset_info.json"
        
        if not info_file.exists():
            raise FileNotFoundError(
                f"未找到 dataset_info.json: {info_file}\n"
                f"请确保数据集目录正确，或先运行 m8b_build_h.py 生成数据集"
            )
        
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        # 提取并覆盖参数
        old_seq_len = cls.seq_len
        old_pred_len = cls.pred_len
        
        cls.seq_len = info.get('seq_len', cls.seq_len)
        cls.pred_len = info.get('pred_len', cls.pred_len)
        cls._dataset_dir = str(dataset_path)
        cls._loaded_from_dataset = True
        
        if verbose:
            print(f"\n{'=' * 70}")
            print("ModelConfig: 从数据集自动加载参数")
            print(f"{'=' * 70}")
            print(f"  数据集目录: {dataset_dir}")
            print(f"  seq_len: {old_seq_len} → {cls.seq_len}")
            print(f"  pred_len: {old_pred_len} → {cls.pred_len}")
            
            # 显示其他有用信息
            if 'feature_dim' in info:
                print(f"  feature_dim: {info['feature_dim']}")
            if 'samples' in info:
                samples = info['samples']
                print(f"  样本数: train={samples.get('train', '?')}, "
                      f"val={samples.get('val', '?')}, test={samples.get('test', '?')}")
            if 'granularity' in info:
                print(f"  数据粒度: {info['granularity']}")
            print(f"{'=' * 70}\n")
        
        return cls

    @classmethod
    def reset_to_default(cls):
        """
        重置 seq_len 和 pred_len 为默认值
        用于在切换数据集时清除之前的加载状态
        """
        cls.seq_len = 336
        cls.pred_len = 72
        cls._dataset_dir = None
        cls._loaded_from_dataset = False
        print("✓ ModelConfig 已重置为默认值")
        return cls

    @classmethod
    def get_dataset_dir(cls) -> str:
        """获取当前加载的数据集目录"""
        return cls._dataset_dir

    @classmethod
    def to_dict(cls):
        """导出可记录的模型配置字典"""
        return {
            'd_model': cls.d_model,
            'nhead': cls.nhead,
            'num_layers': cls.num_layers,
            'dim_feedforward': cls.dim_feedforward,
            'dropout': cls.dropout,
            'weight_net_hidden_dim': cls.weight_net_hidden_dim,
            'weight_net_num_heads': cls.weight_net_num_heads,
            'weight_net_num_layers': cls.weight_net_num_layers,
            'weight_net_dropout': cls.weight_net_dropout,
            'temperature_init': cls.temperature_init,
            'temperature_trainable': cls.temperature_trainable,
            'seq_len': cls.seq_len,
            'output_dim': cls.output_dim,
            'pred_len': cls.pred_len,
            'activation': cls.activation,
            'max_seq_length': cls.max_seq_length,
            'positional_encoding_type': cls.positional_encoding_type,
            'weight_l1_lambda': cls.weight_l1_lambda,
            'layer_norm_eps': cls.layer_norm_eps,
            'stage1_epochs': cls.stage1_epochs,
            'stage2_epochs': cls.stage2_epochs,
            'stage3_epochs': cls.stage3_epochs,
            'input_embedding_type': cls.input_embedding_type,
            'save_attention_weights': cls.save_attention_weights,
            'save_dynamic_weights': cls.save_dynamic_weights,
            'dataset_dir': cls._dataset_dir,
            'loaded_from_dataset': cls._loaded_from_dataset
        }
 
    # ==================== 验证参数 ====================

    @classmethod
    def validate(cls):
        """验证配置参数的合法性"""
        assert cls.d_model % cls.nhead == 0, \
            f"d_model ({cls.d_model}) 必须能被 nhead ({cls.nhead}) 整除"

        assert cls.weight_net_hidden_dim % cls.weight_net_num_heads == 0, \
            f"weight_net_hidden_dim ({cls.weight_net_hidden_dim}) 必须能被 weight_net_num_heads ({cls.weight_net_num_heads}) 整除"

        assert cls.temperature_init > 0, \
            f"temperature_init 必须大于 0，当前值: {cls.temperature_init}"

        assert cls.dropout >= 0 and cls.dropout < 1, \
            f"dropout 必须在 [0, 1) 范围内，当前值: {cls.dropout}"

        assert cls.activation in ['relu', 'gelu', 'silu'], \
            f"不支持的激活函数: {cls.activation}"

        print("✓ 配置参数验证通过")
        return True

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 70)
        print("Model Configuration")
        print("=" * 70)
        
        # 显示数据集加载状态
        if cls._loaded_from_dataset:
            print(f"\n【数据集加载状态】")
            print(f"  - 已从数据集加载: ✓")
            print(f"  - 数据集目录: {cls._dataset_dir}")
        else:
            print(f"\n【数据集加载状态】")
            print(f"  - 已从数据集加载: ✗ (使用默认值)")
        
        print("\n【序列参数】")
        print(f"  - seq_len:          {cls.seq_len}")
        print(f"  - pred_len:         {cls.pred_len}")
        
        print("\n【Transformer Backbone】")
        print(f"  - d_model:          {cls.d_model}")
        print(f"  - nhead:            {cls.nhead}")
        print(f"  - num_layers:       {cls.num_layers}")
        print(f"  - dim_feedforward:  {cls.dim_feedforward}")
        print(f"  - dropout:          {cls.dropout}")

        print("\n【Weight Network】")
        print(f"  - hidden_dim:       {cls.weight_net_hidden_dim}")
        print(f"  - num_heads:        {cls.weight_net_num_heads}")
        print(f"  - num_layers:       {cls.weight_net_num_layers}")
        print(f"  - temperature:      {cls.temperature_init} (trainable={cls.temperature_trainable})")

        print("\n【Output】")
        print(f"  - output_dim:       {cls.output_dim}")
        print(f"  - pred_len:         {cls.pred_len}")

        print("\n【Training】")
        print(f"  - Stage 1 epochs:   {cls.stage1_epochs}")
        print(f"  - Stage 2 epochs:   {cls.stage2_epochs}")
        print(f"  - Stage 3 epochs:   {cls.stage3_epochs}")

        print("\n【Regularization】")
        print(f"  - L1 lambda:        {cls.weight_l1_lambda}")

        print("=" * 70 + "\n")


# 配置验证
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# ModelConfig 测试")
    print("#" * 70)
    
    # 测试默认配置
    print("\n--- 测试1: 默认配置 ---")
    ModelConfig.validate()
    ModelConfig.print_config()
    
    # 测试从数据集加载 (如果存在)
    print("\n--- 测试2: 从数据集加载 ---")
    test_dirs = [
        "data/output/m8b_xsw_hourly_1day",
        "data/output/m8b_xsw_hourly_v2",
        "data/output/m8b_eedscy_hourly_1day"
    ]
    
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            print(f"\n尝试加载: {test_dir}")
            try:
                ModelConfig.from_dataset(test_dir)
                ModelConfig.print_config()
                ModelConfig.reset_to_default()
            except Exception as e:
                print(f"  加载失败: {e}")
            break
    else:
        print("  未找到可用的测试数据集目录")
    
    print("\n" + "#" * 70)
    print("# 使用示例")
    print("#" * 70)
    print("""
# 在训练脚本中使用:

from config.m9_config import ModelConfig

# 方式1: 从数据集目录自动加载
dataset_dir = "data/output/m8b_xsw_hourly_1day"
ModelConfig.from_dataset(dataset_dir)

# 方式2: 手动设置 (不推荐)
# ModelConfig.seq_len = 168
# ModelConfig.pred_len = 24

# 然后正常使用
model = build_model(feature_map, ModelConfig)
""")