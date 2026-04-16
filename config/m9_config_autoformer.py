# -*- coding: utf-8 -*-
"""
M9 Autoformer 模型配置文件
Autoformer-based Dynamic Fusion Predictor Configuration
用途: 定义 Autoformer 骨干网络的模型超参数、架构配置
改进: 将主干网络从 Vanilla Transformer Encoder 替换为 Autoformer (Encoder-Decoder)
      保留 ContextAwareWeightNet 动态权重融合机制
"""

import json
import os
from pathlib import Path


class ModelConfig:
    """
    Autoformer 动态融合预测器配置类
    主干: Autoformer (Auto-Correlation + Series Decomposition)
    融合: ContextAwareWeightNet (1+Tanh 残差激活)
    """

    # ==================== Autoformer 核心架构参数 ====================

    # Autoformer Backbone 参数
    d_model = 64          # 隐藏层维度 (Embedding 维度)
    nhead = 1             # 多头注意力 / Auto-Correlation 头数
    num_encoder_layers = 3  # Autoformer Encoder 层数
    num_decoder_layers = 3 # Autoformer Decoder 层数
    dim_feedforward = 2 * d_model  # 前馈网络隐藏层维度
    dropout = 0.1         # Dropout 比率

    # ==================== Autoformer 特有参数 ====================

    # Series Decomposition (序列分解)
    moving_avg_kernel = 13  # 移动平均核大小 (用于趋势-季节分解)

    # Auto-Correlation Mechanism
    autocorrelation_factor = 2  # Top-k 周期选择因子 (c in paper)
    # 实际 top_k = ceil(autocorrelation_factor * log(seq_len))

    # ==================== 权重网络参数 ====================

    # Context-Aware Weight Network 参数 (与原版保持一致)
    weight_net_hidden_dim = 128
    weight_net_num_heads = 4
    weight_net_num_layers = 2
    weight_net_dropout = 0.1

    # Temperature 参数
    temperature_init = 1.0
    temperature_trainable = True

    # ==================== 输出层参数 ====================
    seq_len = 168         # 输入序列长度 (默认值, 会被 from_dataset 覆盖)
    label_len = 48        # Decoder 标签长度 (Autoformer 特有: Decoder 的已知部分)
    output_dim = 1        # 输出维度 (单变量预测)
    pred_len = 24         # 预测长度 (默认值, 会被 from_dataset 覆盖)

    # ==================== 消融实验参数 ====================

    # ablation_mode 消融模式:
    #   'full'           - 全开: 情感数据 × 权重生成网络权重 → 主干网络 (默认)
    #   'no_weight_net'  - 无权重网络: 情感数据 × 1 → 主干网络 (权重网络输出恒为1)
    #   'no_sentiment'   - 全关: 情感数据与权重网络均不参与, 仅环境特征进入主干网络
    ablation_mode = 'no_sentiment'

    # ==================== 激活函数 ====================

    activation = 'relu'

    # ==================== 位置编码参数 ====================

    max_seq_length = 5000
    positional_encoding_type = 'sinusoidal'

    # ==================== 正则化参数 ====================

    weight_l1_lambda = 0.001
    layer_norm_eps = 1e-5

    # ==================== 训练相关参数 ====================

    stage1_epochs = 10
    stage2_epochs = 10
    stage3_epochs = 20

    # ==================== 输入嵌入参数 ====================

    input_embedding_type = 'linear'

    # ==================== 可视化参数 ====================

    save_attention_weights = True
    save_dynamic_weights = True

    # ==================== 数据集目录 (用于自动加载) ====================

    _dataset_dir = None
    _loaded_from_dataset = False

    # ==================== 从数据集自动加载参数 ====================

    @classmethod
    def from_dataset(cls, dataset_dir: str, verbose: bool = True):
        """
        从数据集目录的 dataset_info.json 自动加载 seq_len 和 pred_len
        同时自动计算 label_len (默认为 pred_len 的 2/3 或最小 24)
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

        old_seq_len = cls.seq_len
        old_pred_len = cls.pred_len
        old_label_len = cls.label_len

        cls.seq_len = info.get('seq_len', cls.seq_len)
        cls.pred_len = info.get('pred_len', cls.pred_len)

        # 自动计算 label_len: Decoder 需要的已知标签长度
        # 通常设为 pred_len // 2, 但不小于 24 (经验值)
        cls.label_len = max(cls.pred_len // 2, min(24, cls.pred_len))

        cls._dataset_dir = str(dataset_path)
        cls._loaded_from_dataset = True

        if verbose:
            print(f"\n{'=' * 70}")
            print("ModelConfig (Autoformer): 从数据集自动加载参数")
            print(f"{'=' * 70}")
            print(f"  数据集目录: {dataset_dir}")
            print(f"  seq_len: {old_seq_len} → {cls.seq_len}")
            print(f"  pred_len: {old_pred_len} → {cls.pred_len}")
            print(f"  label_len: {old_label_len} → {cls.label_len} (auto)")

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
        """重置为默认值"""
        cls.seq_len = 336
        cls.pred_len = 72
        cls.label_len = 48
        cls._dataset_dir = None
        cls._loaded_from_dataset = False
        print("✓ ModelConfig (Autoformer) 已重置为默认值")
        return cls

    @classmethod
    def get_dataset_dir(cls) -> str:
        return cls._dataset_dir

    @classmethod
    def to_dict(cls):
        """导出可记录的模型配置字典"""
        return {
            'model_type': 'autoformer',
            'd_model': cls.d_model,
            'nhead': cls.nhead,
            'num_encoder_layers': cls.num_encoder_layers,
            'num_decoder_layers': cls.num_decoder_layers,
            'dim_feedforward': cls.dim_feedforward,
            'dropout': cls.dropout,
            'moving_avg_kernel': cls.moving_avg_kernel,
            'autocorrelation_factor': cls.autocorrelation_factor,
            'weight_net_hidden_dim': cls.weight_net_hidden_dim,
            'weight_net_num_heads': cls.weight_net_num_heads,
            'weight_net_num_layers': cls.weight_net_num_layers,
            'weight_net_dropout': cls.weight_net_dropout,
            'temperature_init': cls.temperature_init,
            'temperature_trainable': cls.temperature_trainable,
            'seq_len': cls.seq_len,
            'label_len': cls.label_len,
            'output_dim': cls.output_dim,
            'pred_len': cls.pred_len,
            'ablation_mode': cls.ablation_mode,
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

        assert cls.moving_avg_kernel % 2 == 1, \
            f"moving_avg_kernel 必须为奇数，当前值: {cls.moving_avg_kernel}"

        assert cls.label_len <= cls.seq_len, \
            f"label_len ({cls.label_len}) 不能大于 seq_len ({cls.seq_len})"

        assert cls.ablation_mode in ['full', 'no_weight_net', 'no_sentiment'], \
            f"ablation_mode 必须为 'full', 'no_weight_net' 或 'no_sentiment'，当前值: {cls.ablation_mode}"

        print("✓ 配置参数验证通过 (Autoformer)")
        return True

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 70)
        print("Model Configuration (Autoformer)")
        print("=" * 70)

        if cls._loaded_from_dataset:
            print(f"\n【数据集加载状态】")
            print(f"  - 已从数据集加载: ✓")
            print(f"  - 数据集目录: {cls._dataset_dir}")
        else:
            print(f"\n【数据集加载状态】")
            print(f"  - 已从数据集加载: ✗ (使用默认值)")

        print("\n【序列参数】")
        print(f"  - seq_len:          {cls.seq_len}")
        print(f"  - label_len:        {cls.label_len}  (Decoder 已知部分)")
        print(f"  - pred_len:         {cls.pred_len}")

        print("\n【Autoformer Backbone】")
        print(f"  - d_model:          {cls.d_model}")
        print(f"  - nhead:            {cls.nhead}")
        print(f"  - encoder_layers:   {cls.num_encoder_layers}")
        print(f"  - decoder_layers:   {cls.num_decoder_layers}")
        print(f"  - dim_feedforward:  {cls.dim_feedforward}")
        print(f"  - dropout:          {cls.dropout}")

        print("\n【Autoformer 特有参数】")
        print(f"  - moving_avg_kernel:     {cls.moving_avg_kernel}")
        print(f"  - autocorrelation_factor: {cls.autocorrelation_factor}")

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

        _ablation_desc = {
            'full': '全开 (情感×权重网络→主干)',
            'no_weight_net': '无权重网络 (情感×1→主干)',
            'no_sentiment': '全关 (仅环境特征→主干)'
        }
        print("\n【消融实验】")
        print(f"  - ablation_mode:    {cls.ablation_mode} ({_ablation_desc.get(cls.ablation_mode, '未知')})")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# ModelConfig (Autoformer) 测试")
    print("#" * 70)

    ModelConfig.validate()
    ModelConfig.print_config()
