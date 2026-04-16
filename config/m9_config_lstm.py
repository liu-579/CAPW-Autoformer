# -*- coding: utf-8 -*-
"""
M9 LSTM 模型配置文件
LSTM-based Dynamic Fusion Predictor Configuration
用途: 定义 LSTM 骨干网络的模型超参数、架构配置
改进: 将主干网络从 Autoformer / Vanilla Transformer 替换为 LSTM
      保留 ContextAwareWeightNet 动态权重融合机制
"""

import json
import os
from pathlib import Path


class ModelConfig:
    """
    LSTM 动态融合预测器配置类
    主干: 双向 LSTM + 全连接输出头
    融合: ContextAwareWeightNet (1+Tanh 残差激活)
    """

    # ==================== LSTM 核心架构参数 ====================

    d_model = 256          # LSTM 隐藏层维度 (hidden_size)
    lstm_num_layers = 2    # LSTM 堆叠层数
    lstm_bidirectional = True  # 是否使用双向 LSTM
    dropout = 0.1          # Dropout 比率 (层间)

    # ==================== 全连接输出头参数 ====================

    fc_hidden_dim = 256    # 输出头隐藏层维度
    fc_num_layers = 2      # 输出头全连接层数

    # ==================== 权重网络参数 ====================

    # Context-Aware Weight Network 参数 (与原版保持一致)
    weight_net_hidden_dim = 32
    weight_net_num_heads = 2
    weight_net_num_layers = 1
    weight_net_dropout = 0.1

    # Temperature 参数
    temperature_init = 1.0
    temperature_trainable = True

    # ==================== 输出层参数 ====================
    seq_len = 168         # 输入序列长度 (默认值, 会被 from_dataset 覆盖)
    output_dim = 1        # 输出维度 (单变量预测)
    pred_len = 24         # 预测长度 (默认值, 会被 from_dataset 覆盖)

    # ==================== 消融实验参数 ====================

    # ablation_mode 消融模式:
    #   'full'           - 全开: 情感数据 × 权重生成网络权重 → 主干网络 (默认)
    #   'no_weight_net'  - 无权重网络: 情感数据 × 1 → 主干网络 (权重网络输出恒为1)
    #   'no_sentiment'   - 全关: 情感数据与权重网络均不参与, 仅环境特征进入主干网络
    ablation_mode = 'full'

    # ==================== 激活函数 ====================

    activation = 'gelu'

    # ==================== 正则化参数 ====================

    weight_l1_lambda = 0.001
    layer_norm_eps = 1e-5

    # ==================== 训练相关参数 ====================

    stage1_epochs = 10
    stage2_epochs = 10
    stage3_epochs = 20

    # ==================== 可视化参数 ====================

    save_attention_weights = False  # LSTM 无注意力权重
    save_dynamic_weights = True

    # ==================== 数据集目录 (用于自动加载) ====================

    _dataset_dir = None
    _loaded_from_dataset = False

    # ==================== 从数据集自动加载参数 ====================

    @classmethod
    def from_dataset(cls, dataset_dir: str, verbose: bool = True):
        """
        从数据集目录的 dataset_info.json 自动加载 seq_len 和 pred_len
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

        cls.seq_len = info.get('seq_len', cls.seq_len)
        cls.pred_len = info.get('pred_len', cls.pred_len)

        cls._dataset_dir = str(dataset_path)
        cls._loaded_from_dataset = True

        if verbose:
            print(f"\n{'=' * 70}")
            print("ModelConfig (LSTM): 从数据集自动加载参数")
            print(f"{'=' * 70}")
            print(f"  数据集目录: {dataset_dir}")
            print(f"  seq_len: {old_seq_len} → {cls.seq_len}")
            print(f"  pred_len: {old_pred_len} → {cls.pred_len}")

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
        cls.seq_len = 168
        cls.pred_len = 24
        cls._dataset_dir = None
        cls._loaded_from_dataset = False
        print("✓ ModelConfig (LSTM) 已重置为默认值")
        return cls

    @classmethod
    def get_dataset_dir(cls) -> str:
        return cls._dataset_dir

    @classmethod
    def to_dict(cls):
        """导出可记录的模型配置字典"""
        return {
            'model_type': 'lstm',
            'd_model': cls.d_model,
            'lstm_num_layers': cls.lstm_num_layers,
            'lstm_bidirectional': cls.lstm_bidirectional,
            'dropout': cls.dropout,
            'fc_hidden_dim': cls.fc_hidden_dim,
            'fc_num_layers': cls.fc_num_layers,
            'weight_net_hidden_dim': cls.weight_net_hidden_dim,
            'weight_net_num_heads': cls.weight_net_num_heads,
            'weight_net_num_layers': cls.weight_net_num_layers,
            'weight_net_dropout': cls.weight_net_dropout,
            'temperature_init': cls.temperature_init,
            'temperature_trainable': cls.temperature_trainable,
            'seq_len': cls.seq_len,
            'output_dim': cls.output_dim,
            'pred_len': cls.pred_len,
            'ablation_mode': cls.ablation_mode,
            'activation': cls.activation,
            'weight_l1_lambda': cls.weight_l1_lambda,
            'layer_norm_eps': cls.layer_norm_eps,
            'stage1_epochs': cls.stage1_epochs,
            'stage2_epochs': cls.stage2_epochs,
            'stage3_epochs': cls.stage3_epochs,
            'save_attention_weights': cls.save_attention_weights,
            'save_dynamic_weights': cls.save_dynamic_weights,
            'dataset_dir': cls._dataset_dir,
            'loaded_from_dataset': cls._loaded_from_dataset
        }

    @classmethod
    def validate(cls):
        """验证配置参数的合法性"""
        assert cls.d_model > 0, \
            f"d_model 必须大于 0，当前值: {cls.d_model}"

        assert cls.lstm_num_layers >= 1, \
            f"lstm_num_layers 必须 >= 1，当前值: {cls.lstm_num_layers}"

        assert cls.weight_net_hidden_dim % cls.weight_net_num_heads == 0, \
            f"weight_net_hidden_dim ({cls.weight_net_hidden_dim}) 必须能被 weight_net_num_heads ({cls.weight_net_num_heads}) 整除"

        assert cls.temperature_init > 0, \
            f"temperature_init 必须大于 0，当前值: {cls.temperature_init}"

        assert cls.dropout >= 0 and cls.dropout < 1, \
            f"dropout 必须在 [0, 1) 范围内，当前值: {cls.dropout}"

        assert cls.activation in ['relu', 'gelu', 'silu'], \
            f"不支持的激活函数: {cls.activation}"

        assert cls.ablation_mode in ['full', 'no_weight_net', 'no_sentiment'], \
            f"ablation_mode 必须为 'full', 'no_weight_net' 或 'no_sentiment'，当前值: {cls.ablation_mode}"

        print("✓ 配置参数验证通过 (LSTM)")
        return True

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 70)
        print("Model Configuration (LSTM)")
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
        print(f"  - pred_len:         {cls.pred_len}")

        print("\n【LSTM Backbone】")
        print(f"  - d_model (hidden): {cls.d_model}")
        print(f"  - num_layers:       {cls.lstm_num_layers}")
        print(f"  - bidirectional:    {cls.lstm_bidirectional}")
        print(f"  - dropout:          {cls.dropout}")

        print("\n【FC Output Head】")
        print(f"  - fc_hidden_dim:    {cls.fc_hidden_dim}")
        print(f"  - fc_num_layers:    {cls.fc_num_layers}")

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
    print("# ModelConfig (LSTM) 测试")
    print("#" * 70)

    ModelConfig.validate()
    ModelConfig.print_config()
