# -*- coding: utf-8 -*-
"""
M9 模型配置文件
Dynamic Fusion Predictor Configuration
用途: 定义模型超参数、架构配置
"""


class ModelConfig:
    """
    动态融合预测器配置类
    支持三阶段训练和可解释性分析
    """

    # ==================== 模型核心架构参数 ====================

    # Transformer Backbone 参数
    d_model = 64  # 隐藏层维度 (Embedding维度)
    nhead = 4  # 多头注意力头数 (必须能被 d_model 整除)
    num_layers = 2  # Transformer Encoder 层数
    dim_feedforward = 256  # 前馈网络隐藏层维度 (通常是 d_model 的 4 倍)
    dropout = 0.2  # Dropout 比率

    # ==================== 权重网络参数 ====================

    # Context-Aware Weight Network 参数
    weight_net_hidden_dim = 64  # 权重网络隐藏层维度
    weight_net_num_heads = 4  # 权重网络的注意力头数
    weight_net_num_layers = 1  # 权重网络的 Transformer 层数
    weight_net_dropout = 0.1  # 权重网络的 Dropout

    # Temperature 参数 (用于 Softmax)
    temperature_init = 1.0  # 初始温度 (1.0 表示标准 Softmax)
    temperature_trainable = True  # 是否将温度设为可学习参数

    # ==================== 输出层参数 ====================

    output_dim = 1  # 输出维度 (单变量预测)
    pred_len = 7  # 预测长度 (未来7天)

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
    ModelConfig.validate()
    ModelConfig.print_config()