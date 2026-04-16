# -*- coding: utf-8 -*-
"""
M10 LSTM 训练脚本 (包装器)
LSTM Training Wrapper

用途: 使用 LSTM 骨干替代 Autoformer / Vanilla Transformer 进行训练。
原理: 复用 m10_train_2.py 中的全部训练逻辑 (CCC Loss, 自适应峰值加权, 早停等)，
      仅替换模型构建函数和模型配置为 LSTM 版本。

使用方式:
    cd scripts && python m10_train_lstm.py

修改训练参数:
    - 训练超参: 修改 config/m10_config.py (TrainingConfig)
    - 模型超参: 修改 config/m9_config_lstm.py (ModelConfig)
"""

import sys
sys.path.append('.')

# ==================== 核心替换: 导入 LSTM 版本 ====================
from m9_model_lstm import build_model as build_model_lstm
from config.m9_config_lstm import ModelConfig as LSTMModelConfig

# 导入训练模块 (包含全部训练逻辑)
import m10_train_2

# ==================== Monkey-Patch: 替换模块级引用 ====================
# 将 m10_train_2 中的模型构建函数和配置类替换为 LSTM 版本
# 这样 main() 中所有引用 build_model / ModelConfig 的地方都会自动使用新版本
m10_train_2.build_model = build_model_lstm
m10_train_2.ModelConfig = LSTMModelConfig


def main(training_overrides=None, model_overrides=None, reset_model_config=True,
         restore_after_run=False, trial=None, trial_metric='best_val_ccc_loss',
         prune_enabled=False):
    """
    LSTM 训练入口 (参数与 m10_train_2.main 完全一致)

    Args:
        training_overrides: 训练配置覆盖字典
        model_overrides: 模型配置覆盖字典
        reset_model_config: 是否在训练前重置模型配置
        restore_after_run: 训练后是否恢复配置
        trial: Optuna trial 对象 (HPO 时使用)
        trial_metric: HPO 优化指标
        prune_enabled: 是否启用 HPO 剪枝

    Returns:
        experiment_record: 实验记录字典
    """
    # 获取当前消融模式
    _ablation_mode = LSTMModelConfig.ablation_mode
    _ablation_desc = {
        'full': '🟢 全开 (情感×权重网络→主干)',
        'no_weight_net': '🟡 无权重网络 (情感×1→主干)',
        'no_sentiment': '🔴 全关 (仅环境特征→主干)'
    }

    print("\n" + "=" * 70)
    print("🔄 使用 LSTM 骨干网络")
    print("   替换: Autoformer / Transformer → LSTM (双向 + FC 输出头)")
    print("   特点: 结构简洁, 训练速度快, 适合消融实验基线")
    print(f"   消融模式: {_ablation_desc.get(_ablation_mode, _ablation_mode)}")
    print("=" * 70)

    return m10_train_2.main(
        training_overrides=training_overrides,
        model_overrides=model_overrides,
        reset_model_config=reset_model_config,
        restore_after_run=restore_after_run,
        trial=trial,
        trial_metric=trial_metric,
        prune_enabled=prune_enabled
    )


if __name__ == '__main__':
    main()
