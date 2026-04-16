# -*- coding: utf-8 -*-
"""
M10 Autoformer 训练脚本 (包装器)
Autoformer Training Wrapper

用途: 使用 Autoformer 骨干替代原版 Vanilla Transformer 进行训练。
原理: 复用 m10_train_2.py 中的全部训练逻辑 (CCC Loss, 自适应峰值加权, 早停等)，
      仅替换模型构建函数和模型配置为 Autoformer 版本。

使用方式:
    cd scripts && python m10_train_autoformer.py

修改训练参数:
    - 训练超参: 修改 config/m10_config.py (TrainingConfig)
    - 模型超参: 修改 config/m9_config_autoformer.py (ModelConfig)
"""

import sys
sys.path.append('.')

# ==================== 核心替换: 导入 Autoformer 版本 ====================
from m9_model_autoformer import build_model as build_model_autoformer
from config.m9_config_autoformer import ModelConfig as AutoformerModelConfig

# 导入训练模块 (包含全部训练逻辑)
import m10_train_2

# ==================== Monkey-Patch: 替换模块级引用 ====================
# 将 m10_train_2 中的模型构建函数和配置类替换为 Autoformer 版本
# 这样 main() 中所有引用 build_model / ModelConfig 的地方都会自动使用新版本
m10_train_2.build_model = build_model_autoformer
m10_train_2.ModelConfig = AutoformerModelConfig


def main(training_overrides=None, model_overrides=None, reset_model_config=True,
         restore_after_run=False, trial=None, trial_metric='best_val_ccc_loss',
         prune_enabled=False):
    """
    Autoformer 训练入口 (参数与 m10_train_2.main 完全一致)

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
    print("\n" + "=" * 70)
    print("🔄 使用 Autoformer 骨干网络")
    print("   替换: Vanilla Transformer Encoder → Autoformer (Encoder-Decoder)")
    print("   新增: Series Decomposition + Auto-Correlation")
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
