# -*- coding: utf-8 -*-
"""
M10 Autoformer 超参数优化脚本 (Optuna Version)
用途:
1. 基于 Optuna + TPE 自动采样 Autoformer 模型的超参数
2. 调用 m10_train_autoformer.main() 执行单次训练
3. 在每个 epoch 上报中间指标，支持自动剪枝
4. 汇总每个 trial 的结果并保存最佳参数

与 m10_hpo_optuna.py 的区别:
- 导入 Autoformer 版训练入口 (m10_train_autoformer.main)
- 搜索空间新增 Autoformer 特有参数: moving_avg_kernel, autocorrelation_factor,
  num_encoder_layers, num_decoder_layers
- 模型参数名适配: num_layers → num_encoder_layers
"""

import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError as exc:
    raise ImportError(
        '未安装 optuna，请先执行: pip install optuna'
    ) from exc

# ==================== 核心替换: 导入 Autoformer 版本 ====================
from m10_train_autoformer import main as train_main
from m10_train_2 import HPOPruneTrial
from config.m10_config import TrainingConfig as BaseTrainingConfig
from config.m9_config_autoformer import ModelConfig as AutoformerModelConfig


class OptunaHPOConfig:
    """Autoformer Optuna 超参数优化配置"""

    # ==================== 手动配置区：基础运行设置 ====================
    STUDY_NAME = 'm10_autoformer_optuna_study_ablation_allno'   # Optuna study 名称 (消融实验专用)
    STUDY_STORAGE_FILENAME = 'm10_autoformer_optuna_study_ablation_no_sentiment.db'
    USE_PERSISTENT_STORAGE = True                       # 是否使用持久化存储（True=SQLite，False=内存）
    LOAD_IF_EXISTS = True                               # 若 study 已存在是否继续（True=续跑，False=报错）

    NUM_TRIALS = 50                                     # 总共运行多少次试验（每次采样一组超参数）
    RANDOM_SEED = 20260312                              # 随机种子，保证可复现性
    SAMPLER = 'tpe'                                     # 采样器类型：'tpe'(贝叶斯优化) / 'random'(随机搜索)
    PRUNER = 'median'                                   # 剪枝器类型：'median' / 'percentile' / 'successivehalving'

    OPTIMIZE_METRIC = 'best_val_ccc_loss'               # 优化目标指标名（从训练结果 record 中读取）
    DIRECTION = 'minimize'                              # 优化方向：'minimize'(越小越好) / 'maximize'(越大越好)
    TRIAL_REPORT_METRIC = 'best_val_ccc_loss'           # 每个 epoch 上报给剪枝器的中间指标名
    ENABLE_PRUNING = True                               # 是否启用剪枝（True=自动终止表现差的 trial）

    # ==================== 手动配置区：输出路径 ====================
    OUTPUT_ROOT = BaseTrainingConfig.BASE_DIR / 'data' / 'output_xsw' / 'hpo'/'auto2_ablation_no_sentiment'
    OUTPUT_NAME_PREFIX = 'm10_autoformer_optuna_ablation_no_sentiment'
    OUTPUT_USE_TIMESTAMP = True                         # 是否在文件夹名中附加时间戳（避免覆盖旧结果）

    # ==================== 手动配置区：快速搜索模式 ====================
    USE_FAST_MODE = True                                # 是否启用快速搜索模式（减少 epoch 数加快筛选）
    FAST_MODE_TRAINING_OVERRIDES = {
        'NUM_EPOCHS': 150,                              # 快速模式下最大训练轮数
        'PATIENCE': 20,                                 # 快速模式下早停耐心值（连续多少轮无改善则停止）
        'SAVE_INTERVAL': 999999,                        # 快速模式下模型保存间隔（设超大值=不中途保存）
    }

    # ==================== 手动配置区：搜索空间开关与候选 ====================
    BASE_BATCH_SIZE_FOR_LR_SCALING = 32                 # 学习率线性缩放的基准 batch size
    ENABLE_BATCH_SIZE_LR_SCALING = False                # 是否按 batch_size/基准 自动缩放学习率

    # ==================== 消融模式 (从 ModelConfig 读取) ====================
    # 当 ablation_mode 为 'no_sentiment' 或 'no_weight_net' 时,
    # 权重网络相关参数不参与 HPO, 减少搜索空间维度, 最大化搜索其他参数
    ABLATION_MODE = AutoformerModelConfig.ablation_mode

    # 权重网络相关的参数名集合 (当消融模式关闭权重网络时, 这些参数不参与 HPO)
    WEIGHT_NET_PARAMS = {
        'weight_net_dropout',
        'weight_net_hidden_dim',
        'weight_net_num_layers',
        'weight_net_num_heads',
    }

    # 训练超参搜索空间 (与原版保持一致)
    TRAIN_SEARCH_SPACE = {
        'LEARNING_RATE': {'type': 'loguniform', 'low': 1e-4, 'high': 2e-3},  # 学习率，对数均匀采样
        'WEIGHT_DECAY': {'type': 'loguniform', 'low': 1e-6, 'high': 5e-5},     # 权重衰减（L2 正则化系数）
        'BATCH_SIZE': {'type': 'categorical', 'values': [32,64]},                  # 批大小
        'PEAK_SIGMA': {'type': 'categorical', 'values': [1.0, 1.5, 2.0]},      # 峰值高斯平滑的标准差
        'PEAK_PENALTY_WEIGHT': {'type': 'categorical', 'values': [4.0, 5.0, 0.6, 8.0,10.0]},       # 峰值区域惩罚权重
        'LOSS_WEIGHT_PEAK_MSE': {'type': 'categorical', 'values': [0.05, 0.1, 0.2, 0.5,0.6,0.7]},# 峰值 MSE 损失占比
        'TEMP_START': {'type': 'categorical', 'values': [1.0, 2.0, 3.0]},      # 温度退火起始温度
        'TEMP_END_RATIO': {'type': 'categorical', 'values': [0.15, 0.25, 0.4, 0.5, 0.8]},# 终止温度 = 起始温度 × 此比例
        'MAX_GRAD_NORM': {'type': 'categorical', 'values': [0.5]},              # 梯度裁剪最大范数
    }

    # 模型超参搜索空间 (Autoformer 版: 新增 Autoformer 特有参数)
    MODEL_SEARCH_SPACE = {
        # --- 共享参数 (与原版类似) ---
        'dropout': {'type': 'categorical', 'values': [0.05, 0.1, 0.2, 0.3]},              # 主干网络 Dropout 率
        'weight_net_dropout': {'type': 'categorical', 'values': [0.05, 0.1, 0.2, 0.3]},   # 权重网络 Dropout 率 (消融时跳过)
        'd_model': {'type': 'categorical', 'values': [64, 128]},                           # Transformer 隐藏维度
        'dim_feedforward': {'type': 'categorical', 'values': [64, 128, 256, 512]},         # FFN 前馈层维度
        'activation': {'type': 'categorical', 'values': ['relu']},                         # 激活函数类型
        'weight_net_hidden_dim': {'type': 'categorical', 'values': [32,64,128]},           # 权重网络隐藏层维度 (消融时跳过)
        'weight_net_num_layers': {'type': 'categorical', 'values': [1, 2]},                # 权重网络层数 (消融时跳过)

        # --- 头数 (需要条件采样: 必须整除 d_model / weight_net_hidden_dim) ---
        'nhead': {'type': 'categorical', 'values': [1, 2, 4, 8]},                         # 多头注意力头数（需整除 d_model）
        'weight_net_num_heads': {'type': 'categorical', 'values': [1, 2, 4, 8]},           # 权重网络注意力头数 (消融时跳过)

        # --- Autoformer 特有参数 ---
        'num_encoder_layers': {'type': 'categorical', 'values': [1,2,3]},           # 编码器层数（替代原版 num_layers）
        'num_decoder_layers': {'type': 'categorical', 'values': [1,2,3]},           # 解码器层数

        'moving_avg_kernel': {'type': 'categorical', 'values': [13, 25]},       # 移动平均核大小（必须为奇数，用于序列分解 SeriesDecomp）

        'autocorrelation_factor': {'type': 'categorical', 'values': [1, 2]},    # Auto-Correlation 周期选择因子（越大保留越多周期信息）
    }

    @classmethod
    def should_skip_weight_net(cls) -> bool:
        """判断当前消融模式是否需要跳过权重网络参数的 HPO"""
        return cls.ABLATION_MODE in ('no_sentiment', 'no_weight_net')

    @classmethod
    def get_storage_url(cls) -> str | None:
        if not cls.USE_PERSISTENT_STORAGE:
            return None
        return f"sqlite:///{(cls.OUTPUT_ROOT / cls.STUDY_STORAGE_FILENAME).resolve().as_posix()}"

    @classmethod
    def get_hpo_root_dir(cls) -> Path:
        scenic = getattr(BaseTrainingConfig, 'scenic_name', 'default')
        suffix = datetime.now().strftime('%Y%m%d_%H%M%S') if cls.OUTPUT_USE_TIMESTAMP else 'manual'
        return cls.OUTPUT_ROOT / f'{cls.OUTPUT_NAME_PREFIX}_{scenic}_{suffix}'


class OptunaSearchOptimizer:
    """Autoformer Optuna 超参数搜索优化器"""

    def __init__(self, config=OptunaHPOConfig):
        self.config = config
        self.hpo_root_dir = config.get_hpo_root_dir()
        self.hpo_root_dir.mkdir(parents=True, exist_ok=True)
        self.storage = config.get_storage_url()
        self.results = []
        self.study = None

    def _build_sampler(self):
        sampler_name = str(self.config.SAMPLER).lower()
        if sampler_name == 'tpe':
            return optuna.samplers.TPESampler(seed=self.config.RANDOM_SEED)
        if sampler_name == 'random':
            return optuna.samplers.RandomSampler(seed=self.config.RANDOM_SEED)
        raise ValueError(f'不支持的采样器: {self.config.SAMPLER}')

    def _build_pruner(self):
        pruner_name = str(self.config.PRUNER).lower()
        if not self.config.ENABLE_PRUNING:
            return optuna.pruners.NopPruner()
        if pruner_name == 'median':
            return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=2)
        if pruner_name == 'percentile':
            return optuna.pruners.PercentilePruner(50.0, n_startup_trials=5, n_warmup_steps=15, interval_steps=2)
        if pruner_name == 'successivehalving':
            return optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=2)
        raise ValueError(f'不支持的剪枝器: {self.config.PRUNER}')

    def _suggest_from_spec(self, trial, name, spec):
        spec_type = spec['type']
        if spec_type == 'categorical':
            return trial.suggest_categorical(name, spec['values'])
        if spec_type == 'uniform':
            return trial.suggest_float(name, spec['low'], spec['high'])
        if spec_type == 'int':
            return trial.suggest_int(name, spec['low'], spec['high'], step=spec.get('step', 1))
        if spec_type == 'loguniform':
            return trial.suggest_float(name, spec['low'], spec['high'], log=True)
        raise ValueError(f'不支持的采样类型: {spec_type}')

    def _sample_conditional_heads(self, trial, param_name, base_candidates, hidden_value):
        valid_heads = [head for head in base_candidates if head <= hidden_value and hidden_value % head == 0]
        if not valid_heads:
            raise ValueError(
                f'{param_name}: hidden_value={hidden_value} 与任何候选 head {base_candidates} 均不整除，'
                f'请检查搜索空间配置'
            )
        conditional_name = f'{param_name}_for_{hidden_value}'
        return trial.suggest_categorical(conditional_name, valid_heads)

    def _sample_parameters(self, trial):
        """
        采样超参数 (Autoformer 版)
        与原版区别: 模型参数中使用 num_encoder_layers / num_decoder_layers
        替代原版的 num_layers

        消融模式适配:
        当 ablation_mode 为 'no_sentiment' 或 'no_weight_net' 时,
        权重网络相关参数 (weight_net_*) 不参与 HPO 采样,
        减少搜索空间维度, 让 Optuna 集中搜索其他参数的最优组合
        """
        training_overrides = {}
        model_overrides = {}
        skip_weight_net = self.config.should_skip_weight_net()

        if skip_weight_net:
            print(f'  ⚙️ 消融模式: {self.config.ABLATION_MODE} → 跳过权重网络参数 HPO')

        # 训练参数采样 (与原版一致)
        for name, spec in self.config.TRAIN_SEARCH_SPACE.items():
            if name == 'TEMP_END_RATIO':
                continue
            training_overrides[name] = self._suggest_from_spec(trial, f'train__{name}', spec)

        temp_end_ratio = self._suggest_from_spec(
            trial,
            'train__TEMP_END_RATIO',
            self.config.TRAIN_SEARCH_SPACE['TEMP_END_RATIO']
        )
        training_overrides['TEMP_END'] = round(training_overrides['TEMP_START'] * temp_end_ratio, 6)

        # 模型参数采样 (需要跳过条件采样的头数参数 + 消融模式下跳过权重网络参数)
        skip_params = {'nhead', 'weight_net_num_heads'}
        for name, spec in self.config.MODEL_SEARCH_SPACE.items():
            if name in skip_params:
                continue
            # 消融模式下跳过权重网络参数
            if skip_weight_net and name in self.config.WEIGHT_NET_PARAMS:
                continue
            model_overrides[name] = self._suggest_from_spec(trial, f'model__{name}', spec)

        # 条件采样: nhead (必须整除 d_model)
        d_model = model_overrides['d_model']
        model_overrides['nhead'] = self._sample_conditional_heads(
            trial,
            'model__nhead',
            self.config.MODEL_SEARCH_SPACE['nhead']['values'],
            d_model
        )

        # 条件采样: weight_net_num_heads (仅在非消融模式下)
        if not skip_weight_net:
            hidden_dim = model_overrides['weight_net_hidden_dim']
            model_overrides['weight_net_num_heads'] = self._sample_conditional_heads(
                trial,
                'model__weight_net_num_heads',
                self.config.MODEL_SEARCH_SPACE['weight_net_num_heads']['values'],
                hidden_dim
            )

        return training_overrides, model_overrides

    def _build_trial_overrides(self, trial, training_overrides, model_overrides):
        train_overrides = copy.deepcopy(training_overrides)
        model_ovr = copy.deepcopy(model_overrides)

        if self.config.USE_FAST_MODE:
            train_overrides.update(copy.deepcopy(self.config.FAST_MODE_TRAINING_OVERRIDES))

        if self.config.ENABLE_BATCH_SIZE_LR_SCALING:
            base_batch = float(self.config.BASE_BATCH_SIZE_FOR_LR_SCALING)
            lr_scale = float(train_overrides['BATCH_SIZE']) / base_batch
            train_overrides['LEARNING_RATE'] = float(train_overrides['LEARNING_RATE']) * lr_scale

        trial_dir = self.hpo_root_dir / f'trial_{trial.number:03d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        train_overrides['SAVE_DIR'] = trial_dir
        train_overrides['RUN_NAME'] = f'autoformer_optuna_trial_{trial.number:03d}'
        train_overrides['EXPERIMENT_LOG_CSV'] = self.hpo_root_dir / 'optuna_experiment_results.csv'

        return train_overrides, model_ovr, trial_dir

    @staticmethod
    def _paths_to_str(obj):
        """递归将字典/列表中所有 Path 对象转换为字符串"""
        if isinstance(obj, dict):
            return {k: OptunaSearchOptimizer._paths_to_str(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [OptunaSearchOptimizer._paths_to_str(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    def _build_search_space_signature(self) -> dict:
        """构建当前搜索空间的参数签名 (消融模式下排除权重网络参数)"""
        sig = {}
        skip_weight_net = self.config.should_skip_weight_net()

        for name, spec in self.config.TRAIN_SEARCH_SPACE.items():
            key = f'train__{name}'
            if spec['type'] == 'categorical':
                sig[key] = sorted(str(v) for v in spec['values'])
            else:
                sig[key] = {'low': spec['low'], 'high': spec['high']}

        for name, spec in self.config.MODEL_SEARCH_SPACE.items():
            if name in {'nhead', 'weight_net_num_heads'}:
                continue
            # 消融模式下跳过权重网络参数的签名
            if skip_weight_net and name in self.config.WEIGHT_NET_PARAMS:
                continue
            key = f'model__{name}'
            if spec['type'] == 'categorical':
                sig[key] = sorted(str(v) for v in spec['values'])
            else:
                sig[key] = {'low': spec['low'], 'high': spec['high']}

        return sig

    def _check_search_space_compatibility(self):
        """检查历史参数分布与当前搜索空间是否兼容"""
        completed_or_failed = [
            t for t in self.study.trials
            if t.params
        ]
        if not completed_or_failed:
            print('✅ [搜索空间检查] 当前 study 无历史 trial，跳过兼容性检查')
            return

        current_sig = self._build_search_space_signature()
        print(f'\n🔍 [搜索空间检查] 当前 study 已有 {len(self.study.trials)} 个历史 trial，开始兼容性检查...')

        conflicts = []
        from optuna.distributions import CategoricalDistribution
        for trial in completed_or_failed[:1]:
            for param_name, dist in trial.distributions.items():
                if param_name not in current_sig:
                    continue
                if isinstance(dist, CategoricalDistribution):
                    hist_choices = sorted(str(c) for c in dist.choices)
                    curr_choices = current_sig.get(param_name)
                    if isinstance(curr_choices, list) and hist_choices != curr_choices:
                        conflicts.append(
                            f'  ⚡ 参数 [{param_name}]\n'
                            f'     历史 choices : {hist_choices}\n'
                            f'     当前 choices : {curr_choices}'
                        )

        if conflicts:
            conflict_str = '\n'.join(conflicts)
            db_path = self.storage or '(内存)'
            raise RuntimeError(
                f'\n{"=" * 70}\n'
                f'❌ [搜索空间兼容性冲突] 当前搜索空间与历史 study 不兼容！\n'
                f'{"=" * 70}\n'
                f'{conflict_str}\n\n'
                f'解决方案：修改 STUDY_NAME 或删除旧数据库: {db_path}\n'
                f'{"=" * 70}'
            )

        print(f'✅ [搜索空间检查] 兼容性验证通过')

    def _extract_score(self, record):
        metric_name = self.config.OPTIMIZE_METRIC
        if metric_name not in record:
            raise KeyError(f'训练结果中不存在优化目标: {metric_name}')
        return float(record[metric_name])

    def _save_running_summary(self):
        running_csv = self.hpo_root_dir / 'optuna_summary_running.csv'
        if self.results:
            pd.DataFrame(self.results).to_csv(running_csv, index=False, encoding='utf-8-sig')

    def _save_summary(self, best_result):
        summary_csv = self.hpo_root_dir / 'optuna_summary.csv'
        summary_json = self.hpo_root_dir / 'best_trial.json'
        study_csv = self.hpo_root_dir / 'optuna_trials.csv'

        summary_df = pd.DataFrame(self.results)
        if not summary_df.empty:
            ascending = self.config.DIRECTION == 'minimize'
            summary_df = summary_df.sort_values(by=self.config.OPTIMIZE_METRIC, ascending=ascending, na_position='last')
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

        trials_df = self.study.trials_dataframe(attrs=('number', 'value', 'state', 'params', 'user_attrs'))
        trials_df.to_csv(study_csv, index=False, encoding='utf-8-sig')

        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(best_result, f, ensure_ascii=False, indent=4, default=str)

        print(f'\n📦 Optuna 汇总已保存: {summary_csv}')
        print(f'🧾 Study 明细已保存: {study_csv}')
        print(f'🏆 最佳试验已保存: {summary_json}')

    def _objective(self, trial):
        """Optuna 目标函数: 采样参数 → 训练 → 返回分数"""
        sampled_training, sampled_model = self._sample_parameters(trial)
        train_overrides, model_overrides, trial_dir = self._build_trial_overrides(
            trial, sampled_training, sampled_model
        )

        # 消融模式: 强制将 ablation_mode 传递给模型, 确保训练时一致
        model_overrides['ablation_mode'] = self.config.ABLATION_MODE

        trial.set_user_attr('trial_dir', str(trial_dir))
        trial.set_user_attr('training_overrides',
                            self._paths_to_str(copy.deepcopy(train_overrides)))
        trial.set_user_attr('model_overrides',
                            self._paths_to_str(copy.deepcopy(model_overrides)))
        trial.set_user_attr('storage', self.storage)
        trial.set_user_attr('model_type', 'autoformer')
        trial.set_user_attr('ablation_mode', self.config.ABLATION_MODE)

        print('\n' + '-' * 80)
        print(f'🚀 开始 Autoformer Optuna Trial {trial.number + 1}/{self.config.NUM_TRIALS}')
        print(f'   消融模式: {self.config.ABLATION_MODE}'
              f'{" (权重网络参数不参与HPO)" if self.config.should_skip_weight_net() else ""}')
        print('-' * 80)
        print('训练参数覆盖:')
        print(json.dumps(train_overrides, ensure_ascii=False, indent=2, default=str))
        print('模型参数覆盖 (Autoformer):')
        print(json.dumps(model_overrides, ensure_ascii=False, indent=2, default=str))

        try:
            record = train_main(
                training_overrides=train_overrides,
                model_overrides=model_overrides,
                reset_model_config=True,
                restore_after_run=True,
                trial=trial,
                trial_metric=self.config.TRIAL_REPORT_METRIC,
                prune_enabled=self.config.ENABLE_PRUNING
            )
            score = self._extract_score(record)
            trial.set_user_attr('status', 'success')
            trial.set_user_attr('score', score)
            trial.set_user_attr('best_epoch', record.get('best_epoch'))
            trial.set_user_attr('epochs_completed', record.get('epochs_completed'))
            trial.set_user_attr('record', record)

            trial_result = {
                'trial_index': trial.number,
                'trial_dir': str(trial_dir),
                'status': 'success',
                self.config.OPTIMIZE_METRIC: score,
                'final_test_rmse': record.get('final_test_rmse'),
                'final_test_mae': record.get('final_test_mae'),
                'final_test_mape': record.get('final_test_mape'),
                'final_test_pearson_r': record.get('final_test_pearson_r'),
                'final_test_r2': record.get('final_test_r2'),
                'best_epoch': record.get('best_epoch'),
                'epochs_completed': record.get('epochs_completed'),
                'training_overrides': json.dumps(
                    train_overrides, ensure_ascii=False, default=str),
                'model_overrides': json.dumps(
                    model_overrides, ensure_ascii=False, default=str)
            }
            self.results.append(trial_result)
            self._save_running_summary()
            print(f'✅ Trial {trial.number} 完成，'
                  f'{self.config.OPTIMIZE_METRIC}={score:.6f}')
            return score

        except HPOPruneTrial as exc:
            trial.set_user_attr('status', 'pruned')
            trial.set_user_attr('prune_reason', str(exc))
            trial_result = {
                'trial_index': trial.number,
                'trial_dir': str(trial_dir),
                'status': 'pruned',
                self.config.OPTIMIZE_METRIC: None,
                'final_test_rmse': None,
                'final_test_mae': None,
                'final_test_mape': None,
                'final_test_pearson_r': None,
                'final_test_r2': None,
                'best_epoch': None,
                'epochs_completed': None,
                'training_overrides': json.dumps(
                    train_overrides, ensure_ascii=False, default=str),
                'model_overrides': json.dumps(
                    model_overrides, ensure_ascii=False, default=str),
                'error': str(exc)
            }
            self.results.append(trial_result)
            self._save_running_summary()
            print(f'✂️ Trial {trial.number} 被剪枝: {exc}')
            raise TrialPruned(str(exc)) from exc

        except Exception as exc:
            trial.set_user_attr('status', 'failed')
            trial.set_user_attr('error', str(exc))
            trial_result = {
                'trial_index': trial.number,
                'trial_dir': str(trial_dir),
                'status': 'failed',
                self.config.OPTIMIZE_METRIC: None,
                'final_test_rmse': None,
                'final_test_mae': None,
                'final_test_mape': None,
                'final_test_pearson_r': None,
                'final_test_r2': None,
                'best_epoch': None,
                'epochs_completed': None,
                'training_overrides': json.dumps(
                    train_overrides, ensure_ascii=False, default=str),
                'model_overrides': json.dumps(
                    model_overrides, ensure_ascii=False, default=str),
                'error': str(exc)
            }
            self.results.append(trial_result)
            self._save_running_summary()
            print(f'❌ Trial {trial.number} 失败: {exc}')
            raise

    def run(self):
        """运行 Autoformer 超参数优化"""
        print('\n' + '=' * 80)
        print('开始 Autoformer 超参数优化 (Optuna + TPE + Pruning)')
        print('=' * 80)
        print(f'模型类型: Autoformer (Encoder-Decoder + Series Decomposition)')
        _ablation_desc = {
            'full': '全开 (情感×权重网络→主干)',
            'no_weight_net': '无权重网络 (情感×1→主干)',
            'no_sentiment': '全关 (仅环境特征→主干)'
        }
        print(f'消融模式: {self.config.ABLATION_MODE} ({_ablation_desc.get(self.config.ABLATION_MODE, "未知")})')
        if self.config.should_skip_weight_net():
            print(f'  → 权重网络相关参数不参与 HPO: {self.config.WEIGHT_NET_PARAMS}')
        print(f'HPO 输出目录: {self.hpo_root_dir}')
        print(f'Study 存储: {self.storage if self.storage else "内存模式"}')
        print(f'Trial 数量: {self.config.NUM_TRIALS}')
        print(f'优化目标: {self.config.OPTIMIZE_METRIC} ({self.config.DIRECTION})')
        print(f'Trial 上报指标: {self.config.TRIAL_REPORT_METRIC}')
        print(f'剪枝功能: {"开启" if self.config.ENABLE_PRUNING else "关闭"}')
        print(f'快速搜索模式: {"开启" if self.config.USE_FAST_MODE else "关闭"}')
        print('=' * 80 + '\n')

        sampler = self._build_sampler()
        pruner = self._build_pruner()

        self.study = optuna.create_study(
            study_name=self.config.STUDY_NAME,
            storage=self.storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.config.DIRECTION,
            load_if_exists=self.config.LOAD_IF_EXISTS
        )

        self._check_search_space_compatibility()

        self.study.optimize(
            self._objective,
            n_trials=self.config.NUM_TRIALS,
            catch=(Exception,)
        )

        completed_trials = [
            trial for trial in self.study.trials if trial.value is not None
        ]
        if not completed_trials:
            raise RuntimeError(
                '所有 Trial 均失败或被剪枝，未找到可用超参数组合'
            )

        best_trial = self.study.best_trial
        best_result = {
            'model_type': 'autoformer',
            'trial_index': best_trial.number,
            'trial_dir': best_trial.user_attrs.get('trial_dir'),
            'status': best_trial.user_attrs.get('status', 'success'),
            self.config.OPTIMIZE_METRIC: best_trial.value,
            'training_overrides': best_trial.user_attrs.get(
                'training_overrides'),
            'model_overrides': best_trial.user_attrs.get('model_overrides'),
            'best_epoch': best_trial.user_attrs.get('best_epoch'),
            'epochs_completed': best_trial.user_attrs.get('epochs_completed'),
            'full_record': best_trial.user_attrs.get('record')
        }

        self._save_summary(best_result)

        print('\n' + '=' * 80)
        print('Autoformer 超参数优化完成')
        print(f'最佳 Trial: {best_result["trial_index"]}')
        print(f'最佳分数 {self.config.OPTIMIZE_METRIC}: '
              f'{best_result[self.config.OPTIMIZE_METRIC]}')
        print(f'最佳 Trial 目录: {best_result["trial_dir"]}')
        print('=' * 80)
        return best_result


if __name__ == '__main__':
    optimizer = OptunaSearchOptimizer()
    optimizer.run()
