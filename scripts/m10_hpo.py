# -*- coding: utf-8 -*-
"""
M10 自动超参数优化脚本
用途:
1. 基于随机搜索自动采样超参数
2. 调用 m10_train_2.main() 执行单次训练
3. 汇总每个 trial 的结果并保存最佳参数

说明:
- 当前采用随机搜索(Random Search)，不依赖第三方 AutoML 库
- 后续如需升级，可在此基础上平滑迁移到 Optuna
"""

import copy
import json
import math
import random
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

from m10_train_2 import main as train_main
from config.m10_config import TrainingConfig as BaseTrainingConfig
from config.m9_config import ModelConfig as BaseModelConfig


class HPOConfig:
    """自动超参数优化配置"""

    NUM_TRIALS = 12
    RANDOM_SEED = 20260306
    OPTIMIZE_METRIC = 'best_val_ccc_loss'
    DIRECTION = 'minimize'

    # 是否使用较短训练进行粗搜索
    USE_FAST_MODE = True
    FAST_MODE_TRAINING_OVERRIDES = {
        'NUM_EPOCHS': 80,
        'PATIENCE': 12,
        'SAVE_INTERVAL': 999999,
        'RUN_NAME': 'hpo_fast_mode'
    }

    # 随机搜索空间
    SEARCH_SPACE = {
        'training': {
            'LEARNING_RATE': {'type': 'loguniform', 'low': 1e-4, 'high': 3e-3},
            'WEIGHT_DECAY': {'type': 'loguniform', 'low': 1e-6, 'high': 1e-3},
            'BATCH_SIZE': {'type': 'categorical', 'values': [16, 32, 64]},
            'PEAK_SIGMA': {'type': 'categorical', 'values': [1.0, 1.5, 2.0, 2.5]},
            'PEAK_PENALTY_WEIGHT': {'type': 'categorical', 'values': [2.0, 3.0, 5.0, 8.0]},
            'LOSS_WEIGHT_PEAK_MSE': {'type': 'categorical', 'values': [0.05, 0.1, 0.2, 0.5]},
            'TEMP_START': {'type': 'categorical', 'values': [1.0, 2.0, 3.0]},
            'TEMP_END': {'type': 'categorical', 'values': [0.3, 0.5, 0.8, 1.0]},
            'MAX_GRAD_NORM': {'type': 'categorical', 'values': [0.5, 1.0, 2.0]},
        },
        'model': {
            'dropout': {'type': 'categorical', 'values': [0.05, 0.1, 0.2, 0.3]},
            'weight_net_dropout': {'type': 'categorical', 'values': [0.05, 0.1, 0.2, 0.3]},
            'd_model': {'type': 'categorical', 'values': [32, 64, 128]},
            'num_layers': {'type': 'categorical', 'values': [1, 2, 3]},
            'weight_net_hidden_dim': {'type': 'categorical', 'values': [16, 32, 64]},
            'weight_net_num_layers': {'type': 'categorical', 'values': [1, 2]},
            'activation': {'type': 'categorical', 'values': ['relu', 'gelu', 'silu']},
        }
    }

    @classmethod
    def get_hpo_root_dir(cls) -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        scenic = getattr(BaseTrainingConfig, 'scenic_name', 'default')
        return BaseTrainingConfig.BASE_DIR / 'data' / 'output' / 'hpo' / f'm10_hpo_{scenic}_{timestamp}'


class RandomSearchOptimizer:
    def __init__(self, config=HPOConfig):
        self.config = config
        self.rng = random.Random(config.RANDOM_SEED)
        self.hpo_root_dir = config.get_hpo_root_dir()
        self.hpo_root_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def _sample_from_spec(self, spec):
        spec_type = spec['type']
        if spec_type == 'categorical':
            return self.rng.choice(spec['values'])
        if spec_type == 'uniform':
            low = spec['low']
            high = spec['high']
            return self.rng.uniform(low, high)
        if spec_type == 'int':
            low = spec['low']
            high = spec['high']
            step = spec.get('step', 1)
            return self.rng.randrange(low, high + 1, step)
        if spec_type == 'loguniform':
            low = math.log(spec['low'])
            high = math.log(spec['high'])
            return math.exp(self.rng.uniform(low, high))
        raise ValueError(f"不支持的采样类型: {spec_type}")

    def _sample_parameters(self):
        training_overrides = {}
        model_overrides = {}

        for name, spec in self.config.SEARCH_SPACE['training'].items():
            training_overrides[name] = self._sample_from_spec(spec)

        for name, spec in self.config.SEARCH_SPACE['model'].items():
            model_overrides[name] = self._sample_from_spec(spec)

        # 约束修正：保证 Transformer 相关维度合法
        d_model = model_overrides['d_model']
        valid_nheads = [head for head in [1, 2, 4, 8] if head <= d_model and d_model % head == 0]
        model_overrides['nhead'] = self.rng.choice(valid_nheads)

        hidden_dim = model_overrides['weight_net_hidden_dim']
        valid_weight_heads = [head for head in [1, 2, 4, 8] if head <= hidden_dim and hidden_dim % head == 0]
        model_overrides['weight_net_num_heads'] = self.rng.choice(valid_weight_heads)

        feedforward_candidates = [2 * d_model, 4 * d_model]
        model_overrides['dim_feedforward'] = self.rng.choice(feedforward_candidates)

        # 温度退火的基本合法性
        if training_overrides['TEMP_END'] > training_overrides['TEMP_START']:
            training_overrides['TEMP_END'] = training_overrides['TEMP_START']

        return training_overrides, model_overrides

    def _build_trial_overrides(self, trial_index, training_overrides, model_overrides):
        train_overrides = copy.deepcopy(training_overrides)
        model_overrides = copy.deepcopy(model_overrides)

        if self.config.USE_FAST_MODE:
            train_overrides.update(copy.deepcopy(self.config.FAST_MODE_TRAINING_OVERRIDES))

        trial_dir = self.hpo_root_dir / f'trial_{trial_index:03d}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        train_overrides['SAVE_DIR'] = trial_dir
        train_overrides['RUN_NAME'] = f'hpo_trial_{trial_index:03d}'
        train_overrides['EXPERIMENT_LOG_CSV'] = self.hpo_root_dir / 'hpo_experiment_results.csv'

        return train_overrides, model_overrides, trial_dir

    def _extract_score(self, record):
        metric_name = self.config.OPTIMIZE_METRIC
        if metric_name not in record:
            raise KeyError(f"训练结果中不存在优化目标: {metric_name}")
        return record[metric_name]

    def _is_better(self, current_score, best_score):
        if best_score is None:
            return True
        if self.config.DIRECTION == 'minimize':
            return current_score < best_score
        return current_score > best_score

    def _save_summary(self, best_result):
        summary_csv = self.hpo_root_dir / 'hpo_summary.csv'
        summary_json = self.hpo_root_dir / 'best_trial.json'

        summary_df = pd.DataFrame(self.results)
        if not summary_df.empty:
            ascending = self.config.DIRECTION == 'minimize'
            summary_df = summary_df.sort_values(by=self.config.OPTIMIZE_METRIC, ascending=ascending)
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(best_result, f, ensure_ascii=False, indent=4)

        print(f"\n📦 HPO 汇总已保存: {summary_csv}")
        print(f"🏆 最佳试验已保存: {summary_json}")

    def run(self):
        print("\n" + "=" * 80)
        print("开始自动超参数优化 (Random Search)")
        print("=" * 80)
        print(f"HPO 输出目录: {self.hpo_root_dir}")
        print(f"Trial 数量: {self.config.NUM_TRIALS}")
        print(f"优化目标: {self.config.OPTIMIZE_METRIC} ({self.config.DIRECTION})")
        print(f"快速搜索模式: {'开启' if self.config.USE_FAST_MODE else '关闭'}")
        print("=" * 80 + "\n")

        best_result = None
        best_score = None

        for trial_index in range(1, self.config.NUM_TRIALS + 1):
            print("\n" + "-" * 80)
            print(f"🚀 开始 Trial {trial_index}/{self.config.NUM_TRIALS}")
            print("-" * 80)

            sampled_training, sampled_model = self._sample_parameters()
            train_overrides, model_overrides, trial_dir = self._build_trial_overrides(
                trial_index, sampled_training, sampled_model
            )

            print("训练参数覆盖:")
            print(json.dumps(train_overrides, ensure_ascii=False, indent=2, default=str))
            print("模型参数覆盖:")
            print(json.dumps(model_overrides, ensure_ascii=False, indent=2, default=str))

            try:
                record = train_main(
                    training_overrides=train_overrides,
                    model_overrides=model_overrides,
                    reset_model_config=True,
                    restore_after_run=True
                )
                score = self._extract_score(record)
                trial_result = {
                    'trial_index': trial_index,
                    'trial_dir': str(trial_dir),
                    'status': 'success',
                    self.config.OPTIMIZE_METRIC: score,
                    'final_test_rmse': record.get('final_test_rmse'),
                    'final_test_pearson_r': record.get('final_test_pearson_r'),
                    'best_epoch': record.get('best_epoch'),
                    'epochs_completed': record.get('epochs_completed'),
                    'training_overrides': json.dumps(train_overrides, ensure_ascii=False, default=str),
                    'model_overrides': json.dumps(model_overrides, ensure_ascii=False, default=str)
                }
                self.results.append(trial_result)

                print(f"✅ Trial {trial_index} 完成，{self.config.OPTIMIZE_METRIC}={score:.6f}")

                if self._is_better(score, best_score):
                    best_score = score
                    best_result = copy.deepcopy(trial_result)
                    best_result['full_record'] = record
                    print(f"🏆 Trial {trial_index} 成为当前最优")
            except Exception as exc:
                trial_result = {
                    'trial_index': trial_index,
                    'trial_dir': str(trial_dir),
                    'status': 'failed',
                    self.config.OPTIMIZE_METRIC: None,
                    'final_test_rmse': None,
                    'final_test_pearson_r': None,
                    'best_epoch': None,
                    'epochs_completed': None,
                    'training_overrides': json.dumps(train_overrides, ensure_ascii=False, default=str),
                    'model_overrides': json.dumps(model_overrides, ensure_ascii=False, default=str),
                    'error': str(exc)
                }
                self.results.append(trial_result)
                print(f"❌ Trial {trial_index} 失败: {exc}")

            interim_csv = self.hpo_root_dir / 'hpo_summary_running.csv'
            pd.DataFrame(self.results).to_csv(interim_csv, index=False, encoding='utf-8-sig')

        if best_result is None:
            raise RuntimeError('所有 Trial 均失败，未找到可用超参数组合')

        self._save_summary(best_result)

        print("\n" + "=" * 80)
        print("自动超参数优化完成")
        print(f"最佳 Trial: {best_result['trial_index']}")
        print(f"最佳分数 {self.config.OPTIMIZE_METRIC}: {best_result[self.config.OPTIMIZE_METRIC]}")
        print(f"最佳 Trial 目录: {best_result['trial_dir']}")
        print("=" * 80)
        return best_result


if __name__ == '__main__':
    optimizer = RandomSearchOptimizer()
    optimizer.run()
