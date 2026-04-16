# -*- coding: utf-8 -*-
"""
模块 4：基线预训练模型超参数优化脚本 (Optuna Version)

用途:
1. 基于 Optuna + TPE 自动采样超参数
2. 调用 m4_train_baseline_2.main() 执行单次训练
3. 在每个 epoch 上报中间指标，支持自动剪枝 (MedianPruner)
4. 汇总每个 trial 的结果并保存最佳参数

搜索空间包含:
- 学习率 (LEARNING_RATE)
- 权重衰减 (WEIGHT_DECAY)
- 批次大小 (BATCH_SIZE)
- Dropout 比率 (HIDDEN_DROPOUT)
- MSE 损失权重 (MSE_WEIGHT)
- Pearson 损失权重 (PEARSON_WEIGHT)
- 预热比例 (WARMUP_RATIO)
- 最大梯度范数 (MAX_GRAD_NORM)

运行方式:
    python scripts/m4_hpo_optuna.py
"""

import copy
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import torch

# 路径设置
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

from m4_train_baseline_2 import main as train_main
from m4_train_baseline_2 import _OptunaPrunedException
from config.m4_config import BaselineConfig


# ==================== HPO 配置 ====================
class M4OptunaHPOConfig:
    """M4 基线模型 Optuna 超参数优化配置"""

    # ==================== 基础运行设置 ====================
    STUDY_NAME = 'm4_baseline_hpo_study'
    STUDY_STORAGE_FILENAME = 'm4_baseline_hpo_study.db'
    USE_PERSISTENT_STORAGE = True       # True: 使用 SQLite 持久化存储, 支持断点续跑
    LOAD_IF_EXISTS = True               # True: 加载已有 study 继续实验

    NUM_TRIALS = 50                     # 总搜索试验次数
    RANDOM_SEED = 42
    SAMPLER = 'tpe'                     # 采样器: 'tpe' 或 'random'
    PRUNER = 'median'                   # 剪枝器: 'median', 'percentile', 'nop'

    # ==================== 优化目标 ====================
    OPTIMIZE_METRIC = 'best_val_pearsonr'   # 优化目标指标 (来自 train 返回值)
    DIRECTION = 'maximize'                   # 最大化 Pearson 相关系数
    TRIAL_REPORT_METRIC = 'avg_pearsonr'     # 每个 epoch 上报给 Optuna 的中间指标
    ENABLE_PRUNING = True                    # 是否启用 Optuna 中间剪枝

    # ==================== 输出路径 ====================
    OUTPUT_ROOT = BaselineConfig.PROJECT_ROOT / 'data' / 'output' / 'hpo'
    OUTPUT_NAME_PREFIX = 'm4_baseline_hpo'
    OUTPUT_USE_TIMESTAMP = True

    # ==================== 快速搜索模式 ====================
    # HPO 阶段通常不需要跑满所有 epoch，可以减少 epoch 和耐心值加速搜索
    USE_FAST_MODE = True
    FAST_MODE_TRAINING_OVERRIDES = {
        'NUM_EPOCHS': 30,               # HPO 快速模式下最大训练轮数
        'EARLY_STOPPING_PATIENCE': 8,   # HPO 快速模式下早停耐心值
        'SAVE_CHECKPOINT_EVERY_EPOCH': False,  # HPO 不需要每轮保存
        'RESUME_TRAINING': False,        # HPO 每个 trial 从头训练
        'FORCE_RESTART': True,           # 强制从头开始
    }

    # ==================== 搜索空间定义 ====================
    SEARCH_SPACE = {
        # 学习率（对数均匀分布）
        'LEARNING_RATE': {
            'type': 'loguniform',
            'low': 1e-6,
            'high': 5e-4,
        },
        # 权重衰减（对数均匀分布）
        'WEIGHT_DECAY': {
            'type': 'loguniform',
            'low': 1e-5,
            'high': 0.1,
        },
        # 批次大小
        'BATCH_SIZE': {
            'type': 'categorical',
            'values': [8, 16, 32],
        },
        # Dropout 比率
        'HIDDEN_DROPOUT': {
            'type': 'categorical',
            'values': [0.05, 0.1, 0.15, 0.2, 0.3],
        },
        # MSE 损失权重
        'MSE_WEIGHT': {
            'type': 'uniform',
            'low': 0.1,
            'high': 5.0,
        },
        # Pearson 损失权重
        'PEARSON_WEIGHT': {
            'type': 'uniform',
            'low': 0.1,
            'high': 5.0,
        },
        # 预热比例
        'WARMUP_RATIO': {
            'type': 'categorical',
            'values': [0.0, 0.05, 0.1, 0.15, 0.2],
        },
        # 最大梯度范数
        'MAX_GRAD_NORM': {
            'type': 'categorical',
            'values': [0.5, 1.0, 2.0, 5.0],
        },
    }

    @classmethod
    def get_storage_url(cls) -> Optional[str]:
        """获取 Optuna Study 的存储 URL"""
        if not cls.USE_PERSISTENT_STORAGE:
            return None
        cls.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(cls.OUTPUT_ROOT / cls.STUDY_STORAGE_FILENAME).resolve().as_posix()}"

    @classmethod
    def get_hpo_root_dir(cls) -> Path:
        """获取 HPO 实验输出根目录"""
        suffix = datetime.now().strftime('%Y%m%d_%H%M%S') if cls.OUTPUT_USE_TIMESTAMP else 'manual'
        return cls.OUTPUT_ROOT / f'{cls.OUTPUT_NAME_PREFIX}_{suffix}'


# ==================== Optuna 搜索优化器 ====================
class M4OptunaSearchOptimizer:
    """M4 基线模型 Optuna 超参数搜索优化器"""

    def __init__(self, config=M4OptunaHPOConfig):
        self.config = config
        self.hpo_root_dir = config.get_hpo_root_dir()
        self.hpo_root_dir.mkdir(parents=True, exist_ok=True)
        self.storage = config.get_storage_url()
        self.results = []
        self.study = None

    def _build_sampler(self):
        """构建 Optuna 采样器"""
        sampler_name = str(self.config.SAMPLER).lower()
        if sampler_name == 'tpe':
            return optuna.samplers.TPESampler(seed=self.config.RANDOM_SEED)
        if sampler_name == 'random':
            return optuna.samplers.RandomSampler(seed=self.config.RANDOM_SEED)
        raise ValueError(f'不支持的采样器: {self.config.SAMPLER}')

    def _build_pruner(self):
        """构建 Optuna 剪枝器"""
        if not self.config.ENABLE_PRUNING:
            return optuna.pruners.NopPruner()
        pruner_name = str(self.config.PRUNER).lower()
        if pruner_name == 'median':
            return optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            )
        if pruner_name == 'percentile':
            return optuna.pruners.PercentilePruner(
                50.0,
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            )
        if pruner_name == 'nop':
            return optuna.pruners.NopPruner()
        raise ValueError(f'不支持的剪枝器: {self.config.PRUNER}')

    def _suggest_from_spec(self, trial, name: str, spec: Dict) -> Any:
        """从搜索空间规格中采样参数"""
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

    def _sample_parameters(self, trial) -> Dict:
        """从搜索空间中采样所有参数"""
        training_overrides = {}
        for name, spec in self.config.SEARCH_SPACE.items():
            training_overrides[name] = self._suggest_from_spec(trial, name, spec)
        return training_overrides

    def _build_trial_overrides(self, trial, training_overrides: Dict):
        """构建单次 trial 的完整覆盖配置"""
        train_overrides = copy.deepcopy(training_overrides)

        # 应用快速模式覆盖
        if self.config.USE_FAST_MODE:
            train_overrides.update(copy.deepcopy(self.config.FAST_MODE_TRAINING_OVERRIDES))

        # 为每个 trial 设置独立的输出目录
        trial_dir = self.hpo_root_dir / f'trial_{trial.number:03d}'
        trial_dir.mkdir(parents=True, exist_ok=True)

        train_overrides['OUTPUT_DIR'] = trial_dir
        train_overrides['CHECKPOINT_DIR'] = trial_dir / 'checkpoints'
        train_overrides['FIGURES_DIR'] = trial_dir / 'figures'
        train_overrides['METRICS_FILE'] = trial_dir / 'metrics.json'
        train_overrides['LOG_FILE'] = trial_dir / 'training.log'

        return train_overrides, trial_dir

    @staticmethod
    def _paths_to_str(obj):
        """递归将字典中 Path 对象转换为字符串（用于 JSON 序列化）"""
        if isinstance(obj, dict):
            return {k: M4OptunaSearchOptimizer._paths_to_str(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [M4OptunaSearchOptimizer._paths_to_str(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    @staticmethod
    def _cleanup_gpu():
        """Trial 结束后强制清理 GPU 缓存，防止跨 trial 内存泄漏"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def _extract_score(self, record: Dict) -> float:
        """从训练结果中提取优化目标分数"""
        metric_name = self.config.OPTIMIZE_METRIC
        if metric_name not in record:
            raise KeyError(f'训练结果中不存在优化目标: {metric_name}，'
                           f'可用键: {list(record.keys())}')
        return float(record[metric_name])

    def _save_running_summary(self):
        """保存实时汇总（每完成一个 trial 即更新）"""
        running_csv = self.hpo_root_dir / 'hpo_summary_running.csv'
        if self.results:
            pd.DataFrame(self.results).to_csv(running_csv, index=False, encoding='utf-8-sig')

    def _save_summary(self, best_result: Dict):
        """保存最终汇总"""
        summary_csv = self.hpo_root_dir / 'hpo_summary.csv'
        summary_json = self.hpo_root_dir / 'best_trial.json'
        study_csv = self.hpo_root_dir / 'optuna_trials.csv'

        # 保存所有 trial 汇总
        summary_df = pd.DataFrame(self.results)
        if not summary_df.empty:
            ascending = self.config.DIRECTION == 'minimize'
            summary_df = summary_df.sort_values(
                by=self.config.OPTIMIZE_METRIC,
                ascending=ascending,
                na_position='last'
            )
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

        # 保存 Optuna study 明细
        trials_df = self.study.trials_dataframe(
            attrs=('number', 'value', 'state', 'params', 'user_attrs')
        )
        trials_df.to_csv(study_csv, index=False, encoding='utf-8-sig')

        # 保存最佳 trial 参数
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(best_result, f, ensure_ascii=False, indent=4, default=str)

        print(f'\n📦 HPO 汇总已保存: {summary_csv}')
        print(f'🧾 Study 明细已保存: {study_csv}')
        print(f'🏆 最佳试验已保存: {summary_json}')

    def _objective(self, trial) -> float:
        """Optuna 优化目标函数"""
        # 采样超参数
        sampled_params = self._sample_parameters(trial)
        train_overrides, trial_dir = self._build_trial_overrides(trial, sampled_params)

        # 记录 trial 属性
        trial.set_user_attr('trial_dir', str(trial_dir))
        trial.set_user_attr(
            'training_overrides',
            self._paths_to_str(copy.deepcopy(train_overrides))
        )

        print('\n' + '-' * 80)
        print(f'🚀 开始 Optuna Trial {trial.number + 1}/{self.config.NUM_TRIALS}')
        print('-' * 80)
        print('采样的超参数:')
        for k, v in sampled_params.items():
            print(f'  {k}: {v}')
        print()

        try:
            record = train_main(
                training_overrides=train_overrides,
                restore_after_run=True,
                trial=trial,
                trial_metric=self.config.TRIAL_REPORT_METRIC,
                prune_enabled=self.config.ENABLE_PRUNING
            )

            score = self._extract_score(record)

            # 记录 trial 结果
            trial.set_user_attr('status', 'success')
            trial.set_user_attr('score', score)
            trial.set_user_attr('best_epoch', record.get('best_epoch'))
            trial.set_user_attr('epochs_completed', record.get('epochs_completed'))

            trial_result = {
                'trial_index': trial.number,
                'trial_dir': str(trial_dir),
                'status': 'success',
                self.config.OPTIMIZE_METRIC: score,
                'test_avg_pearsonr': record.get('avg_pearsonr'),
                'test_avg_rmse': record.get('avg_rmse'),
                'test_avg_mae': record.get('avg_mae'),
                'test_avg_accuracy': record.get('avg_accuracy'),
                'best_epoch': record.get('best_epoch'),
                'epochs_completed': record.get('epochs_completed'),
            }
            # 添加各维度指标
            for dim in BaselineConfig.DIMENSIONS:
                trial_result[f'test_{dim}_pearsonr'] = record.get(f'{dim}_pearsonr')
                trial_result[f'test_{dim}_accuracy'] = record.get(f'{dim}_accuracy')

            # 添加采样的超参数
            for k, v in sampled_params.items():
                trial_result[f'param_{k}'] = v

            self.results.append(trial_result)
            self._save_running_summary()
            print(f'✅ Trial {trial.number} 完成，'
                  f'{self.config.OPTIMIZE_METRIC}={score:.4f}')
            return score

        except _OptunaPrunedException as exc:
            trial.set_user_attr('status', 'pruned')
            trial.set_user_attr('prune_reason', str(exc))

            trial_result = {
                'trial_index': trial.number,
                'trial_dir': str(trial_dir),
                'status': 'pruned',
                self.config.OPTIMIZE_METRIC: None,
                'test_avg_pearsonr': None,
                'test_avg_rmse': None,
                'test_avg_mae': None,
                'test_avg_accuracy': None,
                'best_epoch': None,
                'epochs_completed': None,
                'error': str(exc),
            }
            for k, v in sampled_params.items():
                trial_result[f'param_{k}'] = v

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
                'test_avg_pearsonr': None,
                'test_avg_rmse': None,
                'test_avg_mae': None,
                'test_avg_accuracy': None,
                'best_epoch': None,
                'epochs_completed': None,
                'error': str(exc),
            }
            for k, v in sampled_params.items():
                trial_result[f'param_{k}'] = v

            self.results.append(trial_result)
            self._save_running_summary()
            print(f'❌ Trial {trial.number} 失败: {exc}')
            raise

        finally:
            # 每个 trial 结束后（无论成功/剪枝/失败）强制清理 GPU 内存
            self._cleanup_gpu()
            print(f'🧹 Trial {trial.number} GPU 内存已清理')

    def run(self) -> Dict:
        """运行 HPO 优化流程"""
        print('\n' + '=' * 80)
        print('模块 4：基线预训练模型超参数优化 (Optuna + TPE)')
        print('=' * 80)
        print(f'HPO 输出目录: {self.hpo_root_dir}')
        print(f'Study 存储: {self.storage if self.storage else "内存模式"}')
        print(f'Trial 数量: {self.config.NUM_TRIALS}')
        print(f'优化目标: {self.config.OPTIMIZE_METRIC} ({self.config.DIRECTION})')
        print(f'Trial 上报指标: {self.config.TRIAL_REPORT_METRIC}')
        print(f'剪枝功能: {"开启" if self.config.ENABLE_PRUNING else "关闭"}')
        print(f'快速搜索模式: {"开启" if self.config.USE_FAST_MODE else "关闭"}')
        print(f'\n搜索空间:')
        for name, spec in self.config.SEARCH_SPACE.items():
            if spec['type'] == 'categorical':
                print(f'  {name}: {spec["values"]}')
            else:
                print(f'  {name}: [{spec["low"]}, {spec["high"]}] ({spec["type"]})')
        print('=' * 80 + '\n')

        # 构建采样器和剪枝器
        sampler = self._build_sampler()
        pruner = self._build_pruner()

        # 创建或加载 Optuna Study
        self.study = optuna.create_study(
            study_name=self.config.STUDY_NAME,
            storage=self.storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.config.DIRECTION,
            load_if_exists=self.config.LOAD_IF_EXISTS
        )

        # 开始优化
        self.study.optimize(
            self._objective,
            n_trials=self.config.NUM_TRIALS,
            catch=(Exception,)
        )

        # 汇总结果
        completed_trials = [t for t in self.study.trials if t.value is not None]
        if not completed_trials:
            raise RuntimeError('所有 Trial 均失败或被剪枝，未找到可用超参数组合')

        best_trial = self.study.best_trial
        best_result = {
            'trial_index': best_trial.number,
            'trial_dir': best_trial.user_attrs.get('trial_dir'),
            'status': best_trial.user_attrs.get('status', 'success'),
            self.config.OPTIMIZE_METRIC: best_trial.value,
            'training_overrides': best_trial.user_attrs.get('training_overrides'),
            'best_epoch': best_trial.user_attrs.get('best_epoch'),
            'epochs_completed': best_trial.user_attrs.get('epochs_completed'),
            'best_params': dict(best_trial.params),
        }

        self._save_summary(best_result)

        # 打印最终结果
        print('\n' + '=' * 80)
        print('超参数优化完成')
        print('=' * 80)
        print(f'总 Trial 数: {len(self.study.trials)}')
        print(f'成功 Trial 数: {len(completed_trials)}')
        print(f'最佳 Trial: #{best_result["trial_index"]}')
        print(f'最佳 {self.config.OPTIMIZE_METRIC}: '
              f'{best_result[self.config.OPTIMIZE_METRIC]:.4f}')
        print(f'\n最佳超参数:')
        for k, v in best_trial.params.items():
            print(f'  {k}: {v}')
        print(f'\n最佳 Trial 目录: {best_result["trial_dir"]}')
        print(f'HPO 汇总目录: {self.hpo_root_dir}')
        print('=' * 80)

        return best_result


if __name__ == '__main__':
    optimizer = M4OptunaSearchOptimizer()
    optimizer.run()
