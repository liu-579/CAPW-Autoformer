# -*- coding: utf-8 -*-
"""
M10 从 HPO 结果中加载最优超参，执行全量训练脚本
用途:
  - 读取指定 trial 的 best_trial.json 或 trial_{N} 目录下的参数
  - 自动提取 training_overrides / model_overrides
  - 去除 HPO 专用的快速搜索覆盖（NUM_EPOCHS=150 等），恢复全量训练配置
  - 调用 m10_train_2.main() 执行全量训练

使用方法:
  1. 修改下方 ===================== 手动配置区 ===================== 中的路径
  2. 直接运行本脚本：python scripts/m10_train_from_hpo.py

切换 trial 只需修改一行：
  TRIAL_SOURCE = ...  （指向任意 best_trial.json 或 trial 目录）
"""

import json
import sys
import copy
from pathlib import Path
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from m10_train_2 import main as train_main

# ===================== 手动配置区（只需改这里）=====================

# --- [1] 指定来源 trial 路径 ---
# 可以是：
#   1. best_trial.json 文件路径     → 读取整个 HPO 运行的最优 trial
#   2. 某个 trial_NNN 目录路径      → 读取该 trial 的参数（目录内需含 best_trial.json 或可从 HPO 汇总读取）
# 示例（切换 trial 只改这一行）：
a=2
if a==1:
    TRIAL_SOURCE = (
        PROJECT_ROOT
        / 'data/output/hpo/m10_optuna_7day_eedscy_20260307_234015/best_trial.json'
    )
if a==2:
    TRIAL_SOURCE = (
         PROJECT_ROOT
        / 'data/output/hpo/m10_optuna_7day_eedscy_20260309_180350/optuna_summary.csv'
    )

    # --- [2] 指定要从 summary.csv 中选取哪个 trial（仅当 TRIAL_SOURCE 指向 csv 时有效）---
    #   填写 trial 编号（整数），例如 39 表示 trial_039
    SELECT_TRIAL_INDEX = 64

# --- [3] 全量训练覆盖参数（在 HPO 快速模式基础上恢复）---
FULL_TRAIN_OVERRIDES = {
    'NUM_EPOCHS':    1000,    # 恢复完整轮次（HPO 快速模式为 150）
    'PATIENCE':      50,      # 恢复完整早停（HPO 快速模式为 20）
    'SAVE_INTERVAL': 5,       # 恢复正常保存间隔（HPO 快速模式为 999999）
}

# --- [4] 输出目录（留空则自动生成带时间戳的目录）---
OUTPUT_SAVE_DIR = ''   # 例如: 'data/output/h_out_eedscy/m10_trial046_full'

# ===================== 手动配置区 END =====================


# ---- HPO 快速搜索模式注入的 key，全量训练需清理/替换 ----
_HPO_INTERNAL_KEYS = {'SAVE_DIR', 'RUN_NAME', 'EXPERIMENT_LOG_CSV',
                      'NUM_EPOCHS', 'PATIENCE', 'SAVE_INTERVAL'}


def _load_from_best_trial_json(path: Path) -> dict:
    """从 best_trial.json 读取参数"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    training_overrides = data.get('training_overrides', {})
    model_overrides = data.get('model_overrides', {})
    trial_index = data.get('trial_index', '?')
    trial_dir = data.get('trial_dir', '?')
    return {
        'trial_index': trial_index,
        'trial_dir': trial_dir,
        'training_overrides': training_overrides,
        'model_overrides': model_overrides,
        'source_file': str(path),
    }


def _load_from_summary_csv(path: Path, trial_index: int) -> dict:
    """从 optuna_summary.csv 读取指定 trial 的参数"""
    import pandas as pd
    df = pd.read_csv(path, encoding='utf-8-sig')
    row = df[df['trial_index'] == trial_index]
    if row.empty:
        available = df['trial_index'].tolist()
        raise ValueError(
            f'在 {path} 中未找到 trial_index={trial_index}\n'
            f'可用 trial 编号: {available}'
        )
    row = row.iloc[0]
    training_overrides = json.loads(row['training_overrides'])
    model_overrides = json.loads(row['model_overrides'])
    return {
        'trial_index': trial_index,
        'trial_dir': row.get('trial_dir', '?'),
        'training_overrides': training_overrides,
        'model_overrides': model_overrides,
        'source_file': str(path),
    }


def load_trial_params(source: Path, select_trial_index: int) -> dict:
    """根据文件类型自动分发加载逻辑"""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f'TRIAL_SOURCE 路径不存在: {source}')

    if source.is_dir():
        # 如果给的是 trial 目录，尝试找同级的 best_trial.json 或 optuna_summary.csv
        candidate_best = source.parent / 'best_trial.json'
        candidate_summary = source.parent / 'optuna_summary.csv'
        # 从目录名提取 trial 编号
        dir_name = source.name  # 例如 trial_046
        if dir_name.startswith('trial_'):
            try:
                idx = int(dir_name.split('_')[1])
            except Exception:
                idx = select_trial_index
        else:
            idx = select_trial_index

        if candidate_summary.exists():
            print(f'📂 检测到 trial 目录，自动从同级 optuna_summary.csv 加载 trial_{idx:03d}')
            return _load_from_summary_csv(candidate_summary, idx)
        elif candidate_best.exists():
            print(f'📂 检测到 trial 目录，自动从同级 best_trial.json 加载')
            return _load_from_best_trial_json(candidate_best)
        else:
            raise FileNotFoundError(
                f'在 {source.parent} 中未找到 best_trial.json 或 optuna_summary.csv'
            )

    suffix = source.suffix.lower()
    if suffix == '.json':
        return _load_from_best_trial_json(source)
    elif suffix == '.csv':
        return _load_from_summary_csv(source, select_trial_index)
    else:
        raise ValueError(f'不支持的文件类型: {suffix}，仅支持 .json 或 .csv')


def build_full_train_overrides(
    loaded: dict,
    full_train_overrides: dict,
    output_save_dir: str,
) -> tuple[dict, dict]:
    """
    清理 HPO 内部 key，合并全量训练配置，确定最终 SAVE_DIR / RUN_NAME
    """
    training = copy.deepcopy(loaded['training_overrides'])
    model = copy.deepcopy(loaded['model_overrides'])
    trial_index = loaded['trial_index']

    # 1. 清理 HPO 快速模式注入的内部 key
    for k in _HPO_INTERNAL_KEYS:
        training.pop(k, None)

    # 2. 合并全量训练覆盖（NUM_EPOCHS、PATIENCE、SAVE_INTERVAL）
    training.update(full_train_overrides)

    # 3. 设置输出目录
    if output_save_dir:
        save_dir = Path(output_save_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = (
            PROJECT_ROOT
            / f'data/output/h_out_eedscy/m10_full_trial{trial_index:03d}_{timestamp}'
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    training['SAVE_DIR'] = save_dir
    training['RUN_NAME'] = f'full_trial_{trial_index:03d}'
    training['EXPERIMENT_LOG_CSV'] = save_dir / 'experiment_results.csv'

    return training, model


def print_summary(loaded: dict, training: dict, model: dict):
    """打印本次全量训练配置摘要"""
    print('\n' + '=' * 70)
    print('M10 全量训练（从 HPO 结果加载超参）')
    print('=' * 70)
    print(f'  来源文件    : {loaded["source_file"]}')
    print(f'  Trial 编号  : {loaded["trial_index"]}')
    print(f'  原 Trial 目录: {loaded["trial_dir"]}')
    print(f'\n  训练参数覆盖:')
    for k, v in training.items():
        print(f'    {k:30s} = {v}')
    print(f'\n  模型参数覆盖:')
    for k, v in model.items():
        print(f'    {k:30s} = {v}')
    print('=' * 70 + '\n')


def main():
    # 1. 加载 trial 参数
    loaded = load_trial_params(
        source=Path(TRIAL_SOURCE),
        select_trial_index=SELECT_TRIAL_INDEX,
    )

    # 2. 构建全量训练参数
    training, model = build_full_train_overrides(
        loaded=loaded,
        full_train_overrides=copy.deepcopy(FULL_TRAIN_OVERRIDES),
        output_save_dir=OUTPUT_SAVE_DIR,
    )

    # 3. 打印摘要
    print_summary(loaded, training, model)

    # 4. 执行全量训练
    result = train_main(
        training_overrides=training,
        model_overrides=model,
        reset_model_config=True,
        restore_after_run=False,   # 全量训练不需要恢复配置
        trial=None,
        prune_enabled=False,
    )

    # 5. 打印结果
    print('\n' + '=' * 70)
    print('全量训练完成')
    print('=' * 70)
    keys_to_print = [
        'best_val_ccc_loss', 'final_test_ccc_loss',
        'final_test_rmse', 'final_test_mae', 'final_test_mape',
        'final_test_pearson_r', 'final_test_r2',
        'best_epoch', 'epochs_completed',
    ]
    for k in keys_to_print:
        if k in result:
            print(f'  {k:30s} = {result[k]}')
    print(f'  模型保存目录: {training["SAVE_DIR"]}')
    print('=' * 70)
    return result


if __name__ == '__main__':
    main()
