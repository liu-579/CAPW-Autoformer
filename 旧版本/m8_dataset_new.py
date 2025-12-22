import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pymysql
import pickle
import sys

# 导入你的配置文件
try:
    import 旧版本.m8_config as cfg
except ImportError:
    print("错误: 未找到 m8_config.py，请确保配置文件在同一目录下。")
    sys.exit(1)


class ChineseCategoricalEncoder:
    """专门处理中文类别特征的编码器"""

    def __init__(self):
        self.encoders = {}
        self.feature_mappings = {}

    def fit_transform(self, df, categorical_columns):
        """对中文类别特征进行编码"""
        df_encoded = df.copy()

        # 过滤掉不在df中的列
        valid_cols = [c for c in categorical_columns if c in df.columns]

        for col in valid_cols:
            # 填充缺失值
            if df_encoded[col].isnull().any():
                df_encoded[col] = df_encoded[col].fillna('未知')

            # 确保是字符串
            df_encoded[col] = df_encoded[col].astype(str)

            # 编码
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col])

            # 保存
            self.encoders[col] = encoder
            self.feature_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

        return df_encoded

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'encoders': self.encoders, 'mappings': self.feature_mappings}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.encoders = data['encoders']
            self.feature_mappings = data['mappings']


def preprocess_tourism_data():
    """
    基于 m8_config 配置的数据预处理全流程
    """
    # 0. 初始化目录
    cfg.create_output_directories()

    print(f"🚀 开始处理数据...")
    print(f"📌 目标景区: {cfg.TARGET_SCENIC_NAME}")
    print(f"📌 数据库: {cfg.DB_CONFIG['database']} | 表: {cfg.INPUT_TABLE}")

    # 1. 连接数据库并读取数据 (动态SQL)
    conn = pymysql.connect(**cfg.DB_CONFIG)

    # 动态构建查询：只查询目标景区的记录
    # 注意：这里假设表中有 scenic_name 字段，如果没有，请修改 WHERE 条件
    query = f"""
    SELECT *
    FROM {cfg.INPUT_TABLE}
    WHERE scenic_name = '{cfg.TARGET_SCENIC_NAME}'
    ORDER BY date ASC
    """

    try:
        df = pd.read_sql(query, conn)
        print(f"✅ 数据读取成功，原始 Shape: {df.shape}")
    except Exception as e:
        print(f"❌ 数据库读取失败: {e}")
        conn.close()
        return
    finally:
        conn.close()

    if df.empty:
        raise ValueError(f"未查询到景区 '{cfg.TARGET_SCENIC_NAME}' 的数据，请检查数据库。")

    # 2. 数据清洗与筛选
    # 处理日期
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        dates = df['date'].values  # 保存日期用于后续对齐（如果需要）

    # 排除不需要的列
    cols_to_drop = [c for c in cfg.EXCLUDE_COLS if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    # 3. 自动识别特征类型
    all_cols = df_clean.columns.tolist()
    target_col = cfg.TARGET_COL

    if target_col not in all_cols:
        raise ValueError(f"目标列 '{target_col}' 不在数据中，现有列: {all_cols}")

    # 3.1 类别特征 (从配置读取)
    cat_cols = [c for c in cfg.CATEGORICAL_COLS if c in all_cols]

    # 3.2 情感特征 (根据配置的关键词识别，例如 'score')
    sentiment_cols = [
        c for c in all_cols
        if cfg.SENTIMENT_FEATURE_KEYWORD in c
           and c not in cat_cols
           and c != target_col
    ]

    # 3.3 其他数值特征 (剩下的数值列)
    # 排除掉已经是类别、情感、目标的列
    remaining = set(all_cols) - set(cat_cols) - set(sentiment_cols) - {target_col}
    # 简单的逻辑：假设剩下的都是数值上下文特征（如 temperature, humidity）
    context_numeric_cols = list(remaining)

    print("\n🔍 特征分组结果:")
    print(f"  - 目标变量: {target_col}")
    print(f"  - 类别特征 ({len(cat_cols)}): {cat_cols}")
    print(f"  - 情感特征 ({len(sentiment_cols)}): {sentiment_cols}")
    print(f"  - 环境/数值特征 ({len(context_numeric_cols)}): {context_numeric_cols}")

    # 4. 特征工程处理

    # 4.1 处理类别特征
    cat_encoder = ChineseCategoricalEncoder()
    if cat_cols:
        df_clean = cat_encoder.fit_transform(df_clean, cat_cols)
        # 保存类别编码器映射表到 json (为了可读性)
        import json
        map_path = cfg.get_feature_map_filename()
        # 将 numpy 类型转为 python原生类型以便 json 序列化
        serializable_map = {}
        for col, mapping in cat_encoder.feature_mappings.items():
            serializable_map[col] = {str(k): int(v) for k, v in mapping.items()}

        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_map, f, ensure_ascii=False, indent=2)
        print(f"✅ 类别映射表已保存: {map_path}")

    # 4.2 处理缺失值 (数值列前向填充)
    numeric_cols = sentiment_cols + context_numeric_cols + [target_col]
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(method='bfill')

    # 5. 数据归一化/标准化
    # 根据配置 SCALER_FEATURE_RANGE 选择 Scaler，这里为了通用性，
    # 我们将 Target 单独 Scaler，特征统一 Scaler

    feature_scaler = MinMaxScaler(feature_range=cfg.SCALER_FEATURE_RANGE)
    target_scaler = MinMaxScaler(feature_range=cfg.SCALER_FEATURE_RANGE)

    # 组合所有输入特征列
    input_features_cols = sentiment_cols + context_numeric_cols + cat_cols

    # 拟合 transform
    if input_features_cols:
        df_clean[input_features_cols] = feature_scaler.fit_transform(df_clean[input_features_cols])

    df_clean[[target_col]] = target_scaler.fit_transform(df_clean[[target_col]])

    # 保存 Scaler
    scaler_path = cfg.get_scaler_filename()
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'feature_names': input_features_cols,
            'target_name': target_col
        }, f)
    print(f"✅ 模型Scaler已保存: {scaler_path}")

    # 6. 构建时间序列样本 (Sliding Window)
    X, y = [], []
    seq_len = cfg.SEQ_LEN
    pred_len = cfg.PRED_LEN

    data_x = df_clean[input_features_cols].values
    data_y = df_clean[target_col].values

    print(f"\n✂️  开始切分序列 (Lookback={seq_len}, Predict={pred_len})...")

    for i in range(len(df_clean) - seq_len - pred_len + 1):
        _x = data_x[i: i + seq_len]
        _y = data_y[i + seq_len: i + seq_len + pred_len]
        X.append(_x)
        y.append(_y)

    X = np.array(X)
    y = np.array(y)

    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")

    # 7. 数据集划分 (Train/Val/Test)
    total_samples = len(X)
    train_size = int(total_samples * cfg.TRAIN_RATIO)
    val_size = int(total_samples * cfg.VAL_RATIO)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    print(f"\n📊 数据集划分完成:")
    print(f"  - Train: {len(X_train)}")
    print(f"  - Val:   {len(X_val)}")
    print(f"  - Test:  {len(X_test)}")

    # 8. 保存数据 (按照 config 定义的文件名)
    filenames = cfg.get_dataset_filenames()

    np.save(filenames['train_x'], X_train)
    np.save(filenames['train_y'], y_train)
    np.save(filenames['val_x'], X_val)
    np.save(filenames['val_y'], y_val)
    np.save(filenames['test_x'], X_test)
    np.save(filenames['test_y'], y_test)

    print(f"\n💾 所有数据已保存至: {cfg.DATASET_OUTPUT_DIR}")

    # 打印一些元数据供检查
    print("\n🔍 最终特征顺序 (Indices):")
    for idx, name in enumerate(input_features_cols):
        print(f"  {idx}: {name}")

    return {
        'feature_names': input_features_cols,
        'config': cfg
    }


if __name__ == "__main__":
    # 打印配置信息
    cfg.print_config_summary()

    # 执行预处理
    preprocess_tourism_data()

    print("\n✨ 预处理任务全部完成!")