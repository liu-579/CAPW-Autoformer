# 旅游景区舆情分析与客流预测系统 — 模块总结文档

本项目是一个完整的旅游景区数据分析流水线，涵盖从评论数据清洗、情感分析、特征工程到时序客流预测的全流程。每个模块（M1-M11）均有对应的脚本文件（`scripts/`）和配置文件（`config/`），部分模块同时提供本地文件版和 MySQL 数据库版两套实现。

---

## M1 — 评论数据清洗

| 项目 | 内容 |
|------|------|
| 脚本 | `m1_clean_comments.py`（本地Excel版）、`m1_sql_clean.py`（MySQL版） |
| 配置 | `m1_config.py`、`m1_sql_config.py` |
| 作用 | 从 Excel / MySQL 读取原始旅游评论数据，执行文本清洗（HTML解码、去URL、去Emoji、去@用户、去话题标签、去重），输出标准化的 JSONL 或写回数据库 |

**主要数据指标：**
- 最小文本长度：10 字符（本地版）/ 2 字符（SQL版）
- 最大文本长度：5000 字符
- 中文字符最小占比：0.3（本地版）
- 去重字段：作者 + 正文
- 输入：`乐山大佛、黄鹤楼评价数据.xlsx` / MySQL 表 `yq_24_2`, `yq_23_xsw_2`, `yq_23_eedscy_2`
- 输出：`m1_cleaned_comments.jsonl` / MySQL 表 `yq_clean_all_3`

---

## M2 — AI 多维情感评分

| 项目 | 内容 |
|------|------|
| 脚本 | `m2_ai_setment.py`（本地版）、`m2_sql_ai_swtment.py`（MySQL版） |
| 配置 | `m2_config.py`、`m2_sql_config.py` |
| 作用 | 调用 LLM API（GPT/DeepSeek）对每条评论进行五维度情感打分，支持异步并发、速率限制、指数退避重试 |

**主要数据指标：**
- 情感维度（5维）：景色、交通、美食、门票/消费、服务
- 评分范围：-2（极差）~ +2（极好），0 表示中性或未提及
- 并发数：10 workers，速率限制 50 req/min
- 重试策略：最多 3 次，指数退避
- 输入：M1 清洗后的评论
- 输出：`m2_sentiment_scores.json` / `m2_sql_ai_results.json`

---

## M3 — 数据集构建与划分

| 项目 | 内容 |
|------|------|
| 脚本 | `m3_build_dataset.py`（本地版）、`m3_sql_build_dataset.py`（MySQL版） |
| 配置 | `m3_config.py`、`m3_sql_config.py` |
| 作用 | 将 M1 清洗后的评论与 M2 的情感评分进行对齐合并，执行数据质量检查，按比例划分训练/验证/测试集，生成数据卡和压缩包 |

**主要数据指标：**
- 数据集划分比例：训练 80% / 验证 10% / 测试 10%
- 随机种子：42
- 合法分数范围：[-2, 2]
- KL 散度阈值：< 0.05（检验训练集与测试集分布一致性）
- 维度映射：scenery→景色, transportation→交通, food→美食, ticket/consumption→门票/消费, service→服务
- 输出：`dataset_v1.jsonl`, `train.csv`, `val.csv`, `test.csv`, `data_card.md`

---

## M4 — BERT 多维情感回归模型训练与推理

| 项目 | 内容 |
|------|------|
| 脚本 | `m4_train_baseline_1.0.py`（v1训练）、`m4_train_baseline_2.0.py`（v2训练，组合损失）、`m4_b_inference.py`（本地推理）、`m4_sql_b_inference.py`（MySQL推理）、`m4_text.py`（交互式单句测试） |
| 配置 | `m4_config.py`（训练配置）、`m4_b_config.py`（本地推理配置）、`m4_sql_b_config.py`（MySQL推理配置） |
| 作用 | 基于预训练中文 BERT 模型（bert-base-chinese / macbert / RoBERTa）微调多维情感回归模型，对评论进行五维度情感打分 |

**主要数据指标：**
- 预训练模型：`hfl/chinese-roberta-wwm-ext`、`hfl/chinese-macbert-base`、`bert-base-chinese`
- 最大序列长度：128 / 512
- 输出维度：5（五维情感）
- 训练轮数：最多 150 epochs
- 学习率：2e-5，权重衰减 0.01
- 损失函数：v1 使用 MSE，v2 使用 MSE + Pearson 组合损失
- 早停耐心值：30 epochs，监控 avg_pearsonr
- 标签归一化：-2→0.0, -1→0.25, 0→0.5, 1→0.75, 2→1.0
- 评估指标：Pearson 相关系数、RMSE、MAE、离散化准确率
- 合格阈值：平均 Pearson r ≥ 0.85
- 推理批次大小：32 / 64
- 输出：每条评论的五维情感分数（整数 -2~2）

---

## M5 — 日度情感聚合

| 项目 | 内容 |
|------|------|
| 脚本 | `m5_daily_aggregation.py`（本地版）、`m5_sql_daily_aggregation.py`（MySQL版，贝叶斯平滑） |
| 配置 | `m5_config.py`、`m5_sql_config.py` |
| 作用 | 将 M4 按条的情感分析结果聚合为按天统计的情感均值，处理缺失日期的填充 |

**主要数据指标：**
- 聚合粒度：按天（daily）
- 情感维度：景色、交通、美食、消费、服务
- 缺失值填充策略：7 天移动平均 → 全局均值兜底
- 贝叶斯平滑分位数：0.25（SQL版动态 K 值）
- 景区列表：响沙湾、鄂尔多斯草原
- 小数精度：3 位
- 输出：`m5_sentiment_daily_int.jsonl` / MySQL 表 `yq_sentiment_daily_2`

---

## M6 — 历史客流数据 ETL

| 项目 | 内容                                                                |
|------|-------------------------------------------------------------------|
| 脚本 | `m6_clean_history.py`（日度版）、`m6_clean_history_hourly.py`（小时版）      |
| 配置 | `m6_config.py`（日度版）、`m6_h_config.py`（小时版）                         |
| 作用 | 从 MySQL 读取混合粒度的历史客流数据，按景区分离，提取日 度/小时度客流基础信息，解析天气 JSON 字段，生成标准化特征表 |

**主要数据指标：**
- 数据源表：`history_passenger_flow_data`
- 景区 ID：`9019043`（响沙湾）、`19012822`（鄂尔多斯草原）
- 粒度标识：`d`（天）、`h`（小时）
- 提取特征：客流人数、星期、节假日、天气状况、温度、风向、风力、湿度
- 天气聚合规则：天气状况取众数、温度取最高/最低、风力/湿度取均值
- 缺失数据填充：线性插值（小时版窗口=3小时）
- 输出：MySQL 表 `scenic_spot_{id}_daily` / `history_hourly_{id}`

---

## M7 — 数据融合与特征工程

| 项目 | 内容 |
|------|------|
| 脚本 | `m7_b_data_fusion.py`（小时尺度融合） |
| 配置 | `m7_config.py`（日度版）、`m7_b_config.py`（小时版）、`m7c_config.py`（淡旺季分析） |
| 作用 | 将日度情感数据与物理特征数据（客流、天气）融合，构建滞后特征和移动平均特征；M7c 使用 Jenks 自然断点法进行淡旺季分类 |

**主要数据指标：**
- 日度版滞后天数：[1, 2, 3, 7] 天
- 小时版滞后小时数：[1, 2, 3, 24, 48, 168] 小时
- 移动平均窗口：日度 7 天 / 小时 24 小时
- 情感维度字段：scenery_score, transportation_score, food_score, consumption_score, service_score
- 淡旺季分类：3 类（淡季/平季/旺季），Jenks 自然断点法，GVF 阈值 > 0.8
- 财年定义：10月1日起始
- 景区映射：响沙湾→`history_daily_xsw` / `history_hourly_xsw`，鄂尔多斯草原→`history_daily_eedscy` / `history_hourly_eedscy`
- 输出：MySQL 表 `fused_training_data_2`（日度）/ `fused_training_data_hourly`（小时）

---

## M8 — 数据清洗与张量构建

| 项目 | 内容 |
|------|------|
| 配置 | `m8a_config.py`（日度清洗）、`m8a_h_config.py`（小时清洗）、`m8b_config.py`（日度张量）、`m8b_config_h.py`（小时张量 v1）、`m8b_config_h_2.py`（小时张量 v2） |
| 作用 | M8a：对融合后的宽表进行二次清洗（中文文本列编码、周期性特征生成、节假日特征、常量列移除）；M8b：将清洗后的数据构建为时序张量（滑动窗口），划分训练/验证/测试集，执行混合归一化策略 |

**主要数据指标（M8a 清洗）：**
- 文本列处理：day_of_week→周期性编码, holiday→chinese_calendar, weather_cond/wind_dir→LabelEncoder
- 淡旺季特征：season_code (0/1/2), One-Hot 编码 (is_low_season, is_normal_season, is_peak_season)

**主要数据指标（M8b 张量构建）：**
- 日度版：输入窗口 30 天，预测 7 天
- 小时版 v1：输入窗口 168 小时（7天），预测 24 小时（1天）
- 小时版 v2：输入窗口 336 小时（14天），预测 72 小时（3天）
- 数据集划分：训练 70% / 验证 15% / 测试 15%
- 归一化策略：混合策略（情感列保持原值，Target 列 Log1p + MinMax/Standard，其他列 MinMax）
- 情感维度分组关键词：scenery, transportation, food, consumption, service（顺序与模型权重对应）
- 输出：`train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy`, `test_x.npy`, `test_y.npy`, `feature_map.json`, `scalers.pkl`

---

## M9 — 动态融合预测模型定义

| 项目 | 内容 |
|------|------|
| 配置 | `m9_config.py`（主配置）、`m9_config_flash.py`（轻量版） |
| 作用 | 定义 Dynamic Fusion Predictor 模型架构参数，基于 Transformer Encoder + Context-Aware Weight Network，支持三阶段训练和可解释性分析 |

**主要数据指标：**
- Transformer Backbone：d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.2
- 轻量版：d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1
- 权重网络：hidden_dim=64, num_heads=2, num_layers=1
- 温度参数：初始 1.0，可学习
- 输出维度：1（单变量客流预测）
- 预测长度：72 小时（主配置）/ 7 天（轻量版）
- 输入序列长度：336 小时（主配置）
- 位置编码：正弦余弦（sinusoidal）
- L1 正则化系数：0.001
- 三阶段训练：阶段1（骨干网络）10 epochs → 阶段2（权重网络）10 epochs → 阶段3（联合微调）20 epochs

---

## M10 — 模型训练

| 项目 | 内容 |
|------|------|
| 脚本 | `m10_train.py`（三阶段渐进式训练）、`m10_train_2.py`（CCC Loss + 自适应峰值加权版） |
| 配置 | `m10_config.py` |
| 作用 | 执行三阶段渐进式训练流程（预热骨干→权重网络学习→联合微调），支持温度退火、多种损失函数、早停策略 |

**主要数据指标：**
- 总训练轮数：1000 epochs
- 三阶段分配：Phase 1（预热）1 epoch → Phase 2（权重）1 epoch → Phase 3（微调）998 epochs
- 批次大小：64
- 学习率：1e-5，权重衰减 1e-4
- 温度退火：2.0 → 0.5（线性）
- 损失函数：
  - v1：MSE + Pearson 组合损失
  - v2：CCC Loss（一致性相关系数）+ 自适应加权 MSE
- 自适应峰值加权：阈值 = Mean + 1.5×Std，惩罚权重 3.0×
- 学习率调度：Cosine Annealing（最小 1e-7）
- 早停：patience=50, min_delta=1e-4
- 梯度裁剪：max_norm=1.0
- 评估指标：RMSE、MAE、MAPE、Pearson r（均在反归一化后的真实尺度上计算）
- 输出：`best_model_unified.pth`, `training_log.csv`, `vis_trend_final_unified.png`

---

## M11 — 模型测试与评估

| 项目 | 内容 |
|------|------|
| 脚本 | `m11_test.py` |
| 配置 | `m11_config.py` |
| 作用 | 加载训练好的模型，在测试集上进行推理评估，生成性能指标报告和多种可视化图表 |

**主要数据指标：**
- 测试场景：响沙湾（xsw）/ 鄂尔多斯草原（eedscy）
- 推理批次大小：64
- 温度参数：0.01（接近 argmax）
- 评估指标：RMSE、MAE、MAPE、Pearson r
- 分时段指标：按小时、按天统计
- 峰值预测性能：高于 90% 分位数的样本单独评估
- 可视化输出：预测值 vs 真实值折线图、散点图、误差分布图、残差图
- 输出：`test_metrics.json`, `predictions.npz`, `test_report.txt`, 多张 PNG 图表

---

## 整体流水线概览

```
M1 评论清洗 → M2 AI情感评分 → M3 数据集构建 → M4 BERT情感模型训练/推理
                                                         ↓
M6 历史客流ETL ──────────────────→ M7 数据融合+特征工程 ← M5 日度情感聚合
                                         ↓
                                   M7c 淡旺季分析
                                         ↓
                                   M8a 数据清洗 → M8b 张量构建
                                                       ↓
                                                 M9 模型定义
                                                       ↓
                                                 M10 模型训练
                                                       ↓
                                                 M11 模型测试
```

- **上游（M1-M5）**：处理舆情评论数据，从原始文本到结构化的日度情感分数
- **中游（M6-M8）**：处理物理特征数据（客流、天气），与情感数据融合，构建模型可用的时序张量
- **下游（M9-M11）**：定义、训练和评估基于 Transformer 的动态融合客流预测模型

目标景区：**响沙湾**、**鄂尔多斯草原**
