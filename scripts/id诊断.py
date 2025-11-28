# 创建 scripts/debug_ids.py
import json

# 检查评论文件
print("=" * 60)
print("检查评论文件 ID 格式")
print("=" * 60)
with open(r'D:\learning\shiyan1\data\输出\m1_cleaned_comments.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 3:  # 只看前3条
            data = json.loads(line)
            print(f"示例 {i+1}: ID={data['id']}, ID类型={type(data['id'])}")

# 检查评分文件
print("\n" + "=" * 60)
print("检查评分文件 ID 格式")
print("=" * 60)
with open(r'D:\learning\shiyan1\data\输出\m2_sentiment_scores.json', 'r', encoding='utf-8') as f:
    scores_data = json.load(f)
    # 获取前3个key
    for i, key in enumerate(list(scores_data.keys())[:3]):
        print(f"示例 {i+1}: ID={key}, ID类型={type(key)}")
        print(f"  内容: {scores_data[key]}")