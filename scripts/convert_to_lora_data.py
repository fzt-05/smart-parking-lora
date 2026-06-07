import pandas as pd
import json
from pathlib import Path

DATA_PATH = "/Users/fziteng/Documents/Code/python/smart-parking-lora/data/parking_data.csv"
OUTPUT_DIR = Path("/Users/fziteng/Documents/Code/python/smart-parking-lora/data")

df = pd.read_csv(DATA_PATH)

samples = []

for _, row in df.iterrows():
    prompt = (
        "你是智慧停车场车流预测助手。"
        "请根据当前停车场历史数据，预测下一小时进入车辆数量和占用率。\n\n"
        f"时间：{row['time']}\n"
        f"星期：{row['weekday']}\n"
        f"天气：{row['weather']}\n"
        f"当前进入车辆数：{row['enter_count']}\n"
        f"当前离开车辆数：{row['leave_count']}\n"
        f"当前占用率：{row['occupancy_rate']}%\n\n"
        "请输出预测结果。"
    )

    completion = (
        f"预测下一小时进入车辆数为 {row['next_enter_count']} 辆，"
        f"预测下一小时停车场占用率为 {row['next_occupancy_rate']}%。"
    )

    samples.append({
        "prompt": prompt,
        "completion": completion
    })

split_index = int(len(samples) * 0.8)

train_data = samples[:split_index]
valid_data = samples[split_index:]

with open(OUTPUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(OUTPUT_DIR / "valid.jsonl", "w", encoding="utf-8") as f:
    for item in valid_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("数据转换完成")
print(f"训练集数量：{len(train_data)}")
print(f"验证集数量：{len(valid_data)}")
print("输出文件：")
print(OUTPUT_DIR / "train.jsonl")
print(OUTPUT_DIR / "valid.jsonl")