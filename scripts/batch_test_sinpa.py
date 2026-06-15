import json
import re
import subprocess
from pathlib import Path


BASE_DIR = Path("/Users/fziteng/Documents/Code/python/smart-parking-lora")

MODEL_PATH = BASE_DIR / "models" / "qwen2.5-3b-mlx"
ADAPTER_PATH = BASE_DIR / "adapters" / "sinpa-qwen3b-lora"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)


cases = [
    {
        "name": "morning_peak",
        "lot_id": "lot_0",
        "history": [0.52, 0.49, 0.45, 0.41, 0.38, 0.35, 0.31, 0.28, 0.25, 0.22, 0.20, 0.18],
        "time_of_day": 32,
        "weekday": 1,
        "is_holiday": 0,
        "temperature": 0.15,
        "humidity": -0.22,
        "windspeed": 0.08,
        "utilization_type": 2,
        "planning_area": 5,
        "road_density": 0.31,
        "latitude": 0.12,
        "longitude": -0.09,
    },
    {
        "name": "evening_peak",
        "lot_id": "lot_1",
        "history": [0.42, 0.39, 0.35, 0.32, 0.29, 0.25, 0.22, 0.19, 0.17, 0.15, 0.14, 0.13],
        "time_of_day": 72,
        "weekday": 5,
        "is_holiday": 0,
        "temperature": 0.10,
        "humidity": 0.36,
        "windspeed": -0.04,
        "utilization_type": 3,
        "planning_area": 8,
        "road_density": 0.47,
        "latitude": 0.08,
        "longitude": -0.11,
    },
    {
        "name": "night_leave",
        "lot_id": "lot_2",
        "history": [0.18, 0.20, 0.25, 0.31, 0.37, 0.44, 0.50, 0.56, 0.62, 0.68, 0.72, 0.76],
        "time_of_day": 84,
        "weekday": 6,
        "is_holiday": 0,
        "temperature": -0.05,
        "humidity": 0.18,
        "windspeed": 0.02,
        "utilization_type": 1,
        "planning_area": 3,
        "road_density": 0.22,
        "latitude": 0.05,
        "longitude": -0.04,
    },
]


def build_prompt(case):
    return f"""你是智慧停车场车位可用性预测助手。
请根据新加坡停车场过去3小时的历史数据和当前特征，预测未来3小时的停车可用性变化。

停车场编号：{case["lot_id"]}
过去12个时间步的停车可用性：{case["history"]}
当前时间段编号：{case["time_of_day"]}，范围为0到95，表示一天中的15分钟时间片。
星期编号：{case["weekday"]}，范围为0到6。
是否节假日：{case["is_holiday"]}
温度特征：{case["temperature"]}
湿度特征：{case["humidity"]}
风速特征：{case["windspeed"]}
停车场利用类型编号：{case["utilization_type"]}
规划区域编号：{case["planning_area"]}
道路密度特征：{case["road_density"]}
纬度特征：{case["latitude"]}
经度特征：{case["longitude"]}

请只输出未来12个时间步的停车可用性预测结果，格式如下：
未来12个时间步的停车可用性预测为：[0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61]。
"""


def run_model(prompt):
    cmd = [
        "mlx_lm.generate",
        "--model", str(MODEL_PATH),
        "--adapter-path", str(ADAPTER_PATH),
        "--prompt", prompt,
        "--max-tokens", "200",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


def extract_availability_list(text):
    match = re.search(r"\[([0-9.,\s]+)\]", text)

    if not match:
        return []

    numbers = re.findall(r"\d+\.\d+|\d+", match.group(1))

    values = []
    for num in numbers[:12]:
        value = float(num)
        value = max(0.0, min(1.0, value))
        values.append(round(value, 4))

    return values


def get_occupancy_level(predicted_availability):
    if not predicted_availability:
        return "unknown"

    avg_availability = sum(predicted_availability) / len(predicted_availability)

    if avg_availability <= 0.25:
        return "high"
    elif avg_availability <= 0.55:
        return "medium"
    else:
        return "low"


def get_congestion_penalty(predicted_availability):
    if not predicted_availability:
        return 0.0

    avg_availability = sum(predicted_availability) / len(predicted_availability)

    if avg_availability <= 0.25:
        return 0.5
    elif avg_availability <= 0.55:
        return 0.25
    else:
        return 0.08


def main():
    for case in cases:
        print(f"正在测试场景：{case['name']}")

        prompt = build_prompt(case)
        raw_output = run_model(prompt)
        predicted_availability = extract_availability_list(raw_output)

        result = {
            "model": "Qwen2.5-3B-Instruct + LoRA",
            "dataset": "SINPA",
            "scene": case["name"],
            "lot_id": case["lot_id"],
            "history_steps": 12,
            "future_steps": 12,
            "input_history_availability": case["history"],
            "predicted_availability": predicted_availability,
            "predicted_occupancy_level": get_occupancy_level(predicted_availability),
            "congestion_penalty": get_congestion_penalty(predicted_availability),
            "raw_model_output": raw_output
        }

        output_path = RESULT_DIR / f"predict_{case['name']}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"已保存：{output_path}")


if __name__ == "__main__":
    main()