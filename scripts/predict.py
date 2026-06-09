import argparse
import json
import subprocess
import re
from pathlib import Path

BASE_DIR = Path("/Users/fziteng/Documents/Code/python/smart-parking-lora")

MODEL_PATH = BASE_DIR / "models" / "qwen2.5-0.5b-mlx"
ADAPTER_PATH = BASE_DIR / "adapters" / "parking-lora"


def build_prompt(time, weekday, weather, enter_count, leave_count, occupancy_rate):
    return f"""你是智慧停车场车流预测助手。请根据当前停车场历史数据，预测下一小时进入车辆数量和占用率。

时间：{time}
星期：{weekday}
天气：{weather}
当前进入车辆数：{enter_count}
当前离开车辆数：{leave_count}
当前占用率：{occupancy_rate}%

请只输出预测结果，格式如下：
预测下一小时进入车辆数为xx辆，预测下一小时停车场占用率为xx%。
"""


def run_model(prompt):
    cmd = [
        "mlx_lm.generate",
        "--model", str(MODEL_PATH),
        "--adapter-path", str(ADAPTER_PATH),
        "--prompt", prompt,
        "--max-tokens", "100",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def extract_numbers(text):
    numbers = re.findall(r"\d+", text)

    if len(numbers) >= 2:
        predicted_enter_count = int(numbers[-2])
        predicted_occupancy_rate = int(numbers[-1])
    else:
        predicted_enter_count = None
        predicted_occupancy_rate = None

    return predicted_enter_count, predicted_occupancy_rate


def get_congestion_level(occupancy_rate):
    if occupancy_rate is None:
        return "unknown"
    if occupancy_rate >= 85:
        return "high"
    elif occupancy_rate >= 60:
        return "medium"
    else:
        return "low"


def main():
    parser = argparse.ArgumentParser(description="智慧停车场车流预测脚本")

    parser.add_argument("--time", required=True, help="当前时间，例如 08:00")
    parser.add_argument("--weekday", required=True, help="星期，例如 Monday")
    parser.add_argument("--weather", required=True, help="天气，例如 Sunny")
    parser.add_argument("--enter_count", type=int, required=True, help="当前进入车辆数")
    parser.add_argument("--leave_count", type=int, required=True, help="当前离开车辆数")
    parser.add_argument("--occupancy_rate", type=int, required=True, help="当前占用率")

    args = parser.parse_args()

    prompt = build_prompt(
        args.time,
        args.weekday,
        args.weather,
        args.enter_count,
        args.leave_count,
        args.occupancy_rate
    )

    raw_output = run_model(prompt)

    predicted_enter_count, predicted_occupancy_rate = extract_numbers(raw_output)

    result = {
        "time": args.time,
        "weekday": args.weekday,
        "weather": args.weather,
        "current_enter_count": args.enter_count,
        "current_leave_count": args.leave_count,
        "current_occupancy_rate": args.occupancy_rate,
        "predicted_enter_count": predicted_enter_count,
        "predicted_occupancy_rate": predicted_occupancy_rate,
        "congestion_level": get_congestion_level(predicted_occupancy_rate),
        "raw_model_output": raw_output.strip()
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()