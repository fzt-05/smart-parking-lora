import json
import re
import subprocess
from pathlib import Path


BASE_DIR = Path("/Users/fziteng/Documents/Code/python/smart-parking-lora")

MODEL_PATH = BASE_DIR / "models" / "qwen2.5-3b-mlx"
ADAPTER_PATH = BASE_DIR / "adapters" / "sinpa-qwen3b-lora"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)


def build_prompt():
    """
    第一版先使用固定样例输入。
    后续可以改成从命令行参数、JSON文件或后端API接收输入。
    """

    history_availability = [
        0.21, 0.25, 0.29, 0.31,
        0.34, 0.36, 0.40, 0.42,
        0.43, 0.45, 0.46, 0.48
    ]

    prompt = f"""你是智慧停车场车位可用性预测助手。
请根据新加坡停车场过去3小时的历史数据和当前特征，预测未来3小时的停车可用性变化。

停车场编号：lot_0
过去12个时间步的停车可用性：{history_availability}
当前时间段编号：32，范围为0到95，表示一天中的15分钟时间片。
星期编号：1，范围为0到6。
是否节假日：0
温度特征：0.15
湿度特征：-0.22
风速特征：0.08
停车场利用类型编号：2
规划区域编号：5
道路密度特征：0.31
纬度特征：0.12
经度特征：-0.09

请只输出未来12个时间步的停车可用性预测结果，格式如下：
未来12个时间步的停车可用性预测为：[0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61]。
"""
    return prompt


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
    """
    从模型输出中提取预测数组。
    如果模型没有严格按格式输出，就返回空列表，避免程序崩溃。
    """

    match = re.search(r"\[([0-9.,\s]+)\]", text)

    if not match:
        return []

    number_text = match.group(1)
    numbers = re.findall(r"\d+\.\d+|\d+", number_text)

    values = []

    for num in numbers[:12]:
        value = float(num)
        value = max(0.0, min(1.0, value))
        values.append(round(value, 4))

    return values


def get_occupancy_level(predicted_availability):
    """
    这里用停车可用性反推拥堵程度：
    可用性越低，说明越拥堵。
    """

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
    """
    给队员C路径规划模块使用。
    可用性越低，拥堵惩罚越高。
    """

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
    prompt = build_prompt()
    raw_output = run_model(prompt)
    predicted_availability = extract_availability_list(raw_output)

    result = {
        "model": "Qwen2.5-3B-Instruct + LoRA",
        "dataset": "SINPA",
        "lot_id": "lot_0",
        "history_steps": 12,
        "future_steps": 12,
        "predicted_availability": predicted_availability,
        "predicted_occupancy_level": get_occupancy_level(predicted_availability),
        "congestion_penalty": get_congestion_penalty(predicted_availability),
        "raw_model_output": raw_output
    }

    output_path = RESULT_DIR / "predict_demo.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n预测结果已保存到：{output_path}")


if __name__ == "__main__":
    main()