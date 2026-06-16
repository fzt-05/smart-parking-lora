import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


BASE_DIR = Path("/Users/fziteng/Documents/Code/python/smart-parking-lora")

MODEL_PATH = BASE_DIR / "models" / "qwen2.5-3b-mlx"
ADAPTER_PATH = BASE_DIR / "adapters" / "sinpa-qwen3b-lora"

RESULT_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

SINPA_TRAIN_PATH = BASE_DIR / "real_datasets" / "SINPA" / "train.npz"

# 这里会自动尝试几个常见位置，你只要把 parking_final.add.xml 放到其中一个位置即可
PARKING_ADD_CANDIDATES = [
    BASE_DIR / "data" / "sumo" / "parking_final.add.xml",
    BASE_DIR / "parking_final.add.xml",
    BASE_DIR / "routing" / "parking_final" / "parking_final.add.xml",
]

RESULT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def find_parking_add_file():
    """
    查找小组 SUMO 停车场附加文件 parking_final.add.xml。
    """
    for path in PARKING_ADD_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "没有找到 parking_final.add.xml。\n"
        "请把 parking_final.add.xml 放到以下任意一个位置：\n"
        "1. data/sumo/parking_final.add.xml\n"
        "2. 项目根目录 parking_final.add.xml\n"
        "3. routing/parking_final/parking_final.add.xml"
    )


def load_parking_metadata_from_sumo():
    """
    从 SUMO 的 parking_final.add.xml 中读取停车场信息。

    主要读取：
    1. parkingArea id
    2. lane
    3. roadsideCapacity
    4. startPos / endPos
    5. width / length / angle

    total_lots = 所有 parkingArea 的 roadsideCapacity 总和。
    """

    add_path = find_parking_add_file()

    tree = ET.parse(add_path)
    root = tree.getroot()

    parking_spaces = []
    total_lots = 0

    for parking_area in root.findall("parkingArea"):
        parking_id = parking_area.get("id")
        lane = parking_area.get("lane")

        capacity = int(float(parking_area.get("roadsideCapacity", "1")))
        start_pos = float(parking_area.get("startPos", "0"))
        end_pos = float(parking_area.get("endPos", "0"))

        width = float(parking_area.get("width", "0"))
        length = float(parking_area.get("length", "0"))
        angle = float(parking_area.get("angle", "0"))

        total_lots += capacity

        parking_spaces.append({
            "parking_area_id": parking_id,
            "lane": lane,
            "capacity": capacity,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "width": width,
            "length": length,
            "angle": angle
        })

    metadata = {
        "lot_id": "parking_final",
        "lot_name": "SUMO智慧停车场",
        "source_file": str(add_path),
        "total_lots": total_lots,
        "parking_space_count": len(parking_spaces),
        "parking_spaces": parking_spaces
    }

    metadata_path = DATA_DIR / "parking_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def load_sinpa_data():
    """
    读取 SINPA 真实训练数据。
    x shape = (12167, 12, 1687, 12)

    本项目中：
    SINPA 用于提供历史停车变化模式；
    SUMO parking_final.add.xml 用于提供小组停车场容量和车位结构。
    """
    data = np.load(SINPA_TRAIN_PATH)
    return data["x"]


def get_sinpa_lot_capacity(x, lot_index):
    """
    估计 SINPA 中某个停车场的容量。
    用该停车场历史最大可用车位数量近似表示。
    """
    capacity = float(np.max(x[:, :, lot_index, 0]))

    if capacity <= 0:
        capacity = 1.0

    return round(capacity, 4)


def time_step_to_clock(time_step):
    """
    将 0-95 的时间片编号转换为 HH:MM。
    每个时间片为 15 分钟。
    """
    time_step = time_step % 96
    total_minutes = time_step * 15
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def find_sample_by_time_step(x, target_time_step, lot_index=0):
    """
    从 SINPA 中查找指定时间段的样本。

    x[:, -1, lot_index, 1] 表示当前样本最后一个历史时间步的 time_of_day。
    time_of_day 范围为 0-95。
    """
    for sample_index in range(x.shape[0]):
        current_time_step = int(round(float(x[sample_index, -1, lot_index, 1])))

        if current_time_step == target_time_step:
            return sample_index

    return 0


def convert_lots_to_ratio(available_lots_list, lot_capacity):
    """
    将可用车位数量转换为 0~1 之间的可用性比例。
    """
    if lot_capacity <= 0:
        return [0.0 for _ in available_lots_list]

    return [
        round(max(0.0, min(1.0, value / lot_capacity)), 4)
        for value in available_lots_list
    ]


def scale_sinpa_history_to_sumo_capacity(x, sample_index, sinpa_lot_index, sumo_capacity):
    """
    从 SINPA 读取历史可用车位序列，并按比例映射到小组 SUMO 停车场容量。

    例如：
    SINPA 某停车场容量约 50，历史可用 25，比例是 0.5；
    小组 SUMO 停车场容量是 694，则映射后可用车位约为 347。
    """

    sinpa_capacity = get_sinpa_lot_capacity(x, sinpa_lot_index)

    lot_history = x[sample_index, :, sinpa_lot_index, :]
    latest = lot_history[-1]

    sinpa_history_available_lots = [
        round(float(value), 4)
        for value in lot_history[:, 0]
    ]

    history_ratio = convert_lots_to_ratio(
        sinpa_history_available_lots,
        sinpa_capacity
    )

    sumo_history_available_lots = [
        round(ratio * sumo_capacity, 4)
        for ratio in history_ratio
    ]

    case = {
        "sample_index": sample_index,
        "sinpa_lot_index": sinpa_lot_index,
        "sinpa_capacity": sinpa_capacity,

        "history_available_lots": sumo_history_available_lots,
        "history_availability_ratio": history_ratio,

        "time_of_day": int(round(float(latest[1]))),
        "weekday": int(round(float(latest[2]))),
        "is_holiday": int(round(float(latest[3]))),

        "temperature": round(float(latest[4]), 4),
        "humidity": round(float(latest[5]), 4),
        "windspeed": round(float(latest[6]), 4),

        "utilization_type": int(round(float(latest[7]))),
        "planning_area": int(round(float(latest[8]))),
        "road_density": round(float(latest[9]), 4),

        "latitude": round(float(latest[10]), 4),
        "longitude": round(float(latest[11]), 4),
    }

    return case


def build_example_values(history_available_lots, lot_capacity):
    """
    根据真实历史数据动态生成 prompt 示例。
    避免固定示例导致模型照抄。
    """
    if not history_available_lots:
        start_value = lot_capacity * 0.5
    else:
        start_value = history_available_lots[-1]

    if len(history_available_lots) >= 2:
        trend = history_available_lots[-1] - history_available_lots[-2]
    else:
        trend = 0

    example_values = []

    for i in range(12):
        value = start_value + trend * (i + 1)
        value = max(0.0, min(lot_capacity, value))
        example_values.append(round(value, 1))

    return example_values


def build_prompt(case, parking_metadata, start_time_step):
    """
    构造大模型输入。
    这里预测的是小组 SUMO 停车场未来12个时间步的可用车位数量。
    """

    lot_capacity = parking_metadata["total_lots"]
    example_values = build_example_values(
        case["history_available_lots"],
        lot_capacity
    )

    prompt = f"""你是智慧停车场车位可用性预测助手。
请根据停车场过去3小时的历史数据和当前特征，预测未来3小时的停车可用性变化。

停车场编号：{parking_metadata["lot_id"]}
停车场名称：{parking_metadata["lot_name"]}
停车场总容量：{lot_capacity}
过去12个时间步的可用车位数量：{case["history_available_lots"]}
过去12个时间步的停车可用性比例：{case["history_availability_ratio"]}
当前时间段编号：{start_time_step}，范围为0到95，表示一天中的15分钟时间片。
星期编号：{case["weekday"]}，范围为0到6。
是否节假日：{case["is_holiday"]}
温度特征：{case["temperature"]}
湿度特征：{case["humidity"]}
风速特征：{case["windspeed"]}
停车场利用类型编号：{case["utilization_type"]}
规划区域编号：{case["planning_area"]}
道路密度特征：{case["road_density"]}

请只输出未来12个时间步的可用车位数量预测结果。

注意：
1. 这里的停车可用性表示“可用车位数量”，不是0到1之间的比例；
2. 输出可以大于1，但不能为负数；
3. 输出数值不能超过停车场总容量；
4. 请不要输出百分号；
5. 不要输出解释文字，只输出指定格式；
6. 输出数值应结合历史趋势，不要照抄示例。

格式如下：
未来12个时间步的可用车位数量预测为：{example_values}。
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


def extract_available_lots_list(text, lot_capacity):
    """
    从模型输出中提取未来12个时间步的可用车位数量。

    如果输出负数或超过容量，返回空列表，让系统走兜底预测。
    """
    match = re.search(r"\[([0-9.,\s]+)\]", text)

    if not match:
        return []

    number_text = match.group(1)
    numbers = re.findall(r"\d+\.\d+|\d+", number_text)

    if len(numbers) < 12:
        return []

    values = []

    for num in numbers[:12]:
        value = float(num)

        if value < 0 or value > lot_capacity:
            return []

        values.append(round(value, 4))

    return values


def fallback_prediction(history_available_lots, lot_capacity):
    """
    如果模型输出不合理，则使用最近趋势进行兜底预测。
    """
    if not history_available_lots:
        return [round(lot_capacity * 0.5, 4)] * 12

    last_value = history_available_lots[-1]

    if len(history_available_lots) >= 2:
        trend = history_available_lots[-1] - history_available_lots[-2]
    else:
        trend = 0

    max_change_per_step = max(1.0, lot_capacity * 0.08)

    if trend > max_change_per_step:
        trend = max_change_per_step
    elif trend < -max_change_per_step:
        trend = -max_change_per_step

    result = []

    for i in range(12):
        value = last_value + trend * (i + 1)
        value = max(0.0, min(lot_capacity, value))
        result.append(round(value, 4))

    return result


def get_occupancy_level_from_ratio(availability_ratio):
    """
    根据可用性比例判断拥堵等级。
    可用性越低，说明越拥堵。
    """
    if availability_ratio <= 0.25:
        return "high"
    elif availability_ratio <= 0.55:
        return "medium"
    else:
        return "low"


def availability_ratio_to_penalty(availability_ratio):
    """
    可用性比例 -> 拥堵惩罚系数。
    """
    if availability_ratio <= 0.25:
        return 0.5
    elif availability_ratio <= 0.55:
        return 0.25
    else:
        return 0.08


def predict_next_12_steps(case, parking_metadata, start_time_step):
    """
    调用模型预测未来12个时间步。
    """
    lot_capacity = parking_metadata["total_lots"]

    prompt = build_prompt(
        case=case,
        parking_metadata=parking_metadata,
        start_time_step=start_time_step
    )

    raw_output = run_model(prompt)

    predicted_available_lots = extract_available_lots_list(
        raw_output,
        lot_capacity
    )

    used_fallback = False

    if len(predicted_available_lots) < 12:
        predicted_available_lots = fallback_prediction(
            case["history_available_lots"],
            lot_capacity
        )
        used_fallback = True
        raw_output = raw_output + "\n[系统提示] 模型输出未解析出合理的12步结果，已使用趋势兜底结果。"

    predicted_availability_ratio = convert_lots_to_ratio(
        predicted_available_lots,
        lot_capacity
    )

    return predicted_available_lots, predicted_availability_ratio, raw_output, used_fallback


def build_24h_prediction():
    """
    生成未来24小时预测结果。

    现在的逻辑：
    1. 从 SUMO parking_final.add.xml 中读取小组停车场总容量；
    2. 从 SINPA 中读取不同时间段的历史变化模式；
    3. 将 SINPA 历史比例映射到小组停车场容量；
    4. 每3小时重新读取一次真实历史模式，避免滚动预测一路变成0；
    5. 拼接成未来24小时的96个时间片。
    """

    parking_metadata = load_parking_metadata_from_sumo()
    x = load_sinpa_data()

    # 第一版使用 SINPA 中第0个停车场作为历史变化参考模式
    sinpa_lot_index = 0

    daily_prediction = []
    raw_model_outputs = []

    for round_index in range(8):
        current_start_time_step = round_index * 12

        sample_index = find_sample_by_time_step(
            x=x,
            target_time_step=current_start_time_step,
            lot_index=sinpa_lot_index
        )

        case = scale_sinpa_history_to_sumo_capacity(
            x=x,
            sample_index=sample_index,
            sinpa_lot_index=sinpa_lot_index,
            sumo_capacity=parking_metadata["total_lots"]
        )

        (
            predicted_available_lots,
            predicted_availability_ratio,
            raw_output,
            used_fallback
        ) = predict_next_12_steps(
            case=case,
            parking_metadata=parking_metadata,
            start_time_step=current_start_time_step
        )

        raw_model_outputs.append({
            "round": round_index + 1,
            "start_time_step": current_start_time_step,
            "start_time": time_step_to_clock(current_start_time_step),
            "sinpa_sample_index": sample_index,
            "used_fallback": used_fallback,
            "raw_model_output": raw_output
        })

        for i in range(12):
            time_step = current_start_time_step + i
            available_lots = predicted_available_lots[i]
            availability_ratio = predicted_availability_ratio[i]

            item = {
                "time": time_step_to_clock(time_step),
                "time_step": time_step,
                "predicted_available_lots": available_lots,
                "predicted_availability_ratio": availability_ratio,
                "predicted_occupancy_level": get_occupancy_level_from_ratio(
                    availability_ratio
                ),
                "congestion_penalty": availability_ratio_to_penalty(
                    availability_ratio
                )
            }

            daily_prediction.append(item)

    average_congestion_penalty = round(
        sum(item["congestion_penalty"] for item in daily_prediction) / len(daily_prediction),
        4
    )

    result = {
        "model": "Qwen2.5-3B-Instruct + LoRA",
        "dataset": "SINPA + SUMO parking_final",
        "data_source": {
            "training_data": "real_datasets/SINPA/train.npz",
            "parking_structure": parking_metadata["source_file"]
        },

        "lot_id": parking_metadata["lot_id"],
        "lot_name": parking_metadata["lot_name"],
        "lot_capacity": parking_metadata["total_lots"],
        "parking_space_count": parking_metadata["parking_space_count"],

        "time_interval_minutes": 15,
        "total_steps": 96,
        "prediction_range": "24h",

        "daily_prediction": daily_prediction,
        "average_congestion_penalty": average_congestion_penalty,
        "raw_model_outputs": raw_model_outputs
    }

    return result


def main():
    result = build_24h_prediction()

    output_path = RESULT_DIR / "predict_24h.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n24小时预测结果已保存到：{output_path}")

    metadata_path = DATA_DIR / "parking_metadata.json"
    print(f"停车场元数据已保存到：{metadata_path}")


if __name__ == "__main__":
    main()