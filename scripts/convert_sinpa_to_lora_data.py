import numpy as np
import json
from pathlib import Path

BASE_DIR = Path("/Users/fziteng/Documents/Code/python/smart-parking-lora")

TRAIN_PATH = BASE_DIR / "real_datasets" / "SINPA" / "train.npz"
VAL_PATH = BASE_DIR / "real_datasets" / "SINPA" / "val.npz"

OUTPUT_DIR = BASE_DIR / "data" / "sinpa_lora"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_NAMES = [
    "parking_availability",
    "time_of_day",
    "weekday",
    "is_holiday",
    "temperature",
    "humidity",
    "windspeed",
    "utilization_type",
    "planning_area",
    "road_density",
    "latitude",
    "longitude"
]


def round_list(values, digits=4):
    return [round(float(v), digits) for v in values]


def build_sample(x_item, y_item, lot_id):
    """
    x_item: shape = (12, 1687, 12)
    y_item: shape = (12, 1687, 1)
    lot_id: 停车场编号
    """

    lot_history = x_item[:, lot_id, :]   # shape = (12, 12)
    lot_future = y_item[:, lot_id, 0]    # shape = (12,)

    history_pa = round_list(lot_history[:, 0])
    future_pa = round_list(lot_future)

    latest = lot_history[-1]

    time_of_day = int(round(latest[1]))
    weekday = int(round(latest[2]))
    is_holiday = int(round(latest[3]))

    temperature = round(float(latest[4]), 4)
    humidity = round(float(latest[5]), 4)
    windspeed = round(float(latest[6]), 4)

    utilization_type = int(round(latest[7]))
    planning_area = int(round(latest[8]))
    road_density = round(float(latest[9]), 4)
    latitude = round(float(latest[10]), 4)
    longitude = round(float(latest[11]), 4)

    prompt = (
        "你是智慧停车场车位可用性预测助手。"
        "请根据新加坡停车场过去3小时的历史数据和当前特征，预测未来3小时的停车可用性变化。\n\n"
        f"停车场编号：lot_{lot_id}\n"
        f"过去12个时间步的停车可用性：{history_pa}\n"
        f"当前时间段编号：{time_of_day}，范围为0到95，表示一天中的15分钟时间片。\n"
        f"星期编号：{weekday}，范围为0到6。\n"
        f"是否节假日：{is_holiday}\n"
        f"温度特征：{temperature}\n"
        f"湿度特征：{humidity}\n"
        f"风速特征：{windspeed}\n"
        f"停车场利用类型编号：{utilization_type}\n"
        f"规划区域编号：{planning_area}\n"
        f"道路密度特征：{road_density}\n"
        f"纬度特征：{latitude}\n"
        f"经度特征：{longitude}\n\n"
        "请输出未来12个时间步的停车可用性预测结果。"
    )

    completion = (
        f"未来12个时间步的停车可用性预测为：{future_pa}。"
    )

    return {
        "prompt": prompt,
        "completion": completion
    }


def convert_npz_to_jsonl(npz_path, output_path, max_time_samples=300, max_lots=10):
    data = np.load(npz_path)

    x = data["x"]
    y = data["y"]

    print(f"正在处理：{npz_path}")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")

    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        time_sample_count = min(max_time_samples, x.shape[0])
        lot_count = min(max_lots, x.shape[2])

        for i in range(time_sample_count):
            for lot_id in range(lot_count):
                item = build_sample(x[i], y[i], lot_id)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

    print(f"已生成：{output_path}")
    print(f"样本数量：{count}")


def main():
    convert_npz_to_jsonl(
        TRAIN_PATH,
        OUTPUT_DIR / "train.jsonl",
        max_time_samples=300,
        max_lots=10
    )

    convert_npz_to_jsonl(
        VAL_PATH,
        OUTPUT_DIR / "valid.jsonl",
        max_time_samples=60,
        max_lots=10
    )


if __name__ == "__main__":
    main()