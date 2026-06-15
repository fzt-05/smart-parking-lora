# 队员A 预测结果 JSON 格式说明

## 一、用途

该 JSON 文件是智慧停车协同调度系统中大模型车位可用性预测模块的输出结果。

预测结果后续主要提供给两个模块：

1. 队员D：用于封装后端 `/predict` 接口；
2. 队员C：用于路径规划中的拥堵惩罚系数计算。

## 二、模型与数据来源

- 基座模型：Qwen2.5-3B-Instruct
- 微调方式：LoRA
- 微调工具：MLX-LM
- 数据集：SINPA 新加坡停车场可用性数据集
- 输入历史长度：12 个时间步
- 预测未来长度：12 个时间步

SINPA 数据以 15 分钟为一个时间步，因此：

```text
12 个历史时间步约等于过去 3 小时；
12 个未来时间步约等于未来 3 小时。
## 三、JSON 字段说明

| 字段名 | 类型 | 含义 |
|---|---|---|
| model | string | 使用的模型和微调方式 |
| dataset | string | 使用的数据集 |
| scene | string | 测试场景名称 |
| lot_id | string | 停车场编号 |
| history_steps | int | 历史输入时间步数量 |
| future_steps | int | 预测未来时间步数量 |
| input_history_availability | list[float] | 输入的历史停车可用性序列 |
| predicted_availability | list[float] | 预测未来 12 个时间步的停车可用性 |
| predicted_occupancy_level | string | 预测拥堵等级，low / medium / high |
| congestion_penalty | float | 拥堵惩罚系数，供路径规划模块使用 |
| raw_model_output | string | 大模型原始输出文本 |

## 四、示例

```json
{
  "model": "Qwen2.5-3B-Instruct + LoRA",
  "dataset": "SINPA",
  "scene": "morning_peak",
  "lot_id": "lot_0",
  "history_steps": 12,
  "future_steps": 12,
  "input_history_availability": [0.52, 0.49, 0.45, 0.41, 0.38, 0.35, 0.31, 0.28, 0.25, 0.22, 0.2, 0.18],
  "predicted_availability": [0.16, 0.15, 0.14, 0.14, 0.13, 0.13, 0.12, 0.12, 0.12, 0.11, 0.11, 0.1],
  "predicted_occupancy_level": "high",
  "congestion_penalty": 0.5,
  "raw_model_output": "未来12个时间步的停车可用性预测为：[...]"
}
## 五、拥堵等级划分规则

由于 SINPA 的预测目标是停车可用性，因此可用性越低，表示停车场越拥堵。

当前第一版划分规则如下：

| 平均预测可用性 | 拥堵等级 | 拥堵惩罚系数 |
|---|---|---|
| <= 0.25 | high | 0.5 |
| 0.25 - 0.55 | medium | 0.25 |
| > 0.55 | low | 0.08 |
其中，`congestion_penalty` 字段将提供给路径规划模块，用于调整路径成本。停车可用性越低，系统认为停车场越拥堵，因此惩罚系数越高。