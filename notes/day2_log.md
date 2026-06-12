# 第二天实验记录

## 一、今日目标

第二天的主要目标是在第一天完成 LoRA 微调流程的基础上，进一步验证微调后的模型是否可以正常用于预测，并将模型输出整理成后续系统模块能够调用的 JSON 格式。

本阶段的重点不是继续扩大训练规模，而是完成从“模型训练完成”到“预测模块可调用”的转化。也就是让微调后的 Qwen 小模型能够根据当前停车场状态，输出下一小时进入车辆数、下一小时车位占用率以及拥堵等级，为后续路径规划模块和后端接口封装提供数据基础。

## 二、开发环境

- 电脑系统：macOS
- 开发工具：VS Code、Terminal
- Python 版本：Python 3.13
- 项目路径：`/Users/fziteng/Documents/Code/python/smart-parking-lora`
- 微调工具：MLX-LM
- 基座模型：Qwen2.5-0.5B-Instruct
- LoRA 权重路径：`adapters/parking-lora`
- 模型路径：`models/qwen2.5-0.5b-mlx`

当前项目目录结构如下：

```text
smart-parking-lora/
├── data
│   ├── parking_data.csv
│   ├── train.jsonl
│   └── valid.jsonl
├── scripts
│   ├── convert_to_lora_data.py
│   └── predict.py
├── models
│   └── qwen2.5-0.5b-mlx
├── adapters
│   └── parking-lora
├── results
│   └── predict_result.json
└── notes
    ├── day1_log.md
    └── day2_log.md
```

## 三、完成内容

### 1. 检查第一天微调产物

首先检查了第一天生成的训练数据、验证数据、基座模型文件和 LoRA adapter 权重文件，确认项目目录结构完整，后续可以继续进行模型推理测试。

主要检查内容包括：

```text
data/train.jsonl
data/valid.jsonl
models/qwen2.5-0.5b-mlx
adapters/parking-lora
```

其中，`train.jsonl` 和 `valid.jsonl` 是由停车场历史数据转换得到的微调数据，`parking-lora` 是第一天训练完成后保存的 LoRA adapter 权重。

### 2. 测试微调后模型的推理效果

在确认 LoRA adapter 存在后，使用 MLX-LM 对微调后的模型进行了基础推理测试。测试输入包括当前时间、星期、天气、进入车辆数、离开车辆数和当前占用率，模型需要输出下一小时的车流预测结果。

测试输入示例：

```text
时间：08:00
星期：Monday
天气：Sunny
当前进入车辆数：45
当前离开车辆数：12
当前占用率：78%
```

模型能够根据输入内容生成预测文本，说明第一天训练得到的 LoRA adapter 可以被正常加载和调用，微调后的模型具备初步预测能力。

### 3. 编写预测脚本 predict.py

为了让模型预测流程更加规范，今天编写了 `scripts/predict.py` 脚本。该脚本通过命令行参数接收当前停车场状态信息，并自动构造 prompt，调用微调后的 Qwen 模型进行预测。

脚本支持输入的参数包括：

```text
--time              当前时间
--weekday           星期
--weather           天气
--enter_count       当前进入车辆数
--leave_count       当前离开车辆数
--occupancy_rate    当前车位占用率
```

通过该脚本，可以避免每次都手动输入较长的模型调用命令，提高后续测试和接口封装的便利性。

### 4. 整理预测结果 JSON 格式

为了方便后续队员D进行后端接口封装，也方便队员C在路径规划模块中使用预测结果，今天将模型输出整理为了 JSON 格式。

预测结果主要包含以下字段：

```json
{
  "time": "08:00",
  "weekday": "Monday",
  "weather": "Sunny",
  "current_enter_count": 45,
  "current_leave_count": 12,
  "current_occupancy_rate": 78,
  "predicted_enter_count": 52,
  "predicted_occupancy_rate": 82,
  "congestion_level": "medium",
  "raw_model_output": "预测下一小时进入车辆数为 52 辆，预测下一小时停车场占用率为 82%。"
}
```

其中，`predicted_enter_count` 表示预测下一小时进入车辆数，`predicted_occupancy_rate` 表示预测下一小时停车场占用率，`congestion_level` 表示根据预测占用率划分出的拥堵等级。

### 5. 保存预测结果文件

在测试过程中，将模型预测结果保存到了 `results` 目录下，方便后续查看和提交。

保存路径为：

```text
results/predict_result.json
```

这样后续进行模块联调时，可以直接读取该 JSON 文件，也可以由后端模块进一步封装成 API 返回结果。

## 四、测试样例

今天主要设计了三类测试样例，用于验证模型在不同场景下的输出情况。

### 1. 早高峰场景

```bash
python scripts/predict.py \
  --time 08:00 \
  --weekday Monday \
  --weather Sunny \
  --enter_count 45 \
  --leave_count 12 \
  --occupancy_rate 78
```

该场景模拟工作日早高峰，车辆进入数量较多，停车场占用率处于较高水平。模型需要预测下一小时占用率是否继续上升。

### 2. 晚高峰场景

```bash
python scripts/predict.py \
  --time 18:00 \
  --weekday Friday \
  --weather Rainy \
  --enter_count 80 \
  --leave_count 25 \
  --occupancy_rate 90
```

该场景模拟周五晚高峰和雨天情况，停车需求较高，当前占用率已经达到较高水平。模型需要判断后续是否会进入高拥堵状态。

### 3. 夜间离场场景

```bash
python scripts/predict.py \
  --time 21:00 \
  --weekday Sunday \
  --weather Cloudy \
  --enter_count 20 \
  --leave_count 55 \
  --occupancy_rate 60
```

该场景模拟夜间车辆离场较多的情况，模型需要预测停车场占用率是否下降。

## 五、今日产出

第二天完成的主要产出如下：

```text
scripts/predict.py
results/predict_result.json
notes/day2_log.md
```

同时，第一天得到的以下文件继续作为今天实验的基础：

```text
data/train.jsonl
data/valid.jsonl
models/qwen2.5-0.5b-mlx
adapters/parking-lora
```

## 六、遇到的问题与解决方法

### 1. 预测结果需要标准化

模型原始输出是自然语言文本，不方便后端和路径规划模块直接使用。因此今天对输出结果进行了处理，将预测内容整理为 JSON 格式。

解决方法：在 `predict.py` 中增加结果解析逻辑，将模型输出中的预测车辆数和占用率提取出来，并补充拥堵等级字段。

### 2. 预测结果需要保存

一开始模型输出只显示在终端中，不会自动形成文件，不方便后续记录和联调。

解决方法：将终端输出重定向保存到 `results/predict_result.json` 文件中，后续也可以在脚本中加入自动保存逻辑。

### 3. 当前预测精度仍有限

目前使用的数据集规模较小，主要是为了跑通微调和推理流程，因此预测结果只能作为初步验证，暂时不能代表真实停车场预测效果。

解决方法：后续需要使用更完整的历史停车数据，增加不同时间段、天气、工作日和节假日场景，提高模型对停车需求变化的学习能力。

## 七、实验总结

第二天主要完成了微调结果的调用与预测模块封装工作。相比第一天只完成数据转换和 LoRA 微调流程，今天进一步将模型结果整理成可以被其他模块使用的形式。

目前已经能够通过命令行输入停车场当前状态，并得到下一小时车流预测结果。预测结果以 JSON 格式输出，具备后续接入 FastAPI 后端接口和路径规划模块的基础条件。

本次实验说明，队员A负责的大模型车流预测模块已经从“训练验证阶段”进入到“可调用模块阶段”。虽然当前数据规模较小，模型预测效果仍需进一步优化，但整体技术链路已经基本跑通。

## 八、明日计划

1. 增加更多训练样本，扩充停车场历史数据；
2. 优化 `predict.py`，让输出 JSON 更稳定；
3. 增加多组测试结果并记录模型表现；
4. 尝试根据预测占用率计算拥堵惩罚系数；
5. 与队员D确认后端接口需要的 JSON 字段；
6. 与队员C确认路径规划模块需要的预测数据格式。