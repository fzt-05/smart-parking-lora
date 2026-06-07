# 队员A 第一天实验记录

## 一、今日目标

本人负责智慧停车协同调度系统中的 AI 大模型微调模块，主要任务是基于停车场历史数据完成 LoRA 微调流程，为后续车流预测接口提供模型基础。

第一天的目标是完成 Mac 本地环境搭建、数据集格式转换、Qwen 小模型测试以及 LoRA 微调流程验证，确保后续可以继续基于更完整的数据集进行训练和优化。

## 二、开发环境

- 电脑系统：macOS
- 开发工具：VS Code、Terminal
- Python 版本：Python 3.13
- 项目路径：`/Users/fziteng/Documents/Code/python/smart-parking-lora`
- 微调工具：MLX-LM
- 基座模型：Qwen2.5-0.5B-Instruct
- 主要依赖：`mlx-lm`、`pandas`、`numpy`、`transformers`、`peft`、`torch`

项目目录结构如下：

```text
smart-parking-lora/
├── data
│   ├── parking_data.csv
│   ├── train.jsonl
│   └── valid.jsonl
├── scripts
│   └── convert_to_lora_data.py
├── models
│   └── qwen2.5-0.5b-mlx
├── adapters
│   └── parking-lora
└── notes
    └── day1_log.md
```

## 三、完成内容

### 1. 创建项目目录

首先创建了 `smart-parking-lora` 项目目录，并按照数据、脚本、模型、LoRA 权重和实验记录进行分类管理，方便后续继续开发和提交。

主要目录包括：

- `data`：用于存放停车场历史数据和训练数据；
- `scripts`：用于存放数据转换脚本；
- `models`：用于存放基座模型；
- `adapters`：用于存放 LoRA 微调后的 adapter 权重；
- `notes`：用于记录每日实验过程。

### 2. 配置 Python 虚拟环境

在项目目录下创建并激活了 Python 虚拟环境，避免项目依赖和系统环境混在一起。随后安装了 MLX-LM、pandas、numpy、transformers、peft、torch 等相关依赖。

由于本人使用的是 Mac 电脑，传统 Unsloth 微调方式对 CUDA/NVIDIA GPU 依赖较强，因此本项目第一阶段改为使用更适合 Mac 的 MLX-LM 工具链完成 LoRA 微调。

### 3. 准备停车场历史数据

在 `data` 目录下准备了 `parking_data.csv` 文件，数据字段包括：

- `time`：当前时间；
- `weekday`：星期；
- `weather`：天气；
- `enter_count`：当前进入车辆数；
- `leave_count`：当前离开车辆数；
- `occupancy_rate`：当前车位占用率；
- `next_enter_count`：下一小时进入车辆数；
- `next_occupancy_rate`：下一小时车位占用率。

该数据用于模拟停车场在不同时间、天气和占用率下的车流变化情况，为后续 LoRA 微调提供训练样本。

### 4. 编写数据转换脚本

编写了 `scripts/convert_to_lora_data.py` 脚本，将 CSV 格式的停车场历史数据转换为适合大模型微调的 JSONL 格式。

转换后的每条数据包含 `prompt` 和 `completion` 两部分：

- `prompt`：输入当前停车场状态，包括时间、星期、天气、进入车辆数、离开车辆数和当前占用率；
- `completion`：输出下一小时进入车辆数和下一小时车位占用率预测结果。

脚本运行后成功生成：

```text
data/train.jsonl
data/valid.jsonl
```

其中 `train.jsonl` 用于模型训练，`valid.jsonl` 用于训练过程中的验证。

### 5. 测试 Qwen 小模型

完成数据转换后，使用 Qwen2.5-0.5B-Instruct 作为第一阶段基座模型。该模型参数规模较小，更适合在 Mac 本地进行初步验证。

通过 MLX-LM 对模型进行转换和测试，确认模型可以在本地环境中正常加载和推理，为后续 LoRA 微调打下基础。

### 6. 完成 LoRA 微调流程

在完成数据准备和模型测试后，使用 MLX-LM 对 Qwen2.5-0.5B-Instruct 进行了 LoRA 微调实验。

微调命令中指定了训练数据路径、batch size、训练轮数、学习率和 adapter 保存路径。训练完成后，成功在 `adapters/parking-lora` 目录下生成 LoRA adapter 权重。

本次微调的重点不是追求预测精度，而是验证 Mac 本地是否能够跑通完整流程：

```text
CSV 数据集 → JSONL 微调数据 → Qwen 基座模型 → LoRA 微调 → adapter 权重保存
```

目前该流程已经基本跑通。

## 四、今日产出

今日完成的主要产出如下：

```text
data/parking_data.csv
data/train.jsonl
data/valid.jsonl
scripts/convert_to_lora_data.py
models/qwen2.5-0.5b-mlx
adapters/parking-lora
notes/day1_log.md
```

其中，`train.jsonl` 和 `valid.jsonl` 是后续继续微调和优化模型的基础数据文件；`parking-lora` 是本次 LoRA 微调后的 adapter 权重文件夹。

## 五、遇到的问题与解决方法

### 1. Mac 本地不适合直接使用 Unsloth

原计划是跑通 Unsloth 官方 LoRA 微调示例，但 Unsloth 对 CUDA/NVIDIA GPU 环境依赖较强，而本人使用的是 Mac，因此直接使用 Unsloth 容易出现兼容问题。

解决方法：将技术路线调整为 MLX-LM，在 Mac 本地完成 Qwen 小模型的 LoRA 微调。

### 2. Python 环境和依赖识别问题

在 VS Code 中曾出现 `pandas`、`sklearn` 无法解析的问题，主要原因是解释器没有选择到项目中的 `.venv` 虚拟环境。

解决方法：在 VS Code 中选择项目路径下的解释器：

```text
/Users/fziteng/Documents/Code/python/smart-parking-lora/.venv/bin/python
```

同时在终端中激活项目虚拟环境后再安装依赖。

### 3. 模型下载和转换较慢

在下载 Qwen 模型和进行模型转换时，可能会受到网络环境影响，出现连接超时或下载不完整的问题。

解决方法：清理不完整缓存，并使用镜像源或重新下载模型，确保模型文件完整后再进行转换和微调。

## 六、实验总结

第一天主要完成了 AI 大模型微调模块的基础验证工作。通过本次实验，已经初步跑通了从停车场历史数据准备、数据格式转换、基座模型测试到 LoRA 微调的完整流程。

本次实验说明，在 Mac 本地环境下，可以使用 MLX-LM 替代传统 CUDA 微调方案，实现轻量级 Qwen 模型的 LoRA 微调。虽然当前使用的数据规模较小，预测效果还需要进一步优化，但整体流程已经具备继续扩展的基础。

后续可以在队长提供更完整的 `historical_data.csv` 后，替换当前模拟数据，并继续增加训练样本数量，提高模型对高峰时段车流变化和停车场占用率变化的预测能力。

## 七、明日计划

1. 使用更多停车场历史数据扩充训练集；
2. 测试微调后模型在不同时间段、天气和占用率条件下的预测结果；
3. 编写 `predict.py` 推理脚本，实现输入当前车流状态，输出预测结果；
4. 将预测结果整理为 JSON 格式，方便后续交给队员D进行后端接口封装；
5. 为队员C提供拥堵预测结果，用于碳感知路径规划模块。