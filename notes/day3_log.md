# 队员A 第三天实验记录

## 一、今日目标

第三天的主要目标是在前两天完成模拟数据 LoRA 微调和基础预测脚本的基础上，引入真实停车场数据集，并将模型升级到更大的基座模型，为后续形成第一版可展示预测模块做准备。

本日重点包括三部分：

1. 查找并确认适合本项目的真实停车数据集；
2. 将 SINPA 真实停车可用性数据转换为适合 LoRA 微调的 JSONL 格式；
3. 将基座模型从 Qwen2.5-0.5B-Instruct 升级到 Qwen2.5-3B-Instruct，并完成模型转换和微调验证。

## 二、开发环境

- 电脑系统：macOS
- 芯片：Apple M3 Max
- 内存：36GB
- 开发工具：VS Code、Terminal
- Python 版本：Python 3.13
- 微调工具：MLX-LM
- 原基座模型：Qwen2.5-0.5B-Instruct
- 升级后基座模型：Qwen2.5-3B-Instruct
- 数据集：SINPA 新加坡停车场可用性数据集
- 项目路径：`/Users/fziteng/Documents/Code/python/smart-parking-lora`

## 三、完成内容

### 1. 查找真实停车场数据集

前两天主要使用模拟停车场历史数据完成 LoRA 微调流程验证。虽然模拟数据可以用于跑通流程，但在项目展示和技术文档中说服力不足，因此第三天开始查找真实停车场数据集。

经过筛选，最终选择 SINPA 数据集作为本项目第一版真实数据来源。SINPA 是一个面向新加坡停车场可用性预测的数据集，包含多个停车场在不同时间段下的停车可用性信息，并结合了时间、天气、区域、道路密度、经纬度等多源特征，和本项目“智慧停车协同调度系统”的车位可用性预测任务较为匹配。

最开始只在仓库中找到了 `lots_location.csv` 文件，该文件只包含停车场经纬度信息，不能直接用于 LoRA 微调。随后查看 README 文件，确认核心数据需要从 Hugging Face 数据集页面单独下载，主要包括：

```text
train.npz
val.npz
test.npz
```

下载完成后，将数据文件放入项目目录：

```text
real_datasets/SINPA/
├── train.npz
├── val.npz
└── test.npz
```

### 2. 检查 SINPA 数据结构

为了确认数据是否可以用于 LoRA 微调，编写并运行了 `scripts/inspect_sinpa.py` 脚本，对 `train.npz` 文件进行检查。

检查结果如下：

```text
文件中的数组名：
['x', 'y']

x: shape=(12167, 12, 1687, 12), dtype=float64
y: shape=(12167, 12, 1687, 1), dtype=float64
```

根据 shape 可以理解为：

```text
x：历史输入数据
y：未来预测目标

12167：训练样本数量
12：历史时间步，每个时间步间隔约15分钟，12步约为过去3小时
1687：停车场数量
12：每个停车场的特征维度
```

`y` 的含义为未来 12 个时间步的停车可用性预测目标，也就是未来约 3 小时的停车可用性变化。

这说明 SINPA 数据集能够用于构建“过去3小时停车场状态 → 未来3小时停车可用性”的预测任务。

### 3. 将 SINPA 数据转换为 LoRA 微调格式

原始 SINPA 数据是四维数组，不能直接输入大语言模型进行 LoRA 微调。因此第三天编写了 `scripts/convert_sinpa_to_lora_data.py` 脚本，将 `.npz` 数据转换为 JSONL 格式。

转换逻辑如下：

```text
输入：
某个停车场过去12个时间步的停车可用性序列
当前时间段编号
星期编号
是否节假日
温度、湿度、风速
停车场利用类型
规划区域
道路密度
纬度、经度

输出：
未来12个时间步的停车可用性预测结果
```

由于原始数据规模较大，如果全部转换会产生大量样本，不利于 Mac 本地快速微调。因此第一版只选取部分样本进行验证：

```text
前300个时间样本 × 前10个停车场
```

转换完成后生成：

```text
data/sinpa_lora/train.jsonl
data/sinpa_lora/valid.jsonl
```

该数据集后续用于替代前两天的模拟数据，作为真实数据版 LoRA 微调输入。

### 4. 升级基座模型

前两天使用的是 Qwen2.5-0.5B-Instruct，优点是体积小、运行快，适合验证流程。但为了提高模型表达能力和项目展示效果，第三天开始升级更大的基座模型。

结合本机配置：

```text
Apple M3 Max + 36GB 内存
```

选择 Qwen2.5-3B-Instruct 作为第一版正式实验模型。

升级路线如下：

```text
Qwen2.5-0.5B-Instruct：用于验证流程
Qwen2.5-3B-Instruct：用于正式实验和第一版成果
```

在下载模型时，曾遇到 Hugging Face 下载不稳定、缓存锁文件、终端多行命令粘贴错误等问题。最终通过先下载 Hugging Face 格式模型，再转换为 MLX 格式的方式完成了模型准备。

模型保存路径为：

```text
models/qwen2.5-3b-hf
models/qwen2.5-3b-mlx
```

其中：

```text
qwen2.5-3b-hf：原始 Hugging Face 格式模型
qwen2.5-3b-mlx：转换后的 MLX 格式模型
```

### 5. 测试 Qwen2.5-3B 模型推理

模型转换完成后，使用 MLX-LM 对 Qwen2.5-3B-Instruct 进行了基础推理测试，确认模型可以正常加载并输出结果。

测试命令主要用于验证：

```text
模型文件是否完整
MLX 格式是否转换成功
Mac 本地是否可以正常运行 3B 模型
```

测试成功后，说明 3B 模型可以作为后续 LoRA 微调的基座模型。

### 6. 基于 SINPA 数据进行 LoRA 微调

完成真实数据转换和 3B 模型准备后，使用 MLX-LM 对 Qwen2.5-3B-Instruct 进行了 LoRA 微调。

微调数据路径为：

```text
data/sinpa_lora
```

LoRA adapter 保存路径为：

```text
adapters/sinpa-qwen3b-lora
```

本次微调的重点是验证完整技术链路：

```text
SINPA 真实数据集
→ JSONL 微调数据
→ Qwen2.5-3B-Instruct 基座模型
→ MLX-LM LoRA 微调
→ adapter 权重保存
→ 微调后模型推理测试
```

目前该流程已经跑通，说明队员A负责的大模型预测模块已经从“模拟数据验证”升级到了“真实数据微调验证”。

## 四、今日产出

第三天完成的主要产出如下：

```text
real_datasets/SINPA/train.npz
real_datasets/SINPA/val.npz
real_datasets/SINPA/test.npz

scripts/inspect_sinpa.py
scripts/convert_sinpa_to_lora_data.py

data/sinpa_lora/train.jsonl
data/sinpa_lora/valid.jsonl

models/qwen2.5-3b-hf
models/qwen2.5-3b-mlx

adapters/sinpa-qwen3b-lora
notes/day3_log.md
```

其中，`data/sinpa_lora/train.jsonl` 和 `data/sinpa_lora/valid.jsonl` 是后续继续进行真实数据微调和模型优化的基础；`adapters/sinpa-qwen3b-lora` 是基于真实数据和 3B 模型训练得到的 LoRA 权重。

## 五、遇到的问题与解决方法

### 1. 只找到停车场位置文件，缺少动态数据

最开始只找到 `lots_location.csv`，其中仅包含停车场的经纬度信息，不能直接用于车位可用性预测。

解决方法：继续阅读 SINPA 项目的 README，确认核心数据集需要从 Hugging Face 单独下载，最终获取到 `train.npz`、`val.npz`、`test.npz` 三个文件。

### 2. 原始数据格式不能直接用于 LoRA 微调

SINPA 原始数据是 `.npz` 格式，内部为四维数组，无法直接输入大语言模型。

解决方法：编写 `convert_sinpa_to_lora_data.py`，将数组数据转换为包含 `prompt` 和 `completion` 的 JSONL 格式，使其能够用于 LoRA 微调。

### 3. 数据量过大，Mac 本地处理压力较大

SINPA 的训练集包含 12167 个时间样本、1687 个停车场，如果全部展开会产生大量样本，不适合作为第一版本地实验。

解决方法：第一版只选取前 300 个时间样本和前 10 个停车场，先完成真实数据微调流程验证。后续可根据训练时间和效果逐步扩大样本范围。

### 4. Qwen2.5-3B 模型下载不稳定

下载模型时出现连接失败、缓存锁文件、资源定位失败等问题。

解决方法：清理不完整缓存，改用本地下载后转换的方式，先将模型下载到 `models/qwen2.5-3b-hf`，再使用 `mlx_lm.convert` 转换到 `models/qwen2.5-3b-mlx`。

### 5. 终端多行命令粘贴错误

执行多行命令时，终端出现 `zsh: bad pattern` 错误，原因是复制命令时混入了特殊字符。

解决方法：使用单行命令重新执行，避免多行粘贴导致的转义字符错误。

## 六、实验总结

第三天的工作主要完成了两个重要升级：一是引入真实停车场数据集 SINPA，二是将基座模型从 Qwen2.5-0.5B 升级到 Qwen2.5-3B。

相比前两天使用模拟数据跑通流程，今天的工作更加接近项目真实需求。SINPA 数据集包含真实停车场可用性数据和多源特征，能够更好地支撑智慧停车系统中的车位可用性预测任务。Qwen2.5-3B 模型相比 0.5B 模型具有更强的语义理解和表达能力，也更适合作为第一版正式实验模型。

目前已经完成：

```text
真实数据下载
真实数据结构分析
真实数据 JSONL 转换
3B 模型转换
3B 模型推理测试
3B + LoRA 微调验证
```

这说明队员A模块已经具备形成第一版预测模块的基础。下一步需要将微调后的模型封装为可调用脚本，输出标准 JSON 结果，方便后端模块和路径规划模块使用。

## 七、明日计划

明天的主要目标是完成第一版可展示预测模块。

计划完成以下任务：

1. 编写正式版 `scripts/predict_sinpa.py`；
2. 使用 Qwen2.5-3B + SINPA LoRA adapter 进行单次预测；
3. 将模型输出整理为标准 JSON 格式；
4. 输出字段包括 `lot_id`、`predicted_availability`、`predicted_occupancy_level`、`congestion_penalty` 和 `raw_model_output`；
5. 编写批量测试脚本，生成早高峰、晚高峰、夜间离场等场景预测结果；
6. 将预测结果保存到 `results` 目录；
7. 编写 `predict_output_format.md`，说明预测结果字段含义；
8. 准备第一版模块汇报材料，交给队长、队员C和队员D进行后续对接。