# 队员A工作更新记录：基于大模型的停车可用性预测模块

## 一、工作背景

本项目围绕“基于 IPv6 + 多源感知的智慧停车协同调度系统”展开，我主要负责队员A部分，即大模型预测模块。该模块的核心任务是根据停车场历史状态和环境特征，预测未来一段时间内的停车可用性变化，并进一步生成拥堵惩罚系数，为队员C的路径规划模块提供决策依据。

在前期工作中，已经完成了 SINPA 真实停车数据集的读取、特征理解、数据转换，以及 Qwen2.5-3B 模型的本地部署和 LoRA 微调。后续工作重点从单纯的数据集实验，转向与小组实际 SUMO 停车场场景的结合，使预测结果能够真正服务于本项目中的停车场仿真环境。

## 二、真实数据与小组停车场结构的结合

本阶段使用了两类数据：

```text
1. SINPA 真实停车数据集
2. 小组自建 SUMO 停车场文件 parking_final.add.xml
```

SINPA 数据主要用于提供真实停车场历史变化模式。数据中包含停车场过去 12 个时间步的历史状态，以及未来 12 个时间步的目标预测值。每个时间步间隔为 15 分钟，因此 12 个时间步对应 3 小时。

小组的 `parking_final.add.xml` 文件则用于提供本项目实际使用的 SUMO 停车场结构。该文件中包含大量 `<parkingArea>` 节点，每个节点对应一个车位，记录了车位编号、所在 lane、起止位置、车位尺寸、角度以及 `roadsideCapacity` 等信息。通过解析这些节点，可以统计出小组停车场的真实车位容量，并提取每个车位在仿真路网中的位置。

这样处理后，模型不再只针对 SINPA 数据集中的停车场编号进行预测，而是将 SINPA 中的停车变化比例映射到小组 SUMO 停车场容量上，使预测结果更贴合小组最终使用的停车场场景。

## 三、停车场结构信息提取

为了让预测模块适配小组自己的停车场，我对 `batch_test_sinpa.py` 进行了修改，使其能够自动读取 `parking_final.add.xml` 文件，并从中提取停车场基础信息。

程序主要提取以下内容：

```text
parking_area_id：车位编号
lane：车位所在车道
capacity：单个 parkingArea 的容量
start_pos：车位在车道上的起始位置
end_pos：车位在车道上的结束位置
width：车位宽度
length：车位长度
angle：车位角度
```

同时，程序会统计所有 `parkingArea` 的 `roadsideCapacity` 总和，作为小组停车场的总容量。这样，预测结果中的 `lot_capacity` 就不再是 SINPA 数据中的估算容量，而是小组 SUMO 停车场对应的真实容量。

运行脚本后，会生成停车场结构信息文件：

```text
data/batch_sumo_parking_info.json
```

该文件可以交给队员C和队员D使用。队员C可以根据其中的 `lane` 和 `parking_area_id` 确定目标车位位置；队员D可以将该文件作为后端或数据库中的停车位基础信息。

## 四、三个高峰场景预测

最终阶段没有采用完整 24 小时预测作为主要输出，而是选择了三个更适合当前系统联调和报告展示的典型场景：

```text
morning_peak：早高峰场景
evening_peak：晚高峰场景
night_leave：夜间高峰 / 夜间离场场景
```

选择这三个场景的原因是，它们更能体现不同停车需求下停车场可用性变化的差异，也更方便与路径规划模块进行联调。每个场景预测未来 12 个时间步，每个时间步为 15 分钟，因此单个场景覆盖未来 3 小时。

脚本运行后会生成以下三个预测结果文件：

```text
results/predict_morning_peak.json
results/predict_evening_peak.json
results/predict_night_leave.json
```

每个 JSON 文件中主要包含以下字段：

```text
scene：预测场景
lot_id：停车场编号
lot_capacity：停车场总容量
input_history_available_lots：过去12个时间步的可用车位数
input_history_availability_ratio：过去12个时间步的可用性比例
predicted_available_lots：未来12个时间步的预测可用车位数
predicted_availability_ratio：未来12个时间步的预测可用性比例
predicted_occupancy_level：预测拥堵等级
congestion_penalty_series：未来12个时间步的拥堵惩罚系数
average_congestion_penalty：平均拥堵惩罚系数
raw_model_output：模型原始输出
```

其中，`congestion_penalty_series` 是后续路径规划中最重要的字段。队员C可以根据车辆预计到达时间，选择对应时间步的惩罚系数，并将其加入路径成本计算。

例如：

```text
cost = distance × emission_factor × (1 + congestion_penalty)
```

这样，路径规划不仅考虑距离和碳排放，还可以考虑停车场未来一段时间的拥堵程度。

## 五、预测异常问题与修正

在测试过程中，曾经出现过模型输出结果不合理的问题。比如模型会照着 prompt 示例输出固定序列：

```text
[149.0, 148.0, 147.0, ...]
```

但某些停车场容量只有 50 或 54，导致程序把所有超过容量的预测值都强行截断为最大容量，最终 JSON 中出现大量完全相同的数据。这种结果显然不适合作为报告展示或系统联调使用。

针对这个问题，我对代码进行了两方面修改。

首先，prompt 中不再使用固定示例值，而是根据当前历史可用车位数动态生成示例。这样可以避免模型机械照抄示例，让输出结果更接近当前场景的实际变化趋势。

其次，解析模型输出时，如果发现预测值为负数，或者超过停车场总容量，程序不再直接截断，而是将其判定为异常输出，并启用趋势兜底预测。兜底预测会根据最近历史变化趋势生成未来 12 个时间步的预测结果，从而避免出现全部变成容量上限或全部异常的情况。

经过修改后，三个场景的预测结果更加合理，也更适合后续报告分析和模块对接。

## 六、与队员C的对接

队员C主要负责路径规划模块，因此我需要提供给他的不是模型训练过程，而是预测结果和停车场结构信息。

需要提供的文件包括：

```text
results/predict_morning_peak.json
results/predict_evening_peak.json
results/predict_night_leave.json
data/batch_sumo_parking_info.json
```

其中，三个 `predict_*.json` 文件分别对应早高峰、晚高峰和夜间离场三个场景。队员C主要读取其中的：

```text
congestion_penalty_series
average_congestion_penalty
predicted_availability_ratio
```

如果车辆预计 30 分钟后到达停车场，则可以取第 `30 / 15 = 2` 个时间步对应的拥堵惩罚系数，用于路径成本计算。

`batch_sumo_parking_info.json` 则提供车位结构信息，包括每个车位的编号、所在 lane 和位置等。队员C可以根据该文件确定目标车位所在车道，再结合预测结果中的拥堵惩罚系数进行路径规划。

## 七、与队员D的对接

队员D主要负责后端、数据库或系统集成相关内容。对于队员D，我提供这些文件的作用是让他能够把预测模块结果接入整个系统，而不是让他重新进行预测。

其中：

```text
data/batch_sumo_parking_info.json
```

可以作为停车场基础结构数据，导入数据库或作为后端接口返回内容。它告诉系统当前停车场一共有多少车位、每个车位在哪里、属于哪条 lane。

三个预测结果文件：

```text
results/predict_morning_peak.json
results/predict_evening_peak.json
results/predict_night_leave.json
```

则可以作为大模型预测模块输出结果，由后端保存或封装为接口，供路径规划模块和前端展示模块调用。

因此，队员D可以用这些文件完成以下工作：

```text
1. 建立停车位基础信息表；
2. 保存三个场景下的预测结果；
3. 为队员C提供预测结果接口；
4. 为前端展示停车场未来拥堵程度提供数据；
5. 完成系统整体联调。
```

## 八、结果图表制作

为了方便后续撰写报告，我根据三个场景的预测结果制作了多张图表。图表主要用于展示不同场景下停车场可用性和拥堵惩罚系数的变化趋势。

生成的图表包括：

```text
chart_1_predicted_available_lots.png
chart_2_availability_ratio.png
chart_3_congestion_penalty_series.png
chart_4_average_congestion_penalty.png
chart_5_morning_peak_history_prediction.png
chart_5_evening_peak_history_prediction.png
chart_5_night_leave_history_prediction.png
```

这些图表分别展示了：

```text
1. 三个场景未来可用车位数对比；
2. 三个场景未来可用性比例对比；
3. 三个场景拥堵惩罚系数对比；
4. 三个场景平均拥堵惩罚系数对比；
5. 早高峰历史数据与预测趋势；
6. 晚高峰历史数据与预测趋势；
7. 夜间高峰 / 夜间离场历史数据与预测趋势。
```

从图表结果可以看出，早高峰场景下可用车位数下降较明显，拥堵惩罚系数逐步升高，整体拥堵程度较高；晚高峰场景下可用车位数也呈下降趋势，但整体紧张程度低于早高峰；夜间高峰 / 夜间离场场景整体处于中等拥堵水平。通过这些图表，可以更加直观地展示大模型预测结果对路径规划模块的支撑作用。

## 九、当前阶段成果总结

目前，队员A部分已经完成了从数据处理、模型微调、预测脚本封装，到小组 SUMO 停车场适配、结果 JSON 输出、图表制作和模块对接说明等工作。

当前阶段的主要成果包括：

```text
1. 完成 SINPA 真实数据读取与特征理解；
2. 完成 Qwen2.5-3B + LoRA 微调与推理测试；
3. 完成 batch_test_sinpa.py 批量预测脚本修改；
4. 完成与 SUMO 停车场 parking_final.add.xml 的适配；
5. 生成早高峰、晚高峰、夜间高峰 / 夜间离场三个典型场景预测结果；
6. 生成停车场结构信息 JSON；
7. 完成给队员C和队员D的对接说明；
8. 制作报告所需的预测结果图表。
```

整体来看，预测模块已经从单纯的数据集实验，进一步转向了与小组实际系统场景结合的应用阶段。当前系统主要围绕三个典型高峰场景输出预测结果，为路径规划模块提供未来 3 小时内的可用车位变化和拥堵惩罚系数。后续如果系统需要扩展到全天预测或实时预测，可以在现有三个场景预测流程的基础上继续拓展。