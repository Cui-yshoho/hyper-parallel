# GR00T N1.7 Recipe

以昇腾 NPU 为目标，将 LeRobot 的 GR00T N1.7 模型和数据处理接入 Hyper。首轮完成下游 SFT、策略导出与 LIBERO 评测，再根据实际接入经验提取公共组件。

## 当前进展

已确定 GR00T N1.7 与 OpenWAM-α 为首轮模型，并完成 GR00T 的初步源码分析。当前分支保存总体方案与首个 Recipe 的接入设计，尚未增加训练实现或运行训练实验。

接下来需要确定昇腾验证环境的软件版本及本地权重、processor 和数据路径，再实现模型与数据薄适配。训练正确性、FSDP、自恢复和 LIBERO 任务效果均按下文顺序验证。

## 首版范围

- 复用 LeRobot 的模型、pre／postprocessor、动作表示、归一化统计及冻结策略。
- 由 Hyper 组织训练、梯度更新、checkpoint 和分布式执行。
- 先验证 NPU 单卡前向、反向和一次参数更新，再验证多卡 FSDP；TP／CP／EP 另行接入。
- 使用已有本地权重和数据，训练入口不隐式下载资源。

默认保持上游冻结视觉与语言骨干、训练 projector 和动作分支的方式。参数使用 FP32，计算使用 BF16，梯度归约使用 FP32；具体可训练参数集合需与上游配置逐项对应。

## 组件与调用关系

```text
LeRobotDataset
    ↓ 上游 preprocessor：图像、语言、状态、动作与有效位置
GR00T N1.7：视觉语言骨干 → projector／动作分支
    ↓ 上游 flow-matching 损失
Hyper：梯度累积 → 梯度裁剪 → 优化器与调度器更新
    ↓
训练 checkpoint + 可部署策略与处理资产
    ↓ 上游 postprocessor：动作解码与反归一化
LIBERO 闭环评测
```

模型接入放在模型侧，Recipe 组织模型与数据配置；Trainer 只依赖通用构建和训练步骤接口。首版不将 GR00T 的字段或类型分支写进通用 Trainer。

## 需要落实的接入边界

| 部分 | 现有行为 | 接入要求 |
|---|---|---|
| 模型构建 | LeRobot `GrootPolicy` 组合自定义 `GR00TN17` 与 Transformers 视觉语言骨干 | 保留完整架构和权重名称，明确 policy 外层与可分片模型的关系 |
| 前向与 FSDP | `GrootPolicy.forward` 直接调用内部模型的 `.forward` | 选择实际经过 `__call__` 的分片边界，验证冻结和可训练分支的参数加载、反向与重分片 |
| 数据处理 | 上游 processor 负责动作 mask、机器人形态、视觉语言输入和归一化 | 复用处理流程，验证字段、shape、device 与动作时序；不用 VLM 文本 packing 替代动作 batch |
| 损失 | 动作速度预测的逐元素 MSE，经 `action_mask` 屏蔽后除以有效元素数 | 保留上游目标；明确梯度累积与 DP 的归约语义，避免再次按文本 token 数归一化 |
| NPU Attention | 上游 FlashAttention 是可选路径 | 首轮关闭 CUDA FlashAttention，检查实际选择的 Attention 后端；用 NPU 前反向验证可用性，再评估融合算子 |
| 导出 | 推理依赖模型配置、权重、pre／postprocessor 与统计信息 | 分别验证训练状态恢复和上游策略重载，不能只验证权重文件保存成功 |

## 实现与验收顺序

1. **固定环境和资产**：确认 NPU 型号、Torch／TorchNPU／CANN、Transformers／LeRobot 版本，以及权重、processor 和数据路径。
2. **实现模型与数据薄适配**：用同一份真实 batch 比较上游路径与 Hyper 接入路径，保持输入处理、mask、冻结参数和损失语义一致。
3. **验证单卡更新**：检查有限 loss、应训练参数的梯度和更新、冻结参数保持不变，以及图像和动作字段的实际参与情况。
4. **接入 FSDP**：在相同有效 batch 与可重放输入下比较单卡和多卡的损失、梯度和更新，并验证 checkpoint 保存与恢复。
5. **导出与评测**：用上游策略接口重载导出资产，接入 LIBERO，分别报告训练正确性与任务成功率。

先完成 GR00T Recipe 所需的最小接口；待 OpenWAM-α 接入后，再比较是否应沉淀为公共具身 Trainer／Kit。

## 代码参考

- [LeRobot GR00T 配置](https://github.com/huggingface/lerobot/blob/b07adcc5e3ce42c26a248236dbb3571f84c28c43/src/lerobot/policies/groot/configuration_groot.py)
- [Policy 封装与训练前向](https://github.com/huggingface/lerobot/blob/b07adcc5e3ce42c26a248236dbb3571f84c28c43/src/lerobot/policies/groot/modeling_groot.py)
- [模型与动作损失](https://github.com/huggingface/lerobot/blob/b07adcc5e3ce42c26a248236dbb3571f84c28c43/src/lerobot/policies/groot/groot_n1_7.py)
- [Processor](https://github.com/huggingface/lerobot/blob/b07adcc5e3ce42c26a248236dbb3571f84c28c43/src/lerobot/policies/groot/processor_groot.py)

Hyper 复用点以 `157be990` 的 `VLMTrainer`、`BaseTrainer` 为起点。模型构建后已经包含并行处理，接入时需要保持构建与分片只有一个所有者。`BaseTrainer.prepare_optimizer_step` 包含文本 token 归一化，具身训练需要独立确定损失归约，再复用梯度裁剪与优化器更新能力。
