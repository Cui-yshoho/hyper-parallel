# GR00T N1.7 Recipe

以昇腾 NPU 为目标，将 LeRobot 的 GR00T N1.7 模型和数据处理接入 Hyper。首轮完成下游 SFT、策略导出与 LIBERO 评测，再根据实际接入经验提取公共组件。

## 当前进展

GR00T N1.7 已接入现有 VLM Trainer。用单卡重放同一份 LIBERO batch 和 LeRobot 基线的 RNG，两个 AdamW 参数组、loss、梯度范数、采样的裁剪后梯度及参数更新量均已对齐。

双卡 FSDP 完成了有效动作数不等和梯度累积的单步对照。从第 1 步 checkpoint 重启后，第 2 步与连续训练的数据位置、RNG、loss、梯度范数和采样的参数更新、AdamW 状态一致。第 2 步策略已导出并由 LeRobot 离线严格重载。在 Ascend 单卡和 OSMesa 软件渲染环境中完成了 LIBERO Spatial 任务 0 的一条 280 步闭环，成功数为 0/1。

## 单卡训练基线

在 Python 3.12 或更新版本的昇腾环境安装本仓库和固定版本的 LeRobot。准备原始 NVIDIA GR00T N1.7 分片权重及其处理器配置、统计文件，本地 LeRobot v3.0 格式的 LIBERO Spatial 数据，以及 Hugging Face 缓存中的 `nvidia/Cosmos-Reason2-2B` tokenizer 和 processor 文件。

```bash
python -m pip install -e .
python -m pip install 'lerobot[groot] @ git+https://github.com/huggingface/lerobot.git@b07adcc5e3ce42c26a248236dbb3571f84c28c43'

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/smoke_groot_n17.py \
  --checkpoint /path/to/GR00T-N1.7-3B \
  --processor-dir /path/to/Cosmos-Reason2-2B-processor \
  --dataset-root /path/to/LIBERO_LeRobot_v3/libero_spatial \
  --dataset-repo nvidia/LIBERO_LeRobot_v3 \
  --device npu:0 --seed 42 --baseline-output /path/to/lerobot-step.pt
```

基线固定使用真实数据的首个样本、原始 NVIDIA 权重和 seed 42。权重和优化器使用 FP32，BF16 计算由 GR00T Policy 自己控制。脚本使用 LeRobot 的参数分组和 AdamW preset，并调用 LeRobot 官方的 `update_policy` 来完成反向、梯度裁剪和更新。单步对比使用固定学习率和一个 micro-batch。调度器的行为在循环接入后单独验证。

`--baseline-output` 保存处理后的同一 batch、更新前 CPU/NPU RNG 状态、loss、全局梯度范数、有效动作数、可训练参数名、优化器组配置，以及固定参数采样位置的裁剪后梯度和更新前后值。切换到 Hyper Trainer 时从原始权重重新加载，只在首步重放这个 batch 和 RNG。先比较有效动作数与冻结集合，再比较损失、梯度范数、裁剪后梯度和参数更新量，并检查冻结参数采样值保持一致。

单卡对齐后，用同一有效 batch 对比双卡 FSDP，在两卡有效动作数不等和启用梯度累积时，按整个优化器步上的有效动作元素总数归一化。分别验证 checkpoint 保存与恢复、策略重载和 LIBERO 闭环表现。

## Hyper Trainer 单卡入口

使用 `scripts/train_vl.py` 和 [LIBERO Spatial 配置](../../../hyper_parallel/models/groot_n17/recipes/libero_spatial.yaml)。将下面的资产根目录替换为本地路径，或者直接编辑 YAML 中的资产路径。

```bash
assets=/path/to/assets
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 torchrun --standalone --nproc-per-node=1 \
  scripts/train_vl.py hyper_parallel/models/groot_n17/recipes/libero_spatial.yaml \
  --model.checkpoint="$assets/hf_weights/GR00T-N1.7-3B" \
  --dataset.frame_offsets.checkpoint="$assets/hf_weights/GR00T-N1.7-3B" \
  --model.dataset_root="$assets/data/LIBERO_LeRobot_v3/libero_spatial" \
  --dataset.dataset_root="$assets/data/LIBERO_LeRobot_v3/libero_spatial" \
  --dataset.model_assets.processor_dir="$assets/hf_weights/Cosmos-Reason2-2B-processor"
```

GR00T 使用固定样本数的 DataLoader，组批时保留样本列表。VLM Trainer 的 `get_batch` 阶段堆叠样本字段并运行 LeRobot processor。随后损失组件从动作 mask 统计有效坐标数。单卡且每步只有一个 micro-batch 时，在反向前保留上游已归一化的 loss，以对齐 BF16 反向的舍入行为。AdamW 使用 LeRobot preset 和相同的衰减分组，由 Hyper Trainer 执行梯度裁剪和更新。

## 双卡 FSDP

在上述启动命令中使用 `--nproc-per-node=2`，并追加 `--fsdp_config.dp_shard_size=2 --training.global_batch_size=2`。每卡累积两个 micro-batch 时使用 `--training.global_batch_size=4 --fsdp_config.requires_grad_sync=false`。

视觉块、语言 decoder 层和动作 Transformer 块各自作为子分片单元，外层训练模块负责剩余参数和共享权重。FSDP 保持 FP32 参数和梯度归约，BF16 autocast 由上游 GR00T 管理。验证时对比同一批真实 LIBERO 样本和逐样本 RNG，包含动作窗尾部的 padding。

保存恢复对照使用两个优化器步，每卡累积两个 micro-batch。设置 `--checkpoint.save_ckpt=true --checkpoint.save_steps=1`，在连续运行结束后用同一训练配置和 `--checkpoint.restore_from=/path/to/checkpoints/global_step_1` 重启。逐 rank 核对第 2 步的样本索引、处理后输入、RNG 及采样的梯度和优化器状态。

## 策略导出与闭环评测

导出脚本在 CPU 上读取 DCP 的模型分片，使用 LeRobot 保存策略配置、完整权重、preprocessor、postprocessor 和归一化状态。导出配置引用本地原始 GR00T 权重目录及 Cosmos processor 目录，重载时保留这两项资产的路径。

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/export_groot_n17.py \
  --training-checkpoint /path/to/checkpoints/global_step_2 \
  --checkpoint /path/to/GR00T-N1.7-3B \
  --processor-dir /path/to/Cosmos-Reason2-2B-processor \
  --dataset-root /path/to/LIBERO_LeRobot_v3/libero_spatial \
  --output-dir /path/to/exported-policy
```

在 LeRobot 环境安装 `hf-libero` 并准备 LIBERO 模拟器资产和 MuJoCo 渲染环境。评测入口用官方 LeRobot API 在 CPU 上严格重载权重，然后将策略移动到 Ascend。GR00T 与 OpenWAM 共用 Hyper 的 LIBERO rollout 和指标统计。使用无 GPU 的 Linux 节点时可配置 OSMesa 软件渲染。

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/eval_libero.py --model groot_n17 \
  --policy-dir /path/to/exported-policy \
  --task-ids 0 --episodes 1 --seed 42 \
  --output /path/to/libero-result.json
```

## 首版范围

- 复用 LeRobot 的模型、pre／postprocessor、动作表示、归一化统计及冻结策略。
- 优先复用 Hyper 已有 Trainer，通过必要的扩展组织训练、梯度更新、checkpoint 和分布式执行，保持上游动作损失语义。
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

模型接入放在模型侧，Recipe 组织模型与数据配置；优先沿用现有 Trainer 的执行流程，在必要处增加通用扩展接口。GR00T 的字段与目标语义由接入层承载，避免在通用 Trainer 中增加模型类型分支。

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

GR00T 通过现有 VLM Trainer 和 YAML 配置启动。与 OpenWAM-α 共享的训练能力纳入 Hyper 的公共组件，GR00T 模型接入层保留上游 policy 和 processor 所需的适配。

## 代码参考

- [LeRobot GR00T 配置](https://github.com/huggingface/lerobot/blob/b07adcc5e3ce42c26a248236dbb3571f84c28c43/src/lerobot/policies/groot/configuration_groot.py)
- [Policy 封装与训练前向](https://github.com/huggingface/lerobot/blob/b07adcc5e3ce42c26a248236dbb3571f84c28c43/src/lerobot/policies/groot/modeling_groot.py)
- [模型与动作损失](https://github.com/huggingface/lerobot/blob/b07adcc5e3ce42c26a248236dbb3571f84c28c43/src/lerobot/policies/groot/groot_n1_7.py)
- [Processor](https://github.com/huggingface/lerobot/blob/b07adcc5e3ce42c26a248236dbb3571f84c28c43/src/lerobot/policies/groot/processor_groot.py)

Hyper 复用点以 `157be990` 的 `VLMTrainer`、`BaseTrainer` 为起点。模型构建后已经包含并行处理，接入时需要保持构建与分片只有一个所有者。`BaseTrainer.prepare_optimizer_step` 包含文本 token 归一化，具身训练需要独立确定损失归约，再复用梯度裁剪与优化器更新能力。
