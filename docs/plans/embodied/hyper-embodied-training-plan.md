# Hyper 具身模型训练方案

## 一、背景与参考资料

本方案面向具身模型的**预训练（PT）和监督微调（SFT）**，覆盖数据处理、训练、策略导出与测评。在线强化学习作为后续扩展，本阶段不展开。

### 1. 模型类型与代表模型

**具身智能**面向智能体通过感知和动作与环境交互，是本方案的应用方向。

**世界模型**（World Model，WM）学习环境状态及其变化，用于未来预测、仿真或规划；它可以服务于具身智能，也可以用于视频生成等任务，并不一定输出机器人动作。

**VLA**（Vision-Language-Action，视觉语言动作模型）根据图像、语言指令和机器人状态生成动作。

**WAM**（World-Action Model，世界动作模型）将未来世界预测与动作生成结合起来。不同模型可以联合生成视频和动作，也可以仅在训练时利用视频监督，推理时直接输出动作。

| 类型 | 代表模型 | 主要特点 |
|---|---|---|
| VLA | [π0.5](https://github.com/Physical-Intelligence/openpi) | 根据视觉、语言和状态，通过 flow matching 生成动作块 |
| VLA | [GR00T N1.7](https://huggingface.co/docs/lerobot/groot) | 组合视觉语言骨干与动作 Transformer，支持不同机器人形态，已有 LeRobot 接入 |
| WAM | [FastWAM](https://github.com/yuantianyuan01/FastWAM) | 训练时联合学习视频与动作，基础推理路径省去未来视频生成 |
| WAM | [OpenWAM-α](https://github.com/OpenWAM-Official/OpenWAM) | 视频 DiT 与 ActionDiT 通过联合注意力学习未来视频和动作，在约 6,400 小时的人类第一人称与机器人数据上预训练 |

WAM 将世界建模与动作学习结合起来，是两者的交集；VLA 则不一定显式预测未来世界。因此，框架支持视频生成或世界模型训练，仍需进一步核查其动作训练、策略导出和环境评测能力。

VLA、WAM 描述模型能力，PT、SFT 描述训练阶段。**同一类模型可能采用不同训练目标；不同训练阶段也可能复用相同的训练循环。**

### 2. 代表框架与参考内容

| 框架 | 主要定位 | 本方案参考内容 |
|---|---|---|
| [LeRobot](https://github.com/huggingface/lerobot) | 机器人学习框架，连接数据、策略训练、推理与评测 | Policy／processor 接口、机器人数据、动作处理与策略资产 |
| [Dexbotic](https://github.com/dexmal/dexbotic) | 覆盖 VLA 预训练、微调、推理与评测的开发工具箱 | 实验配置、混合监督，以及复用 HF Trainer 的方式 |
| [OpenWAM-Infra](https://github.com/OpenWAM-Official/OpenWAM) | OpenWAM-α 所属项目的模块化训练、推理与评测基础设施 | OpenWAM-α 的官方数据处理、视频／动作联合目标、预训练权重微调、部署与评测 |
| [RLinf](https://github.com/RLinf/RLinf) | 面向具身与 Agent 的 RL 基础设施，也提供独立 SFT 路径 | 具身模型的分布式 SFT、评测接入及后续 RL 衔接 |
| [VeOmni](https://github.com/ByteDance-Seed/VeOmni) | 多模态与生成模型分布式训练框架 | [Cosmos3-Nano SFT](https://github.com/ByteDance-Seed/VeOmni/pull/1463)：采用 SeedOmni 模块组合，参考其条件编码与在线／离线缓存训练；该路径主要提供视频训练参考 |
| [NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel) | 模型分布式训练框架，已提供具体 WAM 接入 | [Praxis-1](https://github.com/NVIDIA-NeMo/Automodel/pull/4730)：通过 `TrainWAMFinetuneRecipe` 实现视频／动作联合目标、FSDP2 + EP、全量／LoRA 微调、导出与 LIBERO 评测；尚未覆盖 TP／CP／PP，断点恢复仍有显存限制 |
| [FSDPTurbo](https://gitcode.com/Ascend/FSDPTurbo)／[FSDPTraining](https://gitcode.com/Ascend/FSDPTraining) | 并行加速库与配套训练框架，提供图像／音视频生成配方 | [FLUX.3](https://gitcode.com/Ascend/FSDPTraining/blob/main/examples/flux3_image_klein/README.md) 与 [MiniMax-H3](https://gitcode.com/Ascend/FSDPTraining/blob/main/examples/minimax_h3/README.md)：参考模型结构与目标、数据适配、训练编排及并行能力的分工；现有这些配方不包含机器人动作训练与闭环评测 |

本方案结合这些框架的接入经验，**保留官方模型、数据处理与 processor 的语义，使用 Hyper 承载分布式训练和运行管理**。优先复用 Hyper 现有 Trainer，通过模型与数据适配、必要的扩展接口完成 Recipe，保持上游训练目标和损失语义。具体选择由模型的实际调用链决定；随着实现与验证逐步迭代，再判断哪些能力应提取为公共组件、是否需要新增具身 Trainer。

## 二、开发路线

**优先复用已有 Trainer 完成具身训练，将稳定的能力纳入 Hyper 现有公共组件。新增模型优先通过配置接入。**

以四个代表模型驱动开发，按需求差异逐步推进：

| 阶段 | 模型 | 验证目标 |
|---|---|---|
| 第一轮 | GR00T N1.7 | 优先复用 LeRobot 接入，接通 VLA 的数据／processor、动作训练、策略导出与评测 |
| 第一轮 | OpenWAM-α | 从官方预训练权重出发，接通下游 SFT、策略导出与 LIBERO 评测，检验视频／动作联合训练及官方组件接入 |
| 第二轮 | π0.5 | 用不同 VLA 结构和处理流程检验首轮公共接口 |
| 第二轮 | FastWAM | 检验另一种 WAM 的视频／动作联合目标，以及训练和推理路径的差异 |

1. **完成首轮 Recipe**：先完成 GR00T N1.7，再接入 OpenWAM-α 的下游 SFT。各自优先基于已有 Trainer 走通数据、训练、策略导出与评测，保留上游模型与数据处理方式；对已有训练流程的扩展保持最小必要范围。OpenWAM-α 首轮使用公开预训练权重，以 LIBERO 为评测入口；大规模预训练留作后续扩展。
2. **检验与扩展边界**：比较首轮 Recipe 的共性与差异，再通过 π0.5 和 FastWAM 检验可复用接口。明确哪些逻辑属于模型、哪些属于具身任务、哪些由 Hyper 公共能力承载，再按实际需求扩展 PT 等训练阶段。
3. **收敛具身训练的公共组件**：将可复用的数据和 batch 处理、损失归约、优化器接入和并行策略放入 Hyper 现有目录。模型默认通过配置组合已有组件，上游接口的适配由模型接入层承载。多个模型出现独立的训练循环需求时，再评估新增 Trainer。

GR00T N1.7 和 OpenWAM-α 已验证下游 SFT、策略导出和 LIBERO 闭环。首轮整合将共享能力拆入 Hyper 现有组件，用配置组合 LeRobot 或 OpenWAM 官方实现。第二轮模型将检验这些接口能否支持更多 VLA 和 WAM。

首个 Recipe 以昇腾 NPU 为目标，具体接入边界与验收顺序见 [GR00T N1.7 Recipe](groot-n17-recipe.md)。

## 三、接入现有训练流程

训练时选择模型的 YAML 配置，填写权重和数据等资产路径。两个模型都使用固定样本数的 DataLoader，`SampleListCollator` 保留样本顺序并提供 `samples` 和 `sample_count`。模型配置指定如何把原始样本转换为前向所需的输入。

| 模型 | Hyper 接入点 | 主前向和损失 |
|---|---|---|
| GR00T N1.7 | VLM 的 `get_batch` 堆叠 LeRobot 样本并调用 processor，返回 `(model_inputs, loss_inputs)`。 | GR00T 计算原生动作损失。VLM 统计 `action_mask`，沿用有效坐标计数归约。 |
| OpenWAM-α | LeRobot reader 和 `TrajectoryWindow` 读取并转换轨迹窗口，`DiTBatch` 提供组批数据，条件模型调用上游输入编码。 | OpenWAM 在原有 FSDP 根模块内计算视频和动作联合损失，DiT 通过 `ModelOutputLoss` 读取结果。 |

OpenWAM 的条件模型复用 checkpoint 内已冻结的文本编码器和 VAE。DiT 在禁用 autograd 的条件阶段执行编码，编码器按配置移回 CPU。OpenWAM 的时间步采样、加噪和联合损失仍由原有主前向负责。

两个模型共用 LeRobot v3 reader，reader 按字段和源帧偏移读取窗口。GR00T 从 policy 配置推导动作偏移。OpenWAM 的训练 YAML 通过 `TrajectoryWindow` 指定视频采样、相机布局以及 EEF10 动作和状态的归一化与槽位映射。OpenWAM 数据目录需要包含上游生成的 `meta/libero_normalization_stats.npy`，训练直接读取这份统计。官方 reader 仍可用于导出和基线对照。

回归覆盖 GR00T 单卡 LeRobot 基线、两卡梯度累积、OpenWAM 固定样本及两卡 FSDP。固定样本对比 loss、梯度范数和抽样参数更新。

## 四、共用 LIBERO 闭环评测

用同一个命令评估 GR00T 和 OpenWAM 导出的策略。运行环境需要 LeRobot 的 LIBERO 集成、`hf-libero`、模拟器资产和 MuJoCo 渲染后端。OpenWAM 额外需要官方推理实现及其 `benchmarks` 模块可导入。

```bash
python scripts/eval_libero.py --model openwam_alpha \
  --policy-dir /path/to/exported-policy --device npu:0 \
  --suite libero_spatial --task-ids 0 --episodes 1 --seed 42 \
  --output /path/to/libero-result.json
```

评估 GR00T 时把 `--model` 换成 `groot_n17`。评测命令会在本机启动并关闭 OpenWAM 官方推理服务，可用 `--deploy-config` 指定推理设置。入口统一选择任务、初始化环境、传入观测并执行策略动作，输出按任务、任务组和整体汇总的成功率。默认使用同一套环境初始化和任务步数。调试时可以用 `--max-steps` 缩短 rollout。

评估 π0.5 时使用 `--model pi05`，将 `--policy-dir` 指向本地微调 checkpoint 目录，并保留原始 processor 配置和归一化统计。用 `--tokenizer-dir` 指定本地 PaliGemma tokenizer 目录，或将它放在策略目录的 `tokenizer/` 下。π0.5 和 GR00T 共用 LeRobot 的 processor 和 LIBERO 动作桥接。[LeRobot LIBERO 微调 checkpoint](https://huggingface.co/lerobot/pi05_libero_finetuned_v044)

## 五、π0.5 微调和导出

使用通用 VLM 启动模块加载 π0.5 LIBERO 配置。首轮配置冻结视觉语言骨干，训练动作专家和动作投影。共用 LeRobot reader 从策略配置读取动作窗口长度，再由原生 π0.5 processor 对状态、动作和任务指令做预处理。[π0.5 LIBERO 配置](../../../hyper_parallel/models/pi05/recipes/libero_spatial.yaml)

```bash
torchrun --standalone --nproc-per-node=1 --module scripts.train_vl \
  hyper_parallel/models/pi05/recipes/libero_spatial.yaml \
  --model.checkpoint=/path/to/pi05_libero_base \
  --dataset.frame_offsets.checkpoint=/path/to/pi05_libero_base \
  --dataset.dataset_root=/path/to/LIBERO_LeRobot_v3/libero_spatial \
  --dataset.model_assets.tokenizer_dir=/path/to/paligemma-tokenizer \
  --dataset.model_assets.stats_file=/path/to/libero-spatial-pi05-stats.json
```

基础权重采用 LeRobot π0.5 的分位数归一化默认值。统计 JSON 需要包含训练数据的 `observation.state` 和 `action` 的 `q01`、`q99`。原生训练 loss 对整个动作窗口取均值，包含为补齐 episode 尾部而重复的动作。

双卡 FSDP 使用同一份配置。将 `--nproc-per-node` 改为 `2`，并追加 `--fsdp_config.dp_shard_size=2 --training.global_batch_size=2`。冻结的骨干在各卡保留副本，动作专家和投影在根模块分片。联合注意力两侧使用 FP32，梯度以 FP32 归约。

训练时设置 `--checkpoint.save_ckpt=true --checkpoint.save_steps=1` 可保存 HyperParallel checkpoint。导出使用与训练一致的基础策略配置和归一化统计。

```bash
python scripts/export_pi05.py \
  --training-checkpoint /path/to/checkpoints/global_step_1 \
  --base-checkpoint /path/to/pi05_libero_base \
  --tokenizer-dir /path/to/paligemma-tokenizer \
  --stats-file /path/to/libero-spatial-pi05-stats.json \
  --output-dir /path/to/exported-policy
```

导出目录包含 LeRobot 策略、processor 及 tokenizer。将共用评测入口的 `--policy-dir` 指向该目录即可重载。

### 双卡断点恢复对照

使用 LeRobot 提交 `b07adcc5e3ce42c26a248236dbb3571f84c28c43`，在两张 Ascend 910 上比较连续训练 4 步与从第 2 步恢复到第 4 步。两组使用 seed 42、全局 batch size 2、每卡 micro-batch size 1。两卡动作头分片在第 3、4 步均有更新，恢复组和连续组每步的分片 SHA-256 完全一致。

| 步数 | Rank 0 样本 | Rank 1 样本 | 学习率 | Action loss | 梯度范数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 4 | 5 | `2.5e-5` | `1.188276824951172` | `6.911948204040527` |
| 4 | 6 | 7 | `2.5e-5` | `0.57130126953125` | `2.385023593902588` |

此前的双卡单步 checkpoint 已导出为可离线加载的 LeRobot 策略。在共用 LIBERO 入口上，导出策略在 NPU 上以 seed 42 执行 `libero_spatial` 任务 0，第 154 步成功，覆盖一回合。

在已配置的 Ascend 环境中，从仓库根目录执行。验证脚本调用原有 VLMTrainer，在每个优化器步记录真实样本索引、学习率、loss、梯度范数和该 rank 动作头分片的 SHA-256。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
assets=/path/to/assets
out=/path/to/pi05-resume-validation
common=(
  hyper_parallel/models/pi05/recipes/libero_spatial.yaml
  --model.checkpoint="$assets/hf_weights/pi05_libero_base"
  --dataset.frame_offsets.checkpoint="$assets/hf_weights/pi05_libero_base"
  --dataset.dataset_root="$assets/data/LIBERO_LeRobot_v3/libero_spatial"
  --dataset.model_assets.tokenizer_dir="$assets/hf_weights/pi05-paligemma-tokenizer"
  --dataset.model_assets.stats_file="$assets/data/LIBERO_LeRobot_v3/libero_spatial/meta/pi05_stats.json"
  --fsdp_config.dp_shard_size=2
  --training.global_batch_size=2
  --training.train_iters=4
)
torchrun --standalone --nproc-per-node=2 scripts/verify_pi05_resume.py "${common[@]}" \
  --checkpoint.checkpoint_dir="$out/continuous" \
  --checkpoint.save_ckpt=true --checkpoint.save_steps=2
torchrun --standalone --nproc-per-node=2 scripts/verify_pi05_resume.py "${common[@]}" \
  --checkpoint.checkpoint_dir="$out/resumed" --checkpoint.save_ckpt=false \
  --checkpoint.restore_from="$out/continuous/global_step_2"
python scripts/verify_pi05_resume.py compare "$out/continuous" "$out/resumed"
```

### FastWAM 接口对照

对照上游提交 `7faa71108368fbb3b6885649f112af607427a2d4` 的 LIBERO 双相机配置。原生训练以 33 个源帧为窗口，采样第 0、4 至 32 帧作为 9 帧视频，读取 32 步动作和 33 步状态。共用 reader 可用每字段帧偏移表达采样，并返回对应的 padding mask。两路相机 resize 后横向拼成 `224 × 448`，归一化、夹爪约定和动作表示由上游配置及 processor 决定。[FastWAM LeRobot v3 配置](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/data/libero_2cam_lerobot_v30.yaml)

将同一份本地 LIBERO Spatial v3 数据分别交给共用 reader 和 FastWAM 原生 v3 reader。在第 0 个 episode 的首帧、中段和末帧，任务文本、动作、状态、两路视频以及各窗口的 padding mask 完全一致。末帧有 31 个动作、32 个状态和每路 8 个视频帧标记为 padding。

训练可沿用 DiTTrainer 的组批、优化器和 checkpoint 编排。首版适配可以在 FSDP 根 forward 内调用上游 `training_loss`，保留 native 的视频和动作 padding、逐样本平均和 scheduler 权重。上游 `build_inputs` 同时包含 VAE、文本编码和可训练的 proprio 投影。若拆出条件阶段，需要把投影保留在可求导的主前向里。[FastWAM 原生训练目标](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py)

用单层原生 MoT、随机初始化的两路专家、替代 VAE 和合成文本 embedding 在 CPU 上通过了 DiTTrainer 的前反向验证。用带 padding 的真实读入样本验证了两路 loss 为有限正数，proprio 投影获得非零梯度。正式 NPU 验证仍需适配上游 VAE 编码中固定的 `cudagraphs` backend。[FastWAM VAE 调用](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py#L248)

评测可复用 LIBERO 的任务、初始化、rollout 和指标。策略适配层负责按上游约定预处理两路图像和 proprio、调用 `infer_action`、反归一化与转换夹爪值，并缓存每次重规划输出的动作。上游配置每 10 步重规划，初始化后执行 30 步 warmup，Spatial 单回合上限为 400 步。共用 runner 已支持设置后两项。与上游做动作对齐时还要核对初始状态索引与推理随机种子。[FastWAM LIBERO 配置](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/sim_libero.yaml) · [FastWAM LIBERO 评测实现](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/eval_libero_single.py)

从真实 LIBERO 环境取了 3 帧观测。共用 canonical observation 与上游评测的两路图像逐像素一致，8 维状态的最大绝对差为 `7.45e-9`。共用 reader 及评测入口可以直接复用，横向拼图归属等首个 FastWAM processor 对齐后再决定。首个模型级对照需要 Wan2.2-TI2V-5B 组件、FastWAM LIBERO 发布权重及配套归一化统计。[FastWAM 资产说明](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/README.md)

### FastWAM eager VAE 训练入口

训练配置使用共用 LeRobot reader 和上游 processor，横向拼图保留在 FastWAM 适配层。冻结的 Wan VAE 直接调用原生 `encode`，视频和动作专家与 proprio 投影仍在 DiT 主前向中计算梯度。昇腾上将 RoPE 表的拼接放在 FP64 实部、虚部分量上，拼接后恢复上游使用的复数布局。频率表与上游 CPU 路径逐位一致。

用共享 reader 读取同一 episode 的首帧、中间帧和末帧。经过完整 processor 后，视频、动作、状态、提示词和 padding mask 与 FastWAM 原生 reader 一致。用真实 `9 × 224 × 448` 视频、缩小版原生 Wan VAE 和单层 MoT，在昇腾上完成 BF16 前反向和一次参数更新。该验证使用合成文本 embedding。

另用官方 Wan2.2 VAE 权重在昇腾上编码同样尺寸的真实视频，得到 BF16 latent `[1, 48, 3, 14, 28]`。两次完整视频编码逐位一致，视频的首帧 latent 与单图编码也逐位一致。

下面的配置从发布的 LIBERO 策略继续微调。策略文件提供视频和动作专家的权重，本地 Wan 目录提供 VAE、文本编码器和 tokenizer。[训练配置](../../../hyper_parallel/models/fastwam/recipes/libero_spatial.yaml) · [发布权重](https://huggingface.co/yuanty/fastwam)

```bash
torchrun --standalone --nproc-per-node=1 scripts/train_dit.py \
  hyper_parallel/models/fastwam/recipes/libero_spatial.yaml \
  --model.checkpoint=/path/to/fastwam/libero_uncond_2cam224.pt \
  --model.wan_dir=/path/to/Wan2.2-TI2V-5B \
  --model.model_config=/path/to/FastWAM/configs/model/fastwam.yaml \
  --model.data_config=/path/to/FastWAM/configs/data/libero_2cam_lerobot_v30.yaml \
  --dataset.data_transform.data_config=/path/to/FastWAM/configs/data/libero_2cam_lerobot_v30.yaml \
  --dataset.dataset_root=/path/to/lerobot_v30/libero_spatial_no_noops_lerobot \
  --dataset.data_transform.stats_file=/path/to/fastwam/libero_uncond_2cam224_dataset_stats.json
```
