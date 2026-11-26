# Hyper 具身模型训练方案

## 一、背景与参考资料

本方案面向具身模型的**预训练（PT）和监督微调（SFT）**，覆盖数据处理、训练、策略导出与测评。在线强化学习作为后续扩展，本阶段不展开。

### 1. 模型类型与代表模型

**具身智能**面向智能体通过感知和动作与环境交互，是本方案的应用方向。**世界模型（World Model，WM）**学习环境状态及其变化，用于未来预测、仿真或规划；它可以服务于具身智能，也可以用于视频生成等任务，并不一定输出机器人动作。

**VLA（Vision-Language-Action，视觉语言动作模型）**根据图像、语言指令和机器人状态生成动作。

**WAM（World-Action Model，世界动作模型）**将未来世界预测与动作生成结合起来。不同模型可以联合生成视频和动作，也可以仅在训练时利用视频监督，推理时直接输出动作。

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

本方案结合这些框架的接入经验，**保留官方模型、数据处理与 processor 的语义，使用 Hyper 承载分布式训练和运行管理**。训练入口的组织方式参考 Hyper 现有 `VLMTrainer`；模型特有目标与数据适配留在接入层，训练编排和并行执行复用公共能力，具体接口通过 Recipe 逐步确定。

## 二、开发路线

**先做 Recipe，再逐步沉淀为 Kit，通过实际模型接入确认公共边界。**

以四个代表模型驱动开发，按需求差异逐步推进：

| 阶段 | 模型 | 验证目标 |
|---|---|---|
| 第一轮 | GR00T N1.7 | 优先复用 LeRobot 接入，接通 VLA 的数据／processor、动作训练、策略导出与评测 |
| 第一轮 | OpenWAM-α | 从官方预训练权重出发，接通下游 SFT、策略导出与 LIBERO 评测，检验视频／动作联合训练及官方组件接入 |
| 第二轮 | π0.5 | 用不同 VLA 结构和处理流程检验首轮公共接口 |
| 第二轮 | FastWAM | 检验另一种 WAM 的视频／动作联合目标，以及训练和推理路径的差异 |

1. **完成首轮 Recipe**：先完成 GR00T N1.7，再接入 OpenWAM-α 的下游 SFT。各自走通数据、训练、策略导出与评测，复用 Hyper 的公共训练能力，保留上游模型与数据处理方式。OpenWAM-α 首轮使用公开预训练权重，以 LIBERO 为评测入口；大规模预训练留作后续扩展。
2. **检验与扩展边界**：比较首轮 Recipe 的共性与差异，再通过 π0.5 和 FastWAM 检验可复用接口。明确哪些逻辑属于模型、哪些属于具身任务、哪些由 Hyper 公共能力承载，再按实际需求扩展 PT 等训练阶段。
3. **逐步沉淀 Embodied Kit**：随着模型接入，将已验证的共性提取为可复用组件，逐步稳定数据、训练和评测接口，并据此确定 `EmbodiedTrainer` 的职责。各模型的 Recipe 继续保留，用于组织具体配置和模型特有逻辑。

首版以 GR00T N1.7 和 OpenWAM-α 的下游 SFT 完整链路为目标，接口随接入经验收敛，不预先要求所有 VLA／WAM 遵循同一套协议。LeRobot 与模型官方实现均可作为组件来源，由 Hyper 组织训练与并行执行。当前先完成静态接入分析，明确所需组件、改动位置和验证要求；后续再进入代码实现与实验验证。

首个 Recipe 以昇腾 NPU 为目标，具体接入边界与验收顺序见 [GR00T N1.7 Recipe](groot-n17-recipe.md)。
