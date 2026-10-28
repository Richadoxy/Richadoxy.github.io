#!/usr/bin/env python3
"""Build the static Learning Journey blog from bundled and upstream Markdown notes."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import markdown
from markdown.extensions import Extension
from markdown.extensions.toc import slugify
from markdown.inlinepatterns import BACKTICK_RE
from markdown.postprocessors import Postprocessor
from markdown.preprocessors import Preprocessor


@dataclass(frozen=True)
class Post:
    code: str
    source: str
    slug: str
    title: str
    summary: str
    chain: str
    diagram: str | None = None
    published_en: str = "August 2026"
    published_zh: str = "2026.08"
    level: str = "进阶"


POSTS = [
    Post("F01", "27_vla_overview.md", "vla-overview-reading-map", "多模态 VLA 与模仿学习：统一研究范式", "从数据、表示、对齐、任务融合、动作生成到闭环执行，先建立阅读和研究多模态策略的总地图。", "foundations", published_en="September 2026", published_zh="2026.09", level="入门"),
    Post("F02", "31_multimodal_encoder_alignment_and_infonce.md", "multimodal-encoder-alignment-infonce", "从原始模态到公共语义空间", "串联 token、encoder、ViT、CLIP、CLS/readout、projector、语义锚点与 InfoNCE，理解多模态表示怎样建立共同坐标。", "foundations", published_en="September 2026", published_zh="2026.09", level="核心"),
    Post("F03", "32_three_consistency_supervision_paradigms.md", "multimodal-consistency-supervision", "多模态表示的监督配方", "统一跨模态对齐、时间连续性、增强或遮挡一致性，并说明 reconstruction、future prediction 与任务监督的分工。", "foundations", published_en="September 2026", published_zh="2026.09", level="核心"),
    Post("F04", "19_attention_from_embeddings_to_action_expert.md", "19-attention-openclip-tactile-groot", "从对齐表示到 Policy 融合", "把 Transformer 基础放回策略数据流，系统理解 self/cross/joint attention、readout、projector、gate、AdaNorm 与 Action Head。", "foundations", published_en="September 2026", published_zh="2026.09", level="核心"),
    Post("IL01", "23_imitation_learning_basics.md", "imitation-learning-behavior-cloning", "模仿学习与 Behavior Cloning 入门", "从 demonstration、policy 和 BC 出发，理解离线监督、闭环 rollout 与分布偏移。", "imitation", published_en="September 2026", published_zh="2026.09", level="入门"),
    Post("IL02", "24_robot_data_action_semantics.md", "robot-data-action-semantics", "机器人数据：Observation、Action 与时间窗口", "统一 state/action 语义、episode、action chunk、horizon、控制频率、归一化和 mask。", "imitation", published_en="September 2026", published_zh="2026.09", level="入门"),
    Post("IL03", "25_action_generation_families.md", "action-generation-mse-token-diffusion-flow", "动作生成方法：从连续回归到概率生成", "串联回归、自回归、CVAE、GAN、扩散、Flow Matching 与生成一致性，区分训练目标、采样和执行。", "imitation", published_en="September 2026", published_zh="2026.09", level="进阶"),
    Post("IL04", "36_diffusion_policy_and_act.md", "diffusion-policy-and-act", "Diffusion Policy 与 ACT：小型策略的训练和推理", "从 DP 去噪到 ACT 动作生成，用示范例子、分布图和梯度表解释 CVAE、先验、后验、KL 与训练推理的区别。", "imitation", published_en="October 2026", published_zh="2026.10", level="核心"),
    Post("IL05", "37_generative_consistency_and_sampling_distillation.md", "generative-consistency-and-sampling-distillation", "生成一致性与采样蒸馏：从多步到少步生成", "理解生成轨迹、端点一致性、边界条件和 teacher/student，以及少步生成与闭环反馈的关系。", "imitation", published_en="October 2026", published_zh="2026.10", level="进阶"),
    Post("VLA01", "10_flow_matching_training_and_inference.md", "10-flow-matching-training-inference", "Flow Matching：训练、timestep 与推理", "从插值路径、velocity target 和 timestep condition 出发，区分训练采样与推理积分。", "architectures"),
    Post("VLA02", "04_pi05_architecture.md", "04-pi05-architecture", "PI0.5：前向结构与 Action Expert", "拆解 Prefix、Action Expert、联合 attention 与连续动作路径，再连接 FAST/KI 训练专题。", "architectures", "pi05_structure.drawio"),
    Post("VLA03", "42_fast_ki_and_pi_training.md", "fast-ki-pi-training", "FAST 与 KI：动作监督、梯度与微调", "连接动作 token、双损失、两阶段与联合训练，区分 KI、冻结、LoRA 和官方 openpi 微调。", "architectures", "pi05_fast_ki_training.drawio", published_en="October 2026", published_zh="2026.10", level="核心"),
    Post("VLA04", "11_groot_dit_architecture.md", "11-groot-dit-architecture", "GR00T：VLM Memory 与 DiT Action Head", "梳理 VLM memory、state/action encoders、AlternateVLDiT、AdaLN 与 action decoder 的数据流。", "architectures", "groot_n1d7_structure.drawio"),
    Post("VLA05", "14_architecture_comparison_and_reading_guide.md", "14-architecture-comparison-reading-guide", "PI0.5、GR00T、VLASH 与 VLA-JEPA 对照", "把架构、异步框架和 world-model supervision 放到正确层级，并形成统一比较坐标。", "architectures"),
    Post("VLA06", "13_jepa_and_vla_jepa.md", "13-jepa-vla-jepa", "JEPA、World Model 与 VLA-JEPA", "理解 teacher/student/predictor、latent world-model supervision、leakage-free training 和联合 action loss。", "architectures", "vla_jepa_structure.drawio", level="前沿"),
    Post("ENG01", "02_config_system.md", "02-config-system", "配置与数据合同：从 YAML 到 DataConfig", "理解 YAML 如何展开成 TrainConfig、DataConfig，以及 BrainCo policy 层如何定义机器人数据语义。", "engineering"),
    Post("ENG02", "03_data_pipeline_and_batch.md", "03-data-pipeline-and-batch", "从 Dataset 到 Batch", "跟踪一条 LeRobot 样本经过 transform、collate 和 sharding，最终成为模型训练 batch 的全过程。", "engineering"),
    Post("ENG03", "01_train_entry_and_jax.md", "01-train-entry-and-jax", "JAX 训练入口", "从 scripts/train.py 出发，梳理编译缓存、PRNG、mesh、sharding、checkpoint 与训练循环。", "engineering"),
    Post("ENG04", "05_train_state_jit_checkpoint.md", "05-train-state-jit-checkpoint", "TrainState、JIT 与 Checkpoint", "理解参数初始化、优化器状态、JIT 编译边界、分片方式以及训练恢复流程。", "engineering"),
    Post("ENG05", "06_action_dimensions_28d_56d.md", "06-action-dimensions-28d-56d", "28D/56D 双臂双手改造案例", "记录动作空间改变后投影、mask、归一化、权重加载和部署合同的连锁变化。", "engineering"),
    Post("RUN01", "26_online_execution_runtime.md", "vla-online-runtime-sync-async", "VLA 在线执行：同步、异步与 Action Chunk 调度", "从端到端延迟、active/pending chunk 和 handoff 出发，建立在线部署的时间轴。", "runtime", published_en="September 2026", published_zh="2026.09", level="入门"),
    Post("RUN02", "12_vlash_principles_and_async_runtime.md", "12-vlash-principles-async-runtime", "VLASH：Temporal Offset 与异步执行", "理解 offset augmentation、future state、异步 inference/actor 通信与 chunk handoff。", "runtime", "vlash.drawio"),
    Post("RUN03", "07_vlash_integration.md", "07-vlash-integration", "VLASH 在 BrainCo-IL 中的集成", "以当前代码为准，区分已经实现的 temporal offset/future-state condition 与尚未实现的 shared-prefix 多分支训练。", "runtime", "vlash.drawio"),
    Post(
        "TOUCH00",
        "16_tactile_vla_reading_guide.md",
        "16-tactile-vla-reading-guide",
        "触觉 VLA 与模仿学习阅读地图",
        "按表征与数据、Language-Action、力控闭环、灵巧手和前沿专题，组织触觉 VLA 的渐进阅读顺序。",
        "tactile",
        published_en="September 2026",
        published_zh="2026.09",
        level="入门",
    ),
    Post(
        "TOUCH01",
        "15_tvl_paper_reading_notes.md",
        "15-tvl-touch-vision-language-alignment",
        "TVL 精读：从触觉表征到 LLaMA",
        "从 DIGIT 触觉输入出发，串联 CLIP 对齐、InfoNCE、跨模态检索、projector、gate 与 LLaMA 多模态生成。",
        "tactile",
        published_en="September 2026",
        published_zh="2026.09",
    ),
    Post(
        "TOUCH02",
        "18_anytouch_paper_reading_notes.md",
        "18-anytouch-unified-tactile-representation",
        "AnyTouch 精读：统一静态—动态多传感器触觉表征",
        "从TacQuad多传感器数据出发，拆解静态—动态统一输入、遮挡重建、跨模态对齐、Universal Sensor Token与跨传感器匹配。",
        "tactile",
        published_en="September 2026",
        published_zh="2026.09",
    ),
    Post(
        "TOUCH03",
        "17_octopi_paper_reading_notes.md",
        "17-octopi-tactile-property-reasoning",
        "Octopi 精读：从触觉属性到大模型物理推理",
        "从 PHYSICLEAR 人工属性标注出发，拆解 CLIP 触觉编码器、Vicuna 接入、三阶段训练、OPD 中间描述与零样本 PSR。",
        "tactile",
        published_en="September 2026",
        published_zh="2026.09",
    ),
    Post(
        "TOUCH04",
        "20_beyond_sight_fuse_paper_reading_notes.md",
        "20-beyond-sight-fuse-multimodal-policy",
        "Beyond Sight 精读：用语言把触觉与音频接入通用机器人策略",
        "从Octo与PaliGemma两类backbone出发，拆解FuSe的异构传感器token融合、action readout、语言对比与生成监督，以及真实机器人实验。",
        "tactile",
        published_en="September 2026",
        published_zh="2026.09",
    ),
    Post(
        "TOUCH05",
        "28_vla_touch_paper_reading_notes.md",
        "vla-touch-dual-level-tactile-feedback",
        "VLA-Touch 精读：高层触觉规划与低层动作修正",
        "拆解外挂在 RDT 外部的双层触觉反馈：结构化触觉辅助 GPT-4o 规划，连续触觉通过 Interpolant Controller 修正动作。",
        "tactile",
        published_en="September 2026",
        published_zh="2026.09",
        level="前沿",
    ),
    Post(
        "TOUCH06",
        "29_omnivtla_paper_reading_notes.md",
        "omnivtla-semantic-aligned-tactile-sensing",
        "OmniVTLA 精读：语义对齐触觉感知接入 π0",
        "从 ObjTac 和 SA-ViT 预训练出发，分析双触觉 encoder、prefix token 融合、flow matching 与多种机器人本体实验。",
        "tactile",
        published_en="September 2026",
        published_zh="2026.09",
        level="前沿",
    ),
    Post(
        "TOUCH07",
        "30_tactile_vla_physical_knowledge_paper_reading_notes.md",
        "tactile-vla-physical-knowledge-force-control",
        "Tactile-VLA 精读：物理知识、力动作与混合控制",
        "分析触觉历史 token、Tactile-Aware Action Expert、位置—力混合控制，以及 CoT 纠错循环的能力与边界。",
        "tactile",
        published_en="September 2026",
        published_zh="2026.09",
        level="前沿",
    ),
    Post(
        "TOUCH08",
        "35_rdp_paper_reading_notes.md",
        "rdp-reactive-diffusion-policy",
        "RDP 精读：动作块中的触觉闭环与示范质量",
        "从 Fig. 5 的反馈时序和 Fig. 6 的两阶段训练出发，理解慢速潜在扩散、快速触觉解码、TactAR 数据采集及示范质量边界。",
        "tactile",
        published_en="October 2026",
        published_zh="2026.10",
        level="前沿",
    ),
    Post(
        "TOUCH09",
        "34_trex_paper_reading_notes.md",
        "trex-tactile-reactive-dexterous-manipulation",
        "T-Rex 精读：高频触觉反应与级联 Flow Matching",
        "从三专家 MoT、future-latent queries 和时空触觉编码出发，拆解低频视觉规划、高频触觉修正、KV 缓存与分段 Flow Matching。",
        "tactile",
        published_en="September 2026",
        published_zh="2026.09",
        level="前沿",
    ),
    Post("TOUCH10", "39_tactile_input_representation.md", "tactile-input-representation", "从原始触觉到策略条件", "从压阻、三维力和视触觉出发，连接标定、数值 atlas、区域 token、时间对齐与条件融合。", "tactile", published_en="October 2026", published_zh="2026.10", level="核心"),
    Post("TOUCH11", "38_cgp_paper_reading_notes.md", "cgp-contact-grounded-policy", "CGP 精读：状态、触觉与接触一致的控制", "区分未来机器人状态、触觉 latent 和控制命令，拆解联合扩散、接触映射及 Visuotactile DP 对照。", "tactile", published_en="October 2026", published_zh="2026.10", level="进阶"),
    Post("TOUCH12", "40_policy_memory_and_object_centric_tactile_state.md", "policy-memory-and-object-centric-tactile-state", "策略记忆：从历史观测到对象级触觉状态", "串联滑窗、GRU、缓存和 memory tokens，再理解遮挡中的身份 anchor、动态 slot 与接触归属。", "tactile", published_en="October 2026", published_zh="2026.10", level="进阶"),
    Post("TOUCH13", "41_tactile_memory_experiment_design.md", "tactile-memory-experiment-design", "触觉记忆的实验设计：从假设到控制证据", "把数据、辅助监督、公平 baseline、遮挡扰动和联合成功指标连接成可验证的研究过程。", "tactile", published_en="October 2026", published_zh="2026.10", level="实践"),
    Post("RL01", "43_dexterous_rl_overview.md", "dexterous-rl-overview", "灵巧手强化学习：研究范式与学习地图", "从知识树区分学习信号、网络、PPO 与训练框架，再连接灵巧手路线、实验设计和 HORA 精讲。", "reinforcement", published_en="October 2026", published_zh="2026.10", level="入门"),
    Post("RL02", "44_rl_basics_and_training_loop.md", "rl-basics-and-training-loop", "RL 基础：从 Actor–Critic 到 PPO 的数据、目标与梯度", "用网络图、数值例子和逐段伪代码连接 reward、return、价值预测、GAE 与 PPO，明确训练目标和梯度边界。", "reinforcement", published_en="October 2026", published_zh="2026.10", level="核心"),
    Post("RL03", "45_teacher_student_and_sim_to_real.md", "teacher-student-and-sim-to-real", "Teacher–Student：策略蒸馏、特权信息与部署", "从特权输入与监督目标组织知识迁移，区分动作、分布与 latent 蒸馏，再连接在线适应和真机部署。", "reinforcement", published_en="October 2026", published_zh="2026.10", level="进阶"),
    Post("RL04", "46_hora_paper_walkthrough.md", "hora-paper-walkthrough", "HORA 精讲：从特权 PPO 策略到本体感知在线适应", "将 RL 通识落实到 HORA 的三阶段网络、历史适应、控制接口与对照实验，区分论文设计和公开代码实现。", "reinforcement", published_en="October 2026", published_zh="2026.10", level="精读"),
]


LEGACY_REDIRECTS = {
    "08-transformer-latent-encoder-decoder": "multimodal-encoder-alignment-infonce",
    "model-training-forward-loss-backward": "vla-overview-reading-map",
    "clip-contrastive-representation-learning": "multimodal-encoder-alignment-infonce",
    "09-attention-multi-expert-conditioning": "19-attention-openclip-tactile-groot",
}


CHAINS = {
    "foundations": {
        "title": "第一卷：基础概念",
        "title_en": "Part 1: Foundations",
        "label_zh": "基础概念",
        "label_en": "Foundations",
        "description": "先建立多模态 VLA 与模仿学习的统一范式，再依次理解表示接口、公共空间、监督配方与 Policy 融合。",
    },
    "imitation": {
        "title": "第二卷：模仿学习",
        "title_en": "Part 2: Imitation Learning",
        "label_zh": "模仿学习",
        "label_en": "Imitation Learning",
        "description": "从示范与 Behavior Cloning 出发，连接机器人数据、DP / ACT、概率生成目标与少步采样。",
    },
    "architectures": {
        "title": "第三卷：VLA 核心架构",
        "title_en": "Part 3: VLA Architectures",
        "label_zh": "VLA 架构",
        "label_en": "VLA Architectures",
        "description": "从 Flow Matching 与 PI0.5 前向结构进入 FAST/KI 的动作监督和梯度路径，再对照 GR00T、VLASH 与 VLA-JEPA。",
    },
    "engineering": {
        "title": "第四卷：OpenPI / BrainCo 工程实践",
        "title_en": "Part 4: OpenPI / BrainCo Engineering",
        "label_zh": "训练工程",
        "label_en": "Engineering",
        "description": "从配置、数据和 JAX 训练闭环进入 28D/56D 双臂双手改造。",
    },
    "runtime": {
        "title": "第五卷：推理与部署",
        "title_en": "Part 5: Runtime and Deployment",
        "label_zh": "推理部署",
        "label_en": "Runtime",
        "description": "理解在线延迟、同步与异步执行、chunk handoff，以及 VLASH 的时间对齐方法。",
    },
    "tactile": {
        "title": "第六卷：触觉表征与触觉 VLA",
        "title_en": "Part 6: Tactile Representation and VLA",
        "label_zh": "触觉 VLA",
        "label_en": "Tactile VLA",
        "description": "连接触觉输入、跨模态表示、动作与闭环反馈，再进入 CGP、对象级记忆和实验设计；阅读地图提供基础与论文两条路线。",
    },
    "reinforcement": {
        "title": "第七卷：强化学习与灵巧操作",
        "title_en": "Part 7: Reinforcement Learning and Dexterity",
        "label_zh": "强化学习",
        "label_en": "RL",
        "description": "沿研究地图、Actor–Critic 与 PPO 通识、Teacher–Student 框架进入 HORA 精讲，用符号、网络图和实验建立从交互学习到真机适应的知识链。",
    },
}


PAPER_LINKS = {
    "Fu et al. - 2024 - A Touch Vision and Language Dataset for Multimodal Alignment.pdf": "https://proceedings.mlr.press/v235/fu24b.html",
    "Cheng et al. - 2024 - Touch100k A Large-Scale Touch-Language-Vision Dataset.pdf": "https://arxiv.org/abs/2406.03813",
    "Yu et al. - 2024 - Octopi Object Property Reasoning with Large Tactile-Language Models.pdf": "https://arxiv.org/abs/2405.02794",
    "Feng et al. - 2025 - AnyTouch Unified Static-Dynamic Representation across Visuo-tactile Sensors.pdf": "https://arxiv.org/abs/2502.12191",
    "Jones 等 - 2025 - Beyond Sight Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding.pdf": "https://arxiv.org/abs/2501.04693",
    "Hao et al. - 2025 - TLA Tactile-Language-Action Model for Contact-Rich Manipulation.pdf": "https://arxiv.org/abs/2503.08548",
    "Zhang et al. - 2025 - VTLA Vision-Tactile-Language-Action Model with Preference Learning.pdf": "https://arxiv.org/abs/2505.09577",
    "Bi et al. - 2025 - VLA-Touch Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback.pdf": "https://arxiv.org/abs/2507.17294",
    "Cheng et al. - 2025 - OmniVTLA Vision-Tactile-Language-Action Models with Semantic-Aligned Tactile Sensing.pdf": "https://arxiv.org/abs/2508.08706",
    "Huang et al. - 2025 - Tactile-VLA Unlocking Physical Knowledge for Tactile Generalization.pdf": "https://arxiv.org/abs/2507.09160",
    "Yu et al. - 2025 - ForceVLA Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation.pdf": "https://arxiv.org/abs/2505.22159",
    "Ye et al. - 2025 - Learning to Feel the Future DreamTacVLA.pdf": "https://arxiv.org/abs/2512.23864",
    "Huang et al. - 2026 - TaF-VLA Tactile-Force Alignment in VLA Models.pdf": "https://arxiv.org/abs/2601.20321",
    "Zhao et al. - 2026 - FD-VLA Force-Distilled Vision-Language-Action Model.pdf": "https://arxiv.org/abs/2602.02142",
    "Li et al. - 2026 - ForceVLA2 Hybrid Force-Position Control with Force Awareness.pdf": "https://arxiv.org/abs/2603.15169",
    "Zhang et al. - 2026 - TacVLA Contact-Aware Tactile Fusion for Robust VLA Manipulation.pdf": "https://arxiv.org/abs/2603.12665",
    "Li et al. - 2026 - AT-VLA Adaptive Tactile Injection for Enhanced Feedback Reaction.pdf": "https://arxiv.org/abs/2605.07308",
    "Zhang et al. - 2026 - Feeling the Unexpected ResTacVLA via Residual Tactile Representation.pdf": "https://arxiv.org/abs/2607.03387",
    "Wang et al. - 2026 - ViTaR Visuo-Tactile Residual Adaptation for Foundation VLA Manipulation.pdf": "https://arxiv.org/abs/2608.15816",
    "Zhou et al. - 2026 - TacForcing Streaming Action Generation with Execution-Time Tactile Feedback.pdf": "https://arxiv.org/abs/2608.25798",
    "Cui et al. - 2025 - End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy.pdf": "https://arxiv.org/abs/2511.00139",
    "Tang et al. - 2026 - Towards Human-Like Manipulation with MoDE-VLA.pdf": "https://arxiv.org/abs/2603.08122",
    "Wu et al. - 2026 - Tabero Learning Gentle Manipulation with Vision Touch and Language（复件）.pdf": "https://arxiv.org/abs/2605.27886",
    "Liu et al. - 2025 - MLA A Multisensory Language-Action Model.pdf": "https://arxiv.org/abs/2509.26642",
    "Tian et al. - 2026 - VT-WAM Visual-Tactile World Action Model.pdf": "https://arxiv.org/abs/2607.02503",
    "NeoteAI and Fudan TEAI - 2026 - N0-VTLA Scaling with Latent Tactile Tokens.pdf": "https://arxiv.org/abs/2607.23782",
    "Van der Merwe et al. - 2026 - TAP-VLA Tactile Annotation Prompting for VLA Models.pdf": "https://arxiv.org/abs/2606.29089",
    "Gubernatorov et al. - 2026 - HapticVLA Contact-Rich Manipulation without Inference-Time Tactile Sensing.pdf": "https://arxiv.org/abs/2603.15257",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/xyd/BrainCo/BrainCo-IL/docs/brainco_il_beginner"),
        help="Directory containing the 14 source Markdown files and Draw.io diagrams.",
    )
    parser.add_argument(
        "--site-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Personal homepage root.",
    )
    parser.add_argument(
        "--sync-upstream",
        action="store_true",
        help="Refresh matching bundled Markdown and diagrams from --source before building.",
    )
    return parser.parse_args()


def rewrite_links(text: str, posts_by_source: dict[str, Post]) -> str:
    branch_base = "https://github.com/YINJIAJU1123/BrainCo-IL/blob/feature/vlash-integration/"

    def replace(match: re.Match[str]) -> str:
        label, target = match.group(1), match.group(2)
        normalized_target = target.removeprefix("<").removesuffix(">")
        if normalized_target in PAPER_LINKS:
            target = PAPER_LINKS[normalized_target]
        elif target in posts_by_source:
            target = f"{posts_by_source[target].slug}.html"
        elif target.startswith("../assets/"):
            # Markdown lives in blog/source; generated pages live in blog.
            target = target.removeprefix("../")
        elif target.endswith(".drawio"):
            target = f"assets/diagrams/{Path(target).name}"
        elif target.startswith("../../"):
            target = branch_base + target.removeprefix("../../")
        return f"[{label}]({target})"

    return re.sub(r"\[([^]]+)]\(([^)]+)\)", replace, text)


def reading_time(source_text: str) -> int:
    units = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", source_text)
    return max(2, round(len(units) / 350))


def protect_math(text: str) -> tuple[str, dict[str, str]]:
    """Protect complete formulas after fenced code has been stashed by Markdown."""
    replacements: dict[str, str] = {}
    prefix = "BLOGMATH42"
    while prefix in text:
        prefix += "X"
    code_spans = [
        match.span()
        for match in re.finditer(BACKTICK_RE, text, re.DOTALL)
        if match.group(3)
    ]

    def replace(match: re.Match[str]) -> str:
        if any(start < match.end() and match.start() < end for start, end in code_spans):
            return match.group(0)
        line_start = text.rfind("\n", 0, match.start()) + 1
        if re.match(r"(?: {4}|\t)", text[line_start:match.start()]):
            return match.group(0)
        marker = f"{prefix}{len(replacements)}END"
        replacements[marker] = match.group(0)
        return marker

    protected = re.sub(r"\\\[.*?\\\]|\\\(.*?\\\)", replace, text, flags=re.DOTALL)
    return protected, replacements


def restore_math(text: str, replacements: dict[str, str]) -> str:
    """Restore TeX as text, so comparisons and alignment markers remain safe HTML."""
    for marker, formula in replacements.items():
        text = text.replace(marker, html.escape(formula, quote=False))
    return text


class MathExtension(Extension):
    """Keep TeX opaque to tables and inline Markdown, including TOC labels."""

    def __init__(self) -> None:
        super().__init__()
        self.replacements: dict[str, str] = {}

    def extendMarkdown(self, md: markdown.Markdown) -> None:
        replacements = self.replacements

        class MathPreprocessor(Preprocessor):
            def run(self, lines: list[str]) -> list[str]:
                protected, formulas = protect_math("\n".join(lines))
                replacements.update(formulas)
                return protected.split("\n")

        class MathPostprocessor(Postprocessor):
            def run(self, text: str) -> str:
                return restore_math(text, replacements)

        md.registerExtension(self)
        # Fenced code runs at 25; formulas must be opaque before table parsing.
        md.preprocessors.register(MathPreprocessor(md), "protect_math", 24)
        # TOC rendering invokes postprocessors too, restoring its formula labels.
        md.postprocessors.register(MathPostprocessor(md), "restore_math", 5)

    def reset(self) -> None:
        self.replacements.clear()

    def heading_slug(self, value: str, separator: str) -> str:
        # Preserve links to existing formula headings from the delimiter-only renderer.
        for marker, formula in self.replacements.items():
            value = value.replace(marker, formula)
        # TOC may already have restored formulas while extracting heading text.
        for delimiter, legacy in (
            (r"\[", "MATHDISPLAYOPEN42"),
            (r"\]", "MATHDISPLAYCLOSE42"),
            (r"\(", "MATHINLINEOPEN42"),
            (r"\)", "MATHINLINECLOSE42"),
        ):
            value = value.replace(delimiter, legacy)
        return slugify(value, separator)


def render_markdown(text: str) -> tuple[str, str]:
    math_extension = MathExtension()
    md = markdown.Markdown(
        extensions=["fenced_code", "tables", "toc", "sane_lists", math_extension],
        extension_configs={"toc": {"slugify": math_extension.heading_slug}},
    )
    body = md.convert(text)
    return body, md.toc


def diagram_embed(diagram_name: str, diagram_dir: Path) -> str:
    xml = (diagram_dir / diagram_name).read_text(encoding="utf-8")
    config = {
        "highlight": "#2f7f9f",
        "nav": True,
        "resize": True,
        "toolbar": "zoom layers lightbox",
        "xml": xml,
    }
    encoded = html.escape(json.dumps(config, ensure_ascii=False), quote=True)
    return f"""
      <figure class="architecture-figure">
        <div class="mxgraph" data-mxgraph="{encoded}"></div>
        <figcaption>
          <span data-lang="en">Architecture diagram · <a href="assets/diagrams/{diagram_name}" download>Download the Draw.io source</a></span>
          <span data-lang="zh">配套结构图 · <a href="assets/diagrams/{diagram_name}" download>下载 Draw.io 原文件</a></span>
        </figcaption>
      </figure>
    """


def article_page(
    post: Post,
    body: str,
    toc: str,
    diagram: str,
    previous: Post | None,
    following: Post | None,
    read_minutes: int,
) -> str:
    chain = CHAINS[post.chain]
    viewer_script = '\n    <script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>' if diagram else ""
    math_script = '\n    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>' if r"\[" in body or r"\(" in body else ""
    prev_link = f'<a href="{previous.slug}.html">← {previous.code} · {html.escape(previous.title)}</a>' if previous else "<span></span>"
    next_link = f'<a href="{following.slug}.html">{following.code} · {html.escape(following.title)} →</a>' if following else "<span></span>"
    return f"""<!doctype html>
<html lang="en" data-language="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content="{html.escape(post.summary, quote=True)}" />
    <title>{html.escape(post.title)} | 董希越的学习之路</title>
    <link rel="icon" href="../assets/images/favicon.svg" type="image/svg+xml" />
    <link rel="stylesheet" href="../assets/css/main.css" />
    <link rel="stylesheet" href="../assets/css/blog.css" />
    <script defer src="../assets/js/main.js"></script>{math_script}
  </head>
  <body>
    <header class="site-header">
      <div class="header-inner">
        <a class="site-name" href="../index.html">Xiyue Dong</a>
        <nav class="site-nav blog-top-nav" aria-label="博客导航">
          <a href="index.html"><span data-lang="en">Blog</span><span data-lang="zh">学习之路</span></a>
          <a href="../index.html"><span data-lang="en">Home</span><span data-lang="zh">个人主页</span></a>
          <button class="language-toggle" type="button" aria-label="Switch language">
            <span data-lang="en">中文</span><span data-lang="zh">EN</span>
          </button>
        </nav>
      </div>
    </header>
    <div class="article-shell">
      <aside class="article-sidebar">
        <a class="blog-profile" href="../index.html">
          <img src="../assets/images/profile-xiyue.jpg" alt="董希越" />
          <span><strong><span data-lang="en">Xiyue Dong</span><span data-lang="zh">董希越</span></strong><small>Robotics · VLA</small></span>
        </a>
        <a class="back-to-series" href="index.html"><span data-lang="en">← Back to all posts</span><span data-lang="zh">← 返回知识链目录</span></a>
        <details class="article-toc" open>
          <summary><span data-lang="en">Contents</span><span data-lang="zh">本文目录</span></summary>
          {toc}
        </details>
      </aside>
      <main class="article-main">
        <article class="blog-article">
          <p class="article-series"><span data-lang="en">{html.escape(chain['title_en'])}</span><span data-lang="zh">{html.escape(chain['title'])}</span> · {post.code}</p>
          <h1>{html.escape(post.title)}</h1>
          <p class="article-meta"><span data-lang="en">{html.escape(post.level)} · {post.published_en} · {read_minutes} min read</span><span data-lang="zh">{html.escape(post.level)} · {post.published_zh} · 约 {read_minutes} 分钟阅读</span></p>
          <p class="article-deck">{html.escape(post.summary)}</p>{diagram}
          <div class="article-content">
            {body}
          </div>
          <nav class="post-pagination" aria-label="上一篇和下一篇">
            {prev_link}
            {next_link}
          </nav>
        </article>
      </main>
    </div>
    <footer class="blog-footer">© 2026 Xiyue Dong · Learning Journey</footer>{viewer_script}
  </body>
</html>
"""


def index_page(reading_times: dict[str, int]) -> str:
    entries_by_chain = {chain: [] for chain in CHAINS}
    for post in POSTS:
        chain = CHAINS[post.chain]
        published_year, published_month = post.published_zh.split(".")
        published_zh_long = f"{published_year} 年 {int(published_month)} 月"
        entries_by_chain[post.chain].append(
            f"""<article class="archive-post" data-category="{post.chain}">
              <h3><a href="{post.slug}.html"><span class="archive-category">{post.code}</span> {html.escape(post.title)}</a></h3>
              <p class="archive-reading-time"><span data-lang="en">{html.escape(post.level)} · {reading_times[post.code]} min read</span><span data-lang="zh">{html.escape(post.level)} · 阅读约 {reading_times[post.code]} 分钟</span></p>
              <p class="archive-date"><span data-lang="en">Published: {post.published_en}</span><span data-lang="zh">发布于：{published_zh_long}</span></p>
              <p class="archive-summary">{html.escape(post.summary)}</p>
            </article>"""
        )

    series_cards = "".join(
        f"""<a class="series-card" href="#{chain_key}">
          <span class="series-card-label"><span data-lang="en">{html.escape(chain['title_en'])}</span><span data-lang="zh">{html.escape(chain['title'])}</span></span>
          <span>{html.escape(chain['description'])}</span>
        </a>"""
        for chain_key, chain in CHAINS.items()
    )
    filter_buttons = "".join(
        f"""<button class="archive-filter" type="button" data-filter="{chain_key}" aria-pressed="false"><span data-lang="en">{html.escape(chain['label_en'])}</span><span data-lang="zh">{html.escape(chain['label_zh'])}</span></button>"""
        for chain_key, chain in CHAINS.items()
    )
    series_sections = "".join(
        f"""<section class="archive-year archive-series" id="{chain_key}" data-series="{chain_key}" aria-labelledby="heading-{chain_key}">
          <h2 id="heading-{chain_key}"><span data-lang="en">{html.escape(chain['title_en'])}</span><span data-lang="zh">{html.escape(chain['title'])}</span></h2>
          <p class="archive-series-description">{html.escape(chain['description'])}</p>
          <div class="archive-list">{''.join(entries_by_chain[chain_key])}</div>
        </section>"""
        for chain_key, chain in CHAINS.items()
    )
    return f"""<!doctype html>
<html lang="en" data-language="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="description" content="从机器学习基础、模仿学习到 VLA、OpenPI 工程、异步部署、触觉机器人与灵巧手强化学习的渐进学习路径。" />
    <title>学习之路 | Xiyue Dong</title>
    <link rel="icon" href="../assets/images/favicon.svg" type="image/svg+xml" />
    <link rel="stylesheet" href="../assets/css/main.css" />
    <link rel="stylesheet" href="../assets/css/blog.css" />
    <script defer src="../assets/js/main.js"></script>
    <script defer src="../assets/js/blog-archive.js"></script>
  </head>
  <body>
    <header class="site-header">
      <div class="header-inner">
        <a class="site-name" href="../index.html">Xiyue Dong</a>
        <nav class="site-nav blog-top-nav" aria-label="博客导航">
          <a href="../index.html"><span data-lang="en">Home</span><span data-lang="zh">个人主页</span></a>
          <button class="language-toggle" type="button" aria-label="Switch language">
            <span data-lang="en">中文</span><span data-lang="zh">EN</span>
          </button>
        </nav>
      </div>
    </header>
    <div class="archive-shell">
      <aside class="archive-profile" aria-label="作者信息">
        <a href="../index.html"><img src="../assets/images/profile-xiyue.jpg" alt="董希越" /></a>
        <h1>Xiyue Dong</h1>
        <p class="archive-name-zh">董希越</p>
        <p><span data-lang="en">PhD Student in Robotics</span><span data-lang="zh">机器人方向博士生</span><br />HKUST (Guangzhou)</p>
        <ul class="archive-profile-meta">
          <li><span data-lang="en">Guangzhou, China</span><span data-lang="zh">中国广州</span></li>
          <li><a href="mailto:xdong851@connect.hkust-gz.edu.cn">Email</a></li>
          <li><a href="https://github.com/Richadoxy" target="_blank" rel="me noreferrer">GitHub</a></li>
        </ul>
      </aside>
      <main class="archive-main" id="main-content">
        <header class="archive-intro">
          <h2><span data-lang="en">Learning paths</span><span data-lang="zh">模仿学习、VLA 与强化学习</span></h2>
          <p><span data-lang="en">Seven connected paths through representations, imitation learning, VLA architectures, engineering, runtime, tactile robotics, and reinforcement learning. The RL path connects actor–critic networks, PPO training, teacher–student transfer, and a detailed HORA walkthrough.</span><span data-lang="zh">按知识依赖分成七卷。从多模态表示进入模仿学习、VLA、工程与触觉机器人；RL 路线依次展开 Actor–Critic、PPO 训练闭环、Teacher–Student 与 HORA 精讲。</span></p>
        </header>
        <div class="series-grid">{series_cards}</div>
        <div class="archive-filters" role="group" aria-label="按知识链筛选文章">
          <button class="archive-filter is-active" type="button" data-filter="all" aria-pressed="true"><span data-lang="en">All</span><span data-lang="zh">全部</span></button>
          {filter_buttons}
        </div>
        {series_sections}
        <p class="archive-empty" hidden><span data-lang="en">There are no posts in this category yet.</span><span data-lang="zh">这个分类下暂时还没有文章。</span></p>
      </main>
    </div>
    <footer class="blog-footer">© 2026 Xiyue Dong · Learning Journey</footer>
  </body>
</html>
"""


def redirect_page(target_slug: str) -> str:
    """Keep links to retired foundation articles working after consolidation."""
    target = f"{target_slug}.html"
    return f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url={target}" />
    <link rel="canonical" href="{target}" />
    <title>文章已整合</title>
  </head>
  <body>
    <p>这篇基础文章已整合进新的学习路线。<a href="{target}">前往新文章</a></p>
  </body>
</html>
"""


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    site_root = args.site_root.resolve()
    blog_dir = site_root / "blog"
    source_copy_dir = blog_dir / "source"
    diagram_dir = blog_dir / "assets" / "diagrams"
    source_copy_dir.mkdir(parents=True, exist_ok=True)
    diagram_dir.mkdir(parents=True, exist_ok=True)

    posts_by_source = {post.source: post for post in POSTS}
    reading_times: dict[str, int] = {}
    for post in POSTS:
        upstream = source_dir / post.source
        bundled = source_copy_dir / post.source
        if args.sync_upstream and upstream.exists():
            shutil.copy2(upstream, bundled)
        elif not bundled.exists():
            raise FileNotFoundError(f"No Markdown source found for {post.source}")
    if args.sync_upstream:
        for diagram_path in source_dir.glob("*.drawio"):
            shutil.copy2(diagram_path, diagram_dir / diagram_path.name)

    for index, post in enumerate(POSTS):
        raw = (source_copy_dir / post.source).read_text(encoding="utf-8")
        reading_times[post.code] = reading_time(raw)
        raw_without_title = re.sub(r"^[ \t]*# .+?\n+", "", raw, count=1)
        rewritten = rewrite_links(raw_without_title, posts_by_source)
        body, toc = render_markdown(rewritten)
        diagram = diagram_embed(post.diagram, diagram_dir) if post.diagram else ""
        previous = POSTS[index - 1] if index else None
        following = POSTS[index + 1] if index + 1 < len(POSTS) else None
        output = article_page(post, body, toc, diagram, previous, following, reading_times[post.code])
        (blog_dir / f"{post.slug}.html").write_text(output, encoding="utf-8")

    for legacy_slug, target_slug in LEGACY_REDIRECTS.items():
        (blog_dir / f"{legacy_slug}.html").write_text(redirect_page(target_slug), encoding="utf-8")

    (blog_dir / "index.html").write_text(index_page(reading_times), encoding="utf-8")
    print(f"Built {len(POSTS)} posts in {blog_dir}")


if __name__ == "__main__":
    main()
