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
    Post("F01", "08_transformer_latent_encoder_decoder.md", "08-transformer-latent-encoder-decoder", "Token、Embedding、Encoder/Decoder 与 Latent", "建立 token、hidden width、latent、encoder、decoder 和参数量之间的基础概念坐标系。", "foundations", level="入门"),
    Post("F02", "19_attention_from_embeddings_to_action_expert.md", "19-attention-openclip-tactile-groot", "Transformer 与 Attention：从 QKV 到多模态交互", "区分动态 attention matrix 与可训练参数，并系统理解 self、cross 与 joint attention。", "foundations", published_en="September 2026", published_zh="2026.09", level="入门"),
    Post("F03", "21_clip_and_contrastive_learning.md", "clip-contrastive-representation-learning", "表示学习、CLIP 与对比学习", "从双塔 encoder、global embedding 和 InfoNCE 出发，区分表示对齐、前向融合与下游控制。", "foundations", published_en="September 2026", published_zh="2026.09", level="入门"),
    Post("F04", "09_attention_multi_expert_and_conditioning.md", "09-attention-multi-expert-conditioning", "多模态融合、多专家与条件注入", "比较 token concat、joint/cross attention、multi-expert 与 AdaLN/AdaRMS 条件调制。", "foundations", level="进阶"),
    Post("F05", "22_model_training_basics.md", "model-training-forward-loss-backward", "模型训练基础：Forward、Loss、Backward 与微调", "建立一次参数更新的完整闭环，并区分 freeze、LoRA、主损失、辅助损失及训练专用模块。", "foundations", published_en="September 2026", published_zh="2026.09", level="入门"),
    Post("IL01", "23_imitation_learning_basics.md", "imitation-learning-behavior-cloning", "模仿学习与 Behavior Cloning 入门", "从 demonstration、policy 和 BC 出发，理解离线监督、闭环 rollout 与分布偏移。", "imitation", published_en="September 2026", published_zh="2026.09", level="入门"),
    Post("IL02", "24_robot_data_action_semantics.md", "robot-data-action-semantics", "机器人数据：Observation、Action 与时间窗口", "统一 state/action 语义、episode、action chunk、horizon、控制频率、归一化和 mask。", "imitation", published_en="September 2026", published_zh="2026.09", level="入门"),
    Post("IL03", "25_action_generation_families.md", "action-generation-mse-token-diffusion-flow", "动作生成方法：MSE、Token、Diffusion 与 Flow Matching", "比较四类动作生成目标，区分网络架构、训练 target 与推理采样方式。", "imitation", published_en="September 2026", published_zh="2026.09", level="进阶"),
    Post("VLA00", "27_vla_overview.md", "vla-overview-reading-map", "从模仿学习到 VLA：架构地图与阅读方法", "用统一的数据流和六步检查法定位 VLM、动作头、训练目标与部署系统。", "architectures", published_en="September 2026", published_zh="2026.09", level="入门"),
    Post("VLA01", "10_flow_matching_training_and_inference.md", "10-flow-matching-training-inference", "Flow Matching：训练、timestep 与推理", "从插值路径、velocity target 和 timestep condition 出发，区分训练采样与推理积分。", "architectures"),
    Post("VLA02", "04_pi05_architecture.md", "04-pi05-architecture", "PI0.5：VLM Prefix 与 Action Expert", "拆解 VLM Prefix Expert、Action Expert、联合 attention 和 flow-matching action generation。", "architectures", "pi05_structure.drawio"),
    Post("VLA03", "11_groot_dit_architecture.md", "11-groot-dit-architecture", "GR00T：VLM Memory 与 DiT Action Head", "梳理 VLM memory、state/action encoders、AlternateVLDiT、AdaLN 与 action decoder 的数据流。", "architectures", "groot_n1d7_structure.drawio"),
    Post("VLA04", "14_architecture_comparison_and_reading_guide.md", "14-architecture-comparison-reading-guide", "PI0.5、GR00T、VLASH 与 VLA-JEPA 对照", "把架构、异步框架和 world-model supervision 放到正确层级，并形成统一比较坐标。", "architectures"),
    Post("VLA05", "13_jepa_and_vla_jepa.md", "13-jepa-vla-jepa", "JEPA、World Model 与 VLA-JEPA", "理解 teacher/student/predictor、latent world-model supervision、leakage-free training 和联合 action loss。", "architectures", "vla_jepa_structure.drawio", level="前沿"),
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
]


CHAINS = {
    "foundations": {
        "title": "第一卷：基础概念",
        "title_en": "Part 1: Foundations",
        "label_zh": "基础概念",
        "label_en": "Foundations",
        "description": "先统一 token、encoder、Attention、CLIP、多模态融合和训练过程中的核心词汇。",
    },
    "imitation": {
        "title": "第二卷：模仿学习",
        "title_en": "Part 2: Imitation Learning",
        "label_zh": "模仿学习",
        "label_en": "Imitation Learning",
        "description": "从 demonstration 和 Behavior Cloning 出发，理解机器人数据、action chunk 与不同动作生成目标。",
    },
    "architectures": {
        "title": "第三卷：VLA 核心架构",
        "title_en": "Part 3: VLA Architectures",
        "label_zh": "VLA 架构",
        "label_en": "VLA Architectures",
        "description": "使用统一坐标拆解 Flow Matching、PI0.5、GR00T、VLASH 和 VLA-JEPA。",
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
        "description": "从 TVL 与 AnyTouch 的触觉表征，进入 Octopi 的语言推理和 FuSe 的动作决策。",
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
    """Keep MathJax delimiters from being consumed as Markdown escapes."""
    replacements = {
        "MATHDISPLAYOPEN42": r"\[",
        "MATHDISPLAYCLOSE42": r"\]",
        "MATHINLINEOPEN42": r"\(",
        "MATHINLINECLOSE42": r"\)",
    }
    protected = text
    for marker, delimiter in replacements.items():
        protected = protected.replace(delimiter, marker)
    return protected, replacements


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
    <meta name="description" content="从机器学习基础、模仿学习到 VLA、OpenPI 工程、异步部署与触觉机器人的渐进学习路径。" />
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
          <h2><span data-lang="en">Learning paths</span><span data-lang="zh">从模仿学习到 VLA</span></h2>
          <p><span data-lang="en">A dependency-first map of machine-learning foundations, imitation learning, VLA architectures, engineering, runtime systems, and tactile robotics. Articles are grouped by learning path rather than publication order.</span><span data-lang="zh">这里不再按笔记产生时间排成一条长列表，而是按知识依赖分成六卷。初学者建议从 F01 开始；已有基础的读者可以直接选择 VLA、工程、部署或触觉分支。</span></p>
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
        raw_without_title = re.sub(r"^# .+?\n+", "", raw, count=1)
        rewritten = rewrite_links(raw_without_title, posts_by_source)
        rewritten, math_replacements = protect_math(rewritten)
        md = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "sane_lists"])
        body = md.convert(rewritten)
        for marker, delimiter in math_replacements.items():
            body = body.replace(marker, delimiter)
        diagram = diagram_embed(post.diagram, diagram_dir) if post.diagram else ""
        previous = POSTS[index - 1] if index else None
        following = POSTS[index + 1] if index + 1 < len(POSTS) else None
        output = article_page(post, body, md.toc, diagram, previous, following, reading_times[post.code])
        (blog_dir / f"{post.slug}.html").write_text(output, encoding="utf-8")

    (blog_dir / "index.html").write_text(index_page(reading_times), encoding="utf-8")
    print(f"Built {len(POSTS)} posts in {blog_dir}")


if __name__ == "__main__":
    main()
