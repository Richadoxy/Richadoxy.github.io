# 14. PI0.5、GR00T、VLASH 与 VLA-JEPA 对照

这一章不是再介绍一个新模型，而是把前面的概念压缩成一张知识地图，帮助阅读新 VLA 论文和代码。

## 1. 四者不在完全相同的层级

```text
PI0.5
  = 一套 VLM Prefix + Action Expert 的 VLA 架构

GR00T
  = 一套 VLM memory + embodiment-conditioned DiT 的 VLA 架构

VLASH
  = 可接到现有 VLA 上的训练时间对齐 + 异步部署方法

VLA-JEPA
  = VLM + DiT policy，并增加 latent world-model 训练监督
```

因此：

- PI0.5 与 GR00T 更适合直接比较 backbone/action architecture；
- VLASH 更适合与 synchronous/naive async/RTC 等部署方法比较；
- VLA-JEPA 更适合与 latent-action pretraining、world-model supervision 方法比较。

## 2. 总体数据流

### PI0.5

```text
image + language + discrete state -> Prefix Expert
noisy action + flow t            -> Action Expert
paired joint attention            -> velocity
```

### GR00T

```text
image + language -> VLM memory
state + noisy action -> DiT query tokens
cross/self attention -> velocity
```

### VLASH on PI0.5

```text
fixed image o_t + future state s_(t+δ)
  -> PI0.5
  -> action chunk beginning at t+δ

runtime: execute old chunk while inferring next chunk
```

### VLA-JEPA

```text
current image + language -> Qwen latent/action conditions
  ├-> World Predictor -> future latent loss against frozen V-JEPA2
  └-> DiT Action Head -> flow velocity loss
```

## 3. 核心结构对照

| 维度 | PI0.5 | GR00T N1.7 | VLASH | VLA-JEPA |
| --- | --- | --- | --- | --- |
| 模型/方法层级 | VLA architecture | VLA architecture | async framework | VLA + JEPA training framework |
| VLM | Gemma 2B/PaliGemma prefix | Qwen3-VL/Cosmos backbone | 复用目标 VLA | Qwen3-VL |
| action network | Gemma 300M Action Expert | AlternateVLDiT | 复用目标 VLA | DiT-B Action Head |
| VLM–Action 连接 | 每层 masked joint attention | cross-attention to VLM memory | 不强制改变 | `z_a`/context condition Action Head |
| state 入口 | 默认离散 state prompt | continuous state token | execution-time future state | State Encoder -> DiT |
| timestep | adaRMS condition | Action Encoder fusion + AdaLN | 沿用基础模型 | DiT condition |
| 主要训练 loss | flow matching | flow matching | flow loss不变、改样本 offset | `L_FM + βL_WM` |
| 部署附加模块 | 无 | 无 | async chunk manager | teacher/predictor 移除 |

## 4. 典型 hidden widths

| 位置 | 宽度含义 | 示例 |
| --- | --- | --- |
| PI0.5 Prefix | 每个 image/text token 的 hidden width | 2048 |
| PI0.5 Action Expert | 每个 action position 的 hidden width | 1024 |
| GR00T VLM memory | 每个 vision/language token 的 hidden width | 2048 |
| GR00T DiT | 每个 state/action token 的 hidden width | 1536 = 32×48 |
| VLA-JEPA Qwen | VLM token hidden width | 取决于 Qwen 配置 |
| VLA-JEPA world state | V-JEPA2 target latent width/sequence | 由 target encoder 和多视角组合决定 |

宽度描述一次 activation 的最后一维，不是整个模型参数量，也不是 action dimension。

## 5. State 到底去哪里

```text
PI0.5 默认：
  normalized state -> discrete values -> prompt tokens -> VLM prefix

PI0：
  continuous state -> state_proj -> suffix state token

PI0.5 + VLASH state_cond：
  future state -> StateMLP -> adaRMS condition

GR00T：
  continuous state -> embodiment-specific MLP -> DiT state token

VLA-JEPA：
  proprioceptive state -> State Encoder -> DiT Action Head
  teacher world-state latent 是视频语义状态，不是 robot proprioception
```

最后一条尤其容易混淆：VLA-JEPA 论文中的 `s_t` 常表示 V-JEPA2 world-state latent，而机器人关节 state 是 Action Head 的另一条输入。

## 6. Attention 连接方式

```text
PI0.5：
  Prefix/Action 各自 QKV
  -> 在共同 attention geometry 中按位置联合
  -> mask 控制信息方向

GR00T：
  state/action Q
  -> cross-attend VLM K/V
  -> 与 state/action self-attention 交替

VLA-JEPA world predictor：
  history state latents + latent actions
  -> time-causal attention
  -> future latent prediction
```

看到“VLM 给 Action Expert 提供条件”时，不要马上假设是 cross-attention；它也可能是 joint attention、condition token、AdaNorm 或单个 pooled vector。

## 7. 训练和推理模块要分开画

论文框图经常把训练监督画得很大，但不代表部署也需要这些模块。

| 模块 | 训练 | 推理 |
| --- | --- | --- |
| PI0.5 flow noise/target action | 需要 | 不需要真实 action，只需要初始 noise |
| VLASH temporal offset dataset | 需要 | 不需要 dataset offset sampling |
| VLASH async scheduler | 不属于模型训练 | 需要 |
| VLA-JEPA Frozen V-JEPA2 | 需要 target | 移除 |
| VLA-JEPA World Predictor | 需要 `L_WM` | 移除 |
| DiT/Action Expert | 需要 | 需要 |

阅读任何框图时，先给每条边标注：

```text
training-only
inference-only
both
```

## 8. 常见概念误区

### 误区 1：2B 就是 token 维度

错误。2B 是参数量，2048 才是特定配置中的 token hidden width。

### 误区 2：所有 condition 都是 token

错误。condition 可以是 token、cross-attention K/V、AdaNorm scale/shift/gate，或者直接加到 embedding。

### 误区 3：训练 t 按推理 10 步离散

通常错误。训练随机采连续 t；推理步数是数值积分精度与延迟的选择。

### 误区 4：Action Encoder 等于 Transformer Encoder

错误。它可能只是动作模态适配器。应看输入输出和内部深度。

### 误区 5：VLASH 预测 future image

错误。它保留 stale visual `o_t`，主要 roll forward 可预测的 robot state。

### 误区 6：VLA-JEPA 推理时运行 world model

论文标准部署路径不需要。world predictor 和 frozen target encoder主要用于训练监督。

### 误区 7：PI0.5 multi-expert 是 router MoE

错误。Prefix/Action expert 由模态固定分配，不是 router top-k 选择。

## 9. 阅读一篇新 VLA 论文的顺序

建议不要从模型名字开始，而是依次回答以下问题。

### 第一步：输入与输出

```text
输入哪些相机、语言、state、history？
输出 action position、delta、velocity 还是 torque？
action_dim 和 horizon 是多少？
```

### 第二步：模态如何变成 hidden representations

```text
图像 encoder 是什么？
state 是 token 还是 condition？
noisy action 如何 projection？
内部 width 与原始 action_dim 各是多少？
```

### 第三步：信息如何融合

```text
self-attention？
cross-attention？
joint attention？
AdaLN/AdaRMS？
attention mask 的方向是什么？
```

### 第四步：训练目标

```text
behavior cloning？
flow matching/diffusion？
latent prediction？
world-model loss？
哪些参数被冻结？
```

### 第五步：推理与部署

```text
去噪几步？
输出整个 chunk 还是一个 action？
同步还是异步？
执行多少步后 re-plan？
训练辅助模块是否移除？
```

### 第六步：检查信息泄漏与时间对齐

```text
future frame 是否进入部署可见路径？
训练 state/action 是否对应同一执行时刻？
推理延迟是否使 observation stale？
```

## 10. 工程选型时真正需要测的量

仅知道模型 forward 约 90 ms 不够，还应记录：

- capture 到 request ready；
- serialization/network；
- server queue；
- prefix prefill；
- 全部 denoising steps；
- response decode；
- Actor 收到结果到 chunk handoff；
- 实际 control rate 和 policy rate；
- p50/p95/p99，而不是只有平均值。

VLASH 的 offset 应依据 observation 到实际 handoff 的完整延迟；Action Expert 推理步数则依据精度–延迟权衡。

## 11. 配套图与文档

- [PI0.5 结构图](pi05_structure.drawio)
- [GR00T N1.7 结构图](groot_n1d7_structure.drawio)
- [VLASH 训练与异步运行图](vlash.drawio)
- [VLA-JEPA 结构图](vla_jepa_structure.drawio)
- [PI0.5 代码架构说明](04_pi05_architecture.md)
- [BrainCo-IL VLASH 实现说明](07_vlash_integration.md)

如果只记住一条主线，可以记成：

```text
原始模态
-> token/condition representations
-> attention 与 adaptive modulation
-> action velocity 或 target latent
-> training loss
-> inference integration / runtime scheduling
```
