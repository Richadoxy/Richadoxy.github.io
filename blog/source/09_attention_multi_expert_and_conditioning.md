# 多模态融合、多专家与条件注入

前面的 Attention 专题解释了 Q/K/V、self-attention 和 cross-attention。本篇进一步回答：视觉、语言、state、action 和 timestep 具体可以通过哪些接口影响同一个策略。

## 1. 对齐、融合与条件不是同一个概念

```text
alignment：训练目标约束不同表示在语义空间中的关系
fusion：一次 forward 中，不同信息怎样交换
conditioning：任何影响输出的已知信息及其注入方式
```

例如 TVL 用对比 loss 对齐独立的触觉、视觉和文本 encoder；GR00T 的 action stream 用 cross-attention 融合 VLM memory；flow timestep 则通过 AdaNorm 调节动作网络。

## 2. 独立 Encoder + 表示对齐

```text
touch  -> Touch Encoder  -> z_T
image  -> Vision Encoder -> z_V
text   -> Text Encoder   -> z_L

contrastive loss: matched embeddings close
```

这种方式在 forward 时不要求不同模态的 token 彼此读取，适合：

- 单模态也要独立推理；
- 数据经常缺少某些模态；
- 检索、zero-shot 分类或预训练触觉 encoder；
- 缓存和复用全局 embedding。

它建立语义坐标，但不会自动产生任务相关的细粒度多模态交互。

## 3. Token Concatenation 与 Joint Attention

先将各模态映射到兼容的 hidden width，再拼成一个序列：

```text
[vision tokens, language tokens, touch tokens, state tokens]
                         -> shared Transformer
```

如果 Q/K/V 都来自拼接后的序列，形式上属于 self-attention；因为不同模态 token 在同一 attention matrix 中交流，也常被称为 joint attention 或 early fusion。

Mask 决定信息方向。例如：

```text
bidirectional：所有有效 token 彼此可见
causal：后面位置只能读取允许的历史
block mask：某些模态区段只能单向读取另一部分
```

FuSe-Octo 将 vision、touch、audio、language 和 readout token 放进共享 Transformer，属于这一类。

## 4. Cross-Attention

Cross-attention 显式区分 query stream 和 condition memory：

```text
action/state tokens -> Q
VLM memory          -> K/V
```

它适合以下结构：

- VLM 先独立产生可缓存的 memory；
- Action Network 有自己的 hidden width 和层数；
- 动作 token 需要按层检索视觉语言信息；
- 两个模态的序列长度和表示宽度不同。

Q 与 K 的原始宽度不必相等。各自的投影矩阵会把它们映射到相同 head dimension，再计算点积。

GR00T 的 AlternateVLDiT 就让 state/action query stream 交替读取文字与图像 memory。

## 5. PI0.5 的 Masked Joint Attention

PI0.5 不是“Gemma 2B 完整结束后输出一个向量，再交给 Gemma 300M”。两路 expert 在 paired layers 中协同：

```text
Prefix Expert hidden [B,Lp,2048] -> prefix-specific Q/K/V
Action Expert hidden [B,H,1024]   -> action-specific Q/K/V
                                ↓
                  common attention geometry
```

Mask 保证：

```text
Prefix query -> Prefix K/V
Action query -> Prefix + Action K/V
```

因此 action suffix 能读取图像、语言和 state，而 prefix 不会被 noisy action 污染。

## 6. Projector 与 Pooled Condition

并非所有系统都保留完整 token sequence。另一种路线是先得到全局表示，再映射到目标模型空间：

```text
tactile global embedding
-> MLP Projector
-> LLM hidden width
-> multimodal condition/token
```

Projector 同时解决维度和表示空间不兼容。它可以只是线性层，也可以是两层 MLP、Q-Former 或 resampler。

一个 pooled vector 成本低，但可能丢失局部接触位置、时间变化和 patch 对应关系。

## 7. Gate 与 FiLM

Gate 控制新模态对原 backbone 的影响强度：

\[
h'=h+g\cdot F(h,m)
\]

如果 \(g\) 零初始化，训练开始时模型保持原始行为，之后逐渐学会使用新输入。TVL-LLaMA 使用的 gated adapter 可以从这个角度理解。

FiLM 则根据 condition 产生通道级 scale 和 shift：

\[
h'=\gamma(c)\odot h+\beta(c)
\]

它不要求 condition 占据序列位置。

## 8. AdaLN 与 AdaRMS

Diffusion/flow timestep 通常是全局噪声阶段，更适合调节整层计算：

```text
t -> sinusoidal embedding -> MLP -> condition c
```

然后生成：

```text
scale, shift, optional gate = Linear(c)
normalized = Norm(x)
modulated = normalized * (1 + scale) + shift
```

AdaLN 使用 LayerNorm，AdaRMS 使用 RMSNorm。condition 没有序列位置，也不产生自己的 Q/K/V。

可以把 token condition 理解为“加入会议的参与者”，把 AdaNorm 理解为“调整整间会议室的工作模式”。

## 9. State 与 Timestep 可以走不同路径

同一个语义在不同模型中可能采用不同接口：

```text
PI0.5 default state：离散化后进入 VLM prompt
PI0 state：连续 state token
PI0.5 + VLASH：future state 进入 AdaRMS condition
GR00T state：embodiment-specific State Encoder -> DiT token
VLA-JEPA state：State Encoder -> Action Head
```

Flow timestep 也可以：

- 与 action embedding 拼接；
- 加到 action token；
- 通过 AdaLN/AdaRMS 调制；
- 同时使用 token fusion 和 adaptive normalization。

“作为 condition”不能唯一确定网络结构，必须继续看注入位置。

## 10. Multi-expert 不一定是 Routed MoE

典型稀疏 MoE：

```text
token -> router -> top-k experts -> weighted merge
```

PI0.5 multi-expert：

```text
image/text/state prefix -> Prefix Expert
noisy action suffix     -> Action Expert
```

后者由 token 区段固定选择参数，没有 router 和 top-k。它是 expert-specific weights 的协同 Transformer，但不是常见的 sparse routed MoE。

## 11. 几类模型的统一位置

| 模型 | 表示对齐 | 前向融合 | 其他条件 |
| --- | --- | --- | --- |
| TVL Encoder | Touch/Vision/Text InfoNCE | encoder 间不交换 token | 无 |
| AnyTouch | 多模态对比 + 跨传感器匹配 | Touch Encoder 内 self-attention | sensor token |
| Octopi | 属性分类与语言监督 | tactile embeddings 插入 Vicuna 序列 | 无显式 gate |
| FuSe-Octo | 语言对比/生成辅助 loss | 多模态 joint attention | readout tokens |
| PI0.5 | 机器人 flow supervision | prefix/action masked joint attention | timestep AdaRMS |
| GR00T | 机器人 flow supervision | DiT cross-attention + self-attention | timestep AdaLN |

## 12. 设计或阅读时的检查表

1. 输入是 global embedding 还是 token sequence？
2. 不同模态在 forward 的哪一层第一次交换信息？
3. 它们使用 self、cross、joint attention，还是只有对比 loss？
4. condition 是否占据序列位置？
5. 是否存在 projector、gate、FiLM 或 AdaNorm？
6. attention mask 允许哪些方向？
7. 新模态能否在缺失其他模态时独立工作？
8. 训练 loss 是否迫使策略真正使用新增模态？

一句话总结：

> Alignment 规定表示应该靠近谁，Attention 规定一次前向读取谁，Conditioning 规定哪些已知信息以什么接口影响输出。
