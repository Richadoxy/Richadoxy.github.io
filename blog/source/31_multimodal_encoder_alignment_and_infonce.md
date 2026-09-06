# 从原始模态到公共语义空间：Encoder、ViT、CLIP 与 InfoNCE

> 本文承接统一研究范式，聚焦数据进入 Policy 之前的表示层与对齐层：原始信号怎样成为 token，ViT 与 CLIP 是什么关系，CLS/readout 与 projector 做什么，以及 InfoNCE 如何用锚点和正负样本塑造公共空间。

> 阅读位置：[统一研究范式](27_vla_overview.md) → 本文 → [多模态表示的监督配方](32_three_consistency_supervision_paradigms.md) → [从对齐表示到 Policy 融合](19_attention_from_embeddings_to_action_expert.md)。

## 0. 一张总图

```text
Raw inputs
image / tactile image / force sequence / text
                        ↓
Modality-specific interface
tokenizer / patch embedding / MLP
                        ↓
Encoder
ViT / CNN / text Transformer / temporal Transformer
                        ↓
Local tokens + global CLS/readout
                        ↓
Projector + normalization
                        ↓
Shared embedding space
                        ↓
InfoNCE / retrieval / zero-shot classification
```

这条链路回答的是：

> 每种模态怎样得到独立、可比较、可复用的表示？

它还没有回答：

> 多种模态怎样在一次 forward 中联合生成动作？

后一个问题属于 F04 的 Policy Fusion。

---

## 1. 从原始信号到 Token

### 1.1 不同触觉输入不是同一种数据

触觉论文中的输入可能是：

| 输入 | 例子 | 常见入口 |
| --- | --- | --- |
| 视触觉 RGB | DIGIT、GelSight 内部相机 | CNN/ViT |
| 深度或表面法向 | 由触觉图像重建的几何 | CNN/ViT |
| Marker displacement/flow | 标记点位移 | CNN/ViT/显式特征 |
| 二维力阵列 | Paxini force field | reshape/stitch 后用 ViT |
| 数值力序列 | normal/shear/6D wrench | MLP、1D CNN、时序 Transformer |
| 触觉视频 | 按压、滑动的连续帧 | Video Transformer/temporal pooling |

不能只看到 `tactile encoder` 就认为输入形式相同。输入形式决定 encoder 能利用哪些结构：

- 图像和力阵列具有二维空间邻域；
- 视频和力历史具有时间顺序；
- 单一全局力值没有局部接触位置。

### 1.2 Tokenizer、Embedding 和 Encoder

三者处于不同位置：

```text
原始输入
  ↓ tokenizer / patchify
输入单元
  ↓ embedding / input projection
初始hidden vectors
  ↓ encoder blocks
上下文化tokens
```

文本 tokenizer 把字符串变成 token IDs；ViT patchify 把图像分块。它们负责划分输入单元，不等于已经理解内容。

Encoder 通过多层网络让每个 token 结合上下文，输出 hidden representations。

### 1.3 一个 Token 与 Hidden Width

如果模型宽度为 \(d=2048\)，一个 token 的 hidden state 是：

\[
h_i\in\mathbb R^{2048}
\]

这表示一个 token 由 2048 个标量组成，不是“2048 个 token”。一组 \(N\) 个 token 才写成：

\[
H\in\mathbb R^{N\times2048}
\]

---

## 2. ViT：常见的图像与视触觉 Encoder

### 2.1 ViT 做什么

以 \(224\times224\) 图像和 \(14\times14\) patch 为例：

```text
224×224 image
   ↓ 切成14×14 patches
16×16 = 256 patches
   ↓ 线性投影
256 patch tokens
   ↓ 加CLS与position embedding
257 tokens
   ↓ Transformer self-attention blocks
contextual visual/tactile tokens
```

ViT 是网络架构，不规定预训练方式。同一个 ViT 可以使用：

- ImageNet 分类；
- CLIP 图文对比；
- MAE masked reconstruction；
- 触觉属性标签；
- 机器人 action loss。

### 2.2 CLS Token

CLS 是一个可学习的特殊输入位置：

```text
[CLS, patch₁, patch₂, ..., patch₂₅₆]
                 ↓ self-attention
hCLS逐层读取patch信息
```

初始 CLS embedding 对所有样本相同；经过 Transformer 后，最终 \(h_{CLS}\) 会随输入改变，可用作整张图的 global representation。

### 2.3 CLS、Mean Pooling 与 Readout

全局表示可以来自：

- CLS token；
- 所有 patch token 的均值；
- attention pooling；
- learnable readout/query tokens。

CLS 可以视为经典的 readout token，但机器人 Policy 中的 action readout 通常更加任务化。CLS 总结一张输入，action readout 则汇总生成动作所需的信息。

### 2.4 为什么 ViT 能处理触觉

ViT 不要求输入一定是互联网照片。只要触觉信号能形成二维 patch，就能使用同一计算结构：

```text
GelSight/DIGIT RGB       → patches → tactile ViT
stitched force field     → patches → force ViT
tactile frame sequence   → spatial + temporal tokens
```

但使用相同架构不代表自然图像和触觉已经对齐。训练目标仍需要克服 domain gap。

---

## 3. CLIP 与 ViT 不是同一层级

### 3.1 CLIP 是双 Encoder 训练体系

```text
Image → Image Encoder → global image embedding zV
Text  → Text Encoder  → global text embedding zL
```

正确图文配对被拉近，不匹配图文被推远。

CLIP 的 image encoder 可以是 ResNet 或 ViT；现代 checkpoint 常使用 ViT。因此：

| 名称 | 含义 |
| --- | --- |
| Transformer | 通用网络架构 |
| ViT | 图像 Transformer encoder |
| CLIP | 图像—文本双塔与对比学习方法 |
| CLIP ViT-L/14 | 用 CLIP 目标训练的 ViT-L/14 图像塔 |
| OpenCLIP | CLIP 的开源训练实现和模型集合 |

一句话：

> ViT 规定视觉 encoder 怎样计算；CLIP 规定图像与文本表示怎样共同训练。

### 3.2 CLIP 不是生成式 VLM

CLIP 的典型输出是用于相似度的 global embeddings：

```text
image/text → embedding → cosine similarity
```

它本身不负责长文本生成。LLaMA、Vicuna、Gemma 等语言模型则处理 token sequence 并生成文本。把触觉接入 LLM 时，通常还需要 projector、adapter 或多模态 connector。

### 3.3 为什么使用 CLIP 权重初始化触觉 ViT

```text
CLIP Vision ViT
      ↓ copy/init weights
Tactile ViT
      ↓ tactile adaptation
tactile encoder
```

这种初始化能提供通用纹理、边缘、空间和视觉语义先验，但不能自动消除自然 RGB 与触觉成像之间的域差异。

使用 `CLIP backbone` 也不等于后续一定继续做 CLIP 对齐。例如 Octopi 使用 CLIP ViT 初始化，但其 encoder 适配主要由物理属性分类监督。

---

## 4. Projector：对齐接口，不负责自动对齐语义

不同 encoder 的输出宽度可能不同：

```text
Touch Encoder  → 768D
Vision Encoder → 1024D
Text Encoder   → 512D
```

为计算相似度，使用各自 projector：

\[
z_T=P_T(E_T(T)),\quad
z_V=P_V(E_V(V)),\quad
z_L=P_L(E_L(L))
\]

映射到统一维度后再归一化：

\[
\bar z=\frac{z}{\|z\|_2}
\]

归一化后的点积等价于余弦相似度：

\[
\operatorname{sim}(z_i,z_j)=\bar z_i^\top\bar z_j
\]

Projector 只解决接口问题。真正塑造公共空间的是 loss 和配对数据。

还需要区分两个用途：

| Projector 用途 | 目标空间 |
| --- | --- |
| Contrastive projector | CLIP/InfoNCE 共享 embedding space |
| LLM/VLA connector | LLM 或 Policy 的 hidden width |

二者都叫 projector，但目标完全不同。

---

## 5. 公共语义空间到底是什么

给定三模态样本 \((T_i,V_i,L_i)\)：

```text
Touch    → ET → zT
Vision   → EV → zV
Language → EL → zL
```

对齐希望：

\[
z_T\approx z_V\approx z_L
\]

“约等于”表示：

- 匹配样本余弦相似度高；
- 不匹配样本相似度低；
- 可以进行 touch-to-text、touch-to-vision 检索；
- 相似物体或属性形成相近邻域。

它不表示：

- 三个 encoder 的内部结构相同；
- patch tokens 逐位置相等；
- 所有物理信息都被保留；
- 对齐后已经学会机器人动作。

### 5.1 对齐通常发生在 Global Embedding

```text
patch/token sequence
      ↓ CLS/pooling
global representation
      ↓ projector + normalize
contrastive embedding
```

因此全局检索成功，不代表局部触觉区域与视觉位置已经精确对应。细粒度 token alignment 需要更强的空间标注或专门匹配结构。

---

## 6. 锚点：谁定义公共空间的坐标

### 6.1 固定视觉—语言锚点

最清楚的设置是：

```text
frozen CLIP Vision Encoder ─→ zV
frozen CLIP Text Encoder   ─→ zL
trainable Tactile Encoder  ─→ zT
```

已有 \(z_V,z_L\) 的几何结构基本不动，触觉表示被拉入该空间。

### 6.2 锚点不一定是语言

| 锚点或监督 | 更容易获得 |
| --- | --- |
| Language | 开放词汇属性与对象语义 |
| Vision | 对象身份、几何、位置与外观 |
| Cross-sensor pair | 设备不变的触觉属性 |
| Temporal relation | 接触状态的连续性 |
| Reconstruction target | 局部结构与高信息量细节 |
| Robot action | 对控制任务有用的表示 |

语言是常用锚点，但不是天然最优锚点。它可能压缩无法被文字充分描述、却对控制关键的滑移和瞬时力变化。

### 6.3 锚点也可能缓慢移动

| 参数策略 | 空间行为 |
| --- | --- |
| text/vision 全冻结 | 稳定固定锚点 |
| text 冻结、vision LoRA | 以语言为主，视觉适应新 domain |
| 三个 encoder 都训练 | 各模态共同形成新空间 |

如果所有 encoder 都更新，就没有绝对固定的坐标原点；“锚点”只是哪个预训练结构被更强地保留。

---

## 7. InfoNCE 怎样完成对齐

### 7.1 正样本与负样本

一个 batch 有 \(N\) 组同步 touch-vision 数据：

\[
(T_1,V_1),\ldots,(T_N,V_N)
\]

对于 \(T_i\)：

- \(V_i\) 是正样本；
- 其他 \(V_j,j\ne i\) 通常作为 batch 内负样本。

相似度为：

\[
s_{ij}=\frac{\bar z_{T_i}^{\top}\bar z_{V_j}}{\tau}
\]

### 7.2 单向 InfoNCE

\[
\mathcal L_{T\rightarrow V}
=
-\frac{1}{N}\sum_{i=1}^{N}
\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{N}\exp(s_{ij})}
\]

它相当于：

> 给定 \(T_i\)，在 batch 的 \(N\) 个视觉候选中分类出正确的 \(V_i\)。

### 7.3 双向与三模态 InfoNCE

通常对称计算：

\[
\mathcal L_{TV}
=
\frac{
\mathcal L_{T\rightarrow V}
+
\mathcal L_{V\rightarrow T}}
{2}
\]

三模态可以组合三条边：

\[
\mathcal L_{\mathrm{align}}
=
\alpha_{TV}\mathcal L_{TV}
+
\alpha_{TL}\mathcal L_{TL}
+
\alpha_{VL}\mathcal L_{VL}
\]

```text
Touch  ↔ Vision
Touch  ↔ Language
Vision ↔ Language
```

### 7.4 Temperature

\(\tau\) 控制 softmax 的尖锐程度：

- 小 \(\tau\)：更严格区分最相似候选；
- 大 \(\tau\)：相似度分布更平滑。

### 7.5 为什么不能只拉近正样本

如果两个可训练 encoder 只优化：

\[
\|z_T-z_V\|_2^2
\]

所有输入输出同一个常数向量也能使 loss 很低，这叫 representation collapse。

InfoNCE 同时拉近正确配对、推远错误配对，因此保存判别结构。非对比方法则需要 stop-gradient、teacher–student、variance regularization 或 reconstruction 等其他防坍塌机制。

---

## 8. 正负样本定义比公式名字更重要

跨模态研究真正困难的是定义“相同”：

| 正样本定义 | 学到的关系 |
| --- | --- |
| 同一物体 | 对象身份和材料共性 |
| 同一时间点 | 视觉变化与当前接触同步 |
| 同一接触位置 | 局部几何对应 |
| 同一属性描述 | 开放词汇物理语义 |
| 同一位置、不同传感器 | 跨设备不变性 |

负样本也可能有问题：两块不同海绵虽然是不同实例，却共享 `soft/porous` 语义。如果机械地当成负样本，会制造 false negatives。

阅读论文时应优先问：

1. 正样本按对象、时间还是接触位置配对？
2. batch 内其他样本是否真的是负样本？
3. 配对是否严格同步？
4. 文本描述是实测属性、人工标注还是 LLM 推测？

---

## 9. 缺失模态和星形对齐

### 9.1 三元组不是必要条件

如果数据分别包含：

```text
Touch + Vision → 计算TV loss
Touch + Text   → 计算TL loss
Vision + Text  → 计算VL loss
```

可以按样本存在的模态选择 loss，而不必丢弃不完整数据。

### 9.2 星形对齐

以视觉作为中心：

```text
                 Text
                   ↑
Audio ←──────── Vision ────────→ Touch
                   ↓
                 Depth
```

即使 Touch 与 Text 没有直接配对，也期待它们通过共同视觉锚点间接接近。这能降低数据采集成本，但传递关系不保证非中心模态精确对齐。

拥有完整 touch-vision-language 三元组时，可以直接增加 \(TL\) 边，减少间接对齐误差。

---

## 10. 哪些参数更新

常见策略包括：

### 10.1 只训练新模态

```text
Vision Encoder  ❄
Text Encoder    ❄
Tactile Encoder 🔥
Projector       🔥
```

优点是稳定保留原语义空间；缺点是所有 domain adaptation 压力都落在触觉侧。

### 10.2 部分适配锚点

```text
Text Encoder             ❄
Vision Encoder LoRA/top blocks 🔥
Tactile Encoder          🔥
```

让视觉适应机器人或触觉采集 domain，同时限制语义空间漂移。

### 10.3 端到端共同训练

灵活性最高，但需要更多数据和正则，否则可能遗忘预训练知识或破坏已有视觉—语言对齐。

### 10.4 预训练后接 Policy

对齐 encoder 可以：

- 在 Policy 训练时冻结；
- 使用小学习率或 LoRA；
- 与 Action Expert 一起 full fine-tune。

后两种可能产生 representation drift，因此不能默认预训练的对齐性质在 action fine-tuning 后完整保留。

---

## 11. 对齐成功应该怎样验证

### 11.1 Retrieval

```text
Touch → 从候选Text/Vision中找正确配对
```

Top-1 表示第一名正确，Top-5 表示正确答案出现在前五名。

### 11.2 Zero-shot Classification

把类别写成文本 prompt：

```text
“a soft object”
“a hard object”
```

选择与触觉 embedding 最相似的文本类别。

### 11.3 Linear Probe

冻结 encoder，只训练轻量线性分类器。如果性能良好，说明所需属性已较容易从 embedding 中读取。

### 11.4 下游 Policy

将 encoder 接入策略并测量闭环成功率。只有这一层才能回答表示是否真正帮助机器人控制。

需要避免跨层结论：

- retrieval 好不等于 action 好；
- reconstruction 好不等于控制好；
- action loss 低不等于闭环 rollout 稳健。

---

## 12. 已读论文的对齐路线

| 工作 | Encoder/初始化 | 是否显式建立公共空间 | 主要监督 |
| --- | --- | --- | --- |
| TVL | 新 tactile ViT + OpenCLIP V/T | 是，直接三边对齐 | Touch–Vision–Language InfoNCE |
| Octopi | CLIP ViT + prompt tuning | 不是典型三模态对齐 | 三个物理属性分类 |
| AnyTouch | OpenCLIP-based touch encoder | 是 | 重建、三模态 InfoNCE、跨传感器 matching |
| Beyond Sight/FuSe | 复用 TVL touch encoder | 策略 readout 与文本对齐 | T5 对比 + 语言生成 + action loss |
| OmniVTLA | Pretrained ViT + SA-ViT | SA-ViT 显式对齐 | 三模态 InfoNCE + cross-sensor matching |
| Tactile-VLA | 数值力历史 MLP | 没有独立对齐 | position-force action CFM |
| VLA-Touch | Octopi + 外挂 controller | 没有统一对齐阶段 | 高层规划与低层动作修正 |
| VT-MUSE Stage I | 两个独立 pretrained ViT | 只对齐 Vision–Touch | Cross-modal、temporal、mask consistency |

这张表强调：

> 使用 ViT 或 CLIP 权重不等于做了跨模态对齐；只有数据配对和 loss 明确约束公共空间时，才能称为 alignment stage。

---

## 13. 常见误区

### 误区一：CLIP 就是 ViT

ViT 是架构，CLIP 是双模态训练体系。

### 误区二：Projector 完成语义对齐

Projector 只做映射；loss 和数据关系塑造语义。

### 误区三：三个 Embedding 必须数值完全相同

对齐要求相对相似性和邻域结构，不是逐维复制。

### 误区四：有语言就是以语言为锚点

还需要看 text encoder 是否冻结、语言 loss 权重以及其他模态是否共同更新。

### 误区五：CLIP 初始化就是 CLIP-style 对齐

网络可以用 CLIP 权重初始化，随后只做属性分类或 action learning。

### 误区六：对齐以后自然能输出动作

对齐只建立表示接口，还需要 Policy Fusion 与 action supervision。

---

## 14. 阅读 Encoder/Alignment 工作的检查表

1. 原始输入是 RGB、depth、marker flow、force field 还是数值序列？
2. Encoder 使用什么结构和预训练权重？
3. 输出完整 token sequence 还是 global embedding？
4. Global representation 来自 CLS、pooling 还是 readout？
5. Projector 映射到对比空间还是 LLM/Policy hidden space？
6. 谁是锚点，哪些参数冻结？
7. 正样本按对象、时间还是接触位置定义？
8. 负样本是否存在 false negatives？
9. 对齐是双模态、三角形还是星形？
10. 缺失模态数据如何使用？
11. 对齐效果用 retrieval、probe 还是闭环 policy 评价？
12. Policy fine-tuning 后，对齐空间是否会漂移？

---

## 15. 最终记忆框架

```text
ViT
  = 图像或触觉图像的Transformer Encoder架构

CLIP
  = 图像与文本双Encoder + 对比学习体系

CLS / Pooling / Readout
  = 从token sequence得到全局表示

Projector
  = 将不同输出映射到兼容接口

Anchor
  = 希望保留的已有空间或监督参照

InfoNCE
  = 正确配对相对靠近、错误配对相对远离

Alignment
  = 建立可比较、可检索、可迁移的表示空间
```

本章回答了“怎样把不同模态表示清楚”。下一章继续回答：除了跨模态 InfoNCE，时间、遮挡、重建和任务监督分别规定表示必须保留什么。
