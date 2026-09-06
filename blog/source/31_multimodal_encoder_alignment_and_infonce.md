# 多模态 Encoder 对齐：从 ViT、CLIP、InfoNCE 到触觉表示学习

> 本文从触觉论文中反复出现的 encoder 预训练出发，统一解释 ViT、CLIP、公共 latent space、语义锚点、InfoNCE、对齐与融合，并比较 TVL、Octopi、AnyTouch、Beyond Sight、OmniVTLA、Tactile-VLA、VLA-Touch 与 VT-MUSE 的训练路线。

> 建议先读：[Token、Encoder 与 Latent](08_transformer_latent_encoder_decoder.md)、[模型训练基础](22_model_training_basics.md)、[Attention 专题](19_attention_from_embeddings_to_action_expert.md)、[CLIP 与对比学习](21_clip_and_contrastive_learning.md)和[多模态融合](09_attention_multi_expert_and_conditioning.md)。TVL、AnyTouch、OmniVTLA 等论文在本文后半部分作为对照案例，不要求提前读完。

## 0. 核心结论

触觉论文中反复出现的 encoder 训练可以概括为：

```text
原始模态
  ↓
各自的Encoder
  ↓
Projector映射到相同维度
  ↓
根据配对关系构造正负样本
  ↓
用对比、重建、分类或时间预测Loss组织latent space
  ↓
得到可供检索、理解或控制使用的多模态表示
  ↓
再通过concat / self-attention / cross-attention接入下游任务
```

其中最容易混淆的是五个不同层级：

| 概念 | 回答的问题 |
| --- | --- |
| ViT | 图像或触觉图像如何被 Transformer 编码？ |
| CLIP | 图像和文本 encoder 如何通过对比学习共同训练？ |
| InfoNCE | 正确配对如何靠近，错误配对如何远离？ |
| Alignment | 不同模态是否使用可比较的语义坐标？ |
| Fusion | 完成任务时，多种模态如何交换并联合使用信息？ |

需要先修正两个常见概括：

1. 不是所有触觉论文都有视觉—触觉—语言三模态，也不是都以语言为锚点；
2. 对齐和融合不是同一件事，既可以只对齐不融合，也可以直接融合而不建立通用公共空间。

---

## 1. ViT 是什么

### 1.1 ViT 是 Encoder 架构

ViT，全称 Vision Transformer，是一种用 Transformer 编码二维图像的通用架构。

```text
224×224 RGB图像
      ↓ 切分patch
[patch₁, patch₂, ..., patchₙ]
      ↓ 线性投影并加入CLS、位置编码
[CLS, patch₁, patch₂, ..., patchₙ]
      ↓ 多层Transformer block
输出一组hidden states
```

每个 Transformer block 的主体仍是：

```text
hidden states
    ↓ self-attention
hidden states
    ↓ MLP
new hidden states
```

最终可以使用：

- patch tokens：保存不同局部区域的信息；
- CLS token：汇总整张输入；
- mean pooling：对所有 patch token 求平均得到全局表示。

ViT 只规定网络如何计算，没有规定用什么数据和 loss 训练。同一个 ViT 可以成为：

- ImageNet 图像分类器；
- CLIP 图像 encoder；
- MAE masked reconstruction encoder；
- 视触觉图像 encoder；
- 机器人策略的视觉 encoder。

### 1.2 为什么触觉也经常使用 ViT

DIGIT、GelSight 等视触觉传感器的输出本身就是 RGB 图像：

```text
弹性体接触形变RGB
        ↓
切成patch
        ↓
ViT
        ↓
tactile tokens / tactile embedding
```

力阵列也可以先排布成二维伪图像，再送入 ViT。ViT 本身不知道输入是自然图像、触觉形变图还是力场；训练数据和监督目标决定它最后学到什么。

---

## 2. CLIP 与 ViT 的关系

### 2.1 CLIP 不是 ViT 的同义词

CLIP 通常包含两个 encoder：

```text
图像 → Image Encoder → zV
文本 → Text Encoder  → zL
```

再使用大量图文配对数据进行对比学习，使正确图文表示接近：

\[
z_V\approx z_L
\]

CLIP 的 Image Encoder 可以是：

- ResNet；
- ViT；
- 其他视觉网络。

现在常见的 `CLIP ViT-L/14` 表示：

> 网络结构是 ViT-L/14，参数使用 CLIP 图文对比目标进行预训练。

### 2.2 几个名称的层级

| 名称 | 层级 |
| --- | --- |
| Transformer | 通用序列神经网络架构 |
| ViT | 使用 Transformer 处理图像的 encoder |
| CLIP | 图像—文本双 encoder 及其对比学习体系 |
| CLIP ViT | 经过 CLIP 图文训练的 ViT 图像 encoder |
| OpenCLIP | CLIP 的开源实现、训练代码和权重集合 |

最简单的记忆方式是：

> ViT 是发动机结构；CLIP 是把视觉发动机与文本发动机放在一起训练的方法。

### 2.3 CLIP 权重为什么能初始化触觉 Encoder

常见做法是：

```text
CLIP Vision ViT weights
           ↓ 初始化
      Tactile ViT
           ↓ 触觉数据继续训练
   tactile encoder
```

这种初始化提供了：

- 局部边缘与纹理特征；
- 空间结构先验；
- 已经形成的视觉语义；
- 相比从头训练更稳定的参数起点。

但自然图像与触觉图像存在 domain gap，因此只复制权重不等于模型已经理解触觉，仍需要触觉数据进行适配。

---

## 3. Encoder、Token、Global Embedding 与 Projector

### 3.1 Encoder 输出通常不只一个向量

ViT 内部输出一组 token：

\[
H=[h_{\mathrm{CLS}},h_1,h_2,\ldots,h_N]
\]

其中：

- \(h_i\)：第 \(i\) 个图像 patch 的表示；
- \(h_{\mathrm{CLS}}\)：整张输入的汇总表示。

进行 CLIP 式对齐时，通常不会把所有 patch token 两两比较，而是取一个全局表示：

```text
CLS token
或 mean pooled patch tokens
或专门的readout token
       ↓
global embedding
```

### 3.2 为什么需要 Projector

不同 encoder 的输出宽度可能不同：

```text
Tactile ViT  → 768维
Vision ViT   → 1024维
Text Encoder → 512维
```

要计算跨模态相似度，通常先用 projector 映射到共同维度：

\[
z_T=P_T(E_T(T)),\qquad
z_V=P_V(E_V(V)),\qquad
z_L=P_L(E_L(L))
\]

例如统一成 512 维：

```text
Touch    → Encoder → Projector → 512D
Vision   → Encoder → Projector → 512D
Language → Encoder → Projector → 512D
```

然后进行 L2 normalization：

\[
\bar z=\frac{z}{\|z\|_2}
\]

归一化后的点积就是余弦相似度：

\[
\operatorname{sim}(z_i,z_j)
=
\bar z_i^\top \bar z_j
\]

### 3.3 Projector 不等于完成了对齐

Projector 只解决：

- 向量维度不一致；
- 不同 encoder 输出尺度不同；
- 需要一个适合对比学习的映射空间。

真正让空间产生跨模态结构的是训练 loss。

---

## 4. 什么是多模态对齐

假设数据集中每条样本包含：

\[
(T_i,V_i,L_i)
\]

分别经过 encoder：

```text
Touch    → Tactile Encoder → zT
Vision   → Vision Encoder  → zV
Language → Text Encoder    → zL
```

对齐学习希望同一对象或事件的不同模态在公共 latent space 中互相靠近：

\[
z_T\approx z_V\approx z_L
\]

这不是要求三个向量逐维完全相等，而是希望它们：

- 余弦相似度较高；
- 能够互相检索；
- 具有相似的邻域结构；
- 同时与不匹配样本保持距离。

例如：

```text
柔软海绵的触觉
        ↕ 靠近
柔软海绵的视觉
        ↕ 靠近
“soft porous sponge”

同时远离：
“hard metal cube”
```

所以语言或视觉 embedding 并不是传统意义上逐维回归的 ground truth，而是共同定义 latent space 几何关系的参照。

---

## 5. 什么是“锚点”

“锚点”不是一定出现在网络图中的模块，而是理解训练设计的一种方式。

### 5.1 冻结的 CLIP 视觉—语言空间

假设已经有一个预训练 OpenCLIP：

```text
Vision Encoder → 已有视觉语义空间
Text Encoder   → 已有语言语义空间
```

冻结这两个 encoder，只训练 Tactile Encoder：

```text
                  frozen CLIP space
               ┌─────────────────────┐
vision ───────→ │ zV                  │
language ─────→ │ zL                  │
touch ─encoder→ │ zT ← 被拉进已有空间 │
               └─────────────────────┘
```

此时视觉和语言空间基本固定，可以称为触觉 encoder 的语义锚点。

### 5.2 为什么经常以语言为锚点

语言天然适合表达：

- soft / hard；
- rough / smooth；
- slippery；
- fragile；
- metallic；
- elastic。

如果只使用物体类别标签，模型可能知道“这是海绵”；对齐自然语言后，模型有机会获得更开放的属性语义。

语言锚点的限制是：很多控制相关细节难以被一句标签表达。例如两帧触觉都可以叫 `hard contact`，但其中一帧可能已经发生微小滑移，需要完全不同的控制动作。

### 5.3 为什么视觉也经常是锚点

视觉数据更丰富、同步采集更容易，并且能提供：

- 对象身份；
- 几何和空间关系；
- 接触位置；
- 材料和外观线索。

但视觉不一定看得出真实硬度、摩擦力和内部接触状态。

### 5.4 锚点不一定完全冻结

| 训练设置 | 对“锚点”的理解 |
| --- | --- |
| 冻结 vision/text encoder | 稳定的固定语义坐标 |
| vision 使用 LoRA，text 冻结 | 语言较稳定，视觉允许有限 domain adaptation |
| 三个 encoder 全部更新 | 没有绝对固定坐标，三个空间共同演化 |

因此，“语言是锚点”通常表示训练希望保留语言语义结构，不一定表示语言 encoder 的每个参数都绝对不更新。

---

## 6. InfoNCE：最常见的对齐 Loss

论文里经常出现的不是 `inforce loss`，而是 **InfoNCE loss**，全称 Information Noise-Contrastive Estimation。

### 6.1 正样本与负样本

假设一个 batch 包含 \(N\) 组三模态样本：

```text
(T₁,V₁,L₁)
(T₂,V₂,L₂)
...
(Tₙ,Vₙ,Lₙ)
```

对于触觉 \(T_i\)：

- \(V_i\) 是同步的正确视觉，即正样本；
- batch 中其他 \(V_j,j\ne i\) 是负样本。

首先计算相似度：

\[
s_{ij}
=
\frac{\bar z_{T_i}^{\top}\bar z_{V_j}}{\tau}
\]

其中 \(\tau\) 是 temperature。

触觉检索视觉的 InfoNCE 为：

\[
\mathcal L_{T\rightarrow V}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{N}\exp(s_{ij})}
\]

它可以被理解成一个 batch 内 \(N\) 分类问题：

> 给定触觉 \(T_i\)，从 batch 中的 \(N\) 张视觉图里找出正确的 \(V_i\)。

### 6.2 为什么通常双向计算

触觉找到视觉是一种方向：

\[
\mathcal L_{T\rightarrow V}
\]

视觉找到触觉是另一个方向：

\[
\mathcal L_{V\rightarrow T}
\]

对称损失通常写成：

\[
\mathcal L_{TV}
=
\frac{
\mathcal L_{T\rightarrow V}
+
\mathcal L_{V\rightarrow T}
}{2}
\]

三模态可以形成三条边：

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

如果视觉—语言 encoder 已经由 CLIP 对齐，训练时也可以重点优化包含触觉的两条边。

### 6.3 Temperature 的作用

\[
s_{ij}=\frac{\cos(z_i,z_j)}{\tau}
\]

- \(\tau\) 较小：softmax 更尖锐，更在意最相似与次相似的差异；
- \(\tau\) 较大：分布更平滑，对相似度差异不那么敏感。

它控制对比分类的严格程度，可以是超参数，也可以作为可学习参数。

### 6.4 为什么不只用 MSE 拉近正样本

如果只使用：

\[
\mathcal L=\|z_T-z_V\|_2^2
\]

而两个 encoder 都能自由更新，那么一个退化解是：

```text
所有输入都输出同一个向量
```

所有正样本距离都变成零，但 embedding 没有任何区分能力，这叫 representation collapse。

InfoNCE 同时完成：

```text
正确配对靠近
+
错误配对远离
```

因此能维持样本之间的判别结构。

### 6.5 InfoNCE 的局限

InfoNCE 并不是天然完美：

- batch 中可能出现实际语义相同的“假负样本”；
- batch 太小会导致负样本不足；
- 只对齐全局 embedding 可能忽略局部接触对应；
- 容易保留对象身份，未必保留细微力变化；
- 配对数据噪声会直接污染空间结构。

因此新的方法常加入多正样本、hard negatives、局部 token 对齐、重建、分类或时间预测。

---

## 7. 常见 Encoder 训练 Loss 家族

不同论文的公式很多，但大体可以分成五类。

### 7.1 跨模态对齐

目标是让不同模态表达同一对象或状态。

常见形式：

- InfoNCE；
- cosine consistency；
- triplet loss；
- matching BCE。

例如跨传感器匹配：

\[
\mathcal L_{\mathrm{match}}
=
\operatorname{BCE}(\hat y,y)
\]

模型判断两份触觉是否来自同一物体、同一接触位置。

### 7.2 属性分类监督

例如 hardness、roughness、bumpiness 分类：

\[
\mathcal L_{\mathrm{cls}}
=
-\sum_k y_k\log \hat y_k
\]

分类可以使 embedding 保留指定物理属性，但不等于建立完整的触觉—视觉—语言公共空间。

### 7.3 Masked Reconstruction

```text
masked tactile image
        ↓
Encoder + Decoder
        ↓
reconstructed tactile image
```

通常采用像素 MSE：

\[
\mathcal L_{\mathrm{rec}}
=
\|\hat X-X\|_2^2
\]

它更有利于学习：

- 局部纹理；
- 接触形变；
- 空间结构；
- 传感器成像规律。

这属于单模态 self-supervised representation learning，不是语义对齐。

### 7.4 时间预测与时间对比

常见形式包括：

- 预测下一触觉帧；
- 预测 tactile depth change；
- 让同一轨迹的相邻状态接近；
- 区分不同轨迹或相隔较远的状态。

例如：

\[
\mathcal L_{\mathrm{temp}}
=
\sum_i
\mathcal L_{\mathrm{NCE}}(s_i,s_{i+1})
\]

它希望 encoder 学到接触状态如何随时间演化，而不是只识别单帧物体属性。

### 7.5 下游任务监督

包括：

- action MSE/L1；
- action classification；
- diffusion/flow matching；
- language generation cross-entropy；
- success/failure classification。

它们可以端到端训练 sensor encoder，但形成的是 task-oriented representation，不一定是可跨任务检索的通用对齐空间。

---

## 8. Alignment 与 Fusion 的区别

这是理解多模态论文最重要的分界之一。

### 8.1 Alignment：建立可比较的坐标系

```text
Touch Encoder  → zT
Vision Encoder → zV
Text Encoder   → zL

Loss层比较zT、zV、zL
```

训练完成后，即使三个 encoder 分别运行，它们的 global embedding 也能直接计算相似度。

Alignment 常用于：

- 跨模态检索；
- 开放词汇分类；
- zero-shot 属性判断；
- 将新模态接入已有语义空间。

### 8.2 Fusion：完成任务时交换信息

Joint self-attention：

```text
[Touch tokens, Vision tokens, Language tokens]
                    ↓
              Transformer
```

Cross-attention：

```text
Action tokens作为Q
MUSE memory作为K/V
        ↓
Cross-Attention
```

Fusion 常用于：

- 生成语言描述；
- 物理推理；
- 预测机器人动作；
- 根据触觉反馈调整控制。

### 8.3 对齐不等于融合

\[
z_T\approx z_V\approx z_L
\]

只表示三个模态在公共空间表达相似语义，不会自动产生一个同时包含三者全部细节的新表示。

### 8.4 融合也不保证通用对齐

把 token 拼起来进入 Transformer，并用 action loss 训练：

```text
[T,V,L] → Transformer → action loss
```

只要能预测动作，网络就可能形成任务专用表示，但不保证单独取出的 \(z_T,z_V,z_L\) 可以互相检索。

因此：

> Alignment 关心“不同模态是否使用可比较的语义坐标”；Fusion 关心“当前任务中如何联合使用这些信息”。

---

## 9. 多模态 Encoder 训练的通用范式

### 9.1 选择输入表示

```text
Vision：普通RGB
Touch：触觉RGB / depth / marker flow / force map / 数值力
Text：自然语言描述、属性词或任务指令
```

输入形式决定适合的 tokenizer 和 encoder：

- 图像式触觉常用 CNN/ViT；
- 数值力序列常用 MLP、1D CNN 或时序 Transformer；
- 文本常用 CLIP text encoder、T5 或 LLM embedding。

### 9.2 先学习模态内部结构

```text
触觉图像 → masked reconstruction
触觉视频 → frame prediction
力序列   → temporal modeling
```

这一步回答：

> 触觉自身的局部结构和动态是什么？

### 9.3 选择锚点或参照关系

常见选择：

```text
CLIP视觉—语言空间
冻结的text encoder
视觉encoder
物理属性label
跨传感器配对
机器人action监督
```

锚点决定 latent 更强调语义、空间、触觉属性还是动作相关信息。

### 9.4 映射到公共空间

```text
Encoder → global pooling/CLS → Projector → normalize
```

对齐通常发生在 global embedding 上，不一定是所有 patch token 的逐位置对齐。

### 9.5 定义正样本

正样本可以是：

- 同一对象的视觉—触觉；
- 同一时刻的视觉—触觉；
- 触觉与对应文本；
- 同一接触位置的不同传感器；
- 同一轨迹的相邻状态；
- 同一输入的不同 masking/augmentation。

### 9.6 定义负样本

负样本可以是：

- batch 中其他物体；
- 不同接触位置；
- 不匹配文本；
- 不同任务；
- 不同轨迹或距离较远的状态。

正负样本如何定义往往比 InfoNCE 公式本身更重要，因为它直接规定模型认为什么应该相同、什么必须区分。

### 9.7 组合辅助目标

一个成熟的触觉预训练方案通常不会只使用一个 InfoNCE，而会组合：

```text
跨模态对齐
+ 模态内部重建
+ 时间预测
+ 跨传感器匹配
+ 下游任务监督
```

### 9.8 丢弃训练脚手架并接入下游

```text
预训练：
Encoder + Projector + Decoder + Task Head

部署：
Encoder → classifier / LLM / policy
```

重建 decoder、matching head、privileged posterior 等通常只负责制造训练信号，部署时可以丢弃。

---

## 10. 已读论文的路线对照

| 论文 | 是否有独立 Encoder 对齐 | 主要方法 | 锚点或监督 |
| --- | --- | --- | --- |
| TVL | 是 | 三模态 CLIP/InfoNCE | OpenCLIP 视觉—语言空间 |
| Octopi | 不属于典型三模态对齐 | CLIP ViT + visual prompts + 属性分类 | hardness、roughness、bumpiness 标签 |
| AnyTouch | 是 | 重建/预测 + 三模态 InfoNCE + 跨传感器匹配 | 视觉、文本、跨传感器配对 |
| Beyond Sight / FuSe | 部分有，发生在策略内部 | language readout 与 T5 对比，同时生成传感器描述 | 冻结 T5 和文本标注 |
| OmniVTLA | 是 | AnyTouch 式三模态对齐与跨传感器匹配 | 视觉、文本、触觉配对 |
| Tactile-VLA | 没有独立对齐阶段 | tactile MLP 与 VLA 用 action CFM 端到端训练 | 专家位置—力动作 |
| VLA-Touch | 没有统一对齐阶段 | Octopi 高层反馈 + 外部低层控制 | 规划和控制任务 |
| VT-MUSE Stage I | 是，但只有视觉—触觉 | 跨模态 InfoNCE + 时间对比 + mask consistency | 同步视触觉、相邻时间、同窗口不同遮挡 |
| VT-MUSE Stage II | 更偏时序融合与生成表征 | Temporal Transformer、cross-attention、重建、conditional VAE | masked RGB、tactile depth change、privileged posterior |

---

## 11. TVL：最标准的语义锚点路线

TVL 基本符合下面的直觉：

```text
OpenCLIP Vision Encoder ─→ zV ─┐
OpenCLIP Text Encoder   ─→ zL ─┼→ CLIP/InfoNCE
Tactile ViT             ─→ zT ─┘
```

它保留 OpenCLIP 已有的视觉—语言语义结构，训练触觉 encoder 进入该空间，并直接优化：

\[
\mathcal L
=
\mathcal L_{TV}
+
\mathcal L_{TL}
+
\mathcal L_{VL}
\]

它最直接地回答：

> 触觉能否进入一个已有的开放词汇视觉—语言语义空间？

因此其检索实验是 encoder 对齐的诊断，而不是机器人闭环控制实验。

---

## 12. Octopi：CLIP 初始化不等于 CLIP 对齐

Octopi 使用：

```text
GelSight tactile video
        ↓
CLIP ViT-L/14 backbone
+ Visual Prompt Tuning
        ↓
hardness / roughness / bumpiness heads
```

它复用 CLIP 视觉网络的参数，但 encoder 适配阶段主要由三个物理属性分类标签监督。

因此不能仅因为使用了 CLIP backbone，就认为 Octopi 显式完成了：

\[
z_T\approx z_V\approx z_L
\]

它更接近：

> 利用 CLIP 的视觉初始化，通过属性分类把触觉视频表示调整为适合物理属性理解的表示，再经 projector 接入 Vicuna。

---

## 13. AnyTouch：低层结构与高层语义分阶段训练

AnyTouch 将两类能力分开：

```text
Stage 1：
masked reconstruction + dynamic frame prediction
        ↓
局部触觉结构与接触动态

Stage 2：
Touch ↔ Vision ↔ Text
+ cross-sensor matching
        ↓
高层语义与传感器不变性
```

这体现了一个重要规律：

> 语言对齐擅长提供语义，但不一定保存低层接触细节；重建和时间预测能补充这些细节。

AnyTouch 还通过 sensor-specific token 与 universal sensor token，让同一个 encoder 既能适应设备差异，也能学习跨设备共性。

---

## 14. Beyond Sight：对齐发生在 Policy 内部

Beyond Sight/FuSe 不只关心原始 tactile encoder 是否与语言对齐，而是让策略看过所有传感器后形成的 language readout 与冻结 T5 表示对齐：

```text
vision / touch / audio
        ↓
Octo Transformer
        ↓
language readout
        ↕ contrastive loss
frozen T5 text embedding
```

同时加入 sensory-grounded language generation loss，要求策略不只是“选对文本”，还要能说出感知到的物理属性。

因此 FuSe 的目标更接近：

> 确保新增传感器的信息真的进入策略高层表示，并能影响动作，而不是只预训练一个好看的触觉 embedding。

---

## 15. OmniVTLA：对齐初始化之后再做动作学习

OmniVTLA 的 SA-ViT 使用触觉、视觉和文本配对数据，复用 AnyTouch 式对齐方法：

```text
ObjTac及其他数据
       ↓
Touch ↔ Vision ↔ Text对齐
+ cross-sensor matching
       ↓
SA-ViT初始化
       ↓
接入π0并使用flow-matching action loss微调
```

这里需要区分两个阶段：

1. 对齐预训练告诉 tactile encoder 物体属性与视觉/语言语义；
2. action fine-tuning 告诉 VLA 这些触觉 token 怎样用于动作。

好的对齐初始化不等于已经学会控制机器人。

---

## 16. Tactile-VLA：没有独立对齐阶段

Tactile-VLA 的触觉输入是法向力和剪切力历史：

```text
force history → MLP → tactile token
                      ↓
vision + language + tactile prefix
                      ↓
π0-style Action Expert
                      ↓
position-force action
```

它没有先优化 CLIP/InfoNCE 式的：

\[
z_T\approx z_V\approx z_L
\]

触觉 MLP 与整个策略由 action Conditional Flow Matching 端到端更新。它得到的是 task-oriented tactile representation，而不是专门用于跨模态检索的通用 embedding。

---

## 17. VT-MUSE Stage I：模态、时间和遮挡三种一致性

VT-MUSE Stage I 只有视觉与触觉，没有语言。两个预训练 ViT 分别输出 Visual CLS 和 Tactile CLS，并优化三类关系。

### 17.1 同步跨模态一致性

```text
Visual_t ↔ Tactile_t
```

用双向 InfoNCE 对齐同一时间点的视觉和触觉。

### 17.2 相邻时间一致性

```text
State_t ↔ State_t+1
```

把同步的 Visual/Tactile 表示组成多模态状态，再通过 temporal InfoNCE 让相邻状态在 latent space 中保持连续。

### 17.3 遮挡不变性

```text
同一历史 + visual mask A
同一历史 + visual mask B
```

用 cosine consistency 使两种表示接近，提高视觉缺失时的稳健性。

因此 Stage I 学到三种不变性：

1. 模态改变但交互状态不变；
2. 相邻时刻的交互状态连续；
3. 部分视觉被遮挡时，核心状态表示仍然稳定。

这时主要是在适配视觉和触觉 ViT。Figure 2(b) 的完整 sequential memory、conditional VAE 与重建 decoder 属于 Stage II。

---

## 18. 为什么常见方案会采用“两阶段”

只使用语言或视觉对齐可能损失低层物理细节；只使用重建又可能得到缺乏语义的像素表示。因此很多方法采用：

```text
第一层：模态内部学习
重建 / 预测 / 时序建模
        ↓
学习纹理、形变、接触动态

第二层：跨模态语义学习
视觉 / 文本 / 跨传感器对比
        ↓
学习对象、材料、属性和设备不变性

第三层：下游任务学习
language generation / action loss
        ↓
学习怎样使用这些表示完成任务
```

不同论文不一定严格按这个顺序，但大多数目标都可以放到这三个层级中。

---

## 19. 不同监督目标决定表示保留什么

| 监督目标 | 表示更可能保留 | 可能忽略 |
| --- | --- | --- |
| Language alignment | 材料、硬度、粗糙度等可语言化语义 | 微小滑移、瞬时力变化 |
| Vision alignment | 对象身份、形状、空间与外观 | 视觉不可见的内部接触 |
| Pixel reconstruction | 局部纹理、形变与传感器细节 | 高层语义和任务相关性 |
| Temporal prediction | 接触演化与变化趋势 | 开放词汇语义 |
| Cross-sensor matching | 设备无关的触觉属性 | 设备特有且可能有用的信息 |
| Attribute classification | 指定的物理标签 | 标签没有覆盖的其他性质 |
| Action loss | 当前机器人任务真正需要的信息 | 与当前任务无关的通用语义 |
| Language generation | 足以明确表达属性的信息 | 高频控制细节 |

不存在唯一最好的触觉表示。最佳训练目标取决于下游需求。

---

## 20. 根据下游任务选择路线

### 开放词汇触觉理解

```text
优先：Touch-Language Alignment
补充：Touch-Vision Alignment
```

典型代表：TVL。

### 跨传感器通用 Encoder

```text
优先：Cross-Sensor Matching
+ Universal Sensor Token
+ Reconstruction
```

典型代表：AnyTouch。

### 接触丰富机器人控制

```text
优先：Temporal Modeling
+ Action Supervision
+ Current Tactile Feedback
```

典型代表：VT-MUSE、Tactile-VLA。

### 同时追求语义与动作泛化

```text
先做多模态语义预训练
        ↓
再用机器人示范fine-tune policy
```

典型代表：OmniVTLA。

---

## 21. 阅读 Encoder 论文时的检查清单

以后看到一篇新的触觉 encoder 论文，可以依次问：

### 输入是什么

- RGB 触觉图像？
- depth map？
- marker displacement？
- force array？
- 连续视频还是单帧？

### Encoder 是什么

- CNN、ViT、时序 Transformer 还是 MLP？
- 从头训练还是从 CLIP/ImageNet/MAE 初始化？
- 不同传感器共享参数吗？

### 用什么全局表示

- CLS token？
- mean pooling？
- learnable readout tokens？
- 所有 patch tokens？

### 谁是锚点

- 冻结文本 encoder？
- CLIP 视觉—语言空间？
- 可训练视觉 encoder？
- 属性 label？
- 动作轨迹？

### 正负样本怎样定义

- 同物体？
- 同时刻？
- 同接触位置？
- 同一段描述？
- 同一轨迹相邻状态？

### Loss 属于哪一层

- 语义对齐？
- 模态内部重建？
- 时间预测？
- 跨设备匹配？
- 下游 action/language supervision？

### 哪些参数更新

- anchor 是否冻结？
- tactile encoder 全量更新还是 LoRA/prompt tuning？
- projector 是否训练？
- pretrained backbone 是否会漂移？

### 推理时保留什么

- 只保留 encoder？
- projector 是否保留？
- reconstruction decoder 是否丢弃？
- task head 是否只是训练辅助？

### 最终实验真的证明了什么

- 检索和 linear probing 只能证明表示质量；
- caption 能证明语言表达能力；
- action loss 与真机 rollout 才能证明控制作用；
- 如果同时改变 encoder、policy 和 controller，不能把所有提升都归因于 alignment。

---

## 22. 最终统一知识图

```text
                         ┌──────────────────────────┐
                         │   原始多模态配对数据     │
                         │ Touch / Vision / Text    │
                         └────────────┬─────────────┘
                                      ↓
                   ┌──────────────────────────────────┐
                   │       Modality-specific Encoder  │
                   │  ViT / CNN / MLP / Transformer   │
                   └────────────────┬─────────────────┘
                                    ↓
                   ┌──────────────────────────────────┐
                   │ CLS / Pooling / Readout Token    │
                   │       + Projector + Normalize    │
                   └────────────────┬─────────────────┘
                                    ↓
          ┌─────────────────────────┼──────────────────────────┐
          ↓                         ↓                          ↓
  Cross-modal Alignment       Intra-modal Learning       Task Supervision
  InfoNCE / Matching          Reconstruction / Time      Action / Language
          │                         │                          │
          └─────────────────────────┼──────────────────────────┘
                                    ↓
                     learned multimodal representation
                                    ↓
             ┌──────────────────────┴──────────────────────┐
             ↓                                             ↓
       Representation Use                             Task Fusion
  retrieval / classification             concat / joint attention / cross-attn
                                                           ↓
                                              caption / reasoning / action
```

整个体系可以压缩成一句话：

> **ViT 决定 encoder 怎样计算；CLIP/InfoNCE 决定不同模态怎样在 latent space 中建立相对位置；重建和时间预测决定表示保留哪些低层物理细节；attention 决定下游任务怎样动态读取和融合这些表示；action 或 language loss 最终决定这些表示是否真正对任务有用。**
