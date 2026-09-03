# 《AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-Tactile Sensors》精读笔记

> Y. Feng et al., International Conference on Learning Representations (ICLR) 2025<br>
> 正式论文：[Feng et al. - 2025 - AnyTouch Unified Static-Dynamic Representation across Visuo-tactile Sensors](<Feng et al. - 2025 - AnyTouch Unified Static-Dynamic Representation across Visuo-tactile Sensors.pdf>)<br>
> [arXiv](https://arxiv.org/abs/2502.12191)

## 0. 一句话定位

AnyTouch研究的是：**能否用一个触觉编码器，同时理解不同视触觉传感器、静态接触图像和动态接触视频。**

它不是Octopi那样的大触觉语言模型，也不包含LLaMA或Vicuna。AnyTouch的核心产物是一个统一的触觉表征编码器：

```text
任意一种已见或未见的视触觉传感器
+ 单帧静态接触或短时动态接触
                  ↓
             AnyTouch Encoder
                  ↓
          通用 tactile embedding
                  ↓
分类、检索、生成或机器人策略
```

如果沿用此前的知识坐标：

- TVL重点解决触觉、视觉和语言的语义对齐；
- Octopi重点解决触觉属性描述与物理常识推理；
- AnyTouch重点解决触觉表征在**传感器与时间形态之间的统一**。

它更接近TVL中的“触觉编码器基础设施”，而不是TVL-LLaMA或Octopi的语言生成层。

---

## 1. 为什么需要AnyTouch

视触觉传感器虽然都用相机观察弹性表面的形变，但不同设备的成像差异很大：

- GelSight Mini、DIGIT、DuraGel的外壳、凝胶、照明和视场不同；
- 相同物体、相同接触位置，在不同传感器中的颜色和形变外观可能完全不同；
- 有的数据集只有一张接触图像，有的数据集则记录滑动、按压等动态视频；
- 针对单一传感器训练的encoder，很容易记住设备外观而不是物体的真实触觉属性。

于是过去常见的做法是“一种传感器训练一个模型”。问题在于，换设备以后表示可能失效，跨数据集迁移也很困难。

AnyTouch希望得到：

\[
f\_T(x^{\text{DIGIT}}) \approx
f\_T(x^{\text{GelSight}}) \approx
f\_T(x^{\text{DuraGel}})
\]

这里的“相等”不是像素相同，而是：如果它们观测的是相同物体的相同接触位置，编码后的语义应该相近。

---

## 2. 主要贡献

### 2.1 TacQuad：细粒度对齐的多传感器触觉数据

作者搭建了统一采集平台，用四种传感器采集相互对应的数据：

- GelSight Mini；
- DIGIT；
- DuraGel；
- Tac3D。

TacQuad包含72,606个接触帧，并提供两种对齐粒度：

1. **细粒度时空对齐**：不同传感器依次接触同一物体的同一位置，并尽量保持相同的运动速度和压入深度；
2. **粗粒度空间对齐**：手持不同传感器依次探索同一位置，并加入旋转等更自然的探索动作。

其中细粒度部分覆盖25个物体、30组对齐采集，共17,524帧；粗粒度部分覆盖99个物体、151组采集，共55,082帧。

这里不是四个传感器在同一时刻同时接触，而是通过受控的依次采集建立对应关系。因此它是跨传感器监督的重要近似，但仍会受到接触姿态误差影响。

### 2.2 静态—动态统一输入

AnyTouch把单帧和视频统一成同一种输入格式：

```text
静态图像：复制同一帧 F 次 ─┐
                            ├─> F 帧时空序列 ─> Touch Encoder
动态视频：直接采样 F 帧 ────┘
```

论文实验通常取短序列 `F=3`。这样一个encoder可以同时接收：

- 只有静态触觉图像的旧数据集；
- 包含按压、滑动等变化的动态触觉视频。

### 2.3 Universal Sensor Token

作者为不同传感器设置sensor-specific token，同时加入一个universal sensor token：

```text
训练前期：更多使用设备专属token
训练后期：逐渐提高替换为通用token的概率，最高到0.75
```

直觉上，专属token先帮助模型理解每种设备的“方言”；通用token随后迫使模型寻找不同设备之间共有的触觉语义。遇到训练中没见过的新传感器时，模型使用通用token。

### 2.4 两阶段预训练

AnyTouch把低层触觉结构学习与高层语义对齐分开：

```text
Stage 1：遮挡重建 + 动态下一帧预测
                ↓
学习局部形变、纹理与接触运动

Stage 2：触觉—视觉—语言对比学习 + 跨传感器匹配
                ↓
学习语义一致性与传感器不变性
```

---

## 3. 数据集和Octopi有什么不同

“两篇论文都是给物理属性打标签，再让GPT生成描述”这个概括对Octopi较接近，但对AnyTouch不够准确。

| 维度 | Octopi / PHYSICLEAR | AnyTouch / TacQuad及预训练集合 |
| --- | --- | --- |
| 主要目标 | 触觉属性语言化与物理推理 | 跨传感器、静态—动态统一表征 |
| 传感器 | 单一GelSight设置 | 多种视触觉传感器 |
| 人工核心标签 | 硬度、粗糙度、凹凸程度三个离散属性 | 关键监督是配对的触觉、视觉、文本与跨传感器对应关系 |
| GPT作用 | 将结构化属性扩写为OPD等语言样本 | 根据视觉、触觉和已有描述生成或补全自由文本属性描述，并进行人工检查 |
| 输出模型 | Vicuna驱动的语言模型 | 通用tactile encoder |

因此，AnyTouch并不是先给每条数据人工标三个固定属性，再扩写成所有训练样本。它整合九个数据来源、五类传感器，共约248万帧触觉数据；由于不同数据集的模态并不齐全，作者采用“有哪一对模态，就计算哪一对loss”的方式利用它们。

TacQuad本身还采集了对应的普通视觉图像。GPT-4o被用于补足触觉属性的自然语言描述，描述可能涉及材料、纹理、粗糙度、硬度和接触位置等，而不是固定的三个分类标签。

还要区分“TacQuad里采集了四种传感器”和“主encoder训练使用了哪些数据”：论文的主预训练集合使用TacQuad粗粒度子集中的DIGIT、GelSight Mini和DuraGel触觉图像；Tac3D力场主要用于细粒度跨传感器生成实验，并被视为encoder未见传感器。

一个重要限制是：LLM根据外观推测触觉属性时可能产生幻觉。人工检查可以降低问题，但这种文本仍不等价于严格的物理测量。

---

## 4. Figure 2应该怎么看

Figure 2可以从左到右分成四类模块：

```text
输入构造
  ↓
Touch Encoder（最终要保留的主体）
  ↓
训练辅助模块：Touch Decoder / Task Head
  ↓
监督目标：像素重建、跨模态对齐、跨传感器匹配
```

不要把图里所有模块都当作推理部署时必须运行的网络。Touch Decoder和Task Head主要用于预训练时制造学习信号。

### 4.1 输入和token

触觉帧先被分成时空patch，再投影成token。序列中还加入：

- patch tokens：承载局部接触图像；
- sensor tokens：告诉模型当前设备或要求其使用通用表示；
- CLS token：汇聚全局触觉表示。

这些token进入基于OpenCLIP ViT构建的Transformer encoder。

### 4.2 图中的“多传感器融合”是什么

AnyTouch不是把四种传感器的图像在推理时拼在一起输入。更准确地说：

```text
不同传感器的样本分别经过同一个encoder
        ↓
利用对齐样本和loss约束它们的embedding
        ↓
一个传感器也能独立产生通用表示
```

所以它融合的是训练数据与表示空间，不是要求机器人同时装四种传感器。

---

## 5. Stage 1：像素级触觉结构学习

Stage 1采用masked modeling。约75%的触觉patch被遮挡，encoder只能看到少量内容，decoder负责恢复原始帧。

### 5.1 静态图像重建

对于复制成多帧的静态输入，Touch Decoder根据可见token恢复被遮挡的像素，并与真实图像计算MSE：

\[
\mathcal{L}\_{\mathrm{rec}}^S
= \lVert \hat{X}\_S-X\_S \rVert\_2^2
\]

这迫使encoder理解局部纹理与接触形变，而不是简单复制像素。

### 5.2 动态视频重建与预测

对动态输入，模型既要重建被遮挡的视频patch，也要预测下一帧：

\[
\mathcal{L}\_{\mathrm{stage1}}
= \mathcal{L}\_{\mathrm{rec}}^S
+ \mathcal{L}\_{\mathrm{rec}}^D
+ \mathcal{L}\_{\mathrm{pred}}^D
\]

因此，你此前说的“真实frame对比重建frame建立loss”是对的，但还要加上动态分支的下一帧预测。下一帧监督让模型学习按压、滑动过程中形变如何变化，而不只是单帧外观。

### 5.3 Touch Decoder推理时还用吗

通常不用。它类似MAE中的重建decoder，是预训练脚手架：

```text
预训练：Encoder + Decoder -> 重建loss
下游任务：只取Encoder -> tactile embedding -> 新任务头/策略
```

除非下游任务本身就是触觉生成，否则部署时保留的是Touch Encoder。

---

## 6. Stage 2：语义对齐与传感器不变性

Stage 2不是简单地“前半段用属性label监督，后半段再做别的”。它包含两个并行目标。

### 6.1 多模态对比对齐

触觉、视觉和文本由各自的encoder独立编码：

```text
Touch Encoder  -> x_T
Vision Encoder -> x_V
Text Encoder   -> x_L
```

然后对存在配对关系的模态计算双向InfoNCE对比损失：

\[
\mathcal{L}\_{\mathrm{align}}
= \alpha\_{TV}\mathcal{L}\_{TV}
+ \alpha\_{TL}\mathcal{L}\_{TL}
+ \alpha\_{VL}\mathcal{L}\_{VL}
\]

论文设置中，触觉—视觉和触觉—语言的权重为1，视觉—语言为0.2。目标是让同一样本的跨模态全局embedding接近，让不匹配样本远离。

这里的文本描述是监督信号之一，但不是“几个固定属性分类label”。它为触觉表示提供自然语言语义锚点。

### 6.2 Modality-missing-aware learning

九个数据集并非都有触觉、视觉和文本三种模态。AnyTouch不会因为某项模态缺失就丢掉整个样本，而是：

```text
有Touch + Vision -> 计算TV loss
有Touch + Text   -> 计算TL loss
有Vision + Text  -> 计算VL loss
```

这使模型能够利用异构旧数据集，而不要求每条数据都有完整三元组。

### 6.3 跨传感器匹配

对于TacQuad中的跨设备配对样本：

- 正样本：同一物体、同一接触位置、不同传感器；
- 负样本：不同物体或不同接触位置。

两份embedding做逐元素乘法，再送入MLP Task Head预测是否匹配，并以二元交叉熵训练：

\[
\mathcal{L}\_{\mathrm{stage2}}
= \mathcal{L}\_{\mathrm{align}}
+ \lambda \mathcal{L}\_{\mathrm{match}}
\]

这个任务直接打击“按传感器外观聚类”的捷径，要求encoder保留接触内容。

### 6.4 Task Head推理时还用吗

普通触觉表征推理时不用。它也是预训练辅助头：

```text
训练时：判断两段触觉是否来自相同接触
部署时：丢弃匹配头，保留统一encoder
```

下游做材料分类、抓取判断或机器人控制时，再接各自的新head或policy。

---

## 7. Encoder里有没有Attention

有。AnyTouch的Touch Encoder基于OpenCLIP-Large的ViT/Transformer，因此内部使用multi-head self-attention：

```text
[CLS, sensor tokens, patch tokens]
                ↓
         Transformer blocks
                ↓
每个token从同一序列的其他token聚合信息
```

动态输入中，不同时间和空间patch也能通过self-attention交流，所以模型可以关联“上一帧的局部形变”和“下一帧的位移变化”。

但触觉、视觉和文本encoder之间没有用cross-attention互相读取token。它们通过对比loss在全局embedding层面对齐。原因是AnyTouch需要每种模态可以独立编码，尤其是推理时只有触觉也要工作。

这部分与[Attention专题：从OpenCLIP到触觉模型与GR00T](19_attention_from_embeddings_to_action_expert.md)中的讨论直接相连。

---

## 8. 实验验证了什么

### 8.1 静态触觉理解

作者评估材料、粗糙度、硬度和抓取成功预测等任务。实验目标不只是刷新单项分类准确率，而是验证同一个encoder能否适应不同数据集和不同传感器。

### 8.2 未见传感器迁移

作者还在TACTO、Taxim等预训练未见传感器上进行linear probing。这里冻结encoder，只训练一个轻量线性分类器：

```text
如果简单线性头就能完成任务
    ↓
说明通用encoder已经把有用信息放进embedding
```

Universal Sensor Token正是为这种场景设计的。

### 8.3 表征可视化

t-SNE结果展示了逐步加入训练目标后的变化：

```text
原始CLIP：更容易按传感器设备分群
+ masked modeling：学到触觉局部结构，但设备差异仍明显
+ multimodal alignment：不同设备开始在语义空间混合
+ cross-sensor matching：更倾向按接触内容组织
```

这个实验直观支持了“模型不应只认传感器外观”的核心主张。

### 8.4 动态倒料实验

作者在xArm6、Robotiq夹爪和GelSight Mini上做倒料任务：机器人要根据动态触觉反馈决定继续倾倒、等待或回正。目标倒出60g物料。

论文报告的平均误差大致为：

| 表征 | 平均误差 |
| --- | ---: |
| CLIP | 5.22g |
| T3 | 2.33g |
| AnyTouch去掉动态训练 | 2.45g |
| 完整AnyTouch | **1.56g** |

这个结果说明动态预训练确实提供了静态帧难以表达的接触变化信息。

需要注意：AnyTouch encoder只是策略的感知前端，后面另有模仿学习policy输出动作。不能把该实验解释成“AnyTouch本身就是VLA控制器”。

### 8.5 跨传感器生成

作者还把AnyTouch embedding接到额外decoder上，完成：

- GelSight Mini到DuraGel触觉图像生成；
- GelSight Mini或DIGIT到Tac3D力场的生成。

这验证了统一embedding保留了可供下游重建其他传感器观测的信息。生成器本身不是AnyTouch encoder的固定组成部分。

---

## 9. 与TVL、Octopi的架构对照

| 模型 | 核心输入 | 核心输出 | 跨模态方法 | 是否含LLM | 主要回答的问题 |
| --- | --- | --- | --- | ---: | --- |
| TVL Encoder | DIGIT触觉、视觉、文本 | 对齐embedding | 对比学习 | 否 | 触觉能否进入CLIP语义空间 |
| TVL-LLaMA | 视觉、触觉、文本prompt | 自然语言 | projector + gated adapter | 是 | 能否用语言描述触觉 |
| Octopi | GelSight触觉视频、文本prompt | 属性描述与物理推理 | projector + Vicuna联合序列 | 是 | 能否利用触觉属性进行常识推理 |
| AnyTouch | 多传感器静态/动态触觉 | 通用tactile embedding | 对比对齐 + 跨传感器匹配 | **否** | 一个encoder能否跨设备、跨静态/动态工作 |

因此，AnyTouch可以成为未来大触觉语言模型或VLA的触觉前端，但论文本身没有完成“embedding接入LLM并生成语言”这一步。

---

## 10. 如何理解这篇论文的价值

AnyTouch最有价值的地方不是创造了又一个问答模型，而是处理触觉研究中的基础工程问题：**传感器碎片化。**

如果每个设备都需要独立预训练一个encoder，那么触觉数据很难像互联网图像那样跨机构累计。AnyTouch给出了一条可行路线：

```text
多设备配对数据
+ 低层masked modeling
+ 视觉/语言语义锚点
+ 显式跨传感器匹配
+ 通用sensor token
        ↓
共享的触觉foundation encoder
```

对VLA而言，它最自然的作用是先提供可靠触觉token，再由action expert或多模态Transformer完成任务条件融合和动作生成。

---

## 11. 局限与阅读时应保留的判断

1. **动态窗口很短。** 三帧能够捕获局部变化，但与长时滑动、物体形变历史仍有距离。
2. **静态统一通过复制帧实现。** 这是方便的输入工程，并不等于静态数据真正获得了时间信息。
3. **跨传感器对齐并非绝对精确。** 不同设备是依次接触，微小位置和力度差异不可避免。
4. **文本可能包含外观推断偏差。** GPT-4o生成的触觉描述不等价于力、位移或材料参数的实测值。
5. **“universal”是实验范围内的泛化。** 面对原理差异很大的新触觉传感器，仍可能需要适配。
6. **它不是语言模型，也不是控制策略。** 对物理常识推理要接LLM，对连续动作控制要接policy。

---

## 12. 原文定位索引

| 想核对的内容 | 原文位置 |
| --- | --- |
| TacQuad采集与多传感器对齐 | Section 3，PDF约第4页 |
| Figure 2整体架构 | Section 4开头，PDF约第5页 |
| 静态—动态统一输入与Stage 1 | Sections 4.1–4.3，PDF约第5–6页 |
| 多模态对齐与缺失模态处理 | Section 4.4，PDF约第6页 |
| Universal Sensor Token | Section 4.5，PDF约第6–7页 |
| Cross-sensor Matching | Section 4.6，PDF约第7页 |
| 网络和训练超参数 | Appendix A.6，PDF约第16页 |
| GPT-4o文本生成方式 | Appendix A.7，PDF约第17–18页 |
| 消融与更多分析 | Appendix A.9，PDF约第18–19页 |

---

## 13. 最终记忆框架

可以把AnyTouch压缩成三句话：

1. **输入统一**：把静态图像复制成短序列，与动态视频走同一个ViT触觉encoder。
2. **监督分层**：Stage 1用重建和预测学低层接触结构，Stage 2用对比对齐和跨传感器匹配学语义与设备不变性。
3. **部署轻量**：丢掉Touch Decoder、Vision/Text Encoder和Task Head，只保留AnyTouch Encoder，为分类器、生成器、LLM或机器人策略提供tactile embedding。

对应的数据流是：

```text
多源触觉数据
  -> 静态/动态统一token化
  -> self-attention触觉编码
  -> 像素级预训练
  -> 视觉/语言对齐 + 跨设备匹配
  -> 通用tactile embedding
  -> 下游感知、生成或控制任务
```
