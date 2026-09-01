# 《A Touch, Vision, and Language Dataset for Multimodal Alignment》精读笔记

> Letian Fu et al., ICML 2024 / PMLR 235<br>
> 正式论文：[Fu et al. - 2024 - A Touch Vision and Language Dataset for Multimodal Alignment](<Fu et al. - 2024 - A Touch Vision and Language Dataset for Multimodal Alignment.pdf>)<br>
> [项目主页](https://tactile-vlm.github.io/) · [arXiv](https://arxiv.org/abs/2402.13232)

## 0. 一句话定位

这篇论文不是直接训练机器人策略，而是在解决更上游的问题：**怎样把触觉、视觉和自然语言放入同一个语义空间，并让语言模型根据视觉和触觉生成触感描述。**

它位于触觉机器人学习技术栈的第一层：

```text
触觉数据与表征学习     ← 本文主要解决
        ↓
触觉—视觉—语言融合
        ↓
模仿学习 / VLA策略
        ↓
机器人动作与闭环控制
```

本文没有训练动作策略，也没有灵巧手、行为克隆或机器人任务成功率实验。

---

## 1. 论文试图解决什么问题

以往视觉—触觉研究通常集中在：

- 固定类别的材料或纹理分类；
- 视觉—触觉配对或跨模态预测；
- 使用封闭词表描述 `wood / fabric / metal` 或 `rough / smooth`。

而 VLM/VLA 使用开放自然语言，例如：

```text
soft, slightly compressible, smooth
hard, rough, grainy, rigid
```

主要障碍是缺少大规模的“视觉—触觉—自然语言”三元组。论文的解决路线是：

```text
同步采集视觉和DIGIT触觉
            ↓
人工标注少量数据 + GPT-4V伪标注大量数据
            ↓
训练与CLIP视觉、语言空间对齐的触觉编码器
            ↓
融合视觉和触觉embedding并接入LLaMA
            ↓
生成自然语言触感描述
```

---

## 2. 三项核心贡献

### 2.1 TVL数据集

数据集包含约4.4万组视觉—触觉—语言数据：

- 43,741个有接触的视觉—触觉样本；
- 约10%使用人工触觉标签；
- 约90%使用GPT-4V根据视觉图像生成伪标签；
- 测试集有402个全部由人工标注的样本；
- 共出现254种不同的触觉形容词。

数据分为两部分：

| 子集 | 数量 | 采集方式 | 标签来源 | 特点 |
|---|---:|---|---|---|
| SSVTP | 4,587 | UR5机器人＋DIGIT | 人工 | 实验室环境，数据量小 |
| HCT | 39,154 | 5人使用手持设备采集约20小时 | 主要为GPT-4V | 真实环境、同步采集、更多样 |

论文的意义不是数据量绝对巨大，而是把触觉从固定类别推进到**开放词汇自然语言描述**。

### 2.2 与视觉和语言直接对齐的触觉编码器

论文保留OpenCLIP已有的视觉、文本语义空间，训练新的触觉ViT编码器，并同时计算：

\[
\mathcal L = \mathcal L_{V,T}+\mathcal L_{T,L}+\mathcal L_{V,L}
\]

相较于只通过视觉间接连接触觉和语言，关键是加入了直接的触觉—语言监督 \(\mathcal L_{T,L}\)。

### 2.3 TVL-LLaMA与TVL Benchmark

作者把视觉和触觉特征融合，通过projector和门控adapter注入LLaMA2-7B，让模型生成最多五个触觉形容词，并提出触觉语义描述Benchmark。

---

## 3. 触觉输入究竟是什么

论文使用DIGIT视觉式触觉传感器。它通过内部摄像头拍摄弹性体接触物体后产生的光学形变。

触觉编码器的输入是：

```text
DIGIT内部摄像头拍摄的RGB触觉图像
```

它不是显式的：

- depth map；
- surface normal map；
- force map；
- marker displacement / optical flow；
- 六维力传感器读数。

预处理过程为：

1. 将原始触觉图像零填充成正方形；
2. 可选地减去未接触背景；
3. 使用数据统计量归一化；
4. resize到224×224；
5. 输入ViT-Tiny、ViT-Small或ViT-Base。

数学上可以写成：

\[
T_{RGB}\in\mathbb R^{224\times224\times3}
\xrightarrow{E_T}
z_T\in\mathbb R^d
\]

背景相减为：

\[
T_{diff}=T_{contact}-T_{background}
\]

它可以减弱传感器本体、凝胶外观和内部光照的偏差。论文消融中，加入背景相减后，触觉—文本Top-1从36.3%提高到42.3%。

---

## 4. Token与embedding不能混为一谈

OpenCLIP严格来说是对比学习式视觉—语言双编码器，而不是LLaVA式生成模型。

```text
RGB图像 ─→ Vision Encoder  ─→ 全局视觉embedding z_v
文本描述 ─→ Text Encoder    ─→ 全局语言embedding z_l
触觉图像 ─→ Tactile Encoder ─→ 全局触觉embedding z_t
```

ViT内部确实会产生一组patch tokens，但TVL进行对比学习时主要使用池化、投影和归一化后的**全局embedding**。

需要区分三层概念：

| 名称 | 含义 |
|---|---|
| ViT patch tokens | 图像或触觉图像各局部patch的内部表示 |
| CLIP/TVL global embedding | 用于余弦相似度、对比学习和跨模态检索的全局向量 |
| LLaMA multimodal token | 全局向量经过projector后，变成LLaMA可以接收的条件表示 |

---

## 5. ImageBind的“星形对齐”

ImageBind利用现实中较容易收集的“视觉—其他模态”配对数据，把视觉作为中心节点：

```text
                文本
                  ↑
音频 ←——— 视觉 ———→ 深度
                  ↓
               IMU、热成像
```

它通常不需要为所有非视觉模态两两收集数据。例如声音和文本可以通过视觉间接对齐：

```text
狗叫声embedding ≈ 狗图像embedding
“dog”文本embedding ≈ 狗图像embedding
                ↓
期待狗叫声与“dog”也比较接近
```

问题是，这种传递关系不保证触觉和语言会精确对齐。

TVL拥有视觉、触觉、语言三元组，因此可以进行三边直接对齐：

```text
             视觉
            ↙   ↘
         触觉 ↔ 语言
```

消融结果显示：去掉触觉—语言损失后，触觉—文本Top-1从36.3%降到20.3%，说明只通过视觉间接绑定触觉和语言不够。

---

## 6. 如何把三种模态放入同一个语义空间

对每一个三元组：

\[
(I_i,T_i,L_i)
\]

分别编码并归一化：

\[
z_V=\operatorname{norm}(E_V(I)),\quad
z_T=\operatorname{norm}(E_T(T)),\quad
z_L=\operatorname{norm}(E_L(L))
\]

同一个三元组内的模态是正样本：

```text
T_i ↔ I_i：正样本
T_i ↔ L_i：正样本
I_i ↔ L_i：正样本
```

batch中其他样本作为负样本：

```text
T_i ↔ I_j：负样本，i ≠ j
T_i ↔ L_j：负样本，i ≠ j
```

以触觉—视觉InfoNCE为例：

\[
\mathcal L_{T,V}
=-
\log
\frac{\exp(\operatorname{sim}(z_{T_i},z_{V_i})/\tau)}
{\sum_j\exp(\operatorname{sim}(z_{T_i},z_{V_j})/\tau)}
\]

训练使同一样本的跨模态embedding靠近，不同样本的embedding远离。OpenCLIP已有的视觉和文本空间相当于语义锚点，新的触觉编码器被拉到正确位置：

```text
“rough carpet”文本 ●
                    \
地毯视觉图像       ● —— ● 地毯触觉图像

“smooth glass”文本 ▲
                    \
玻璃视觉图像       ▲ —— ▲ 玻璃触觉图像
```

---

## 7. 开放词汇跨模态检索在证明什么

这个实验是诊断性表征实验，用于判断触觉encoder是否进入了正确的视觉—语言语义位置，并不是最终机器人任务。

### 7.1 触觉→视觉

给定触觉样本 \(T_i\)，从402张候选视觉图像中检索同步配对的 \(I_i\)：

\[
s_{ij}=\cos(E_T(T_i),E_V(I_j))
\]

这里的“触觉→视觉”表示跨模态检索，不表示生成视觉图像。

### 7.2 触觉→文本

给定触觉样本，从候选语言描述中检索相符的触觉语义：

\[
s_{ij}=\cos(E_T(T_i),E_L(L_j))
\]

由于 `rigid / stiff / hard` 等词可能是近义词，论文扩充了同义词，并允许多个语义相近文本作为正确答案。

### 7.3 Top-1和Top-5

- Top-1：正确答案排在相似度第一名；
- Top-5：正确答案出现在相似度前五名。

如果402个候选只有一个正确答案，随机Top-1约为：

\[
1/402\approx0.25\%
\]

随机Top-5约为：

\[
5/402\approx1.24\%
\]

### 7.4 主要结果

| 模型/模态 | Top-1 | Top-5 |
|---|---:|---:|
| OpenCLIP 视觉→文本 | 28.4% | 64.9% |
| TVL 触觉→文本 | 36.7% | 70.3% |
| SSVTP 触觉→视觉 | 0.2% | 0.3% |
| TVL 触觉→视觉 | 79.5% | 95.7% |

“提升29%”主要是36.7相对28.4的相对提升，不是29个百分点。

解释这些数字时应注意：

- OpenCLIP视觉→文本和TVL触觉→文本不是完全相同的输入任务，不能简单理解为“触觉优于视觉”；
- SSVTP只见过较小的实验室数据，在HCT真实环境上存在严重域偏移；
- 很高的触觉→视觉匹配率可能部分使用了接触位置、背景、传感器姿态等同步线索；
- 该实验能证明embedding对齐，但不能证明闭环控制、滑移感知或力控制能力。

---

## 8. CLIP与LLaMA是不同层级的模型

| 模型 | 层级和作用 | 输入 | 输出/目标 |
|---|---|---|---|
| CLIP/OpenCLIP | 感知与跨模态表征 | 图像或文本 | 全局embedding、相似度 |
| TVL触觉编码器 | 触觉表征 | DIGIT RGB图像 | 触觉embedding |
| LLaMA | 语言理解和生成 | 文本token＋多模态条件 | 自回归生成文本token |

可以把CLIP理解成“语义坐标系”，把LLaMA理解成“根据条件组织和生成语言的模型”。

此前讨论的两个系统阶段是：

```text
阶段一：训练触觉编码器
触觉、视觉、语言 → 对齐到CLIP语义空间

阶段二：训练TVL-LLaMA
视觉＋触觉embedding → LLaMA hidden空间 → 生成语言
```

这与TVL-LLaMA内部“先训练projector/gate、再用LoRA微调LLaMA”的两阶段优化流程不是同一个层级。

---

## 9. 视觉和触觉如何注入LLaMA

### 9.1 对齐阶段有三个编码器

```text
视觉图像 I ─→ CLIP Vision Encoder ─→ z_v ∈ R^d
触觉图像 T ─→ TVL Tactile Encoder ─→ z_t ∈ R^d
文本标签 L ─→ CLIP Text Encoder   ─→ z_l ∈ R^d
```

这里是三个编码器输出三个**同为d维的向量**，不是“一个三维向量”。

### 9.2 生成阶段只融合两种感知embedding

```text
视觉图像 ─→ z_v ─┐
                   ├→ 平均 → z_vt
触觉图像 ─→ z_t ─┘
```

\[
z_{VT}=\frac{z_V+z_T}{2}
\]

因为两个embedding已经处在同一语义空间，所以可以做平均。然后通过可训练projector：

\[
m=P(z_{VT})
\]

projector完成：

```text
CLIP/TVL embedding维度与表示空间
                ↓
LLaMA hidden dimension与表示空间
```

### 9.3 Gate是什么

gate可以理解为一个可学习的“多模态信息音量旋钮”。概念上：

\[
h'=h+g\cdot F(h,m)
\]

- \(h\)：LLaMA原有文本hidden state；
- \(m\)：projector输出的视觉—触觉条件；
- \(F\)：多模态adapter产生的修正；
- \(g\)：可学习门控参数。

门控采用零初始化：训练刚开始时 \(g=0\)，模型保持原始LLaMA的行为；训练过程中再逐渐学习何时、在哪些层、以多大强度使用多模态信息。这样可以避免随机初始化的多模态支路立刻破坏LLaMA已有的语言能力。

### 9.4 Language在哪里

语言在两个阶段扮演不同角色：

1. **触觉对齐阶段**：语言标签经过CLIP Text Encoder形成 \(z_L\)，直接监督触觉embedding。
2. **生成阶段**：文本指令经过LLaMA自己的tokenizer；人工触觉描述作为训练目标和生成答案。

CLIP文本embedding通常不与视觉、触觉embedding一起平均。

完整结构：

```text
                      感知支路
视觉图像 ─→ E_v ─→ z_v ─┐
                          ├→ 平均 → Projector → Gate/Adapter ─┐
触觉图像 ─→ E_t ─→ z_t ─┘                                    │
                                                               ↓
文本指令 ─→ LLaMA Tokenizer ─→ 文本tokens ───────────────→ LLaMA
                                                               ↓
                                              生成触觉语言描述
```

---

## 10. 论文做了哪些生成实验

TVL Benchmark要求模型根据视觉和触觉输出最多五个形容词，再由文本版GPT-4比较模型回答与人工标签并打1–10分。

| 模型 | 综合分数 |
|---|---:|
| 最佳开源视觉语言基线ViP-LLaVA-13B | 3.80 |
| GPT-4V | 4.49 |
| SSVTP-LLaMA | 3.54 |
| TVL-LLaMA ViT-Tiny | 4.94 |
| TVL-LLaMA ViT-Small | 4.89 |
| TVL-LLaMA ViT-Base | **5.03** |

论文据此报告：

- 相对GPT-4V提高约12%；
- 相对最佳开源VLM提高约32%。

值得注意的消融结论：

- 去掉触觉—语言损失，触觉文本Top-1由36.3%降至20.3%；
- ViT-Tiny的触觉—文本结果最好，ViT-Base的触觉—视觉结果最好；
- 模型越大不一定越好，伪标签和人工测试标签之间存在分布差异；
- 加入无接触帧主要抑制训练过拟合，对最终测试准确率改善有限；
- 视觉和触觉同时输入通常优于单模态，但差距并不足以证明很强的触觉推理。

评估本身也有局限：90%训练标签来自GPT-4V，而最终又使用文本GPT-4评分，虽然评分参照的是人工标签，仍可能存在模型家族偏好与评估循环问题。

---

## 11. CLIP式编码器在现代VLM中的位置

CLIP完整结构是视觉、文本双塔：

```text
Image Encoder：图像 → embedding
Text Encoder：文本 → embedding
```

这种结构仍广泛用于图文检索、零样本分类、开放词汇感知、数据清洗，以及把触觉、音频、深度等新模态对齐到语言。

但生成式VLM通常只复用CLIP式视觉塔：

```text
视觉编码器 → Projector/Q-Former/Resampler → LLM
文本问题   → LLM Tokenizer ───────────────→ LLM
```

语言由LLM自己处理，因此不一定保留CLIP Text Encoder。现代模型也可能使用SigLIP、DINOv2或自行训练的ViT，而不是原版CLIP。

TVL的融合方式相对简单：它把视觉和触觉各压缩成一个全局embedding，再平均为一个多模态条件。这可能丢失：

- 接触发生在哪个局部区域；
- DIGIT接触斑内部的空间结构；
- 触觉随时间的变化；
- 滑移、剪切和接触力动态；
- 视觉patch和触觉patch之间的局部对应。

因此现代触觉VLM/VLA通常还需要研究多token融合、cross-attention、时序建模和闭环反馈。

---

## 12. 它对触觉模仿学习/VLA的真正作用

未来可以把TVL触觉encoder作为策略的预训练感知前端：

\[
z_t^{touch}=E_{TVL}(T_t)
\]

再与视觉、语言、机器人本体状态一起输入策略：

\[
a_t=\pi(I_t,z_t^{touch},q_t,\text{instruction})
\]

潜在价值包括：

- 不必仅靠少量机器人示范从零学习触觉特征；
- 让触觉概念可以与语言指令连接；
- 为触觉VLA提供可复用的预训练初始化；
- 通过开放词汇标签扩大触觉数据规模。

但本文没有直接验证这些下游收益。它没有：

- action输出；
- expert demonstration或行为克隆；
- manipulation success rate；
- 灵巧手控制；
- 时序触觉策略；
- 力、滑移或闭环接触控制。

另外，语言对齐可能会压缩掉没有被语言标签描述、但对控制很关键的细粒度变化。例如两个触觉帧都可以叫“hard”，但其中一个可能已经产生微小滑移。对语义理解来说它们相近，对控制策略来说却可能需要完全不同的动作。

因此本文最准确的定位是：

> 它建立了触觉进入视觉—语言基础模型的接口，但从“用语言理解触感”到“利用触感控制机器人”之间，仍需要时序建模、动作学习和真实闭环实验。

---

## 13. 术语速查

| 术语 | 本文中的含义 |
|---|---|
| Tactile observation | DIGIT内部摄像头产生的RGB触觉图像 |
| Encoder | 把某种模态转换成特征表示的网络 |
| Embedding | 对比学习和检索使用的全局向量 |
| Patch token | ViT内部对局部图像块的表示 |
| Multimodal token | projector转换后可被LLaMA使用的条件表示 |
| Alignment | 使相同语义的跨模态embedding靠近 |
| InfoNCE | 拉近正样本、推远batch内负样本的对比损失 |
| Top-k | 正确候选是否位于相似度排序前k名 |
| Projector | 将CLIP/TVL空间映射到LLaMA hidden空间 |
| Gate | 控制多模态信息注入强度的可学习机制 |
| Pseudo-label | GPT-4V根据视觉图像生成的触感文本标签 |
