# 《Octopi: Object Property Reasoning with Large Tactile-Language Models》精读笔记

> Samson Yu et al., Robotics: Science and Systems (RSS) 2024<br>
> 正式论文：[Yu et al. - 2024 - Octopi Object Property Reasoning with Large Tactile-Language Models](<Yu et al. - 2024 - Octopi Object Property Reasoning with Large Tactile-Language Models.pdf>)<br>
> [RSS Proceedings](https://roboticsproceedings.org/rss20/p066.html) · [arXiv](https://arxiv.org/abs/2405.02794) · [代码与数据](https://github.com/clear-nus/octopi)

> 推荐先读：[TVL](15_tvl_paper_reading_notes.md)、[表示学习、CLIP 与对比学习](21_clip_and_contrastive_learning.md)和[多模态融合与条件注入](09_attention_multi_expert_and_conditioning.md)。本文重点是触觉属性接口与语言推理。

## 0. 一句话定位

Octopi研究的不是“如何根据触觉直接控制机器人”，而是：**如何让大语言模型理解真实触觉，并结合已有常识对物体的物理属性进行推理。**

它的核心逻辑是：

```text
GelSight触觉视频
        ↓
硬度、粗糙度、凹凸程度等物理事实
        ↓
自然语言属性描述
        ↓
Vicuna已有的物理常识
        ↓
比较、匹配和场景选择
```

例如，相机可能无法可靠判断两个牛油果哪个更成熟，但触觉可以发现左边更软；Vicuna又知道“成熟牛油果通常更软”，于是模型能够选择左边的牛油果。

这里必须先划清边界：Octopi输出的是**语言描述或离散判断**，不是连续机器人动作，因此它是large tactile-language model，而不是VLA策略。

---

## 1. 论文要解决的问题

传统LLM和VLM拥有大量文本和视觉常识，但它们很难直接获得视觉不可辨识的物理信息，例如：

- 一个物体究竟软还是硬；
- 表面是真光滑，还是看起来光滑但摸起来粗糙；
- 两个外观相似的水果哪个更成熟；
- 一个材料在潮湿条件下是否容易抓住。

另一方面，GelSight可以观测接触形变和表面纹理，却不能天然调用语言模型中的常识。Octopi尝试在这两部分之间建立接口：

```text
触觉编码器：负责从接触图像中提取物理证据
Vicuna：负责理解语言并调用常识进行推理
Projector：负责连接两个表示空间
```

因此，Octopi真正验证的是：**只要把触觉证据转化成LLM能够使用的表示和属性语言，已有LLM的常识就能够迁移到触觉物理推理中。**

---

## 2. 三项主要贡献

### 2.1 PHYSICLEAR数据集

作者构建了一个面向触觉物理属性理解的数据集：

- 74种日常物体；
- 408段真实GelSight触觉视频；
- 对应的普通相机采集视频；
- 物体名称和部件名称；
- 超过1,200条物理属性标注；
- 训练、验证、测试按照物体划分为60/7/7，测试物体在训练阶段不可见。

采集包含两种连续探索动作：

```text
按压：观察压力导致的形变
旋转：观察剪切和表面纹理变化
```

### 2.2 CLIP-based触觉视频编码器

作者从CLIP ViT-L/14的视觉编码器出发，通过ViFi-CLIP和Visual Prompt Tuning把它适配成触觉视频编码器，使其能够预测硬度、粗糙度和凹凸程度。

### 2.3 触觉属性描述驱动的物理推理

Octopi把触觉特征接入Vicuna 1.5，并使用Object Property Description（OPD）作为显式中间步骤：

```text
触觉视频
    ↓
“柔软、表面光滑、没有明显凸起”
    ↓
“柔软的牛油果通常更加成熟”
    ↓
选择更可能成熟的牛油果
```

这不是一种新的LLM推理算法，而是通过物理属性语言把触觉感知与Vicuna已有的常识连接起来。

---

## 3. PHYSICLEAR究竟标注了什么

### 3.1 三种人工物理属性

每段触觉数据由三名独立标注者标注，最终使用平均评分。三个属性分别有三个等级：

| 属性 | 含义 | 三个类别 |
|---|---|---|
| Hardness | 表面受压后是否容易变形 | soft / moderately hard / hard |
| Roughness | 手指滑过时的摩擦和粗糙感 | smooth / slightly rough / rough |
| Bumpiness | 表面凸起的明显程度 | no bumps / small bumps / big bumps |

其中：

- 硬度和粗糙度主要依据标注者实际触摸物体后的感受；
- 凹凸程度更多依据GelSight图像中的凸起进行判断；
- 论文报告三个属性的标注者一致性ICC3k分别为0.894、0.979和0.792。

### 3.2 数据集不只是“图像加三个label”

更准确地说，底层数据包含：

```text
GelSight触觉视频
+ 采集过程的普通视频
+ 物体/部件名称
+ hardness标签
+ roughness标签
+ bumpiness标签
```

物体和部件名称会用于Property-object Matching等任务。例如，同一把牙刷可以包含handle和bristles两个不同的触觉部位。

### 3.3 复杂问答是如何构造的

五类任务不是五套独立的人工逐条标注。作者先得到触觉视频、物体名称和三个属性标签，再通过样本组合和prompt模板构造语言任务。

```text
人工物理属性标签
+ 物体/部件名称
+ 不同触觉视频的组合
+ 固定prompt模板
+ ChatGPT 3.5生成并由人工清理的OPD描述
        ↓
训练和评估用的指令—回答样本
```

ChatGPT 3.5主要用于生成OPD的非结构化描述，并不是自由生成全部任务数据。

OPD的结构化部分可以直接由标签套模板得到：

```text
[moderately hard, smooth, no bumps]
        ↓
“Overall, it presents a moderately hard and
smooth surface with no bumps.”
```

PC、PSS和POM则主要通过组合不同视频，再根据属性标签和物体名称确定正确答案。

---

## 4. 五个训练与评估任务

### 4.1 OPD：Object Property Description

输入一段触觉视频，生成物理属性描述：

```text
输入：GelSight触觉视频
输出：物体较软，表面光滑，没有明显凸起
```

OPD同时包含：

- 结构化的三属性总结；
- 更自然、更丰富的非结构化描述。

### 4.2 PC：Property Comparison

给定两段触觉视频，判断一个物体是否比另一个更硬、更粗糙或具有更大的凸起。

完整prompt要求模型先分别描述两个物体，再给出比较结论。

### 4.3 PSS：Property Superlative Selection

给定三段触觉视频，从中选出最软、最硬、最光滑或最粗糙的物体。

### 4.4 POM：Property-object Matching

给定三段触觉视频和三个候选物体名称，将视频与物体匹配。这个任务要求模型把触觉属性与已有物体常识结合起来。

### 4.5 PSR：Property Scenario Reasoning

给定两个物体的触觉观测和一个现实场景，选择物理属性更适合该场景的物体。例如：

- 哪个物体适合清理不粘锅又不划伤锅面；
- 哪个物体在潮湿环境中更容易抓稳；
- 哪个物体适合击碎汽车表面的薄冰。

五个任务的训练关系是：

| 任务 | 训练 | 评估 |
|---|---:|---:|
| OPD | ✓ | ✓ |
| PC | ✓ | ✓ |
| PSS | ✓ | ✓ |
| POM | ✓ | ✓ |
| PSR | ✗ | ✓ |

PSR完全不参加训练，用于测试模型能否把学到的触觉属性和Vicuna的常识迁移到新场景。

---

## 5. Octopi网络架构

### 5.1 整体数据流

```text
5帧GelSight触觉图像
        ↓
CLIP-based tactile encoder
        ↓
触觉特征
        ↓
两层MLP Projector
        ↓
与Vicuna word embedding同维度的tactile embeddings
        ↓
插入语言embedding序列
        ↓
Vicuna 1.5 7B/13B
        ↓
属性描述、比较结果或场景判断
```

### 5.2 触觉编码器

Octopi不是从头初始化一个触觉ViT，而是复用预训练CLIP ViT-L/14：

```text
原始CLIP：自然RGB图像 → CLIP视觉表示

Octopi：GelSight触觉帧 → 适配后的CLIP视觉表示
```

由于触觉视频和互联网自然图像之间存在明显domain gap，作者加入：

- ViFi-CLIP的视频输入方式；
- Visual Prompt Tuning；
- 每个Transformer层中的8个task-specific learnable prompts；
- 一个shared linear layer；
- 硬度、粗糙度、凹凸程度三个分类头。

原始CLIP Transformer backbone保持冻结。

### 5.3 五帧触觉视频输入

原始视频平均约112帧。作者先根据相邻帧像素强度变化选出变化最大的30%帧，再使用5帧作为模型输入：

- 训练时从显著帧中随机采样5帧；
- 测试时从显著区域开始均匀选择5帧。

ViFi-CLIP在编码器属性分类阶段将逐帧特征平均池化为视频级表示。

### 5.4 Projector

Projector参考LLaVA，由两层线性层和中间的GELU组成：

```text
CLIP tactile feature
        ↓
Linear → GELU → Linear
        ↓
Vicuna word embedding dimension
```

它解决的是CLIP触觉特征与Vicuna输入维度及表示空间不一致的问题。

### 5.5 触觉token如何进入Vicuna

普通文字经过：

```text
自然语言prompt
        ↓
Vicuna tokenizer
        ↓
token IDs
        ↓
Vicuna word embedding layer
        ↓
word embeddings
```

`<tact_start>`和`<tact_end>`是两个新增的特殊语言token，拥有新训练的word embeddings。触觉图像本身不经过tokenizer，而是经过CLIP和projector变成连续embedding。

最终序列近似为：

```text
[word embeddings]
<tact_start embedding>
[tactile embedding 1]
[tactile embedding 2]
[tactile embedding 3]
[tactile embedding 4]
[tactile embedding 5]
<tact_end embedding>
[word embeddings]
```

### 5.6 Octopi没有gate

论文没有使用zero-initialized gate或其他显式门控融合模块。触觉embedding在指定位置与文本embedding合并，然后直接输入Vicuna。

Vicuna内部的self-attention会学习语言token和触觉token之间的关系，但这不等同于单独设计的gate。

---

## 6. 三阶段训练流程

### 6.1 第一阶段：Encoder Fine-tuning

这一阶段只使用PHYSICLEAR的触觉视频和三个物理属性标签：

```text
GelSight视频
    ↓
ViFi-CLIP + Visual Prompt Tuning
    ├→ hardness分类头
    ├→ roughness分类头
    └→ bumpiness分类头
```

训练：

- Visual Prompts；
- shared linear layer；
- 三个属性分类头。

冻结：

- 原始CLIP Transformer backbone。

三个分类头同时使用cross-entropy loss。论文没有给出三个loss的具体加权公式，因此只能将其理解为常规的多任务分类训练，不能断言作者使用了某组特殊权重。

### 6.2 第二阶段：Tactile Feature Alignment

完成属性分类后，作者丢弃三个分类头，使用约8K条PHYSICLEAR语言监督样本训练触觉到Vicuna的接口。

训练：

- 两层MLP Projector；
- `<tact_start>`和`<tact_end>`的word embeddings。

冻结：

- tactile encoder；
- Vicuna主体。

目标是让projector输出的触觉embedding能够被被冻结的Vicuna解释。

### 6.3 第三阶段：End-to-end Fine-tuning

作者再使用约3K条PHYSICLEAR指令样本训练：

- Projector；
- word embedding layer；
- Vicuna的LoRA参数。

触觉编码器仍然冻结。因此论文虽然称其为end-to-end fine-tuning，但并不是所有模型参数都更新。

### 6.4 四个语言任务不是四个独立loss

OPD、PC、PSS、POM被组织为统一的指令微调样本：

```text
USER prompt + tactile embeddings
                ↓
        ASSISTANT target
```

一个batch可以混合来自不同任务的样本，并对ASSISTANT目标token使用统一的自回归语言建模损失。论文没有定义：

```text
L = L_OPD + L_PC + L_PSS + L_POM
```

因此，更合适的理解是“共享一个语言生成目标的多任务指令微调”，而不是四个task-specific loss的加权训练。

### 6.5 三阶段可以这样记忆

```text
第一阶段：用三属性标签教CLIP看懂触觉

第二阶段：用8K语言样本教Projector
          把触觉特征翻译到Vicuna输入空间

第三阶段：用3K语言样本微调Projector和Vicuna LoRA
          学会描述、比较、选择和匹配
```

---

## 7. OPD为什么重要

### 7.1 完整Octopi的推理方式

完整模型被要求先描述物理属性，再给出答案。例如PSS：

```text
a) 柔软、粗糙、小凸起
b) 较硬、光滑、大凸起
c) 柔软、略粗糙、小凸起

Conclusion: b) is the smoothest.
```

模型的处理链是：

```text
触觉视频
    ↓
显式物理属性描述
    ↓
比较或常识推理
    ↓
最终答案
```

### 7.2 `w/o OPD`是什么意思

`w/o OPD`去掉中间物理属性描述，要求模型从触觉embedding直接预测最终答案：

```text
完整模型：描述三个物体，然后选择最光滑的一个
w/o OPD：直接选择最光滑的一个
```

这不只是推理时临时换一句prompt。作者还微调了不要求中间属性预测的模型变体，因此它是训练和输出结构的消融实验。

### 7.3 为什么触觉编码器已经有属性，仍需要OPD

“触觉编码器包含属性信息”不等于“Vicuna能稳定地从连续触觉embedding中提取属性并调用文本常识”。如果直接回答，模型要在一步内完成：

```text
识别触觉
+ 提取相关物理属性
+ 判断场景依赖哪个属性
+ 调用常识
+ 比较候选物体
+ 输出答案
```

OPD将问题拆成两步：

```text
第一步：这个物体摸起来怎样？
第二步：具有这些属性的物体是否适合当前场景？
```

它相当于一个显式的语义瓶颈：

- 把连续触觉embedding转换成Vicuna熟悉的语言概念；
- 强迫模型关注与任务相关的物理属性；
- 降低模型绕过触觉、利用语言偏差猜答案的风险；
- 为感知和推理提供可检查的中间结果。

但OPD不是总能带来正确答案。如果第一步将“软”错误描述为“硬”，后续推理也会被错误的中间事实带偏。

---

## 8. PSR能力从哪里来

PSR是论文的重要贡献之一，但它不是通过PSR样本训练出来的，也没有专门的reasoning module或PSR loss。

其能力来自三个部分：

### 8.1 触觉编码器提供物理事实

```text
GelSight视频 → 柔软、粗糙、有凸起
```

### 8.2 OPD、PC、PSS和POM完成触觉—语言grounding

这些训练任务让模型学会：

- 什么触觉模式对应soft或rough；
- 如何比较两个物体；
- 如何从多个物体中选择；
- 如何把物理属性和物体名称联系起来。

### 8.3 Vicuna提供文本预训练常识

Vicuna原本已经可能知道：

```text
成熟牛油果通常更软
粗糙表面在潮湿条件下更容易抓稳
柔软材料不容易划伤不粘锅
硬物更适合击碎薄冰
```

Octopi把触觉事实和这些常识连接起来。

因此，PSR可以描述为：

> 通过人为设计的物理属性中间接口，激活Vicuna已有常识后产生的零样本任务迁移。

作者没有创新一种新的LLM推理算法，但创新了数据、属性接口和评估任务的组合方式，并证明这种组合能够让LLM基于真实触觉进行物理推理。

---

## 9. 关键实验结果

### 9.1 OPD对物理理解任务的影响

| 模型 | PC | PSS | POM |
|---|---:|---:|---:|
| 随机基线 | 33.33% | 33.33% | 16.67% |
| Octopi-7B w/o OPD | 46.51% | 39.88% | 23.23% |
| Octopi-7B | 48.10% | 74.67% | 44.39% |
| Octopi-13B w/o OPD | 40.70% | 39.88% | 18.71% |
| Octopi-13B | 55.06% | 84.00% | 60.43% |

OPD对PSS和POM的提升尤其显著，说明“先描述属性，再回答”比直接从触觉embedding映射到选择结果更加稳定。

### 9.2 PSR场景推理

PSR评估时不提供人工ground-truth属性描述。模型必须先根据触觉自己预测属性，再回答场景问题。

| 方法 | Octopi-7B | Octopi-13B |
|---|---:|---:|
| PSR | 69.57% | 67.39% |
| PSR w/o OPD | 63.04% | 39.13% |
| 随机基线 | 50.00% | 50.00% |

13B模型去掉OPD后从67.39%下降到39.13%，进一步说明显式物理属性是触觉证据和LLM常识之间的重要接口。

值得注意的是，13B并没有在PSR上稳定优于7B。这说明更大的语言模型不必然解决触觉grounding问题，触觉编码和中间属性质量同样关键。

### 9.3 CLIP触觉微调消融

作者比较了：

```text
原始off-the-shelf CLIP
vs.
用PHYSICLEAR物理属性微调后的CLIP
```

微调后的编码器通常在属性预测、PC、PSS和POM上表现更好，说明自然图像预训练并不能自动消除GelSight触觉图像的domain gap。

### 9.4 LoRA端到端微调消融

加入Vicuna LoRA后，模型的属性预测和物理理解任务通常进一步改善，尤其是13B模型。这说明只训练projector不足以完全适配触觉输入，少量语言模型参数微调仍然有价值。

### 9.5 牛油果真实机器人实验

作者把两个GelSight传感器安装到7-DoF Franka Emika Panda上，对10个牛油果采集200个触觉样本，并构造100对成熟度比较。

Octopi-13B得到：

- 三属性同时正确的准确率：35.50%，随机为3.70%；
- 成熟度比较准确率：63.00%，随机为50.00%。

Octopi首先利用常识判断硬度和凹凸程度与成熟度更相关，再根据触觉属性选择更成熟的牛油果。

但该实验不是端到端机器人策略：

```text
Octopi：判断哪个牛油果更成熟
ROS与预编程流程：执行抓取和放置
```

---

## 10. 论文的局限

### 10.1 数据规模仍然较小

74种物体和408段触觉视频足以验证概念，但距离通用触觉基础模型的数据规模仍有较大差距。

### 10.2 属性空间很窄

模型主要围绕三个离散属性：

```text
hardness / roughness / bumpiness
```

没有覆盖温度、重量、摩擦系数、粘性、弹性恢复、剪切方向或精确受力等更复杂物理量。

### 10.3 离散标签损失连续物理信息

“soft / moderately hard / hard”适合语言推理，但无法替代杨氏模量、摩擦系数或力/位移曲线等连续物理测量。

### 10.4 传感器泛化有限

模型主要在GelSight数据上训练，没有充分证明对DIGIT、GelSlim、TacTip或非视觉式触觉传感器的泛化。

### 10.5 探索动作相对固定

数据主要来自按压和旋转。模型没有学习“为了判断某种属性，应该主动执行什么探索动作”。

### 10.6 不是闭环控制策略

Octopi输出语言或离散选择，没有直接预测：

- 机械臂位姿；
- 手指关节动作；
- 抓取力；
- 高频滑移修正。

因此牛油果实验验证的是触觉辅助的高层判断，而不是端到端接触操作能力。

### 10.7 OPD可能传播错误

显式属性描述提高了可解释性和总体准确率，但也形成信息瓶颈。如果属性预测错误，后续常识推理可能建立在错误事实之上。

---

## 11. 与TVL的核心区别

两篇论文都在让语言模型理解触觉，但关注点不同：

```text
TVL：
触觉怎样进入视觉—语言的开放语义空间？

Octopi：
怎样从触觉中提取物理属性，并让LLM利用这些属性推理？
```

| 对比项 | TVL | Octopi |
|---|---|---|
| 触觉编码器 | 随机初始化ViT | 预训练CLIP ViT-L/14 |
| 主要训练目标 | 触觉—视觉—语言对比学习 | 三属性分类与语言指令微调 |
| 触觉输入 | 主要为单帧DIGIT RGB | 5帧GelSight视频 |
| LLM | LLaMA2-7B | Vicuna 1.5 7B/13B |
| 融合方式 | 多模态特征平均、projector和gate | tactile token直接插入文本序列 |
| 中间物理属性 | 不强制显式预测 | 强调OPD中间描述 |
| 主要任务 | 开放词汇检索和触觉描述 | 描述、比较、匹配和场景推理 |

一句话记忆：

```text
TVL建立“触觉语义接口”；
Octopi利用“物理属性接口”完成常识推理。
```

---

## 12. 原文定位索引

下表页码按17页PDF在阅读器中的页码记录：

| 内容 | 原文位置 |
|---|---|
| 论文贡献概述 | PDF第2页，Introduction末尾Contributions |
| 数据采集与人工属性标注 | PDF第3页，Section III.B |
| 五个任务定义 | PDF第4页，Section III.C |
| 哪些任务参与训练 | PDF第5页，Table III |
| Prompt和目标回答示例 | PDF第5页，Table IV |
| PSR三个场景 | PDF第5页，Table V |
| 整体架构 | PDF第4页，Figure 3与Section IV |
| tactile embedding插入方式 | PDF第5页底部，inference process |
| Encoder fine-tuning | PDF第5—6页，Section IV.A |
| Feature alignment | PDF第6页，Section IV.B |
| LoRA端到端微调 | PDF第6页，Section IV.C |
| 8K/3K训练样本和超参数 | PDF第6页，Section V.B |
| OPD消融 | PDF第6—7页，Table VI与Section VI.A |
| PSR与w/o OPD | PDF第7页，Table VII与Section VI.B |
| 牛油果实验 | PDF第7—8页，Section VI.C与Table VIII |
| Encoder和LoRA消融 | PDF第8—9页，Section VII |
| 详细人工标注 | PDF第13页，Appendix A |
| 物体、部件和数据划分 | PDF第14页，Appendix B |
| 完整prompt模板 | PDF第16页，Appendix D |
| Encoder特征分析 | PDF第16—17页，Appendix E |

---

## 13. 最终总结

Octopi的完整故事可以压缩为：

```text
PHYSICLEAR提供真实GelSight视频和三个物理属性标签
                         ↓
用多任务属性分类把CLIP视觉编码器适配成触觉编码器
                         ↓
用Projector把触觉特征映射到Vicuna词向量空间
                         ↓
用OPD、PC、PSS、POM完成触觉—语言属性grounding
                         ↓
先生成显式物理属性，再调用Vicuna已有常识
                         ↓
在没有PSR训练数据的情况下完成场景推理
```

它最重要的设计不是一个复杂的融合层，也不是新的LLM推理算法，而是：

> **用显式物理属性描述，把真实触觉感知和大语言模型的常识推理连接起来。**
