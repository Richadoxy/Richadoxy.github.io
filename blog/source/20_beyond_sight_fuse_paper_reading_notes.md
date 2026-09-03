# 《Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding》精读笔记

> Joshua Jones, Oier Mees, Carmelo Sferrazza, Kyle Stachowicz, Pieter Abbeel, Sergey Levine，IEEE International Conference on Robotics and Automation（ICRA）2025<br>
> 正式论文：[Jones 等 - 2025 - Beyond Sight Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding](<Jones 等 - 2025 - Beyond Sight Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding.pdf>)<br>
> [arXiv](https://arxiv.org/abs/2501.04693) · [项目主页](https://fuse-model.github.io/) · [开源代码](https://github.com/fuse-model/FuSe) · [Hugging Face论文页](https://huggingface.co/papers/2501.04693)

## 0. 一句话定位

Beyond Sight提出的**FuSe不是一个全新的VLA backbone**，而是一套微调已有通用机器人策略的方法：把视觉、触觉和声音都编码成token，交给同一个策略Transformer处理，再用自然语言作为公共语义监督，逼迫策略真正理解新增传感器，而不是继续只依赖预训练时熟悉的视觉和语言。

```text
已有通用机器人策略（Octo / PaliGemma VLA）
                 +
少量带触觉、声音和动作的机器人数据
                 +
动作模仿 + 对比式语言对齐 + 传感器到语言生成
                 ↓
能够根据 vision + touch + audio + language 共同决策的策略
```

如果沿着此前几篇论文的知识链看：

- [TVL](15_tvl_paper_reading_notes.md)解决“触觉如何进入视觉—语言语义空间”；
- [Octopi](17_octopi_paper_reading_notes.md)解决“触觉如何生成属性描述并参与物理推理”；
- [AnyTouch](18_anytouch_paper_reading_notes.md)解决“如何跨传感器、跨静态—动态形式学习统一触觉表征”；
- **FuSe第一次把这条线明确接入通用机器人策略，让触觉和声音参与动作输出。**

---

## 1. 论文究竟在解决什么问题

大规模通用机器人策略通常有丰富的视觉—语言—动作预训练，但几乎没有同步的触觉和音频数据。现在拿一个只有约27K条轨迹的小数据集，给模型加上DIGIT和麦克风，直接做行为克隆，看似可行，实际有三个困难：

1. 新增的触觉、音频encoder要从很少的数据里学会有用表征；
2. 预训练策略已经很擅长看图，微调时容易继续走视觉捷径，**忽略新传感器**；
3. “红色”“柔软”“发出钢琴声”等信息分属于不同模态，需要一套共同语义才能支持组合指令。

例如，袋子里的物体被遮挡后，视觉已经不足以区分；夹爪必须先接触物体，利用触觉判断软硬，才能决定是不是目标。若训练只用动作MSE，即使网络形式上收到了触觉token，它也可能学成：

```text
touch token ──> 几乎不使用
RGB + instruction ──> action
```

FuSe的核心问题因此不是“怎样把一个张量拼进去”，而是：

> 怎样提供足够强的训练信号，让策略知道触觉和声音分别代表什么，并在视觉不充分时真的用它们。

---

## 2. 主要贡献

### 2.1 一套与backbone相对解耦的多传感器微调方法

FuSe在正常动作模仿损失以外加入两条辅助监督：

- multimodal contrastive loss：让传感器观测表示与对应文本语义靠近；
- sensory-grounded language generation loss：让传感器观测直接生成属性描述。

二者都以自然语言作为公共接口，使视觉、触觉和声音不必拥有完全重叠的大规模数据，也可以在高层语义上建立联系。

### 2.2 在两类差异很大的通用策略上验证

作者不是只在一种架构上实验，而是把FuSe用于：

- Octo：机器人原生的multimodal Transformer policy；
- 3B PaliGemma VLA：由互联网预训练VLM扩展而来的生成式VLA。

这说明FuSe更接近一套训练recipe，而不是只能附着在某个特定网络上的模块。

### 2.3 一个同步包含多传感器和动作的真实机器人数据集

作者采集26,866条真实机器人遥操作轨迹，其中包含：

- 第三人称RGB；
- 腕部RGB；
- 夹爪两侧的两个DIGIT触觉传感器；
- 麦克风音频；
- 机器人动作、proprioception与IMU等记录；
- 任务指令以及事后补充的模态属性文本。

这和只有“触觉图像—属性标签”的表征数据集不同：它包含低层机器人动作，因此可以直接训练policy。

### 2.4 新能力与真实机器人验证

微调后的策略可以完成：

- 多模态提示，例如“拿起红色、摸起来柔软并会发出响声的物体”；
- 视觉被遮挡时依靠触觉；
- 根据声音选择按钮；
- 跨模态组合提示；
- 接触后生成物体描述。

论文报告FuSe相对所有考虑的baseline，整体成功率提升超过20个百分点。

---

## 3. 先纠正一个VLA架构上的前提

“VLA = VLM + Action Expert”是一类很重要的架构，但不是全部VLA的统一定义。

可以把常见架构粗分为三类：

| 类型 | 代表 | 动作如何产生 |
| --- | --- | --- |
| VLM直接生成动作token | OpenVLA、PaliVLA | 把动作离散化，像生成文本token一样自回归预测 |
| VLM + 独立Action Expert | π0.5、GR00T N1.x | VLM提供语义memory，Action Expert处理state、noisy action与时间条件 |
| 机器人multimodal Transformer + Action Head | Octo | 视觉、语言、观测token在机器人Transformer中融合，再由动作头解码 |

FuSe论文中的主backbone **Octo属于第三类**。它不是“一个现成VLM后接Action Expert”，其中也没有一个等价于PaliGemma、Qwen-VL或LLaMA的完整VLM。

Octo主要由以下部分组成：

```text
RGB tokenizer / image encoder ─┐
T5 language encoder ───────────┼─> Octo Transformer ─> readout ─> action head
other observation tokenizers ──┘
```

T5-base负责把语言变成token embedding，但不承担多模态推理和动作生成；真正融合任务与观测的是Octo Transformer。

因此，“FuSe把touch、audio embedding和language embedding拼起来送进VLM”只适合非常宽泛的口语描述。更准确的说法是：

> FuSe把各模态编码、投影为Octo共享宽度的token组，在Octo Transformer中用joint self-attention融合，再让不同readout读取融合结果。

---

## 4. Octo中的readout与Action Head

### 4.1 readout是什么

readout不是“直接取Transformer序列的最后一个普通token”，而是一组**可学习的查询token**。它们和观测token一起通过Transformer，在attention中读取任务和观测信息，最终形成面向某个输出目标的hidden state。

可以把它理解为：

```text
大量观测token：描述当前世界发生了什么
readout token：带着“我要为某个输出收集信息”的目的去读这些观测
```

FuSe的Octo实现至少区分：

- action readout：为动作预测汇总信息；
- language readout：为对比式语义对齐和语言生成汇总信息。

开源配置中，额外加入24个language readout tokens。readout先经过共享Transformer，不同head还可用Multi-head Attention Pooling（MAP）再次汇聚。

### 4.2 Action Head是什么

Action Head泛指把策略内部hidden state转换为动作的输出模块。它不一定只是一个单层projector，也可能是：

- MLP回归头；
- 离散动作token生成头；
- diffusion denoiser；
- flow-matching Action Expert。

FuSe-Octo采用的具体路径是：

```text
action readout hidden states
        ↓
MAP attention pooling
        ↓
Dense projection
        ↓
reshape为 [action_horizon=4, action_dim=7]
        ↓
连续动作块
```

所以在这篇论文的Octo实验里，Action Head确实比较接近“attention pooling + 线性投影”，但不能把所有VLA的Action Head都等同于简单MLP。

---

## 5. Figure 2：FuSe的数据流到底是什么

Figure 2可以拆成共享主干和三条训练分支。

### 5.1 各模态先被tokenize

```text
第三人称RGB ─┐
腕部RGB ─────┴─> Octo image tokenizer / encoder ─> visual tokens

左DIGIT差分图 ─┐
右DIGIT差分图 ─┴─> shared pretrained TVL encoder ─> tactile tokens

最近1秒音频 ─> mel spectrogram ─> ResNet26 ─> audio tokens

instruction ─> frozen T5-base ─> language tokens
```

触觉输入不是depth图或marker位移图，而是DIGIT的视触觉RGB图像。论文对每个DIGIT减去无形变背景图，突出接触形变，再把左右两个传感器分别送入**共享权重**的TVL encoder。

所有token会投影到Octo的公共hidden width，按token组放入共享Transformer。它们不是先平均成三个向量；多模态融合发生在Transformer的attention中。

### 5.2 joint self-attention完成融合

假设序列中有：

```text
[task tokens]
[visual tokens, tactile tokens, audio tokens, action readout, language readouts] at t0
[visual tokens, tactile tokens, audio tokens, action readout, language readouts] at t1
```

Octo采用blockwise causal attention规则，让任务token、各时刻观测和readout按照允许的信息方向交流。经过多层Transformer后，readout hidden states已经包含跨模态信息。

这与单独训练三个encoder、只在loss末端比较全局embedding不同。FuSe既在共享策略中做token级融合，又在辅助loss中施加语言对齐。

更多attention背景可参见[Attention专题](19_attention_from_embeddings_to_action_expert.md)。

---

## 6. 三条loss分支

FuSe总目标写成：

\[
\mathcal{L}
= \mathcal{L}\_{\mathrm{BC}}
+ \beta \mathcal{L}\_{\mathrm{gen}}
+ \lambda \mathcal{L}\_{\mathrm{contrast}}
\]

实验中作者取 \(\beta=1\)、\(\lambda=1\)。这不是四种任务各有一个action loss再组合，而是同一个batch上同时计算动作监督和两种语言辅助监督；不同模态组合内部再求平均。

### 6.1 动作模仿损失

```text
instruction + 当前多传感器观测
        ↓
Octo Transformer
        ↓
action readout + MSEActionHead
        ↓
预测动作块 â
        ↓ compare
遥操作ground-truth动作 a
```

其核心是直接回归：

\[
\mathcal{L}\_{\mathrm{BC}}=\lVert \hat{a}-a\rVert^2
\]

这里没有加噪动作、diffusion timestep或反复去噪，推理时一次前向即可得到动作块。

### 6.2 Multimodal Contrastive Loss

这条分支要回答：“只看这些传感器观测，policy内部表示能否对应到正确的自然语言语义？”

```text
一侧：vision / touch / audio的某种组合
      ─> Octo Transformer（不输入真实instruction）
      ─> language readout
      ─> MAP + projection
      ─> z_obs

另一侧：与这段轨迹匹配的正确文本
      ─> frozen T5-base
      ─> pooling + projection
      ─> z_text

CLIP-style contrastive loss：配对样本靠近，batch内不匹配文本远离
```

训练时必须拿掉或置空输入instruction，否则网络可以直接复制文本语义，不需要观察触觉和音频。

这里的几个名词要分开：

- **language readout**：Octo Transformer中的可学习查询token，用来从传感器观测中读出适合语言对齐的表示；
- **T5-base**：独立、冻结的文本encoder，把正确标注转成语言embedding，充当语义锚点；
- **Contrastive Head**：把两侧表示池化并投影到同一512维latent，再计算对称的CLIP/InfoNCE式loss。

这与TVL的区别是：TVL主要对齐独立的原始模态encoder输出，目标近似为

\[
x\_T \approx x\_V \approx x\_L.
\]

FuSe对齐的则是**policy看过传感器之后形成的高层观测表示**与文本语义。它更关心“这些信息是否已进入策略内部”，而不是只得到一个可检索的触觉encoder。

### 6.3 Sensory-grounded Language Generation Loss

这条分支要求模型从传感器主动说出它感知到了什么：

```text
vision / touch / audio的某种组合
        ↓
Octo Transformer（真实instruction置空）
        ↓
24个language readout states
        ↓
generation head + vocabulary projection
        ↓
预测文本token
        ↓ cross entropy
正确属性描述token
```

作者枚举七种非空组合：

```text
V, T, A, V+T, V+A, T+A, V+T+A
```

Octo本身不是LLM，因此这里的生成器是一个轻量附加head，不是LLaMA或Vicuna。开源实现用模态类别token、MAP和词表投影，预测最多24个T5词表token。

这个loss比对比学习更强：对比学习只要求“选对/靠近正确文本”，生成loss要求保留足够明确的属性信息，真正把它说出来。

---

## 7. 最容易混淆的一点：Octo到底有没有diffusion

答案要分成“原始Octo”和“FuSe-Octo实验配置”。

### 7.1 原始Octo：有DDPM式Diffusion Action Head

原始Octo的典型动作头先让Transformer产生action readout condition，再由一个残差MLP式diffusion网络预测动作噪声：

\[
a\_t=\sqrt{\bar{\alpha}\_t}a\_0+
\sqrt{1-\bar{\alpha}\_t}\epsilon,
\]

\[
\mathcal{L}\_{\mathrm{diff}}
=\lVert \epsilon-\epsilon\_\theta(a\_t,t,c)\rVert^2.
\]

它不是π0.5或GR00T那种Transformer Action Expert；noisy action和时间步进入的是动作头内部的diffusion MLP。

### 7.2 FuSe的Octo配置：换成直接MSE Action Head

Beyond Sight正文明确把最终目标中的动作项称为MSE imitation loss；发布代码的`fuse_config.py`也把原始action head替换为`MSEActionHead`，输出4步、每步7维的动作块。

因此：

| 模型 | 动作生成形式 | 监督目标 |
| --- | --- | --- |
| 原始Octo | DDPM diffusion head | 预测噪声 |
| FuSe-Octo | 直接连续动作回归 | 预测动作与示范动作的MSE |
| PaliGemma FuSe | 生成离散动作token | token cross-entropy |
| π0.5 / GR00T | flow-matching Action Expert | 预测velocity field |

论文摘要说FuSe适用于“diffusion-based generalist policies”，指它以Octo这一预训练模型家族为起点；但**FuSe主实验的发布配置本身不靠diffusion采样动作**。仅看摘要或只看原始Octo介绍，很容易把这两个层次混在一起。

---

## 8. “通过语言建立公共语义”究竟是什么意思

视觉、触觉和声音的raw signal完全不同：

```text
视觉像素 ─> yellow / round
触觉形变 ─> soft / rough
音频频谱 ─> piano / metallic clink
```

它们不需要在像素空间相等，也不要求所有模态两两直接重建。作者利用人类语言给这些信息一个共同可组合的坐标系：

```text
“pick the yellow object that feels soft”
“press the button that plays piano”
```

这种设计的作用有两层：

1. **语义锚定**：触觉或声音表示被拉向已有语言语义，而不是只学与动作偶然相关的统计特征；
2. **组合接口**：颜色、材质、声音等概念可以在一句新指令中组合，即使训练数据没有覆盖每一种组合。

因此FuSe不是简单要求：

\[
x\_V=x\_T=x\_A.
\]

它要求不同传感器在与任务有关的高层含义上能被同一语言描述和查询。

---

## 9. 正确文本标注从哪里来

Contrastive Loss和Language Generation Loss的正确文本都已经在微调数据准备阶段生成，不是机器人推理时在线调用ChatGPT，也不是T5自己凭空提供ground truth。

流程大致是：

```text
每段机器人轨迹 / 物体的已知属性
        ↓
人工设计的模态模板
        ↓
vision、touch、audio及其组合描述
        ↓
ChatGPT离线生成保持语义不变的改写版本
        ↓
训练时按概率采样原模板或改写句子
```

例如同一条轨迹可以拥有：

- visual text：“grab the red round object”；
- tactile text：“pick the object that feels soft”；
- audio text：“push the button that plays piano”；
- compositional text：“grab the red object that feels soft”。

论文为每种模态组合准备20种改写模板，开源配置以0.5概率使用rephrased instruction。无法成立的模态组合会通过validity mask排除。

需要注意：这更接近**轨迹/物体级属性描述**，不是每一帧触觉图像都由人工写一段自由文本。ChatGPT的作用是扩写表达方式，不负责从原始传感器中发现ground truth。

---

## 10. PaliGemma版本与Octo版本有什么不同

PaliGemma分支才更接近“VLM式VLA”：

```text
PaliGemma 3B
= SigLIP So400m/14 vision encoder
+ Gemma 2B language model
```

作者以PaliVLA实现为基础，先在OXE机器人数据上预训练，再用FuSe数据微调。新增模态仍使用预训练TVL处理DIGIT、ResNet26处理音频，并投影成语言模型可以接收的token。

与Octo版本相比：

| 维度 | FuSe-Octo | FuSe-PaliGemma |
| --- | --- | --- |
| 核心backbone | 机器人原生Octo Transformer | 互联网预训练PaliGemma VLM |
| language encoder | frozen T5-base | Gemma token embedding / LLM |
| 多模态融合 | Octo token groups + joint attention | multimodal tokens装入VLM序列 |
| 动作输出 | MSE连续动作块 | 离散动作token生成 |
| 语言生成 | 额外轻量generation head | 利用VLM原生生成能力 |

所以FuSe的原则可以迁移，但两个版本并非逐层完全相同。论文希望证明的是：**语言对齐和传感器语义生成这套思想，既适用于机器人Transformer，也适用于生成式VLM backbone。**

---

## 11. 实验设置

### 11.1 机器人和传感器

- WidowX 250六自由度机械臂；
- 5 Hz delta end-effector position control；
- 第三人称相机与腕部相机；
- 夹爪左右各一个DIGIT；
- 麦克风与9-DoF IMU；
- Meta Quest 2遥操作采集；
- 视觉原始分辨率640×480，DIGIT原始分辨率320×240；
- 音频使用最近1秒、44.1 kHz采样的信号；
- 模型观察历史受算力限制约为0.4秒。

论文的数据集记录了proprioception和IMU，但Figure 2及主要FuSe实验着重展示的是vision、touch、audio和language。不能据此把论文夸大为完整利用所有传感器的统一foundation model。

### 11.2 三类真实机器人任务

1. **Tabletop Grasping**：在桌面多个物体中按文本选择并抓取目标；
2. **Shopping Bag**：物体位于纸袋中，第三人称视角严重遮挡，夹爪进入袋中后腕部光照也变差；
3. **Button Pressing**：六个按钮会播放不同声音，指令可以按颜色或声音指定按钮。

抓取任务使用24个训练物体，并测试32个未见物体。每种evaluation scenario执行5次rollout。抓取实验同时记录“到达/接触目标”和“完整抓取”，避免只看最终成功率掩盖感知是否选对目标。

### 11.3 Baselines

- **Octo FuSe**：预训练Octo + 全部传感器 + 两种语言辅助loss；
- **Octo Vision FT**：预训练Octo，但微调只使用已有视觉模态；
- **Octo Scratch**：相同Transformer和全部传感器，从零训练；
- **ResNet Scratch**：更小的ResNet26策略，从零训练，以FiLM注入语言。

这组对照分别检验：大规模机器人预训练是否重要、新增传感器是否重要、提升是否只是网络更大造成的。

---

## 12. 实验结果回答了什么

### 12.1 FuSe在三个任务上总体最好

FuSe在桌面抓取、袋中抓取和声音按钮任务上整体优于从零训练以及只用视觉微调的baseline。优势在Shopping Bag最明显，因为视觉在那里本来就不充分，新触觉模态确实提供了不可替代的信息。

Octo Scratch表现差，说明27K条轨迹不足以从零学好一个大Transformer；ResNet Scratch略强于Octo Scratch但仍明显落后，说明“有全部传感器”不等于“能从小数据中学会组合它们”。

### 12.2 多模态提示消除单模态歧义

论文专门构造一类场景：只看视觉或只摸触觉时，多个物体具有相同属性；只有组合提示才能唯一确定目标。

| 歧义类型 | 指令模态 | Tabletop Reach / Grasp | Bag Reach / Grasp | 平均Reach / Grasp |
| --- | --- | --- | --- | --- |
| 视觉歧义 | V | 0.43 / 0.43 | 0.30 / 0.25 | 0.37 / 0.34 |
| 视觉歧义 | V+T | 0.50 / 0.43 | 0.55 / 0.30 | 0.53 / 0.37 |
| 触觉歧义 | T | 0.40 / 0.40 | 0.35 / 0.30 | 0.38 / 0.35 |
| 触觉歧义 | V+T | 0.40 / 0.40 | 0.50 / 0.30 | 0.45 / 0.35 |

结果有一个值得冷静解读的细节：多模态提示主要提高了**Reach，即选对并接触目标**；完整Grasp提升较小。这说明论文更强地证明了多模态语义选择，而不是已经解决精细接触控制。

### 12.3 Compositional Cross-modal Prompting

作者展示两种组合能力。

简单任务：

```text
“抓取与会播放钢琴声的按钮颜色相同的物体”
```

策略必须把声音概念与按钮视觉颜色关联，再把颜色用于物体抓取。

多步任务：

```text
1. 根据视觉提示按下训练中未见的按钮
2. 听到按钮声音
3. generation head把声音转成语言描述
4. 在另一场景中按下会发出相同声音的训练按钮
```

这是一条由语言连接子任务的pipeline，不应描述成一个完全端到端、不中断的长时程rollout。它的重要性在于展示语言确实可以充当跨模态中间接口。

### 12.4 辅助loss消融

作者在Shopping Bag任务比较：

- 完整FuSe；
- 去掉generation loss；
- 去掉contrastive loss；
- 两者都去掉，只保留action imitation。

完整模型最好，去掉任一条辅助loss都会下降，两条都去掉最差，尤其影响未见物体。这直接支持论文的核心论点：只把sensor token接到policy再做BC，模型并不会自动学会使用它们。

### 12.5 PaliGemma VLA实验

FuSe-PaliGemma在三类任务上与Octo版本有竞争力，并在Shopping Bag等设置中表现较强。这项实验的意义主要不是宣称PaliGemma全面胜过Octo，而是验证FuSe训练思路能够跨越两种完全不同的策略架构。

### 12.6 触觉/多模态描述

策略还可以在交互后生成未见物体的属性描述。这里的“zero-shot”是指新物体、新的语言组合或跨模态组合，不是指模型完全没有经过FuSe微调，也不是换机器人或换一种触觉硬件就能直接使用。

---

## 13. 这篇工作与TVL、Octopi、AnyTouch的关系

| 工作 | 核心输入输出 | 主要解决的问题 | 是否直接输出动作 |
| --- | --- | --- | --- |
| TVL | touch / vision / text → shared embedding与描述 | 触觉—视觉—语言对齐 | 否 |
| Octopi | tactile images + prompt → 属性描述/PSR回答 | 大触觉语言推理 | 否 |
| AnyTouch | 多传感器静态/动态touch → unified embedding | 触觉encoder跨设备泛化 | 否 |
| FuSe | vision + touch + audio + instruction → action / description | 把异构传感器接入通用机器人策略 | **是** |

FuSe直接复用了TVL encoder，说明TVL这类表征模型的价值不仅是检索和caption，还能作为VLA的新传感器前端。

但FuSe并没有把Octopi整体接进policy，也没有调用一个大触觉语言模型先生成描述、再让VLA动作。它选择了更紧耦合的路线：

```text
tactile token ─┐
visual token ──┼─> policy Transformer ─> action
audio token ───┘             │
                             └─> language-grounding auxiliary losses
```

即：语言是训练时的语义桥梁，触觉在执行时仍可直接影响action readout。

---

## 14. 这篇论文的真正作用

我认为Beyond Sight最重要的不是提出了某个更复杂的触觉encoder，而是给出了一个非常清楚的“触觉进入VLA”的工程与学习范式：

1. 用已有预训练触觉encoder降低小数据学习难度；
2. 把新模态变成policy可以消费的token；
3. 在共享Transformer中做动作相关的多模态融合；
4. 除动作监督外，显式要求策略从新模态恢复语言语义；
5. 用遮挡、歧义和组合任务验证模型不是只看RGB。

它把此前“触觉—语言表征对齐”的研究向前推进了一层：

```text
TVL：touch知道“soft”是什么意思
FuSe：policy知道“soft”与当前指令有关，并据此选择和执行动作
```

这也是它在触觉VLA路线中的历史位置：**从感知对齐走向语言条件下的动作决策。**

---

## 15. 局限与阅读时应保持的边界

### 15.1 触觉参与了目标选择，但还不是精细力控

动作控制仍是5 Hz的delta end-effector position，任务以抓取和按按钮为主。论文没有重点研究：

- 接触力闭环；
- slip detection后的实时抓力调整；
- 插拔、装配等高精度接触任务；
- 灵巧手多指协同。

Reach提升大于Grasp提升也印证了它更偏“语义触觉VLA”，不是完整的低层触觉控制方案。

### 15.2 heterogeneous主要指模态异构，不是触觉硬件泛化

论文使用两个DIGIT实例，但没有像AnyTouch那样验证DIGIT、GelSight、Tac3D等不同触觉硬件之间的统一迁移。因此这里的heterogeneous sensors主要是vision、touch、audio等传感器类型不同。

### 15.3 触觉需要接触后才能获得

在抓取任务中，机器人往往需要先reach/touch候选物体，再利用触觉确认。这种interactive perception很合理，但也意味着策略不是在接触前凭触觉远程锁定目标。

### 15.4 语言监督受模板与属性词表限制

ChatGPT增加了语言表达的多样性，却没有增加新的物理ground truth。模型理解的属性范围仍受数据采集和模板定义约束。

### 15.5 训练成本和短历史窗口

作者使用v5e-128 TPU、batch size 1024训练50K步；额外模态增加了算力开销，使观测历史限制在约0.4秒。这对稀疏触觉事件和较长音频模式都可能不够。

---

## 16. 回答本轮几个核心问题

### Q1：Octo里面的VLM是什么？

Octo主干里没有一个独立完整VLM。T5-base只是冻结文本encoder，Octo Transformer负责融合视觉、语言和其他观测。PaliGemma分支才有明确的PaliGemma VLM backbone。

### Q2：FuSe是不是把touch、audio和language embedding拼起来进VLM？

在Octo分支中，更准确地说是：各encoder输出token并投影到公共宽度，作为不同token组输入Octo Transformer，用joint attention融合。不是把每种模态平均成一个embedding后送进某个VLM。

### Q3：Octo/FuSe用flow matching吗？

不用。原始Octo通常用DDPM式diffusion action head；FuSe发布的Octo配置改用直接MSE动作回归；PaliGemma分支生成离散动作token。三者都不是π0.5/GR00T式flow matching。

### Q4：readout和Action Head分别是什么？

readout是进入Transformer、主动读取任务与观测信息的可学习查询token；Action Head是把action readout hidden state转换为实际动作的解码模块。在FuSe-Octo中，它是MAP pooling加Dense连续动作回归。

### Q5：Contrastive Loss里的language readout与T5是什么关系？

language readout从传感器观测一侧提取语义；T5从正确文本一侧产生语义锚点。二者经pooling/projector进入同一latent，配对样本靠近，不配对样本远离。

### Q6：Contrastive和Generation的正确文本是谁标的？

它们来自数据集中事后准备的属性模板与离线ChatGPT改写。训练时已经存在；ChatGPT不参与部署时推理，T5也不是标签生成器。

### Q7：这些辅助head推理时还需要吗？

普通动作执行只需传感器encoder、Octo Transformer、action readout与Action Head。Contrastive Head是训练辅助，可以不运行；Generation Head只有在需要输出物体描述或执行论文中的语言中介组合流程时才使用。

---

## 17. 原文定位索引

| 想核对的内容 | 原文位置 |
| --- | --- |
| 问题动机、语言作为公共模态、贡献 | Abstract与Section I，PDF第1–2页 |
| FuSe总体架构 | Figure 2与Section III，PDF第3页 |
| TVL tactile encoder、audio ResNet26 | Section III “Tactile encoder / Audio encoder”，PDF第3页 |
| contrastive与generation loss | Section III “Auxiliary losses”，PDF第3–4页 |
| 文本模板与ChatGPT改写 | Section III “Language Rephrasing”，PDF第4页 |
| 总loss、训练超参数 | Section III “Implementation Details”，PDF第4页 |
| 机器人、传感器与26,866条轨迹 | Section IV-A，PDF第4页 |
| 三类任务 | Section IV-B，PDF第4–5页 |
| 总体性能 | Section IV-C与Figure 5，PDF第5页 |
| 单模态歧义与多模态提示 | Section IV-D与Table I，PDF第5页 |
| 组合任务 | Section IV-E与Figure 6，PDF第5–6页 |
| 辅助loss消融 | Section IV-F与Figure 7，PDF第6页 |
| PaliGemma VLA迁移 | Section IV-G与Figure 8，PDF第6–7页 |
| 0.4秒历史窗口局限 | Section V，PDF第7页 |

特别说明：**FuSe-Octo使用`MSEActionHead`、4步×7维动作块、24个language readout等实现细节，需要结合开源仓库中的`octo_digit/scripts/configs/fuse_config.py`核对；正文只明确写出MSE imitation loss，并没有逐项展开全部配置。**

---

## 18. 最终心智模型

读完这篇论文，最值得保留的不是“FuSe又加了两个loss”，而是下面这条完整因果链：

```text
预训练Octo已经会“看图 + 听指令 + 做动作”
                    ↓
TVL与ResNet把touch/audio变成可输入的token
                    ↓
共享Transformer让这些token有机会影响action readout
                    ↓
仅用动作MSE时，模型仍可能无视新模态
                    ↓
contrastive loss要求观测表示对应正确语言语义
generation loss要求观测能明确说出属性
                    ↓
新传感器不再只是“接进来了”，而是被赋予可查询、可组合的含义
                    ↓
在遮挡或单模态歧义场景中，touch/audio真正影响动作选择
```

用一句话收束：

> FuSe不是把一个大触觉语言模型外挂到VLA上，而是用语言监督把触觉和声音训练成通用策略内部真正可用的决策信息。
