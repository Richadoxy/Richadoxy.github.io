   # 多模态 VLA 与模仿学习：一套统一的研究范式

> 这一章不是从某个模型名字开始，而是先建立研究地图：一项多模态机器人工作通常怎样组织数据、训练表示、融合模态、生成动作并完成闭环验证。后续所有基础概念与论文精读都放回这张地图中理解。

> 后续阅读顺序：[从原始模态到公共语义空间](31_multimodal_encoder_alignment_and_infonce.md) → [多模态表示的监督配方](32_three_consistency_supervision_paradigms.md) → [从对齐表示到 Policy 融合](19_attention_from_embeddings_to_action_expert.md) → [模仿学习与 Behavior Cloning](23_imitation_learning_basics.md)。

## 0. 先记住整条主线

多模态 VLA 或模仿学习研究通常可以拆成六层：

```text
1. Data
   vision / language / touch / audio / state / action
                         ↓
2. Representation
   tokenizer / encoder / projector → tokens or embeddings
                         ↓
3. Alignment（可选）
   让不同模态进入可比较的语义或物理表示空间
                         ↓
4. Task-conditioned Fusion
   concat / self-attention / cross-attention / gate / conditioning
                         ↓
5. Action Generation
   MLP / autoregressive tokens / diffusion / flow matching
                         ↓
6. Closed-loop Execution
   action chunk / controller / replanning / sensor feedback
```

一篇论文通常不会同时发明六层。它的 contribution 往往集中在其中一两层：

- 新数据集或采集装置；
- 新的触觉/音频 encoder；
- 新的跨模态预训练目标；
- 新的 fusion 或 routing 结构；
- 新的 action decoder；
- 新的控制器或在线执行框架。

阅读时首先要定位它改了哪一层，避免把“换了 encoder”“增加一个 loss”和“提出新 VLA 架构”混为一谈。

---

## 1. 研究对象：从专家轨迹学习条件策略

模仿学习数据通常是一组轨迹：

\[
\tau={(o_t,a_t)}_{t=1}^{T}
\]

多模态 observation 可以写成：

\[
o_t=(I_t,L,T_t,U_t,q_t,c)
\]

其中：

| 符号 | 含义 |
| --- | --- |
| \(I_t\) | 外部、腕部或第一视角 RGB 图像 |
| \(L\) | 自然语言指令或任务描述 |
| \(T_t\) | 触觉 RGB、深度、marker flow、力阵列或力序列 |
| \(U_t\) | 音频等其他传感器信号 |
| \(q_t\) | 关节、末端位姿、夹爪状态等 proprioception |
| \(c\) | 离散任务 ID、机器人 embodiment ID 等条件 |
| \(a_t\) | 专家动作 |

策略学习的是：

\[
\pi_\theta(a_{t:t+K-1}\mid o_{\le t})
\]

也就是根据当前或历史观测，生成未来 \(K\) 步 action chunk。

“VLA”通常表示视觉和语言参与动作生成；加入触觉、音频等模态后，更准确的重点是：

> 不同信息如何形成表示、怎样在当前任务下融合，以及最终怎样改变动作。

---

## 2. 数据层：监督能力的上限首先由数据决定

### 2.1 一条数据不只有 observation 和 action

多模态训练还需要明确：

- 哪些模态在同一时刻同步；
- 是否包含完整时间历史；
- 自然语言是人工写、模板生成还是 LLM 扩写；
- 触觉与视觉是否来自同一对象、同一位置和同一次接触；
- action 是关节位置、末端增量、速度、力还是混合目标；
- 数据是人类手持示范、遥操作、机器人 rollout 还是仿真；
- 哪些信息只在训练时存在。

这些关系不仅是元数据，还直接构造监督信号：

```text
同一时刻的vision-touch      → cross-modal positive pair
同一轨迹的相邻状态          → temporal positive pair
被遮挡帧的真实图像          → reconstruction target
专家动作                    → imitation target
失败原因的文字              → language-generation target
```

### 2.2 多模态数据并不总是完整

现实数据可能只有：

```text
Touch + Vision
Touch + Text
Vision + Language + Action
Vision + Touch + Action
```

因此很多研究的重要问题不是“怎样把所有模态同时输入”，而是：

- 缺失模态样本是否仍能用于预训练；
- 不同数据源能否共享一个 encoder；
- 训练时存在、推理时缺失的模态如何处理；
- 不同采样率怎样时间对齐。

---

## 3. 表示层：先把原始信号变成模型可处理的 token

不同模态需要不同入口：

```text
RGB image      → ViT/CNN                  → visual tokens
language       → tokenizer + embedding    → language tokens
tactile RGB    → tactile ViT/CNN          → tactile tokens
force history  → MLP/1D CNN/Transformer   → force tokens
robot state    → linear/MLP projector      → state tokens
```

这里需要区分：

| 名称 | 作用 |
| --- | --- |
| Tokenizer | 把原始对象切成离散或连续的输入单元 |
| Encoder | 根据上下文把输入单元变成表示 |
| Projector | 对齐维度或接口空间 |
| Patch token | 表示图像/触觉图像的局部区域 |
| CLS/readout token | 汇总一张图或一组 token 的信息 |
| Global embedding | 用于相似度、检索或对比学习的全局向量 |

并不是所有 encoder 输出都必须先压成一个向量：

- CLIP 式对比通常比较 global embeddings；
- 生成式 VLM/VLA 常保留多枚 patch/sensor tokens；
- policy 可以让 action tokens 读取完整的感知 token memory。

更详细的接口关系见[从原始模态到公共语义空间](31_multimodal_encoder_alignment_and_infonce.md)。

---

## 4. 对齐层：建立公共坐标，但它不是所有策略的必要条件

### 4.1 什么是对齐

给定配对数据：

```text
Touch    → ET → zT
Vision   → EV → zV
Language → EL → zL
```

语义对齐希望：

\[
z_T\approx z_V\approx z_L
\]

这里不是逐维相等，而是让正确配对在 latent space 中相互靠近，错误配对远离，从而支持：

- 跨模态检索；
- 开放词汇分类；
- zero-shot 属性理解；
- 将新传感器接入已有视觉—语言知识。

### 4.2 锚点是什么

如果冻结 CLIP 的视觉和文本 encoder，只训练 tactile encoder，那么已有视觉—语言空间就是锚点：

```text
frozen vision-language space
              ↑
    trainable tactile encoder
```

锚点也可以是：

- 冻结文本 encoder；
- 可轻微 LoRA 调整的视觉 encoder；
- 跨传感器对应关系；
- 属性标签；
- 机器人 action supervision。

不同锚点会保留不同信息。语言锚点擅长物体属性语义，时间预测擅长接触变化，动作监督擅长任务相关信息。

### 4.3 对齐是可选的

两种路线都可能成功：

```text
路线A：先预训练对齐
touch ↔ vision ↔ language
          ↓
再接policy并fine-tune

路线B：不做独立对齐
raw sensor encoder + policy
          ↓
直接用action loss端到端学习
```

路线 A 通常有利于数据效率、开放语义和复用；路线 B 更直接地优化控制，但表示未必适合检索或跨任务迁移。

---

## 5. 表示监督层：定义什么相同、什么不同、什么必须保留

表示学习的常见目标可以按问题分类。

| 目标 | 正样本/监督来源 | 希望保留 |
| --- | --- | --- |
| Cross-modal InfoNCE | 同一对象/时刻的不同模态 | 跨模态共同语义 |
| Temporal contrast | 同一轨迹的相邻状态 | 时间连续性 |
| Augmentation consistency | 同一输入的不同增强/遮挡 | 对无关扰动的稳定性 |
| Masked reconstruction | 被遮挡内容的真实值 | 局部结构与细节 |
| Future prediction | 未来帧、深度或触觉变化 | 动态演化 |
| Classification/regression | 属性标签或物理数值 | 指定物理属性 |
| Language generation | 正确描述或推理文字 | 可表达的语义 |
| Action loss | 专家动作 | 与控制任务相关的信息 |

一个常见的分阶段配方是：

```text
Stage A：模态内部结构
reconstruction / future prediction
                 ↓
Stage B：跨模态语义
InfoNCE / matching / consistency
                 ↓
Stage C：下游任务
language generation / action loss
```

这些 loss 不一定需要人工标签。同步关系、相邻时间和不同增强本身就能构造自监督信号。详见[多模态表示的监督配方](32_three_consistency_supervision_paradigms.md)。

---

## 6. Fusion 层：对齐之后，信息还要真正进入 Policy

对齐只说明不同模态具有可比较表示，不意味着它们已经联合决定动作。Policy 还需要一种信息融合机制。

### 6.1 Token concatenation + joint self-attention

```text
[language tokens, vision tokens, tactile tokens, state tokens]
                          ↓
                    Transformer
```

同一个 attention 序列中的 token 可以按 mask 规则互相读取。

### 6.2 Cross-attention

```text
Action tokens = Q
Perception memory = K,V
         ↓
Cross-Attention
```

动作表示根据当前需要，从感知 memory 中检索信息。

### 6.3 Pooled condition 与调制

```text
sensor history → pooled vector
                       ↓
           FiLM / AdaLN / AdaRMS
                       ↓
             调节action network
```

条件向量不一定成为独立 token，也可以缩放或偏移网络中间激活。

### 6.4 Gate 与 residual injection

```text
X' = X + g · CrossAttn(X,M,M)
```

可学习 gate 控制新增模态影响强度，常用小值或零初始化，以免预训练策略在微调初期被随机新分支破坏。

### 6.5 Readout 与 Action Head

```text
multimodal Transformer
          ↓
action readout hidden states
          ↓
Action Head / Decoder
          ↓
robot action chunk
```

Readout token 负责从 Transformer 中汇总动作所需的信息；Action Head 才负责把 hidden states 转换为动作参数、离散 token、噪声或 flow velocity。

这些结构的完整关系见[从对齐表示到 Policy 融合](19_attention_from_embeddings_to_action_expert.md)。

---

## 7. 动作层：网络架构与训练目标不要混为一谈

相同的感知与 fusion backbone 可以连接不同动作生成方法：

| 动作路线 | 训练目标 | 推理方式 |
| --- | --- | --- |
| Direct regression | MSE/L1 对齐专家动作 | 一次前向输出 |
| Discrete action token | next-token cross-entropy | 自回归解码 |
| Diffusion | 预测噪声/score | 多步去噪 |
| Flow matching | 预测速度场 | 数值积分得到动作 |
| Residual correction | 预测基础动作修正量 | 基础策略后置修正 |

因此：

- Transformer 不是一种 action loss；
- Action Expert 不一定使用 flow matching；
- Action Head 不一定只是一个 MLP；
- 使用 VLM 不代表动作一定由语言 token 自回归生成。

动作生成方法的细节放在第二卷和第三卷展开。

---

## 8. 训练图和推理图必须分开画

### 8.1 训练时可能存在的模块

```text
pretrained encoders
+ reconstruction decoder
+ contrastive head
+ privileged posterior
+ language generation head
+ action policy
```

训练中还需要 ground truth：

- 未遮挡图像；
- 未来触觉；
- 专家 action chunk；
- 属性或推理文本；
- success/failure 标签。

### 8.2 推理时通常只保留

```text
deployable encoders
+ necessary projector/fusion modules
+ policy/action decoder
+ low-level controller
```

重建 decoder、matching head、privileged posterior 经常只是训练脚手架。阅读论文时需要逐项问：

- 哪些网络只制造 loss？
- 哪些参数被冻结？
- 哪些模块部署时真正运行？
- 训练时使用的信息在推理时是否可获得？

---

## 9. 三种常见训练组织方式

### 9.1 独立表示预训练，再冻结接 Policy

```text
multimodal data
      ↓ representation losses
Encoder
      ↓ freeze
task-specific Policy
      ↓ action loss
```

优点：表示可复用、任务数据需求低、训练稳定。缺点：冻结表示不一定完全适合特定动作。

### 9.2 预训练初始化，再端到端 Fine-tune

```text
aligned/pretrained Encoder
          ↓ initialize
VLM/Policy/Action Expert
          ↓ action loss updates all or part
```

优点：语义先验能适应控制。缺点：小数据可能破坏原有对齐，即 representation drift。

### 9.3 直接端到端学习

```text
raw multimodal observation
          ↓
Encoder + Fusion + Policy
          ↓ imitation loss
```

优点：目标直接、结构简单。缺点：模型可能忽略新增模态，数据效率与跨任务泛化也可能较弱。

为了防止“接了触觉但策略不用”，研究中常加入：

- modality dropout；
- language grounding auxiliary loss；
- future/reconstruction loss；
- bottleneck 或 routing；
- matched no-sensor ablation。

---

## 10. 多模态 VLA 研究通常在研究什么

可以把论文 contribution 放入下面七个问题。

### 10.1 数据与同步

- 如何采集真正依赖触觉/音频的专家示范？
- 视觉 20 Hz、触觉 100 Hz、控制 50 Hz 怎样对齐？
- 如何生成可靠的语言属性或失败分析？

### 10.2 通用表示

- 一个 encoder 能否跨对象、任务和传感器工作？
- 静态触觉与动态触觉能否统一？
- 触觉应对齐语言、视觉还是动作？

### 10.3 缺失与异构模态

- 不是每条数据都有全部模态，如何训练？
- 训练与部署使用不同传感器怎么办？
- 视觉被遮挡后，触觉能否补足状态？

### 10.4 Policy Fusion

- 新模态在 input、middle layer 还是 output 端注入？
- 使用 joint attention、cross-attention、gate 还是调制？
- 高层语义触觉和低层连续触觉是否需要不同路径？

### 10.5 动作与控制

- 预测位置、速度、力还是混合 action？
- action chunk 怎样生成与在线调度？
- 是否还需要阻抗、位置—力或 residual controller？

### 10.6 泛化与适应

- 是否泛化到未见词、物体、任务、传感器和环境？
- 失败后能否使用触觉重规划？
- 表示预训练是否真的改善闭环成功率？

### 10.7 系统可部署性

- 传感器频率、推理延迟和 action horizon 是否匹配？
- 模型能否在接触快速变化时及时反应？
- slow reasoning 与 fast control 如何协同？

---

## 11. 实验应该与论文声称的层级匹配

| 论文声称 | 至少需要的证据 |
| --- | --- |
| 表示对齐更好 | retrieval、zero-shot classification、linear probe |
| 表示保留局部/动态信息 | reconstruction、future prediction、contact-state probe |
| 新模态真正进入 policy | sensor removal/dropout、matched architecture ablation |
| 动作预测更好 | held-out action error 仍不够，需闭环 rollout |
| 真实机器人更稳健 | 多次 trial、清楚的成功标准和失败类型 |
| 具有 zero-shot 泛化 | train/test 对象、词语、任务或环境严格隔离 |
| VLM 先验被“解锁” | 冻结/随机化/替换 VLM 等因果消融 |

Top-1、重建 PSNR 或训练 action loss 只证明某一层能力，不能直接替代真机闭环结论。

---

## 12. 阅读一篇新论文的统一检查法

### 第一步：定义输入输出

- 输入有哪些模态？
- 每种模态是单帧还是时间窗口？
- 输出是 embedding、文本、action 还是控制目标？

### 第二步：定位表示

- 原始信号怎样 tokenization？
- encoder 来自哪里，是否预训练？
- 输出 global embedding 还是 token sequence？

### 第三步：定位对齐

- 是否有独立 alignment stage？
- 谁是锚点？正负样本怎样定义？
- 使用 InfoNCE、重建、分类还是时间预测？

### 第四步：定位融合

- concat、joint attention、cross-attention 还是 modulation？
- 谁是 Query，谁提供 Key/Value？
- 有没有 gate、readout 或 multi-expert？

### 第五步：定位动作

- Action Head 是回归、token、diffusion 还是 flow？
- action space、horizon 和控制频率是什么？
- 是否有独立低层 controller？

### 第六步：分开训练与推理

- 每个 stage 的数据和 loss 是什么？
- 哪些参数更新、冻结或使用 LoRA？
- 哪些训练辅助模块推理时丢弃？

### 第七步：检查实验闭环

- Baseline 是否参数量和传感器接口匹配？
- Ablation 能否定位每个组件的贡献？
- 表示指标是否转化为 policy rollout 提升？

---

## 13. 将已读触觉工作放回统一地图

| 工作 | 主要贡献所在层级 |
| --- | --- |
| TVL | 数据 + 触觉—视觉—语言对齐 + 语言生成 |
| Octopi | 触觉属性 encoder + LLM 物理属性推理 |
| AnyTouch | 跨传感器、静态—动态统一表示 |
| Beyond Sight / FuSe | 策略内部的异构传感器融合与语言辅助监督 |
| VLA-Touch | 外挂式高层规划 + 低层触觉动作修正 |
| OmniVTLA | 语义对齐触觉 encoder 接入 π0 |
| Tactile-VLA | 触觉 token + 力动作空间 + 混合控制 + CoT 重规划 |
| VT-MUSE | 独立时序视触觉 memory + gated cross-attention policy |

这张表说明：“触觉进入机器人智能”不是一条单一路线，而至少包含：

```text
触觉如何表示
触觉如何获得语义
触觉如何与其他模态交互
触觉如何改变动作
触觉如何进入真实控制闭环
```

---

## 14. 最终心智模型

以后看到任何多模态 VLA 或模仿学习系统，先画下面这张图：

```text
Raw data
vision / language / touch / state / action
              ↓
Modality interfaces
tokenizer / encoder / projector
              ↓
Representation objectives
alignment / reconstruction / temporal / consistency
              ↓
Task-conditioned fusion
self-attn / cross-attn / gate / modulation
              ↓
Action generation
regression / token / diffusion / flow
              ↓
Execution system
chunk scheduler / controller / feedback / replanning
```

然后问两个贯穿始终的问题：

1. **训练信号到底迫使模型学习了什么？**
2. **该信息在推理时通过哪条路径真正影响动作？**

只要这两个问题能回答清楚，一篇复杂多模态论文的大部分结构就不会再混乱。
