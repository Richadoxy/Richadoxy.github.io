# 《Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization》精读笔记

> Jialei Huang, Shuo Wang, Fanqi Lin, Yihang Hu, Chuan Wen, Yang Gao<br>
> arXiv 2025；本文依据 arXiv v1 整理，目前论文正文未注明正式会议或期刊<br>
> [arXiv](https://arxiv.org/abs/2507.09160) · [论文 PDF](https://arxiv.org/pdf/2507.09160) · [项目主页](https://jialeihuang.github.io/tactileVLA.github.io/)

> 推荐先读：[Flow Matching](10_flow_matching_training_and_inference.md)、[VLA 架构地图](27_vla_overview.md)、[VLA-Touch](28_vla_touch_paper_reading_notes.md)和[OmniVTLA](29_omnivtla_paper_reading_notes.md)。本文代表的是“触觉 token 直接进入 π0，并把力纳入 action space”的路线。

## 0. 一句话定位

Tactile-VLA 以 π0 为主体，在 VLM 输入前缀中增加一个由 **MLP 编码的触觉历史 token**，并把 Action Expert 的输出从纯运动动作扩展成“目标位姿 + 目标力”。这些目标经过混合位置—力控制器转换成可执行的位置命令；Tactile-VLA-CoT 则进一步让 VLM 周期性分析任务是否失败、生成纠正语言，再指导下一轮动作。

```text
历史 RGB ── ViT ─────────────────────────────┐
                                            │
语言指令 ── tokenizer / embedding ──────────┤
                                            ├─> 非因果 Transformer 前缀融合
历史法向力与剪切力 ── MLP ── 1 tactile token ┘
                                                       │
机器人状态 + noisy action + flow timestep ────────────┤
                                                       ↓
                                         Tactile-Aware Action Expert
                                                       ↓
                                  [target position, target contact force]
                                                       ↓
                                       Hybrid Position-Force Controller
                                                       ↓
                                                robot command
```

它与 OmniVTLA 的关键区别是：

- OmniVTLA 的重点是**先预训练语义对齐的触觉 ViT 表征**；
- Tactile-VLA 没有额外的触觉—视觉—语言预训练，触觉 encoder 只是简单 MLP；
- Tactile-VLA 的核心重点是**让语言、触觉与力动作共同进入一个端到端控制闭环**。

---

## 1. 论文想解决什么问题

普通 VLA 擅长回答“做什么”：

```text
pick up the apple
insert the USB
wipe the board
```

但接触丰富的操作还需要决定“怎样做”：

```text
轻轻抓，还是用力抓？
插接件需要施加多少力？
擦拭失败后是否应该增大下压力？
```

视觉和语言模型可能知道“水果容易损坏”“铁球又重又硬”“hard 比 softly 力度大”，但普通 VLA 没有把这种抽象知识与真实触觉信号和力控制接口连接起来。

作者的核心假设是：

> VLM 已经从大规模视觉—语言预训练中获得了一部分物理常识。只要用少量力引导示范，把语言、触觉和力动作接入同一策略，就可以激活这些先验，并泛化到未见指令、物体和接触情境。

论文围绕三个问题展开：

1. 能否把 softly、hard 等语言修饰词映射成不同的物理力？
2. 能否利用 VLM 常识，为没见过的物体选择合适抓力？
3. 能否根据触觉反馈诊断失败，并通过推理改变下一次动作？

---

## 2. 主要贡献，以及真正的创新边界

### 2.1 触觉作为原生 VLA 输入

Tactile-VLA 不像 VLA-Touch 那样把触觉放在基础 VLA 外部。它将触觉历史编码成 token，并与视觉和语言 token 一起送入 Transformer 前缀，让 Action Expert 可以在生成动作时直接读取触觉。

因此它属于：

```text
end-to-end tactile-conditioned VLA
```

而不是：

```text
现成 VLA + 外挂 planner / residual controller
```

### 2.2 力被显式加入 action space

Action Expert 不仅输出运动目标，还输出目标接触力：

\[
a_t=[P_{\text{target}},F_{\text{target}}]
\]

这点很重要。触觉不再只是帮助模型“感知”，而是直接影响策略要预测的控制目标。

### 2.3 混合位置—力控制器

神经网络预测的目标力不能直接发送给以位置控制为主的机器人。作者设计了一个低层混合控制器，根据目标力与实测力的误差修正目标位置，把高层的“力意图”变成实际机器人命令。

### 2.4 Tactile-VLA-CoT

基础 Tactile-VLA 是连续感知—动作策略；CoT 版本增加一个较慢的显式推理循环：

```text
检查结果 → 判断成功/失败 → 分析力和滑移 → 生成纠正指令 → 再执行
```

它主要用于处理执行失败和新环境，而不是替代高频 Action Expert。

### 2.5 三类泛化实证

作者分别展示：

- 力相关语言从一个任务迁移到另一个任务；
- 物体物理常识从 seen objects 迁移到 unseen objects；
- 擦白板学到的触觉推理迁移到未见过的黑板。

### 2.6 没有重新发明的部分

以下主体来自 π0：

- 预训练视觉—语言骨干；
- Gemma 2.6B；
- 约 300M 参数的 Action Expert；
- action chunk 与 Conditional Flow Matching；
- VLM 前缀条件化 Action Expert 的基本范式。

因此更准确的评价是：

> Tactile-VLA 是在 π0 上完成触觉输入、力动作和低层力控制的系统性扩展，而不是从头提出一种新的 VLA 主干。

---

## 3. Figure 2：完整架构拆解

### 3.1 视觉路径

论文使用与 π0 类似的预训练 ViT，对最近 \(H\) 帧图像分别编码：

\[
E'_{\mathrm{vis}}(I_{t-H+1}),\ldots,E'_{\mathrm{vis}}(I_t)
\]

每一帧会形成一组视觉 token。历史帧保留了动作和场景随时间的变化。

### 3.2 语言路径

当前语言指令 \(L_t\) 经过语言 tokenizer 与 embedding：

\[
E_{\mathrm{lang}}(L_t)
\]

例如：

```text
“Pull out and insert the USB softly.”
```

语言路径继承预训练 VLM，因而已经具有一定的物体语义、形容词和副词关系知识。

### 3.3 触觉路径

触觉输入不是 GelSight RGB、深度图或 marker 位移图，而是双触觉传感器测得的数值力信号，包括：

- normal force：垂直于接触面的力；
- shear force：沿接触面方向的剪切力。

最近 \(H\) 个时刻的触觉测量先在时间维拼接，再由简单 MLP 压缩成一个 token：

\[
E'_{\psi}([T_{t-H+1},\ldots,T_t])
\]

这个 token 表示一小段接触历史，而不是单个瞬时力。

这里没有：

- CLIP 式触觉—文本预训练；
- AnyTouch/OmniVTLA 式语义对齐触觉 ViT；
- 单独设计的触觉 Transformer；
- 图像式视触觉传感器 encoder。

### 3.4 Token-level fusion

三种模态在 token 序列维拼接成统一前缀：

\[
S_t=[E'_{\mathrm{vis}}(I_{t-H+1}),\ldots,E'_{\mathrm{vis}}(I_t),
E_{\mathrm{lang}}(L_t),E'_{\psi}([T_{t-H+1},\ldots,T_t])]
\]

该前缀使用 non-causal attention。也就是说，前缀中的视觉、语言和触觉 token 可以双向读取彼此，而不是像文本自回归生成那样只能看左侧 token。

因此，它不是只在 loss 层面对齐三个独立 embedding，而是在策略前向传播中真正做 task-conditioned fusion：

```text
语言 token 可读取视觉/触觉上下文
视觉 token 可读取语言/触觉上下文
触觉 token 可读取视觉/语言上下文
```

### 3.5 Tactile-Aware Action Expert 到底“aware”在哪里

它并没有再加一个名为 Tactile-Aware Attention 的特殊算子。“触觉感知”主要来自两点：

1. Action Expert 可以读取包含 tactile token 的多模态前缀；
2. 它预测的 action 多了 force target 维度。

所以可以把它理解成：

```text
π0 Action Expert
  + tactile-conditioned context
  + force-augmented action space
```

而不是一个完全独立的新网络家族。

---

## 4. Tactile-VLA 与 Tactile-VLA-CoT 的区别

| 方面 | Tactile-VLA | Tactile-VLA-CoT |
| --- | --- | --- |
| 核心目标 | 根据多模态输入直接生成位姿与力动作 | 执行失败后分析原因并调整策略 |
| 触觉作用 | 连续控制条件 | 连续控制条件 + 显式推理证据 |
| 输出 | action chunk | 推理文本、纠正指令以及后续 action |
| 时间尺度 | 高频策略执行 | 固定间隔触发的较慢重规划 |
| 额外数据 | 常规成功示范 | 带失败原因与纠正思路的针对性数据 |

### 4.1 CoT 推理循环

论文描述的流程是：

```text
1. 执行一段动作
2. 固定间隔检查任务是否完成
3. 如果失败，结合视觉和触觉解释原因
4. 区分不同力分量的问题
5. 生成新的纠正指令
6. 用新指令重新条件化策略，开始下一次尝试
```

例如：

```text
现象：板上仍有痕迹
判断：抓持力足够，但法向下压力不足
纠正：再次擦拭，并施加更大的向下力
```

### 4.2 CoT 不是高频控制器

CoT 不会在每个控制周期输出关节动作。它负责低频诊断与重规划；高频执行仍由 Action Expert 和混合控制器完成。

### 4.3 论文没有讲清楚的部分

论文称其使用 VLM 自身的 pretrained decoder 生成 internal monologue，但没有充分披露：

- 推理文本的 token-level loss 公式；
- 动作 CFM loss 与语言生成 loss 如何加权；
- 成功/失败判断究竟依赖视觉、触觉还是外部规则；
- 固定触发间隔的具体数值；
- CoT 数据中的文字由谁、以什么模板标注。

因此可以确认“存在语言监督与重规划”，但不能从正文还原完整的联合训练实现。

---

## 5. 训练链路

### 5.1 参数初始化

```text
预训练 π0 参数
├─ 视觉 encoder / VLM：预训练初始化
├─ Gemma 2.6B：预训练初始化
└─ π0 共享策略组件：预训练初始化

新增组件
├─ tactile MLP encoder：随机初始化
└─ modified Action Expert：随机初始化或因输出维度变化重新初始化
```

论文明确称整个模型进行 end-to-end fine-tuning，因此不是只训练 tactile projector 或 LoRA。按正文表述，预训练组件与新增组件都会被更新。

### 5.2 一条普通示范提供什么监督

一条同步示范包含：

- 历史 RGB 图像；
- 自然语言指令；
- 历史法向力与剪切力；
- 动作轨迹；
- 对应的目标位置；
- 对应的目标接触力。

前四类是策略条件，专家位姿和专家力轨迹是模仿学习标签。

### 5.3 Conditional Flow Matching

Tactile-VLA 没有直接用 MSE 一次性回归最终 action chunk，而是沿用 π0 的 flow matching。

设增强后的专家动作序列为：

\[
A^1=[P^1,F^1]
\]

从噪声动作 \(A^0\) 与专家动作之间构造中间状态：

\[
A^\tau=(1-\tau)A^0+\tau A^1
\]

Action Expert 根据多模态前缀、机器人状态、中间动作和时间 \(\tau\)，预测把噪声推向专家动作的速度场。概念上：

\[
\mathcal L_{\mathrm{CFM}}
=
\mathbb E\left[
\left\|v_\theta(A^\tau,\tau,S_t)-u(A^0,A^1)\right\|^2
\right]
\]

这一个 flow matching 目标同时覆盖：

- position/kinematic dimensions；
- force dimensions。

论文并没有设计“一个位置 loss + 一个触觉 loss + 一个力 loss”再手工求和的多任务结构。基础版本的核心监督仍是统一的 action-space CFM loss，只是动作向量增加了力维度。

### 5.4 梯度如何迫使模型使用触觉

如果 tactile token 没有帮助正确预测力动作，force dimensions 上的 flow matching 误差就会增大。梯度会从 Action Expert 反向传播到：

```text
force action output
        ↑
Action Expert
        ↑
multimodal Transformer prefix
        ↑
tactile MLP encoder
```

语言与视觉也参与同一预测，所以模型有机会学习：

```text
“softly” + USB + 当前接触力 → 较小目标力
“hard”   + USB + 当前接触力 → 较大目标力
```

这不是先把三个模态独立对齐到统一 CLIP 空间，再接 action head；它是直接由动作监督形成 task-oriented representation。

### 5.5 CoT 版本的额外监督

CoT 小数据集中，每个失败样本还包含：

- 失败事件；
- 同期多模态观测；
- 对失败原因的语言分析；
- 应如何调整的纠正文本。

语言标注训练 VLM decoder 将触觉现象映射成原因与下一步指令。但论文没有给出精确的联合 loss，不能武断地写成“明确的 CFM + cross-entropy 加权公式”。

---

## 6. 推理链路

### 6.1 基础 Tactile-VLA

```text
最近 H 帧图像 ── ViT ─────────────┐
当前语言指令 ── tokenizer ─────────┤
最近 H 帧力信号 ── tactile MLP ────┤
                                   ↓
                         multimodal prefix
                                   │
机器人状态 + noisy action + τ ─────┤
                                   ↓
                         多步 flow integration
                                   ↓
                     [P_target chunk, F_target chunk]
                                   ↓
                         hybrid controller
                                   ↓
                          robot position commands
```

### 6.2 Tactile-VLA-CoT

```text
基础策略执行
    ↓
到达固定检查间隔
    ↓
VLM 判断任务是否完成
    ├─ 成功 → 结束或进入下一阶段
    └─ 失败 → 读取视觉/触觉
                 ↓
              生成原因
                 ↓
              生成纠正指令
                 ↓
           再次运行基础策略
```

---

## 7. Hybrid Position-Force Controller 做什么

### 7.1 为什么 Action Expert 后面还需要控制器

模型给出的是高层目标：

```text
末端应该到哪里？
接触力应该是多少？
```

机器人执行器通常仍需要稳定、连续的位置或关节命令。尤其在刚性接触时，直接追踪位置可能造成力突增，而直接发送抽象目标力又不一定符合机器人底层接口。

### 7.2 用力误差修正位置

定义：

\[
\Delta F=F_{\mathrm{target}}-F_{\mathrm{measured}}
\]

只有当误差超过阈值 \(\tau\) 时才修正目标位置：

\[
P_{\mathrm{hybrid}}=
P_{\mathrm{target}}+
\begin{cases}
K\Delta F,&\|\Delta F\|>\tau\\
0,&\|\Delta F\|\le\tau
\end{cases}
\]

然后 PID 控制器让机器人跟踪更新后的 \(P_{\mathrm{hybrid}}\)。

直觉上：

```text
目标力比实测力大 → 沿接触方向再压入一些
目标力比实测力小 → 从接触面退回一些
误差很小          → 不反复抖动调整
```

### 7.3 两类力被解耦

作者区分：

| 力 | 控制手段 | 含义 |
| --- | --- | --- |
| net external force | 调整夹爪笛卡尔位置 | 对环境施加的推、压、擦拭力 |
| internal grasping force | 调整夹爪宽度 | 两指如何夹住物体 |

因此“拿稳一个物体”和“把物体压向桌面”不会被混成同一个控制通道。

### 7.4 它不是另一个神经网络

混合控制器是基于误差、增益和 PID 的确定性控制模块：

- 不使用 flow matching；
- 不需要单独的训练数据；
- 没有通过 imitation learning 更新的参数；
- 负责将模型预测的力目标安全地落到机器人控制接口。

---

## 8. 数据采集与数据集

### 8.1 数据是否全部来自 UMI

按 Section 2.4 披露的流程，训练示范来自作者改装的 tactile-aware UMI：

- 便携式手持 3D 打印夹爪；
- GoPro 第一视角相机；
- 夹爪两侧的高分辨率触觉传感器；
- 可记录法向力与剪切力。

操作者直接拿着 UMI 在真实物体上执行动作，并能感受到接触动态，因而示范本身是 force-guided demonstration。

需要区分两种“真机数据”：

| 含义 | 论文是否包含 |
| --- | --- |
| 在真实物理世界中操作真实物体 | 包含，UMI 数据不是仿真数据 |
| 由机械臂实际执行或遥操作得到的训练轨迹 | 论文没有报告 |
| 真机 rollout 失败数据回灌、DAgger 或在线训练 | 论文没有报告 |
| 真实机械臂闭环测试 | 包含，但用于评测 |

正文没有逐项声明所有子数据的采集硬件，但它把 UMI 明确描述为 VLA-T training dataset 的采集系统，全文也没有第二个训练数据源。因此，“训练示范均来自 tactile-aware UMI”是最合理的判断，但应标记为根据全文信息作出的推断。

### 8.2 时间同步

- 触觉原始采样率：100 Hz；
- 视觉采样率：20 Hz；
- 采集前对齐各数据流时间戳；
- 训练前把高频触觉降采样，与相应视觉帧对齐。

最终数据包含同步的：

```text
vision + language + tactile + action trajectory
```

### 8.3 各实验的数据量

| 任务 | 训练示范 |
| --- | ---: |
| USB softly | 100 |
| USB hard | 100 |
| Charger 基础运动，不含 force adverb | 100 |
| Tabletop grasping，6 个 seen objects | 每个 50，共约 300 |
| Whiteboard successful wiping | 100 |
| Whiteboard failed wiping | 100 |

按正文列出的子数据合计约 800 条，但它们来自不同实验设置，不能简单等价为“一个统一混合的 800 条数据集”。

### 8.4 可还原的单条普通样本

下面是根据论文公式和数据章节还原的概念结构，不是作者发布的原始文件格式：

```yaml
vision_history: [I_t-H+1, ..., I_t]       # 约 20 Hz GoPro 图像
tactile_history: [T_t-H+1, ..., T_t]      # 法向/剪切力，100 Hz 后降采样对齐
language_instruction: "Insert the USB softly."
state_or_pose_history: [...]               # 状态/位姿细节未完整披露
target_position_trajectory: [...]
target_force_trajectory: [...]
```

CoT 失败样本还会增加类似信息：

```yaml
outcome: failure
failure_reason: "The downward force was insufficient."
corrective_reasoning: "Apply a stronger normal force and try again."
```

### 8.5 论文有没有真正展示数据样例

没有。

- Figure 4 展示采集硬件，不是单条数据；
- Figure 2 的触觉矩阵只是架构示意；
- Figure 5 和 Figure 7 展示任务场景与结果；
- 正文没有给出 JSON/HDF5 schema、完整时间序列或文件目录。

截至 2026 年 9 月检查项目主页时，Code 和 Data 仍标记为 `Coming Soon`，没有可下载的原始数据样例。

### 8.6 影响复现的缺失细节

论文没有充分报告：

- 触觉传感器具体型号；
- 触觉数组维度及各通道定义；
- 历史窗口 \(H\) 的数值；
- UMI 动作如何标定或 retarget 到机器人坐标系；
- position 与 force action 的归一化和权重；
- action chunk 长度与控制频率；
- CoT 文本标注者、模板和质量控制；
- train/validation split；
- 完整数据存储格式。

---

## 9. 三组实验分别证明什么

### 9.1 RQ1：Tactile-Aware Instruction Following

### 实验设计

训练阶段：

- USB 任务同时提供 `softly` 和 `hard` 两类示范；
- Charger 任务只教基础插拔运动，不提供带力度副词的示范。

测试阶段检验三层泛化：

1. 训练词：`softly`、`hard`；
2. 未见同义词：`gently`、`firmly`、`rigidly`、`harder`；
3. 把在 USB 中学到的词义迁移到 Charger。

### 成功率

| 模型 | USB | Charger |
| --- | ---: | ---: |
| π0-base | 5% | 40% |
| π0-fast | 0% | 25% |
| Tactile-VLA | **35%** | **90%** |

### 力语言映射

在 USB 上，Tactile-VLA 输出：

- softly：0.51 N；
- hard：2.57 N；
- gently：0.75 N；
- firmly：1.98 N；
- rigidly：2.42 N；
- harder：2.94 N。

在未用 force words 训练的 Charger 上：

- softly：4.68 N；
- hard：9.13 N。

这说明模型没有把一个词机械绑定到跨任务固定的绝对牛顿值，而是保留了任务情境，同时让不同副词控制相对强弱。

### 它真正支持的结论

实验较有力地支持：加入触觉和力动作后，语言副词可以系统性调节物理作用力，并跨任务迁移。

但它不能单独证明所有提升都来自“VLM 隐式物理知识”，因为 Tactile-VLA 同时改变了：

- 触觉输入；
- Action Expert 输出空间；
- 训练目标中的力维度；
- 低层混合控制器。

---

### 9.2 RQ2：Tactile-Relevant Common Sense

### 实验设计

物体分为：

- Solid & Heavy：应该用较大抓力；
- Solid & Light：应该用中等抓力；
- Fragile & Light：应该用较小抓力。

训练使用 6 个 seen objects，每个 50 条示范；测试额外加入 6 个 unseen objects。成功标准是一次抓起且没有可观察形变。

### 结果

论文报告 10 次/物体的成功率。Tactile-VLA 在多数 seen 与 unseen 物体上达到 80%–100%，尤其对脆弱物体明显优于 π0：

| OOD 物体 | π0-base | π0-fast | Tactile-VLA |
| --- | ---: | ---: | ---: |
| Nail | 30% | 10% | **100%** |
| Steel Ball | 60% | 70% | **90%** |
| Plastic Toy | 40% | 30% | **80%** |
| Melon | 0% | 10% | **80%** |
| Blueberry | 0% | 0% | **100%** |
| PaperBox | 0% | 0% | **90%** |

Figure 6 还显示模型对重物、普通轻物和易碎物分别产生 hard、medium 和 soft 的抓力分布。

### 它真正支持的结论

结果说明预训练 VLM 与触觉/力控制结合后，对未见物体能选择更合适的抓力。

但“知识一定来自 VLM 常识”仍属于作者解释，不是被严格隔离出来的因果结论。更强的验证应包括：

- 冻结或随机初始化 VLM 的对照；
- 遮挡物体外观、只保留名称的对照；
- 移除 tactile token 但保留 force action/controller 的参数匹配消融；
- 物理性质相近但外观差异大的受控对象集。

---

### 9.3 RQ3：Tactile-Involved Reasoning

### 实验设计

训练只在白板上进行：

- 100 条成功示范；
- 100 条失败示范；
- 失败样本配有“力太小，需要增大到某个水平”等纠正文字。

测试时换成从未训练的黑板。默认擦拭力 3.5 N 无法擦掉粉笔后，CoT 应判断失败、增大下压力再尝试。

### 成功率

| 模型 | 白板 ID | 黑板 OOD |
| --- | ---: | ---: |
| π0-base | 40% | 0% |
| π0-fast | 45% | 0% |
| Tactile-VLA | **80%** | 15% |
| Tactile-VLA-CoT | 75% | **80%** |

CoT 在黑板上把力从默认 3.5 N 增加到 6.7 N；这甚至高于白板示范中的 5 N 纠正水平。

### 为什么基础版本在 OOD 黑板上仍然差

基础 Tactile-VLA 虽能在一个 action chunk 内根据触觉控制力，但它没有明确的“检查任务结果—诊断失败—改变策略”循环。面对材料发生变化的黑板，它可能继续重复训练分布中的默认擦拭方式。

CoT 版本增加了一个较长时间尺度的状态转换：

```text
当前策略无效
    ↓
这是力不足，而不是动作方向错误
    ↓
生成更强下压力的新指令
    ↓
重新运行策略
```

这就是 CoT 实验最主要的价值。

---

## 10. 与前面几篇论文的架构比较

| 工作 | 触觉输入 | 触觉如何接入 | 动作生成 | 主要目标 |
| --- | --- | --- | --- | --- |
| TVL | GelSight 类触觉图像 | 触觉 encoder 对齐 CLIP；再接 LLaMA | 不输出机器人动作 | 触觉—视觉—语言对齐与描述 |
| Octopi | 视触觉图像 | CLIP-based encoder + projector → Vicuna | 语言描述与物理推理，不直接控制 | 物体属性描述和 PSR |
| AnyTouch | 多传感器静态/动态触觉 | 统一触觉 encoder；可接 LLM | 主要做表征和语言理解 | 跨传感器、静动态统一表征 |
| Beyond Sight / FuSe | 力、音频等异构传感器 | 传感器 token 与 Octo token 融合；语言辅助 loss | Octo action readout | 用 language grounding 适配异构传感器 |
| VLA-Touch | GelSight 图像/marker-derived force | VLA 外部高层 planner + 低层 controller | RDT + BRIDGeR 修正 | 模块化双层触觉反馈 |
| OmniVTLA | Paxini 二维力场伪图像 | 双触觉 ViT token 进入 π0 VLM prefix | π0 flow matching | 语义对齐触觉 encoder 对控制的作用 |
| **Tactile-VLA** | 法向力和剪切力历史 | **MLP 压成单 token，进入 π0 前缀** | **力增强的 π0 CFM Action Expert** | **语言—触觉—力动作闭环与 CoT 适应** |

### 10.1 与 OmniVTLA 的核心区别

```text
OmniVTLA：
force field → pretrained ViT + semantic-aligned ViT → many tactile tokens
重点：好的触觉表征初始化能否提升 VLA

Tactile-VLA：
force history → simple MLP → one tactile token
重点：语言、触觉、force action、controller 和 CoT 如何形成闭环
```

OmniVTLA 更像“encoder 研究”；Tactile-VLA 更像“从语义到物理执行的系统研究”。

### 10.2 与 VLA-Touch 的核心区别

```text
VLA-Touch：
RDT 本体不看触觉
触觉在外部帮助 GPT-4o 规划，并由后置控制器修正动作

Tactile-VLA：
触觉 token 进入 VLM 前缀
Action Expert 本身生成目标力
整个策略端到端微调
```

VLA-Touch 的优点是模块化、容易替换基础 VLA；Tactile-VLA 的优点是触觉直接参与动作生成。

### 10.3 与 TVL/Octopi 的核心区别

TVL 和 Octopi 的最终输出主要是语言，并不直接承担机器人实时控制。Tactile-VLA 的输出是动作与目标力，因此监督信号、时间尺度和评价指标完全不同：

```text
TVL/Octopi：触觉 → semantic embedding / language
Tactile-VLA：触觉 + 语言 + 视觉 → action trajectory + force trajectory
```

---

## 11. 对论文结论的审慎评价

### 11.1 最有价值的地方

1. **触觉进入策略内部，而不是只做外挂描述。**
2. **力被显式纳入 action space。** 这是从“感知触觉”走向“控制接触”的关键一步。
3. **语言副词能够跨任务调节力。** USB 到 Charger 的实验故事清楚。
4. **快慢两个时间尺度互补。** Action Expert 负责连续控制，CoT 负责失败后的策略改变。
5. **UMI 让力引导示范更自然。** 操作者直接接触真实物体，比缺乏力反馈的远程遥操作更可能产生与触觉真正相关的数据。

### 11.2 主要局限

#### 缺少关键消融

没有充分拆开比较：

- tactile input；
- force-augmented action；
- hybrid controller；
- end-to-end VLM fine-tuning；
- CoT language supervision。

因此很难量化每个组件单独贡献多少。

#### Baseline 不完全等价

π0-base 和 π0-fast 缺少触觉路径、力输出和混合控制器。它们不仅“少了 tactile token”，而是整个控制接口都不同，因此比较展示了完整系统收益，却不能严格定位收益来源。

#### “解锁 VLM 物理知识”证据仍是间接的

未见物体和未见副词上的泛化与 VLM 先验一致，但论文没有通过冻结、随机化或替换 VLM 等实验严格证明知识来源。

#### 触觉 encoder 很简单且传感器相关

MLP 接收固定格式的力历史，缺少：

- 跨触觉传感器泛化；
- 空间结构建模；
- 大规模触觉预训练；
- 复杂动态接触的长时序建模。

#### 实验规模较小

只有三类任务、少量训练对象和每物体有限测试次数。黑板 CoT 很有代表性，但仍是单一场景的 proof of concept。

#### 复现信息与开源不足

关键超参数、数据格式、标定流程和 CoT 训练细节没有完整给出，项目页当前也未释放代码和数据。

### 11.3 对作者主张的更稳妥表述

作者说：

> VLM 已包含丰富的物理交互知识，少量示范即可将其解锁。

更严谨的表述是：

> 在 π0 的预训练视觉—语言表示上增加触觉输入、力动作监督和混合控制，并进行少量端到端微调，可以在若干接触任务上获得未见副词、物体和表面的显著泛化；实验与“利用 VLM 物理先验”的解释相符，但尚未通过因果消融完全证明。

---

## 12. 最终知识框架

理解这篇论文，可以记住四个层次。

### 表征层

```text
force history → MLP → tactile token
```

它不追求建立通用 CLIP 式触觉语义空间，而是生成面向当前控制任务的触觉条件。

### 策略层

```text
vision + language + tactile + state
                ↓
π0-style Transformer + Action Expert
                ↓
position-force action chunk
```

### 控制层

```text
target force - measured force
                ↓
position correction + PID
                ↓
stable physical contact
```

### 推理层

```text
failure observation + tactile feedback
                ↓
VLM internal monologue
                ↓
corrective instruction
                ↓
new action and force strategy
```

这四层共同构成论文真正的闭环：

\[
\text{语义意图}
\rightarrow
\text{触觉条件化动作}
\rightarrow
\text{实际力控制}
\rightarrow
\text{失败反馈与重规划}
\]

---

## 13. 阅读完应该带走的结论

1. Tactile-VLA 和 Tactile-VLA-CoT 不是两个完全不同的 VLA；后者是在前者上增加周期性语言推理与重规划。
2. 触觉输入是法向/剪切力时间序列，经简单 MLP 压缩成一个 token，并非语义预训练的触觉图像 encoder。
3. Tactile-Aware Action Expert 的关键不是特殊 attention，而是读取 tactile-conditioned prefix，并预测 force-augmented actions。
4. 基础训练使用统一的 Conditional Flow Matching 动作监督，位置与力都在 action space 中。
5. 混合控制器不学习，它把目标力误差转换为位置修正，并分别控制外部接触力与内部抓持力。
6. 训练示范按论文披露来自改装 UMI，是现实世界人类手持示范；论文没有报告加入机械臂 rollout 训练数据。
7. 三组实验分别验证语言力度迁移、未见物体抓力泛化和未见表面上的失败后重规划。
8. 论文的系统故事完整，但关键消融、数据细节与开源仍不足，“解锁 VLM 物理知识”应视为得到实验支持、但未被完全隔离验证的解释。
