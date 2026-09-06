# 《VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback》精读笔记

> Jianxin Bi, Kevin Yuchen Ma, Ce Hao, Mike Zheng Shou, Harold Soh，arXiv 2025<br>
> [arXiv](https://arxiv.org/abs/2507.17294) · [Hugging Face论文页](https://huggingface.co/papers/2507.17294) · [开源代码](https://github.com/jxbi1010/VLA-Touch)

> 推荐先读：[Octopi](17_octopi_paper_reading_notes.md)、[Beyond Sight / FuSe](20_beyond_sight_fuse_paper_reading_notes.md)、[动作生成方法](25_action_generation_families.md)和[VLA架构地图](27_vla_overview.md)。本文代表的是“外挂式触觉增强已有VLA”的路线。

## 0. 一句话定位

VLA-Touch没有训练一个新的端到端触觉VLA，而是在已有RDT视觉—语言—动作策略外增加两条彼此独立的触觉路径：

1. **高层触觉规划**：把触觉转换成语言或数值反馈，由GPT-4o据此选择对象并生成下一条primitive instruction；
2. **低层触觉控制**：把GelSight marker位移汇总成低维力信号，由Interpolant Controller修正RDT生成的动作块。

```text
高层：GelSight → Octopi / 属性分类器 / 力估计 → GPT-4o → 新的primitive instruction

低层：RDT action + RGB + proprioception + tactile force
                              ↓
                    Interpolant Controller
                              ↓
                         refined action
```

因此这篇论文的贡献主要是**模块化框架与系统集成**，不是共享触觉latent space、触觉token注入VLA Transformer或端到端多模态训练。

---

## 1. 论文要解决什么问题

现有VLA主要使用视觉、语言和机器人状态。它可以知道“杯子在哪里”“应该把杯子放到盘子上”，但在接触丰富的任务中会缺少两类信息。

### 1.1 高层决策缺少不可见的物理属性

视觉不一定能判断：

- 杯子是空的还是装了水；
- 两块外观接近的海绵哪块更光滑；
- 两个芒果哪个更软、更成熟。

这些信息会影响机器人究竟应该选择哪个对象、执行哪一步动作。

### 1.2 低层控制缺少实时接触状态

即使选对了对象，机器人仍可能：

- 抓杯子时把杯子推走；
- 放杯子时过早松开夹爪；
- 擦拭时压力不足，只擦掉中央部分；
- 削皮时失去接触，削皮器从芒果表面滑开。

这些问题不能只靠一句高层触觉描述解决，需要连续触觉信号参与动作闭环。

VLA-Touch因此提出：**触觉不应该只有一种接入方式。语义属性适合高层规划，连续接触力适合低层控制。**

---

## 2. 主要贡献

### 2.1 双层触觉反馈框架

论文将触觉划分为两种抽象层级：

| 层级 | 触觉形式 | 消费者 | 解决的问题 |
| --- | --- | --- | --- |
| 高层 | 物理属性语言或数值 | GPT-4o planner | 选择哪个对象、下一步做什么 |
| 低层 | 连续低维力向量 | Interpolant Controller | 怎样抓、怎样保持接触、使用多大力 |

这比简单地把“触觉”当成一个统一模态更接近真实机器人系统：规划需要可解释的属性，控制需要高频数值反馈。

### 2.2 不修改RDT内部结构的触觉增强方式

基础VLA使用RDT-1B。触觉既不进入RDT的视觉编码器，也不作为token进入RDT Transformer。作者通过外部planner和后置controller增强RDT，因此可以复用现有VLA能力。

### 2.3 用BRIDGeR修正VLA动作分布

低层控制器不是只预测一个确定性的动作残差，而是使用stochastic interpolant，从已有信息量的RDT动作分布出发，逐步向专家动作分布运输。

```text
普通diffusion：Gaussian noise → expert-like action

VLA-Touch：RDT action → stochastic bridge → expert-like action
```

RDT已经提供了大体合理的任务动作，控制器只需要利用触觉修正其中的接触误差。

### 2.4 真实机器人任务验证

作者在杯子放置、海绵擦拭和芒果削皮三个任务上分别验证：

- 结构化触觉反馈能否改善GPT-4o规划；
- 触觉动作修正能否改善RDT控制；
- 同时使用高层和低层触觉是否优于只使用其中一层。

---

## 3. Figure 1与完整推理流程

VLA-Touch的一次任务执行可以拆成“规划—执行—接触—重新规划”的循环。

### 3.1 第一次规划

```text
任务目标g + 初始场景图像o_t
              ↓
            GPT-4o
              ↓
primitive instruction I_0
```

例如完整任务是：

```text
找到两块海绵中更光滑的一块，并用它擦除盘子上的墨迹。
```

GPT-4o不会把整项任务一次性交给RDT，而是先产生一个原子动作：

```text
Pick up the left sponge.
```

### 3.2 RDT生成基础动作

```text
视觉 observation + proprioception + instruction I_k
                        ↓
                      RDT-1B
                        ↓
               action chunk a^{RDT}
```

RDT完全不接收触觉。它负责根据视觉和语言产生一个基础动作块。

### 3.3 低层控制器修正动作

Interpolant Controller接收：

- RDT生成的动作块；
- 两路RGB相机特征；
- 机器人末端位姿和夹爪状态；
- GelSight marker得到的低维力。

形式上可写成：

\[
\hat a_{t:t+T_i}
=
\pi_I\left(
a^{\mathrm{RDT}}_{t:t+T_i},
s_t,
m_t
\right)
\]

控制器输出修正后的动作并执行。系统通过滑动窗口逐段修正RDT动作块，而不是一次把完整64步全部开环执行。

### 3.4 接触后获取高层触觉反馈

primitive完成或执行失败后，系统读取触觉信息：

```text
6帧GelSight触觉图像
          ↓
       Octopi
          ↓
roughness / hardness语言描述
```

然后重新提示GPT-4o：

\[
I_{k+1}
=
\mathrm{GPT4o}(g,o_t,L_m,I_k)
\]

GPT-4o结合总任务、原始场景、上一条指令和触觉反馈，产生下一条primitive instruction。

---

## 4. 高层Planner究竟是什么

### 4.1 它不是作者训练的新VLM

Task Planner就是现成的GPT-4o。作者通过system prompt规定：

1. 机器人具有单臂、夹爪和触觉传感器；
2. 每次只输出一个与至多一个物体交互的primitive action；
3. 如果决策需要额外信息，应说明需要获取什么信息；
4. 执行动作后会获得属性或执行结果反馈；
5. Planner要根据反馈继续规划。

因此Planner没有训练数据、没有新增loss，也没有参数更新；它属于inference-time prompting。

### 4.2 Octopi只负责部分属性

Octopi处理6帧GelSight触觉序列，主要推断：

- roughness；
- hardness。

但三个实验并非全部直接使用Octopi生成的完整句子。

#### 海绵粗糙度与芒果硬度

可以使用Octopi输出自然语言，也可以使用其连续属性分类器输出，例如：

```text
Left sponge roughness: 1.62
Right sponge roughness: 7.71
```

论文Appendix B发现：对于差异细微的芒果硬度，连续属性分数比Octopi完整语言描述更有效。完整描述取得60%规划准确率，连续分类结果达到75%。这说明自然语言可能形成信息瓶颈。

#### 杯子重量/力

Octopi没有针对力估计预训练，因此作者直接跟踪GelSight的7×9 marker阵列，将每个marker的位移向量汇总为：

\[
m_t=(X,Y,M)
\]

其中：

- \(X\)：水平方向位移之和；
- \(Y\)：垂直方向位移之和；
- \(M\)：总体位移幅值。

系统把估计值和空杯、满杯的参考值一起交给GPT-4o。严格来说，杯子任务的高层反馈不是Octopi生成的触觉描述，而是数值化marker force feedback。

---

## 5. 低层触觉表示与Controller输入

低层控制器不处理原始GelSight RGB，也不使用Octopi embedding。其触觉输入是marker tracking得到的三维向量：

```text
GelSight RGB
    ↓ OpenCV marker tracking
63个marker的位移
    ↓ 聚合
(X, Y, M)
```

视觉输入来自两个RealSense相机，经过冻结的DINOv2-small获得特征。随后将以下信息组合：

```text
固定相机DINO特征 ─┐
腕部相机DINO特征 ─┤
机器人state ──────┼─> state encoder ─> controller condition z_t
低维触觉力 ───────┘
```

代码中的机器人state/action维度为10，state encoder hidden width为256。论文更重要的概念是：低层触觉是一种紧凑的控制信号，而不是语言语义token。

---

## 6. 数据怎样采集，专家动作从哪里来

作者使用：

- Franka Emika Panda机械臂；
- Robotiq 2F-140夹爪；
- 安装在一个手指上的GelSight Mini；
- 固定俯视RealSense相机；
- 腕部RealSense相机。

数据通过**kinesthetic teaching，即手把手拖动示教**采集，而不是论文中常见的VR、SpaceMouse或主从遥操作。论文没有进一步说明具体的重力补偿或示教控制模式。

传感器以10 Hz记录：

```text
固定相机RGB
腕部相机RGB
GelSight图像
机械臂末端位姿
夹爪状态
人类示教动作
episode对应的语言instruction
```

这里的expert action就是人在拖动机器人完成任务时记录的动作，不是Octopi生成的，也不是另一个已训练expert policy输出的。

### 6.1 采集规模

| 任务 | 子任务 | Episodes | 采集时间 |
| --- | --- | ---: | ---: |
| Cup | Pick 40 + Place 60 | 100 | 约8分钟 |
| Wipe | Pick 40 + Wipe 60 | 100 | 约15分钟 |
| Peel | Pick 60 + Peel 120 | 180 | 约30分钟 |
| 总计 |  | 380 | 约53分钟 |

每个episode还带有对应语言指令，供RDT学习instruction following。

### 6.2 同一批示范被使用两次

第一次用于fine-tune RDT，但移除触觉通道：

```text
RGB + proprioception + instruction → RDT → expert action
```

第二次用于构造Controller数据。作者在每个专家轨迹状态上运行已经fine-tune的RDT，补充生成source action：

\[
D_{\mathrm{VLA}}
=
\left\{
s_{t-n:t},
o^p_{t-n:t},
a^{\mathrm{RDT}}_{t:t+T_a},
a^{\mathrm{Exp}}_{t:t+T_a}
\right\}
\]

其中：

- \(a^{\mathrm{RDT}}\)：同一状态下RDT提出的动作；
- \(a^{\mathrm{Exp}}\)：人类示范中的动作；
- \(T_a=64\)：action chunk horizon；
- \(n=2\)：历史条件长度。

所以Controller数据不是重新采集的第二套示范，而是：

```text
原始人类示范
   ├─ 读取专家动作
   ├─ 读取同步触觉与视觉
   └─ 离线运行RDT，生成对应的RDT动作
```

### 6.3 论文没有充分说明的部分

作者首先还使用“unrelated Franka data”训练RDT 100k steps，以对齐当前Franka的action space，但论文没有明确给出：

- 这部分数据的具体名称；
- episode数量和任务构成；
- train/validation划分；
- 是否公开。

下游三项任务的数据规模有说明，但完整数据集细节和可复现性仍然有限。

---

## 7. 分阶段训练：哪些参数会更新

VLA-Touch不存在一个贯穿GPT-4o、Octopi、RDT和Controller的端到端总loss。训练分为互相独立的阶段。

### 7.1 阶段0：使用现成预训练组件

| 模块 | 状态 |
| --- | --- |
| GPT-4o | API调用，只做prompt inference |
| Octopi | 预训练并冻结 |
| DINOv2-small | 预训练并冻结 |
| marker tracker | OpenCV算法，无学习参数 |

### 7.2 阶段1：RDT action space与任务适配

作者先在未具体说明的Franka数据上fine-tune RDT约100k steps（约80小时），使其适配当前机器人action space。之后，在三个任务的人类示范上分别训练约20k steps（约16小时/任务）。

这一阶段：

- 更新RDT参数；
- 使用RGB、机器人状态和语言；
- 监督信号是人类专家动作；
- 完全不使用触觉输入。

论文宣传语中“without fine-tuning the base VLA”容易产生误解。更准确地说，它不需要为了触觉接入而修改并联合fine-tune RDT内部结构；但Appendix C明确说明，RDT本身仍然进行了机器人和任务适配。

### 7.3 阶段2：生成Controller训练配对

RDT在专家数据中的各个状态上产生动作块：

```text
同一个observation
    ├─ 人类示范 → a_expert
    └─ RDT推理   → a_RDT
```

再把两者与同步触觉、视觉和机器人状态配成训练样本。

### 7.4 阶段3：训练Interpolant Controller

Controller的顶层监督目标可概括为：

\[
\min_\theta
\mathbb E_{D_{\mathrm{VLA}}}
\left[
\left\|
a^{\mathrm{Exp}}
-
\pi_I(a^{\mathrm{RDT}},z;\theta)
\right\|^2
\right]
\]

即让修正动作接近专家动作。

实际BRIDGeR实现通过stochastic-interpolant的速度场、score/bridge相关回归项训练，代码记录`v_loss`、`s_loss`和`b_loss`。论文的MSE表达是对最终训练目标的上层概括，不能简单理解为一次前向后只计算一个普通动作MSE。

该阶段更新：

- state encoder MLP；
- Interpolant / conditional 1D U-Net网络。

该阶段冻结：

- RDT；
- DINOv2；
- Octopi；
- GPT-4o。

### 7.5 总结参数更新关系

| 模块 | 是否更新 | 何时更新 |
| --- | ---: | --- |
| GPT-4o Planner | 否 | 只做prompt inference |
| Octopi | 否 | 使用预训练checkpoint |
| DINOv2-small | 否 | 只提取视觉特征 |
| marker tracker | 否 | 无训练参数 |
| RDT-1B | 是 | 机器人与任务适配阶段 |
| State Encoder | 是 | Controller训练阶段 |
| Interpolant网络 | 是 | Controller训练阶段 |
| LSTM Residual baseline | 是 | 单独训练，仅作基线 |

不存在：

- 触觉—视觉对比对齐loss；
- Planner语言生成训练loss；
- 触觉token进入RDT后的action loss；
- GPT-4o到低层Controller的梯度反传；
- 全系统共享的多模态latent space。

---

## 8. LSTM Residual Controller与Interpolant Controller

两者使用同类输入和同一个专家动作监督，区别是怎样表达“从RDT动作到专家动作”的映射。

### 8.1 LSTM Residual Controller

LSTM接收RDT动作、触觉力、视觉和机器人状态，预测确定性的动作修正量：

\[
\Delta a_t
=
f_{\mathrm{LSTM}}
\left(
a_t^{\mathrm{RDT}},
o_t,s_t,m_t,h_{t-1}
\right)
\]

最终动作是：

\[
\hat a_t=a_t^{\mathrm{RDT}}+\Delta a_t
\]

训练使用直接MSE：

\[
\mathcal L_{\mathrm{LSTM}}
=
\left\|
a_t^{\mathrm{Exp}}-hat a_t
\right\|^2
\]

官方代码会训练observation encoder、force encoder、两层单向LSTM和output head，DINOv2保持冻结。

### 8.2 Interpolant Controller

Interpolant把RDT动作视为source policy \(\pi_0\)，把专家动作视为target policy \(\pi_1\)，建立连续随机过程：

\[
a_\tau
=
I(\tau,a_0,a_1,x)+\gamma(\tau)z
\]

满足：

\[
I(0,a_0,a_1,x)=a_0,
\qquad
I(1,a_0,a_1,x)=a_1
\]

推理时从RDT动作开始，通过若干次迭代向专家动作分布运输。

### 8.3 核心差异

| 对比项 | LSTM Residual | Interpolant |
| --- | --- | --- |
| 思路 | 预测一个动作偏移 | 学习RDT与专家动作分布之间的桥 |
| 输出 | 确定性 \(a+\Delta a\) | 生成式、迭代式动作修正 |
| 主体网络 | LSTM + MLP head | conditional 1D U-Net / BRIDGeR |
| 多模态动作 | MSE容易产生平均解 | 理论上可以保留多个动作模式 |
| 推理成本 | 较低 | 较高 |
| 训练目标 | 直接动作MSE | stochastic-interpolant内部回归目标 |

例如专家示范中既有“从左侧绕开杯子”也有“从右侧绕开杯子”的正确轨迹。确定性MSE可能把两种方式平均成“从正中间推向杯子”；Interpolant则尝试建模多个可能的专家动作模式。

论文把Residual在杯子上的系统性定位偏差，以及削皮时容易提前抬起工具，归因于确定性残差模型难以处理示范数据的多模态性。不过论文没有提供动作分布可视化，这一解释主要由最终成功率间接支持。

---

## 9. 三个真实机器人任务

每个任务都同时需要高层触觉判断和低层触觉控制。

### 9.1 Cup

```text
高层：通过抓取后的力判断杯中是否有水
低层：稳定抓取并放到对应位置，避免滑落或洒水
```

### 9.2 Wipe

```text
高层：触摸两块海绵，选择更光滑的一块
低层：抓住海绵中央，以足够压力擦除盘子上的墨迹
```

### 9.3 Peel

```text
高层：根据硬度选择更软/更成熟或任务要求的芒果
低层：让手持削皮器切入表面，并在运动中持续保持接触
```

---

## 10. 实验一：结构化触觉是否改善规划

作者比较GPT-4o的三种输入：

1. 只有固定相机RGB；
2. RGB加原始GelSight图像；
3. RGB加结构化触觉语言/数值反馈。

每种条件测试20次。

| 属性判断 | 原始触觉图像 | 结构化反馈 |
| --- | ---: | ---: |
| 海绵roughness | 100% | 100% |
| 杯子force/weight | 50% | 90% |
| 芒果hardness | 60% | 75% |

只看场景RGB时，GPT-4o通常回答无法判断；如果强制选择，结果接近随机猜测。

这个实验说明的不是“GPT-4o完全无法看懂GelSight”，因为它对差异明显的粗糙度原始图像也达到100%。更准确的结论是：**对于力和细微硬度等难以直接解释的信号，专用触觉处理后的结构化反馈更容易被通用VLM稳定使用。**

---

## 11. 实验二：哪种低层Controller更有效

作者比较：

- RDT：不使用触觉的基础策略；
- RDT + Residual：LSTM预测动作残差；
- RDT + Interpolant：使用BRIDGeR修正动作分布；
- w/o Touch：Interpolant不使用触觉；
- w/o Vision：Interpolant不使用视觉。

### 11.1 完整结果

| Task | Evaluation | RDT | Residual | Interpolant | w/o Touch | w/o Vision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cup | Pick | 9/20 | 7/20 | **12/20** | 10/20 | 10/20 |
| Cup | Place | 7/20 | 6/20 | **10/20** | 5/20 | 7/20 |
| Wipe | Pick | 11/20 | 15/20 | **17/20** | 15/20 | 15/20 |
| Wipe | Partial | 8/20 | 13/20 | **16/20** | 12/20 | 10/20 |
| Wipe | Full | 5/20 | 6/20 | **12/20** | 7/20 | 8/20 |
| Peel | Pick | 13/20 | 14/20 | **18/20** | 16/20 | 13/20 |
| Peel | Partial | 8/20 | 12/20 | **13/20** | 12/20 | 8/20 |
| Peel | Full | 6/20 | 7/20 | **10/20** | 5/20 | 5/20 |

Interpolant在所有列中取得最好结果，尤其是：

- Wipe完整成功率从RDT的5/20提高到12/20；
- Peel完整成功率从6/20提高到10/20；
- Cup放置从7/20提高到10/20。

触觉对需要维持接触和调节压力的后半段帮助最大；视觉对空间定位和初始抓取仍然重要。

---

## 12. Pick、Partial与Full怎样理解

它们通常是逐级包含的阶段性指标，而不是互斥的三类结果：

\[
\mathrm{Full}\subseteq\mathrm{Partial}\subseteq\mathrm{Pick}
\]

表格中的分母始终是全部20次实验，不是以前一阶段成功数作为分母。

### 12.1 Cup

- **Pick**：成功抓住并抬起杯子；
- **Place**：成功放到目标位置，且没有过早释放、明显滑落或洒水。

Interpolant的12/20 Pick、10/20 Place表示20次中12次抓起，其中10次最终完成放置。

### 12.2 Wipe

- **Pick**：成功抓住正确海绵，最好位于海绵中央；
- **Partial**：至少擦掉部分墨迹，但可能只清理中央区域；
- **Full**：完成整个目标区域的擦除，包括外围部分。

Interpolant得到17/20 Pick、16/20 Partial和12/20 Full，表示17次成功抓取，16次至少产生有效擦拭，12次完整清理。

### 12.3 Peel

- **Pick**：成功抓取/完成工具就位并进入削皮阶段；
- **Partial**：削皮器成功切入并削下部分果皮，但可能很短或中途失去接触；
- **Full**：较完整地保持接触并削下足够长的果皮。

论文报告Residual共削下12片果皮，其中5片是短片，对应12次Partial和7次Full；Interpolant削下13片，其中3片是短片，对应13次Partial和10次Full。

论文没有给出“擦除面积超过多少”或“果皮长度超过多少厘米”这样的量化阈值，因此这些指标主要是人工判定的阶段成功标准，是实验设计中不够严格的一点。

---

## 13. 实验三：为什么两层触觉都需要

作者比较：

| 设置 | 高层规划 | 低层控制 |
| --- | --- | --- |
| w/o Planning | GPT-4o，不使用Octopi反馈 | RDT + Interpolant |
| w/o Control | GPT-4o + Octopi | 只有RDT |
| VLA-Touch | GPT-4o + Octopi | RDT + Interpolant |

最终任务成功率：

| 方法 | Cup | Wipe | Peel |
| --- | ---: | ---: | ---: |
| w/o Planning | 5/20 | 5/20 | 6/20 |
| w/o Control | 6/20 | 5/20 | 4/20 |
| Full VLA-Touch | **9/20** | **12/20** | **7/20** |

这说明：

- 只有低层控制，机器人可能动作稳定但选错对象；
- 只有高层规划，机器人可能选对对象但接触操作失败；
- 双层系统同时解决“做什么”和“怎么做”。

但每组只有20次试验，论文没有报告置信区间或显著性检验，应谨慎看待较小的数字差异。

---

## 14. 与此前几篇工作的架构区别

| 工作 | 触觉接入位置 | 是否进入统一策略 | 主要输出 |
| --- | --- | ---: | --- |
| TVL | CLIP语义空间与LLaMA | 进入触觉语言模型 | 触觉描述 |
| Octopi | CLIP-based触觉encoder接Vicuna | 进入触觉语言模型 | 属性描述与推理 |
| AnyTouch | 统一触觉encoder与LLM接口 | 主要做表征/语言 | 表征、分类和描述 |
| FuSe | 触觉token进入策略Transformer | 是 | 机器人动作 |
| TLA | 触觉与语言进入Qwen2并生成动作token | 是，但没有RGB视觉 | 机器人动作 |
| VLA-Touch | RDT外部Planner和后置Controller | 否 | 新指令 + 修正动作 |

### 14.1 与Octopi的关系

Octopi在VLA-Touch中只是一个冻结的高层触觉解释器。VLA-Touch没有重新训练Octopi，也没有使用Octopi的latent feature进行低层控制。

### 14.2 与FuSe的关系

FuSe属于内部融合：

```text
vision/touch/audio/language tokens
               ↓
       policy Transformer
               ↓
             action
```

VLA-Touch属于外部、晚期融合：

```text
Octopi → GPT-4o → instruction → RDT action
                                      ↓
tactile force + RGB + state → Controller → refined action
```

FuSe试图让policy内部理解多个传感器；VLA-Touch保持policy不懂触觉，通过外部模块改变instruction并修正动作。

### 14.3 与“VLM + Action Expert”式VLA的区别

VLA-Touch没有像π0.5或GR00T那样，让触觉token与VLM memory、state和noisy action在Action Expert内部共同attention。它甚至没有改变RDT的Transformer。Interpolant Controller是RDT外部的第二个动作模型，更像一个learned post-policy corrector。

---

## 15. 优点与局限

### 15.1 优点

1. **模块化**：可以复用GPT-4o、Octopi和已有VLA；
2. **职责清晰**：语言属性用于规划，连续力用于控制；
3. **训练成本相对可控**：触觉不需要进入十亿参数RDT联合训练；
4. **容易迁移思路**：理论上可把RDT替换为其他VLA；
5. **动作起点更好**：Interpolant从RDT动作而非纯噪声开始修正。

### 15.2 局限

1. **不是统一触觉VLA**：RDT本身始终不理解触觉；
2. **没有共享latent space**：高层语言和低层力是两条独立信息通道；
3. **误差会级联**：触觉估计、GPT-4o规划、RDT动作和Controller修正任一环节都可能出错；
4. **语言存在信息瓶颈**：连续硬度分数比完整Octopi描述更好；
5. **依赖云端VLM并存在延迟**：GPT-4o不适合高频闭环；
6. **传感器配置有限**：只有一个手指安装GelSight Mini；
7. **Controller偏任务特定**：每个任务都需要示范和RDT适配，尚不是通用触觉控制器；
8. **低层触觉压缩较强**：63个marker最终只保留\((X,Y,M)\)，空间接触分布被丢失；
9. **实验规模有限**：三个任务、每种条件20次，没有统计显著性分析；
10. **数据披露不足**：前置Franka数据集没有明确名称和规模。

---

## 16. 最终理解

VLA-Touch可以被概括成：

```text
                ┌─ 触觉属性/力数值 ─> GPT-4o ─> 新instruction ─┐
任务 + 视觉 ────┤                                            ├─> RDT
                └────────────────────────────────────────────┘
                                                               ↓
                                         初始action chunk a_RDT
                                                               ↓
RGB + proprioception + tactile force ─> Interpolant Controller
                                                               ↓
                                                     refined action
```

它不是在回答“怎样让一个大模型学会统一的视觉—触觉—语言—动作表示”，而是在回答另一个很实际的问题：

> 如果已经有一个可用但不懂触觉的VLA，怎样尽量不改它的内部结构，也能让触觉同时改善任务决策和接触控制？

答案是给已有VLA安装两个触觉外挂：

- 一个负责根据触觉重新思考任务；
- 一个负责根据触觉修正动作。

从研究路线看，它代表的是**模块化retrofit路线**，而FuSe、TLA以及后续触觉原生VLA代表的是**模型内部融合路线**。前者更容易落地和复用，后者更有机会形成统一表征与端到端能力。

---

## 17. 精读时最值得记住的五点

1. Dual-Level不是两个encoder，而是高层语义规划和低层连续控制；
2. GPT-4o和Octopi都不训练，RDT也不接收触觉；
3. RDT确实做了机器人和任务fine-tune，只是没有为触觉修改其内部结构；
4. Controller的专家动作来自kinesthetic teaching，同一轨迹上的RDT动作通过离线推理补充生成；
5. Interpolant相对Residual的核心优势，是从RDT动作分布向专家动作分布进行生成式修正，而不是只回归一个容易平均化的确定性残差。

## 18. 原文定位

| 内容 | 原文位置 |
| --- | --- |
| 整体双层架构 | Figure 1、Introduction |
| GPT-4o Planner与Octopi | Section 3.1 |
| Interpolant Controller | Section 3.2、Algorithm 1 |
| 任务与baseline | Section 4.1 |
| 高层规划实验 | Section 4.2 |
| RDT、Residual、Interpolant对比 | Section 4.3、Table 1 |
| 双层反馈消融 | Section 4.4、Table 2 |
| 去触觉/去视觉消融 | Section 4.5 |
| Octopi描述与连续分数比较 | Appendix B |
| 数据采集、RDT fine-tune与Controller训练 | Appendix C |
