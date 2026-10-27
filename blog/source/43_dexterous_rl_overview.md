# 灵巧手强化学习：研究范式与学习地图

> 文献核查：2026-10-27。本文按任务、监督来源、学习算法、训练框架和部署接口组织研究地图；代表论文不按引用量排名，近期预印本单独标注。对“多数论文采用什么”只作路线分析，不给出未经统计的比例。

> 本卷顺序：本文 RL01 → [RL 基础与训练闭环](44_rl_basics_and_training_loop.md) RL02 → [Teacher–Student 与部署](45_teacher_student_and_sim_to_real.md) RL03。已有 [VLA 统一研究范式](27_vla_overview.md) 与 [模仿学习基础](23_imitation_learning_basics.md) 分别提供多模态策略和示范学习的背景。

## 0. 先回答两个核心问题

**灵巧手 RL 不是只有“tracking 一条专家数据”和“手调 reward 从零训练”两种选择。** Tracking 本身可以通过 RL 实现：参考轨迹提供目标状态，奖励衡量实际状态与参考的差距，策略通过环境交互寻找能实现这些状态的动作。

**Teacher–student 是组织知识迁移的框架，不是与 PPO、SAC 并列的 RL 算法。** Teacher 可以通过不同 RL 算法训练，也可以来自规划、优化或已有控制器。Student 可以通过动作监督、分布蒸馏、DAgger 或结合 RL 的方式学习。具体选择必须满足观测、动作和数据接口，不能理解成任意实现都能无修改替换。

因此下面这句话完全成立：

> 用 PPO 优化物体轨迹跟踪奖励，得到特权状态 teacher；再用 DAgger 训练只读取视觉、触觉和关节历史的 student。

其中 PPO 是算法，tracking 是奖励与任务定义，teacher–student 是训练框架，DAgger 是交互式模仿学习的数据收集与聚合方法，视觉／触觉是部署观测。这是一个组合示例，不是某篇论文的完整复述。

## 1. 不同术语分别处于哪一层

| 层级 | 回答的问题 | 常见选择 |
| --- | --- | --- |
| 任务 | 要获得什么能力？ | 抓取、手内重定向、工具使用、关节物体操作、双手协作 |
| 任务信号与先验 | 什么算做好，怎样帮助找到成功行为？ | 目标奖励、参考状态、接触约束、示范动作、初始抓姿 |
| 学习算法 | 参数怎样由经验更新？ | PPO、SAC、基于模型的学习方法 |
| 策略结构 | 怎样从输入产生动作？ | MLP、循环策略、Transformer、分层或 residual 策略 |
| 训练组织 | 哪些阶段和模型协作？ | BC 后 RL、teacher–student、specialist–generalist、课程学习 |
| 部署接口 | 真机能看到什么、实际执行什么？ | RGB、触觉、关节历史；关节目标、力矩或底层控制器目标 |

这几层能够组合。比如 goal-conditioned 表示策略以目标为条件，不限定算法；residual 表示输出修正量，也不自动等于 RL。

还要区分任务难度：生成一个静态抓姿、执行抓取、持续换指重定向、完成工具操作，是不同能力。抓姿生成和纯模仿学习论文可以给 RL 提供组件，但不能仅凭名称中有“Dex”就归为 RL 工作。

## 2. 一篇灵巧手 RL 论文的完整流程

~~~text
任务与物理环境
手型、物体、接触、初始状态、成功条件
                 ↓
目标与先验
reward / reference / demonstrations / controller
                 ↓
观测、状态表示与动作接口
关节历史、视觉、触觉；关节目标、力矩、技能或修正量
                 ↓
交互训练
算法 + 探索 + curriculum + reset 分布
                 ↓
可选的策略整合
多专家整合 / 特权信息蒸馏 / 感知适配
                 ↓
部署与独立评测
真机执行、泛化、扰动恢复、失败重试、在线适应
~~~

与 VLA 地图相比，RL 需要额外关注环境、奖励、探索和初始化分布。使用相同的 PPO，改变这些部分，也可能得到完全不同的可学性。

仿真提供真实物体位姿、接触和物理参数，但这些量是否进入最终 actor，必须单独记录。能在特权状态下完成任务，不代表已经得到可部署的视觉／触觉策略。

## 3. 奖励与示范怎样进入学习

### 3.1 目标奖励 RL：让机器人寻找完成目标的方法

例如希望物体持续绕指定轴旋转，可以奖励目标方向角速度，惩罚掉落、偏移和过大动作；不必指定每根手指在每一刻的运动。

代表路线：[Dactyl](https://arxiv.org/abs/1808.00177) 不依赖人类示范，通过仿真 RL 和域随机化实现真机重定向；[HORA](https://haozhi.io/hora/) 进一步研究使用本体感知历史适应不同物体。

这类任务的最终目标容易定义，但奖励仍不能自动解决探索。初始抓持、物体分布、课程和训练规模都可能很关键。

### 3.2 示范辅助 RL：把示范当作探索与策略先验

示范可以用于：

- BC 预训练，使初始策略有机会接近成功状态；
- 在 RL 更新时保留模仿项，减少有用行为过早丢失；
- 提供 replay 数据、初始状态或阶段目标。

[DAPG](https://arxiv.org/abs/1709.10087) 的代表性做法是 BC 初始化，再结合示范项进行 policy gradient 更新。这里示范帮助寻找成功行为，任务回报仍提供改进方向；不要求部署时逐帧输入一条专家轨迹。

### 3.3 Tracking RL：用参考状态规定交互过程

可以用下面的示意奖励统一理解目标与参考：

\[
r_t=w_g r_{\mathrm{task}}(s_t,g)
    +w_r r_{\mathrm{ref}}(s_t,\hat{s}_{\phi_t})
    -w_c c(s_t,a_t).
\]

其中 \(g\) 是任务目标，\(\hat{s}_{\phi_t}\) 是当前参考进度对应的状态，\(c\) 是代价。各方法可以只采用部分项；这不是某篇论文的原始奖励。

跟踪对象可以是手关节、指尖、物体位姿、关节物体开合量、接触关系或关键阶段。物体中心的参考允许机器人用不同于人手的动作实现相同效果。[Human2Sim2Robot](https://proceedings.mlr.press/v305/lum25a.html) 从单个人类 RGB-D 示范提取物体轨迹构造奖励，并用操作前手姿态辅助探索。

| 容易混淆的说法 | 应怎样理解 |
| --- | --- |
| 一条示范就能学 | 一条示范可以定义任务；训练仍可能使用大量仿真交互 |
| 跟踪人类视频 | 视频中的手物状态通常不是机器人专家动作标签 |
| Tracking 就是 BC | BC 学观测到专家动作；tracking RL 通过交互学习实现参考状态的动作 |
| 有反馈就不再是 tracking | 跟踪器可以持续读取实际观测；固定参考也能配合闭环控制 |
| 一个 tracker 就是通用技能 | 要区分单轨迹专用策略、多参考策略和未见参考泛化 |
| 训练中用过示范，推理也需要 | 取决于具体接口；reference-conditioned tracker 与 BC 初始化策略不同 |

阅读时必须查清：部署是否仍需要参考、参考从哪里来、如何推进阶段，以及扰动后是否能调整参考或恢复。

### 3.4 基于模型的方法：先学习动作怎样改变状态

Model-based 方法显式学习或使用动力学模型，再用于预测、规划或策略改进。例如 [PDDM](https://arxiv.org/abs/1909.11652) 将学习到的动力学与在线 MPC 结合，研究真机灵巧操作的数据效率。

它与目标奖励或示范参考可以组合；“使用仿真训练”不自动等于 model-based RL。PPO 在物理仿真中采样，但并不因此在算法内部学习动力学模型。

## 4. 可以与上述方式组合的训练框架

| 框架 | 主要解决什么 | 关键边界 |
| --- | --- | --- |
| Teacher–student | 压缩策略、转换观测接口、整合技能 | 不规定 teacher 必须使用哪种 RL 算法 |
| Specialist–generalist | 将不同任务或物体专家的能力整合进通用策略 | 不保证自动获得未见任务泛化 |
| Curriculum | 从容易的探索、物体或阶段逐步扩大难度 | 最终测试要撤掉仅训练时存在的辅助 |
| 分层／技能原语 | 高层给目标，低层处理精细接触 | 高层任务能力与低层技能能力需分别评价 |
| Residual learning | 在基础控制器或策略上学习修正 | 可以用 RL，也可以用监督学习 |
| Sim-to-real 与适应 | 处理视觉、传感器、动力学和执行器差异 | 域随机化、系统辨识与在线适应不是同一操作 |

[UniDexGrasp++](https://arxiv.org/abs/2304.00464) 组合几何课程与 generalist–specialist 学习；[DextrAH-RGB](https://arxiv.org/abs/2412.01791) 把特权 RL 策略蒸馏为 RGB 策略；[DexGen](https://arxiv.org/abs/2502.04307) 则先用 RL 获得底层操作原语，再训练可由粗粒度人类命令驱动的控制器。

Teacher–student 的动作、潜变量、梯度与部署分工，放在 [RL03](45_teacher_student_and_sim_to_real.md) 单独展开。

## 5. 代表论文与路线时间线

年份按主要发表或公开时间组织；跨年工作给出范围。下表用于理解路线，不比较不同硬件和评测协议下的成功率。

| 时间 | 工作 | 阅读重点 |
| --- | --- | --- |
| 2018，RSS；预印本 2017 | [DAPG：Learning Complex Dexterous Manipulation…](https://arxiv.org/abs/1709.10087) | 示范初始化与 policy gradient 结合；原论文实验主要在仿真 |
| 2018，预印本 | [Learning Dexterous In-Hand Manipulation／Dactyl](https://arxiv.org/abs/1808.00177) | 无人类示范的 RL、域随机化和真实手操作 |
| 2019，预印本 | [Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) | 自动域随机化 ADR、记忆和适应；物理操作与魔方求解规划需区分 |
| 2019，预印本 | [PDDM：Deep Dynamics Models…](https://arxiv.org/abs/1909.11652) | 学习动力学与在线 MPC，关注真实交互数据效率 |
| 2022，CoRL | [HORA](https://haozhi.io/hora/) | 用本体感知历史进行物体属性适应 |
| 2023 | [UniDexGrasp](https://arxiv.org/abs/2303.00938) / [UniDexGrasp++](https://arxiv.org/abs/2304.00464) | 抓姿目标、课程、专家与通用策略整合 |
| 2023，CVPR | [DexArt](https://arxiv.org/abs/2305.05706) | 关节物体任务、点云表示和未见物体泛化 |
| 2023，RSS | [Rotating without Seeing](https://arxiv.org/abs/2303.10880) | 覆盖手掌和手指的接触感知，以及触觉 sim-to-real |
| 2023–2024 | [Robot Synesthesia](https://arxiv.org/abs/2312.01853) | 以点云形式组织视触觉空间信息，改善手内操作 |
| 2023，预印本 | [Eureka](https://arxiv.org/abs/2310.12931) | 自动生成和迭代奖励代码；手部转笔展示在仿真中 |
| 2024–2025 | [DextrAH-RGB](https://arxiv.org/abs/2412.01791) | 特权 RL teacher 到可部署 RGB student |
| 2025，ICLR | [DexTrack](https://meowuu7.github.io/DexTrack/) | 结合 RL、IL 与成功跟踪数据迭代，训练可泛化的 tracker |
| 2025，CoRL | [Human2Sim2Robot](https://proceedings.mlr.press/v305/lum25a.html) | 单个人类视频提供物体轨迹奖励与探索先验 |
| 2025，公开版本 | [DexMachina](https://arxiv.org/abs/2505.24853) | 逐步减弱虚拟物体辅助控制，学习双手功能重定向；该版本主要验证仿真能力 |
| 2025 | [DexterityGen／DexGen](https://arxiv.org/abs/2502.04307) | RL 原语数据与基础控制器，连接粗粒度命令和精细操作 |
| 2025 | [Scaffolding Dexterous Manipulation](https://arxiv.org/abs/2506.19212) | VLM 生成粗略手物参考，residual RL 学习执行 |

### 2026 年的进展：与经典路线分开阅读

| 工作与状态 | 值得跟踪的问题 |
| --- | --- |
| [AdaDexTrack，CVPR 2026](https://janebek.github.io/AdaDexTrack/) | 在通用 tracker 周围加入 RL 调制器，反馈修正参考、物体表示和执行目标 |
| [SimToolReal，RSS 2026](https://www.roboticsproceedings.org/rss22/p151.html) | 用程序生成的工具类物体训练统一目标位姿策略，减少逐工具和逐任务训练 |
| [DexMimic，2026-10 预印本](https://arxiv.org/abs/2610.19089) | 多参考联合训练与 value 引导的 reset 分配 |
| [Towards High-DoF Dexterous Manipulation through VLA Post-Training，2026-09 预印本](https://arxiv.org/abs/2609.19666) | 动作编码、SFT、DAgger 与真机 latent residual RL 的组合 |

这些近期工作说明研究正在向通用 tracking、反馈调制和真机适配推进，不能据此认定它们已经形成与经典论文相同程度的影响。

## 6. 怎样理解当前活跃方向

从上述代表工作可归纳出以下并行问题。这是研究地图，不是论文数量占比。

1. **目标条件通用技能：** 从单个物体旋转扩展到跨物体、跨目标的可复用能力。
2. **人类参考到可执行交互：** 处理形态差异、接触不可行、阶段错位和参考噪声。
3. **多专家到通用策略：** 减少每个物体、每条轨迹训练一套策略的成本。
4. **视触觉与历史状态：** 在遮挡、滑移和部分可观测条件下持续估计接触。
5. **减少人工任务定制：** 自动奖励、参考生成、课程和 reset 分配。
6. **高层目标与低层接触控制：** 将 VLA、人类命令或规划接入可复用底层技能。
7. **真机数据效率与恢复：** 适应仿真误差，减少训练失败成本，支持失败后的继续操作。

这几项可以同时出现在一篇论文中。例如感知创新可以配合现成 PPO 和 teacher–student，而无需提出一个新的 RL 更新公式。

## 7. Contribution 常见地落在哪里

| 创新位置 | 可提出的问题 | 至少需要什么证据 |
| --- | --- | --- |
| 任务与奖励表示 | 物体效果、接触或关键阶段是否比关节轨迹更易迁移？ | 跨任务或跨手型结果，减少任务定制的记录 |
| 探索与课程 | 如何找到首次成功，覆盖困难阶段？ | 同交互预算、多随机种子、学习曲线 |
| 参考与重定向 | 如何把人类运动变成物理可行的机器人行为？ | 任务成功、物理可行性与跟踪误差分别报告 |
| 感知与记忆 | 哪些信息在遮挡后仍能支持动作？ | 同传感器、同历史、同控制频率的闭环对照 |
| 动作空间与控制结构 | 技能空间、分层或 residual 是否更易学？ | 相同预算下的效率、精度与稳定性 |
| 通用策略整合 | 怎样保留多个专家能力并泛化？ | 已见任务保持与未见物体／轨迹测试 |
| Sim-to-real | 哪种建模或适应改善真实执行？ | 真机重复试验、扰动、失败类型与适配成本 |
| 数据与 benchmark | 哪些能力被现有协议忽略？ | 清晰划分、强 baseline、可复现的评测 |

“用了 PPO”“增加触觉”“换成 Transformer”本身只是组件选择。方法贡献需要说明它改变了哪个瓶颈，并由相应层级的证据支持。训练 reward 上升不自动证明任务成功；表示预测更准也不自动证明控制更好。

## 8. 推荐学习路线

### 路线 A：从模仿学习进入 RL

| 阶段 | 阅读 | 学完应能回答 |
| --- | --- | --- |
| 1 | [BC 基础](23_imitation_learning_basics.md) → [RL02](44_rl_basics_and_training_loop.md) | 专家动作标签与交互回报如何分别更新策略？ |
| 2 | DAPG → Dactyl | 示范辅助探索与无示范 RL 有什么区别？ |
| 3 | HORA → [RL03](45_teacher_student_and_sim_to_real.md) | 特权信息如何参与训练，部署怎样适应？ |
| 4 | UniDexGrasp / ++ → DextrAH-RGB | 多物体泛化与感知部署分别怎样获得？ |

### 路线 B：研究 tracking 与复杂操作

先理解 RL02 的 reward、reset 与 curriculum，再读 DexTrack → Human2Sim2Robot → DexMachina，最后看 AdaDexTrack 与 DexMimic。每篇记录“参考什么、谁生成参考、一个策略覆盖多少参考、部署时需要什么”。

### 路线 C：研究触觉与灵巧手

Rotating without Seeing → Robot Synesthesia → [触觉输入表示](39_tactile_input_representation.md) → [对象级触觉记忆](40_policy_memory_and_object_centric_tactile_state.md) → [实验设计](41_tactile_memory_experiment_design.md)。

这一分支关注传感器信息如何改善决策。RL teacher 可以提供技能或监督，但仿真接触力不等于真实触觉读数，特权物体状态也不能直接成为部署输入。

### 路线 D：连接 VLA 与底层控制

先读 [VLA 研究地图](27_vla_overview.md)、[动作接口](24_robot_data_action_semantics.md) 与 [RL03](45_teacher_student_and_sim_to_real.md)，再读 DexGen、Scaffolding 和上述 VLA post-training 预印本。重点分清高层条件、低层动作、各阶段训练目标和实际反馈频率。

## 9. 读新论文时填写这张卡片

~~~text
Task：任务、手型、对象和成功条件
Observation：actor / critic / teacher / student 分别看到什么
Action：输出语义、控制频率和底层控制器
Signal：任务 reward、参考 reward、模仿或辅助 loss
Algorithm：每个阶段使用什么优化算法
Data：示范来源、交互预算、reset 和课程
Transfer：整合什么知识、采用什么蒸馏接口
Inference：部署保留哪些模块，是否还需要参考
Evidence：仿真、真机、未见对象、未见任务和恢复能力
Contribution：真正改变了哪一层，哪些只是沿用组件
~~~

例如，若论文只有“特权 teacher 成功率很高”，它尚未证明真实触觉 student 的闭环能力。若一个策略只跟踪训练过的轨迹，也不能仅凭任务数量称为未见技能泛化。保持这张卡片中的各列分开，才能比较不同研究路线。
