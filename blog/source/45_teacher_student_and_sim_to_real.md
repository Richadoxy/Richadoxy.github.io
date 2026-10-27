# Teacher–Student：策略蒸馏、特权信息与部署

> 阅读位置：[灵巧手 RL 研究地图](43_dexterous_rl_overview.md) → [RL 基础与训练闭环](44_rl_basics_and_training_loop.md) → 本文。这里讨论控制策略的 teacher–student；表示学习中的 EMA teacher 和生成模型的采样蒸馏，需要按各自目标另外理解。

## 0. Teacher–student 是框架，teacher 可以怎样获得

**Teacher–student 描述谁提供学习目标、谁学习这些目标，不限定 teacher 的求解算法。** Teacher 可以是通过 PPO、SAC 或其他合适方法训练的策略；也可以是规划器、轨迹优化器、已有控制器或多个专家组成的集合。

Student 也不一定使用 RL。常见流程是：

~~~text
阶段 A：任务奖励 / 参考 + 环境交互 → RL teacher
阶段 B：teacher 输出的监督 + student 观测 → 模仿学习 student
阶段 C：可选的 RL 微调或在线适应 → 部署
~~~

Teacher 不必比 student 参数更多，也不必始终更强；它可能只是掌握更丰富的训练信息，或者只擅长一个任务。[Policy Distillation](https://arxiv.org/abs/1511.06295) 展示了策略压缩与多任务专家整合，说明这是一种知识迁移组织方式。

### 算法、目标和角色可以自由组合到什么程度

| Teacher 来源 | 提供什么 | 需要注意 |
| --- | --- | --- |
| PPO 等 on-policy RL | 经交互优化的动作或动作分布 | rollout、奖励和价值估计要适配任务 |
| SAC 等 off-policy RL | 可复用经验训练的连续控制策略 | replay 覆盖和价值估计质量影响结果 |
| 示范辅助 RL | 经过 BC 初始化或模仿正则的策略 | 示范和任务奖励的作用要分开 |
| Model-based 学习与 MPC | 利用动力学预测求出的动作 | 规划耗时、模型误差与状态需求 |
| 离线训练得到的策略 | 固定数据支持范围内的行为 | 不能假定未覆盖状态上也能正确示范 |
| 手工控制器／轨迹优化器 | 规则或优化生成的动作标签 | 这些来源不一定属于 RL |
| 多个 specialist | 针对不同目标、物体或参考的专家动作 | 必须给 student 足够的任务条件以消除冲突 |

[PPO](https://arxiv.org/abs/1707.06347) 与 [SAC](https://arxiv.org/abs/1801.01290) 规定不同的策略优化方法；二者都没有把“必须作为 teacher”写进定义。[PDDM](https://arxiv.org/abs/1909.11652) 提供了学习动力学与 MPC 的灵巧控制例子；将其输出作为蒸馏标签在接口上可行，不表示原论文一定采用了 teacher–student。

**算法层面可替换，不表示工程上直接换一个名字就能运行。** 数据收集方式、动作分布、状态输入、奖励尺度和优化器配置都需要重新核对。

## 1. 为什么灵巧手常需要这个框架

仿真容易提供物体位姿、接触、速度和物理参数。先用这些信息获得可靠行为，可以把“动作怎样才有效”和“传感器怎样支持动作”分阶段解决。

设：

- \(o_t\)：部署可获得的观测；
- \(p_t\)：只在训练中可获得的特权信息；
- \(h_t\)：由观测与过去动作构成的历史；
- \(g\)：目标或任务条件。

一种特权动作蒸馏接口是：

\[
a_t^T\sim\pi_T(a_t\mid o_t,p_t,g),\qquad
a_t^S\sim\pi_S(a_t\mid h_t,g).
\]

Teacher 可以使用 \(p_t\)，student 的输入不包含它。训练时可以用 \(p_t\) 为 student 制造标签，但部署时所需输入必须实际存在。

[DextrAH-RGB](https://arxiv.org/abs/2412.01791) 先在仿真训练特权控制策略，再将其蒸馏为 RGB 输入策略，是真实感知接口转换的一个代表案例。

这与单纯压缩模型不同：同观测蒸馏主要改变模型或计算量，特权蒸馏还改变了可获得信息。Student 可能需要记忆，或者使用不同于 teacher 的动作策略才能在部分可观测条件下完成任务。

## 2. Student 可以学动作，也可以学其他目标

### 2.1 动作监督

对确定性连续动作，一个简单目标为：

\[
\mathcal L_{\mathrm{action}}
=\mathbb E_{(h_t,g,a_t^T)\sim D}
\left[\|\mu_S(h_t,g)-a_t^T\|^2\right].
\]

这里 \(D\) 可以来自 teacher rollout，也可以来自 student 访问过、再由 teacher 标注的状态。若使用 teacher 的动作均值作为标签，应明确记录，避免混淆随机探索动作与部署动作。

同一组标签也可以用于训练条件概率模型。使用 Diffusion Policy 或 flow action head，不会自动把 student 的训练变成 RL；训练目标仍可能完全来自动作示范。相关背景见 [动作生成方法](25_action_generation_families.md)。

### 2.2 动作分布蒸馏

若 teacher 与 student 都输出可比较的动作分布，可以使用分布差异作为监督：

\[
\mathcal L_{\mathrm{distill}}
=\mathbb E_D\left[
D_{\mathrm{KL}}\left(
\pi_T(\cdot\mid o_t,p_t,g)
\;\|\;
\pi_S(\cdot\mid h_t,g)
\right)
\right].
\]

KL 的方向、分布参数化和尺度要明确；不是所有生成式策略都有方便计算的显式动作密度。不同动作空间之间也不能直接计算这个 KL。

### 2.3 潜变量或状态估计监督

可以让 teacher 的特权分支形成环境潜变量 \(z_t\)，再训练历史编码器预测它：

\[
z_t=E_T(p_t),\qquad
\hat z_t=A_\phi(h_t),\qquad
\mathcal L_z=\|\hat z_t-z_t\|^2.
\]

部署时把 \(\hat z_t\) 输入底层策略，而不一定复制整个 actor。[RMA](https://arxiv.org/abs/2107.04034) 是腿足机器人中的代表结构；[HORA](https://haozhi.io/hora/) 将快速适应思路用于手内物体旋转。引用前者是解释框架来源，不把它算作灵巧手实验。

此时学的是可用于控制的隐状态或环境属性表示；它不必对应每个物理参数的精确估计。

## 3. BC 与 DAgger：训练状态分布从哪里来

直接 BC 常在 teacher 访问的状态上学习。部署时 student 的小误差会改变下一步观测，逐渐进入训练集没覆盖的区域。

DAgger 的核心是让训练数据逐步覆盖学习者自身造成的状态：

~~~text
1. 用 student 或 teacher/student 混合策略执行
2. 在访问到的状态上查询 teacher 的动作标签
3. 将这些样本加入聚合数据集
4. 更新 student，继续收集
~~~

动作监督仍来自 teacher，而不是直接最大化环境 reward，所以 DAgger 属于交互式模仿学习。它需要能够在学习者访问的状态上查询有效专家。[DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html)

在仿真中，同一时刻的完整状态和传感器观测可以配对：teacher 读真值，student 读渲染图像或触觉，监督对应同一次物理状态。真机上若无法获得 teacher 需要的隐藏状态，就不能假定这个查询接口仍然可用。

## 4. Critic 不等于动作 teacher

这是最容易混淆的两个框架：

| 项目 | 非对称 actor–critic | 特权 teacher–student |
| --- | --- | --- |
| 谁读取特权信息 | Critic 可以读取 | Teacher 可以读取 |
| 谁输出动作 | Actor | Teacher 与 student 各自可以输出 |
| 主要学习信号 | 价值估计与 RL 策略更新 | 动作、分布或潜变量监督 |
| 是否必须单独训练动作专家 | 不需要 | 需要某种能提供目标的专家来源 |
| 部署通常保留 | 可部署观测 actor | Student，或历史估计器加底层策略 |

非对称 actor–critic 让 actor 只读图像等可部署观测，critic 在训练中利用完整状态估值；并不是让 actor 回归 critic 给出的专家动作。[Asymmetric Actor Critic](https://arxiv.org/abs/1710.06542)

二者也能组合。例如先用 actor–critic 算法训练特权 teacher，再训练视觉 student；或让 student 在蒸馏之后使用非对称 actor–critic 做 RL 微调。应按阶段分别画图。

## 5. 训练图、梯度图和推理图

下面是一个通用的“两阶段动作蒸馏”示例，不代表所有论文都冻结同样的模块。

~~~text
阶段 A：teacher 学习
特权状态 + 目标 → teacher actor → 环境 → reward
                                      ↓
                  teacher critic / RL 更新

阶段 B：student 学习
同一环境状态 ──→ 冻结 teacher → 动作标签 ─┐
           └──→ 部署观测 + 历史 → student ─┴→ imitation loss

部署
真实传感器 + 目标 + 必要历史 → student → 底层控制器
~~~

| 阶段／损失 | 主要更新谁 | 哪些组件通常不需要更新 |
| --- | --- | --- |
| Teacher RL actor loss | Teacher actor；共享 encoder 是否更新取决于实现 | 尚未参与训练的 student |
| Teacher value loss | Critic；可能还有共享 encoder | 无关的 student |
| Student imitation loss | Student 的 encoder、记忆和动作头 | 作为固定标签源的 teacher |
| Student 后续 RL 微调 | 被选中的 student 参数和其 critic | 可保持冻结的 teacher／基础模块 |

推理时是否需要参考轨迹，由策略的条件输入决定；是否需要 critic，由部署算法决定。普通策略网络往往不带 critic，在线规划或价值引导系统则可能保留价值模型。

## 6. Tracking 与 teacher–student 怎样组合

Tracking 决定 teacher 要完成什么，不决定 teacher 怎样获得，也不决定 student 怎么训练。

### 示例 A：目标奖励 teacher → 感知 student

~~~text
目标姿态奖励 + PPO + 特权状态
                ↓
             teacher
                ↓ 动作监督 / DAgger
关节历史 + 视觉或触觉 → student
~~~

若换成 SAC，需要改变经验采集和 RL 更新实现，但外层 teacher–student 关系可以保持。

### 示例 B：参考跟踪 teacher → 参考条件 student

~~~text
人类手物参考 → 重定向 / 物体目标
                         ↓ tracking reward
                 RL tracking teacher
                         ↓ 蒸馏
部署观测 + 可获得的参考 → student tracker
~~~

Student 仍然读取参考时，系统还必须提供参考生成、选择和进度机制。把跟踪器蒸馏成一个网络，并没有自动获得“仅输入语言就能执行所有任务”的能力。

### 示例 C：直接训练可部署 actor

~~~text
视觉 / 触觉 → actor → 动作
特权状态   → critic → RL 训练信号
~~~

这是非对称 actor–critic 的一种安排，可以不经过独立 teacher–student 阶段。是否分阶段，取决于可学性、感知成本和实验目的。

## 7. 三个决定蒸馏能否成功的条件

### 7.1 Student 能否从输入中确定合理动作

假设相同观测可能对应“物体向左滑”和“向右滑”，teacher 因能读真值而输出相反的修正动作。没有足够历史或传感器信息的 student 无法区分两种状态；动作 MSE 可能学出不合适的平均值。

增加数据和网络规模不一定解决信息缺失。需要改善观测、引入历史或主动感知，或允许 student 学习更适合自身信息条件的行为。Teacher 的观测更强，也意味着它可能采用 student 无法复现的决策。

### 7.2 动作接口是否相同

需要核对关节顺序、坐标系、绝对／增量目标、归一化、控制周期、动作延迟和底层增益。Teacher 输出力矩、student 输出位置目标时，不能直接做动作 MSE，需要定义合理转换或换用其他监督。

跨手型还涉及不同运动学和接触能力。简单动作蒸馏不会自动消除 embodiment gap。接口背景见 [机器人数据与动作语义](24_robot_data_action_semantics.md)。

### 7.3 Teacher 能否在 student 的失败附近继续指导

只保存成功轨迹适合训练初始策略，但可能缺少滑移、偏离和恢复动作。需要评价专家在这些状态是否仍可靠，再决定是否采用 DAgger、扰动采集或后续 RL 微调。

“Teacher 在自己的 rollout 上表现好”与“能纠正任意 student 状态”是两个能力。

## 8. Online adaptation 不一定在线更新权重

| 机制 | 部署时改变什么 |
| --- | --- |
| 历史编码／循环记忆 | Hidden state 或环境 latent |
| 系统参数在线估计 | 显式参数估计 |
| 在线 RL／梯度微调 | Actor、critic 或部分网络权重 |
| 高层重规划 | 当前目标、参考或技能选择 |

HORA 的历史适应不应简单理解成“真机上继续运行 PPO”。同样，使用新触觉更新记忆，也不等于在部署时训练网络。

Sim-to-real 还需要处理动力学、视觉、触觉和执行器差异。蒸馏解决了知识或接口迁移问题，不自动消除所有仿真差距。

## 9. 与触觉数据生成路线的关系

RL teacher 可以生成机器人技能与监督，随后用于触觉策略。不同的数据阶段应分开记录：

| 数据阶段 | 能提供什么 | 需要验证什么 |
| --- | --- | --- |
| 仿真 teacher rollout | 动作、物体真值、接触及任务标签 | 行为是否有效，是否覆盖恢复状态 |
| 仿真传感器配对 | 与动作同步的渲染视觉或触觉近似 | 是否表达真实传感器的空间、噪声和时序 |
| 受控真机执行／闭环迁移 | 真实触觉、真实命令和真实结果 | 策略输入是否可获得，行为是否仍成功 |
| 部署 student rollout | 自身观测分布下的控制结果 | 独立成功率、掉落、误操作和扰动恢复 |

仿真接触力不能直接冒充真实 taxel 或视触觉图像；特权 teacher 也不能在缺少隐藏状态的真机上直接照搬。需要可部署状态估计、student 或其他经过验证的执行方式。

对于“保持物体 A，同时操作 B”，teacher 可以使用物体真值训练任务技能，而 student 使用视觉身份、全掌触觉和历史。最终要验证对象级接触记忆是否在相同传感器和控制频率下优于全局记忆，不能仅凭 teacher 成功率支持表示贡献。相关实验见 [触觉记忆的实验设计](41_tactile_memory_experiment_design.md)。

## 10. 读论文时记录四个独立答案

1. **Teacher 怎样得到？** 算法、奖励、参考、观测和训练预算。
2. **Student 学什么？** 动作、分布、latent、价值或组合目标；采用什么数据分布。
3. **信息怎样迁移？** 哪些模块共享、冻结、重训，监督能更新哪些参数。
4. **部署运行什么？** 输入能否获得，是否仍需 reference、teacher、planner 或在线更新。

只写“我们采用 teacher–student”不足以描述一个可复现方法。这四个答案才决定它如何与 PPO、SAC、tracking、视觉触觉感知和真实控制组合。
