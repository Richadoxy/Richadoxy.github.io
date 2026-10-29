# Teacher–Student：策略蒸馏、特权信息与部署

> 阅读位置：[灵巧手 RL 研究地图](43_dexterous_rl_overview.md) → [RL 通识与 PPO 训练闭环](44_rl_basics_and_training_loop.md) → 本文 → [HORA 精讲](46_hora_paper_walkthrough.md)。RL02 说明单个 actor–critic 怎样从交互中学习；本文说明不同模型、输入与训练阶段之间怎样传递知识，RL04 再代入 HORA 的具体网络和实验。

## 0. 训练角色与求解算法

**Teacher–student 描述谁提供学习目标、谁学习这些目标，不限定 teacher 的求解算法。** Teacher 可以是通过 PPO、SAC 或其他合适方法训练的策略；也可以是规划器、轨迹优化器、已有控制器或多个专家组成的集合。

“Teacher”是它在后续知识迁移中的角色，不意味着它在初始化时已经会完成任务。一个从随机权重开始的 actor，可以先靠环境奖励训练，再成为 student 的 teacher。特权信息给它更完整的输入，但不自动给出正确动作标签；它自己的 actor loss 和 value loss 仍按 [RL02](44_rl_basics_and_training_loop.md) 的交互闭环产生。

Student 也不一定使用 RL。常见流程是：

~~~text
阶段 A：任务奖励 / 参考 + 环境交互 → RL teacher
阶段 B：teacher 输出的监督 + student 观测 → 模仿学习 student
阶段 C：可选的 RL 微调或在线适应 → 部署
~~~

Teacher 不必比 student 参数更多，也不必始终更强；它可能只是掌握更丰富的训练信息，或者只擅长一个任务。[Policy Distillation](https://arxiv.org/abs/1511.06295) 展示了策略压缩与多任务专家整合，说明这是一种知识迁移组织方式。

### 算法、目标与角色的组合边界

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

域随机化（Domain Randomization，DR）又处在另一层：它规定训练时怎样采样质量、摩擦、尺寸、传感器噪声等环境条件。直接输入状态的 MLP 可以配合 DR，带历史适应器的策略同样可以配合 DR；是否有特权 Encoder、是否做 latent 蒸馏，都不能单独判断一个方法是不是 DR。把“PPO + DR + teacher–student”写在一起，是算法、环境分布和训练框架的组合。

## 1. 从特权观测到部署观测

仿真容易提供物体位姿、接触、速度和物理参数。先用这些信息获得可靠行为，可以把“动作怎样才有效”和“传感器怎样支持动作”分阶段解决。

本文采用以下通用记号。不同论文可以使用不同字母；HORA 将特权信息记作 \(e_t\)，精讲中会给出逐项对应。

| 符号 | 含义 |
| --- | --- |
| \(t\) | 机器人控制时间步 |
| \(o_t\) | 部署可获得的观测，例如关节读数、图像、触觉 |
| \(p_t\) | 只在训练时可获得的特权信息，例如仿真物体状态 |
| \(h_t\) | 由当前及过去观测、过去动作组成的历史或记忆 |
| \(g\) | 目标或任务条件；无显式条件的任务可以省略 |
| \(\pi_T,\pi_S\) | Teacher 与 student 的策略分布，参数分别记作 \(\theta_T,\theta_S\) |
| \(a_t^T,a_t^S\) | Teacher 与 student 输出的动作；上标 \(T,S\) 表示角色 |
| \(D\) | 用于监督训练的数据集或采样分布 |
| \(\mathbb E_D[\cdot]\) | 对从 \(D\) 采样的数据取平均 |
| \(\operatorname{sg}(\cdot)\) | Stop-gradient：保留数值，但不沿该分支反向传播 |

一种特权动作蒸馏接口是：

\[
a_t^T\sim\pi_T(\cdot\mid o_t,p_t,g),\qquad
a_t^S\sim\pi_S(\cdot\mid h_t,g).
\]

Teacher 可以使用 \(p_t\)，student 的输入不包含它。训练时可以用 \(p_t\) 为 student 制造标签，但部署时所需输入必须实际存在。

[DextrAH-RGB](https://arxiv.org/abs/2412.01791) 先在仿真训练特权控制策略，再将其蒸馏为 RGB 输入策略，是真实感知接口转换的一个代表案例。

这与单纯压缩模型不同：同观测蒸馏主要改变模型或计算量，特权蒸馏还改变了可获得信息。Student 可能需要记忆，或者使用不同于 teacher 的动作策略才能在部分可观测条件下完成任务。

### 1.1 信息的角色与可获得性是两个维度

“这是 goal”说明它要求系统完成什么；“这是 privileged information”说明在部署条件下能否获得它。两者不能互相替代。

| 量 | 在控制中的角色 | 可获得性需要核对什么 |
| --- | --- | --- |
| 当前关节角 | 当前观测 | 编码器能否测量，延迟和噪声是多少 |
| 仿真器给出的物体精确位姿 | 当前状态 | 真实部署是否有等价的可靠估计 |
| 用户指定的目标高度 | 任务目标 | 用户或任务系统可以直接提供 |
| 由可获得点云生成的目标抓姿 | 任务目标 | 生成器的全部输入与预处理是否可部署 |
| 目标相对当前物体的误差 | 派生控制条件 | 误差计算是否暗中读取物体真值 |

例如目标位置 \(\mathbf p_g\) 是已知的，但目标误差

\[
\Delta\mathbf p_t=\mathbf p_g-\mathbf p_t^{\mathrm{object}}
\]

还依赖当前物体位置 \(\mathbf p_t^{\mathrm{object}}\)。若它来自仿真器真值，\(\Delta\mathbf p_t\) 就继承了这个信息依赖。即使删掉名为 `object_state` 的输入块，也不能据此断定剩下的 `goal_delta` 不包含特权信息。

因此要从传感器或任务来源一路检查到坐标变换、误差特征和网络输入。网络框图上的标签描述语义，代码中的依赖链决定实际的信息合同。[UniDexGrasp 精读](47_unidexgrasp_paper_walkthrough.md) 将论文的理想接口与发布实现分开核对。

## 2. 监督目标决定 student 的学习内容

Teacher–student 不能只画成两个网络之间的一根箭头。箭头上传递的是动作、概率分布还是 latent，决定了损失比较什么，也决定 student 必须保留哪些模块。

### 2.1 动作监督

对确定性连续动作，一个简单目标为：

\[
\mathcal L_{\mathrm{action}}
=\mathbb E_{(h_t,g,a_t^T)\sim D}
\left[\|m_{\theta_S}(h_t,g)-\operatorname{sg}(a_t^T)\|_2^2\right].
\]

\(m_{\theta_S}\) 是 student 的确定性动作输出或动作分布均值，\(\|\cdot\|_2^2\) 表示动作向量各维误差平方之和。Teacher 动作作为固定标签，损失更新 student 参数 \(\theta_S\)。这里 \(D\) 可以来自 teacher rollout，也可以来自 student 访问过、再由 teacher 标注的状态。若使用 teacher 的动作均值作为标签，应明确记录，避免混淆随机探索动作与部署动作。

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

\(D_{\mathrm{KL}}(P\|Q)\) 是从分布 \(P\) 到 \(Q\) 的 KL 散度，\(\cdot\) 表示遍历动作空间。在此安排中 teacher 分布固定，student 调整 \(\theta_S\) 去接近它。KL 的方向、分布参数化和尺度要明确；不是所有生成式策略都有方便计算的显式动作密度。不同动作空间之间也不能直接计算这个 KL。

### 2.3 潜变量或状态估计监督

可以让 teacher 的特权分支形成环境潜变量 \(z_t\)，再训练历史编码器预测它：

\[
z_t=E_\eta(p_t),\qquad
\hat z_t=F_\omega(h_t),\qquad
\mathcal L_z=\mathbb E_D\left[\|\hat z_t-\operatorname{sg}(z_t)\|_2^2\right].
\]

这里 \(E_\eta\) 是参数为 \(\eta\) 的特权信息编码器，\(z_t\) 是它产生的环境表征；\(F_\omega\) 是参数为 \(\omega\) 的历史估计器，\(\hat z_t\) 是历史推断得到的表征。字母 \(F\) 表示估计模块，避免与 RL02 中的优势函数 \(A\) 混淆。

一种常见安排是先训练带特权分支的控制策略，再固定该策略和 \(E_\eta\)，只优化 \(F_\omega\)。部署时将 \(\hat z_t\) 输入已有的底层策略，未必再训练一套完整的动作 student。[RMA](https://arxiv.org/abs/2107.04034) 是腿足机器人中的代表结构；[HORA](https://arxiv.org/html/2210.04887v1#S3) 将快速适应思路用于手内物体旋转。引用前者是解释框架来源，不把它算作灵巧手实验。

此时学的是可用于控制的隐状态或环境属性表示，不要求每一维都有明确物理单位。如果训练目标改成真实质量、摩擦等参数，就变成显式参数估计这一设计选择；是否继续经过编码器，应查看该方法的实际接口。

| 监督目标 | Student 直接学习的输出 | 部署可能保留的结构 |
| --- | --- | --- |
| 动作 | 下一步命令或动作序列 | Student 策略 |
| 动作分布 | 各动作的相对倾向及不确定性 | Student 概率策略 |
| 环境 latent | 供控制器使用的表示 | 历史估计器 + 已有控制策略 |

这些目标可以组合，但“student”这个角色不能单独决定它一定输出动作，也不能决定它一定使用 PPO。

## 3. BC 与 DAgger：监督数据覆盖的状态

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

### 3.1 MSE 规定比较什么，DAgger 规定在哪里取得标签

同一个动作 MSE 可以在专家离线轨迹上训练，也可以放进 DAgger 循环。仅看到 \(\|a_t^S-a_t^T\|^2\)，无法判断数据是不是 DAgger 收集的。

下面是确定性动作策略的概念伪代码。\(k\) 表示第几轮采集，\(\beta_k\in[0,1]\) 表示该轮执行 teacher 动作的概率；`D` 保存历轮聚合数据。`teacher_action` 和 `student_action` 都读取当前同一物理时刻的信息，返回可直接执行的动作。

~~~python
D = []
for k in range(num_rounds):
    obs, privileged, goal = env.reset()
    for t in range(rollout_steps):
        with no_grad():
            label = teacher_action(obs, privileged, goal)
            action_s = student_action(obs, goal)
        D.append(copy_sample(obs, goal, label))

        # 混合是选择谁的动作；不是默认对两者动作向量求平均
        action = label if uniform_0_1() < beta[k] else action_s
        obs, privileged, goal, done = env.step(action)
        if done:
            obs, privileged, goal = env.reset()

    for obs_b, goal_b, label_b in minibatches(D):
        predicted = student_action(obs_b, goal_b)
        loss = squared_error(predicted, label_b.detach()).mean()
        optimizer_student.zero_grad()
        loss.backward()
        optimizer_student.step()
~~~

这里省略历史状态、归一化与并行环境；`copy_sample` 表示复制数据，避免仿真缓冲区后续更新覆盖旧样本。常见安排会逐渐降低 \(\beta_k\)，让数据越来越接近 student 自身的访问分布。Teacher 不随这段 student loss 更新，环境也不需要对动作可微。

聚合数据可以让后续训练继续覆盖过去版本 student 访问过的状态。实际系统可能采用有限缓冲区或其他采样近似，阅读时应记录其实现，而不是仅凭算法名称假定无限保存全部历史。[原始 DAgger 算法](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)

### 3.2 Latent 监督与交互式采集可以组合

将动作标签换成特权 Encoder 的 latent 标签，改变的是监督对象；让学习者执行还是让专家执行，改变的是状态分布。二者是独立的设计轴：

| 监督对象 | 固定专家数据上的监督 | 学习者访问状态上的监督 |
| --- | --- | --- |
| 动作 | 离线 BC | 专家动作查询；配合跨轮聚合可形成 DAgger |
| 环境 latent | 离线历史估计器拟合 | 在线收集历史并查询特权 latent；是否聚合需另查 |

因此，latent MSE 既不自动等于 DAgger，也不排斥借用数据聚合思想。原始 DAgger 讨论的是专家行为监督；对 latent 学习使用类似收集框架时，应说明扩展了什么，不能把原算法的保证不加条件地迁移过来。

[HORA v0.0.1 的适应代码](https://github.com/HaozhiQi/hora/blob/v0.0.1/hora/algo/padapt/padapt.py) 用估计 latent 的控制器交互，再以特权 latent 做监督，没有采用原始 DAgger 的跨轮数据集聚合循环。更准确的描述是 **on-policy latent supervision，即在当前适应策略访问的状态上学习 latent**；其中 on-policy 描述采样分布，不表示这一步在使用 PPO。UniDexGrasp 的动作蒸馏实例见 [对应精读](47_unidexgrasp_paper_walkthrough.md)。

## 4. Actor–Critic 与 Teacher–Student 的分工

Actor–critic 描述同一个 RL 训练过程中的决策与估值职责；teacher–student 描述跨模型或跨阶段的监督关系。Teacher 自己可以通过 actor–critic 训练，因此一套系统中可以同时存在 teacher actor、teacher critic 和 student。

Critic 的价值预测不是专家动作标签。它通过环境回报构造的目标学习估值，再帮助 actor 形成策略梯度；这一闭环在 [RL02](44_rl_basics_and_training_loop.md) 中展开。以下比较的是特权信息可以放置的两种位置：

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
特权输入 + 目标 → teacher actor → 环境 → 奖励与下一观测
特权输入 + 目标 → teacher critic → 价值预测
上述交互记录与预测 → RL 更新 → teacher 网络参数

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

这里的“固定 teacher”指其权重不参与这个阶段的更新，不表示每次输出相同的标签。输入状态变化，固定权重的 teacher 仍然可以输出不同动作；同样，固定权重的历史估计器也可以输出随历史变化的 latent。

推理时是否需要参考轨迹，由策略的条件输入决定；是否需要 critic，由部署算法决定。普通策略网络往往不带 critic，在线规划或价值引导系统则可能保留价值模型。

## 6. 将目标奖励、参考跟踪与蒸馏组合

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

## 7. 可部署蒸馏的三个条件

### 7.1 观测足以支持合理决策

假设相同观测可能对应“物体向左滑”和“向右滑”，teacher 因能读真值而输出相反的修正动作。没有足够历史或传感器信息的 student 无法区分两种状态；动作 MSE 可能学出不合适的平均值。

增加数据和网络规模不一定解决信息缺失。需要改善观测、引入历史或主动感知，或允许 student 学习更适合自身信息条件的行为。Teacher 的观测更强，也意味着它可能采用 student 无法复现的决策。

### 7.2 动作接口可以对应

需要核对关节顺序、坐标系、绝对／增量目标、归一化、控制周期、动作延迟和底层增益。Teacher 输出力矩、student 输出位置目标时，不能直接做动作 MSE，需要定义合理转换或换用其他监督。

跨手型还涉及不同运动学和接触能力。简单动作蒸馏不会自动消除 embodiment gap。接口背景见 [机器人数据与动作语义](24_robot_data_action_semantics.md)。

### 7.3 监督覆盖 student 的偏离与恢复状态

只保存成功轨迹适合训练初始策略，但可能缺少滑移、偏离和恢复动作。需要评价专家在这些状态是否仍可靠，再决定是否采用 DAgger、扰动采集或后续 RL 微调。

“Teacher 在自己的 rollout 上表现好”与“能纠正任意 student 状态”是两个能力。

## 8. 在线更新表征与在线更新权重

固定参数的网络仍然可以随输入变化而适应环境。以历史估计器为例：

\[
\hat z_t=F_\omega(h_t),\qquad
\hat z_{t+1}=F_\omega(h_{t+1}).
\]

新观测改变了 \(h_{t+1}\)，因此输出 \(\hat z_{t+1}\) 可以不同；两次前向计算使用的是同一组参数 \(\omega\)。这种适应无需在部署循环中调用反向传播或优化器更新。

| 机制 | 部署时改变什么 |
| --- | --- |
| 历史编码／循环记忆 | Hidden state 或环境 latent |
| 系统参数在线估计 | 显式参数估计 |
| 在线 RL／梯度微调 | Actor、critic 或部分网络权重 |
| 高层重规划 | 当前目标、参考或技能选择 |

HORA 使用上述历史表征更新，而不是在真机上继续运行 PPO。它的三阶段计算图、固定模块和真实控制接口见 [HORA 精讲](46_hora_paper_walkthrough.md)。同样，使用新触觉更新记忆，也不等于在部署时训练网络。

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

## 10. 从通用框架进入 HORA

建立完 RL02 的单次训练闭环与本文的阶段关系，再读 HORA，可以将每个设计放回准确的位置：

| 已有知识 | HORA 中需要具体化的接口 |
| --- | --- |
| Actor–critic 用交互回报学习 | 第一阶段的特权输入、动作分布、价值头和 PPO 更新 |
| 特权信息可以形成控制表征 | 物体信息编码器生成的 latent，以及它怎样进入控制策略 |
| Student 可以只学习 latent | 历史适应模块的输入、监督目标、数据收集与固定模块 |
| 部署可以只更新表征 | 用历史估计替换真值编码，控制策略保持已训练权重 |
| 实验应对应方法主张 | 分布内外测试、真机测试，以及移除、替换或冻结模块的对照 |

HORA 第一阶段的 actor 本身使用特权信息，不能仅因存在 critic 就把整套方法叫作“只有 critic 看真值的非对称 actor–critic”。第二阶段主要训练 latent 估计器，也不能按名字假定它重新模仿整套 teacher 动作。[论文训练流程](https://arxiv.org/html/2210.04887v1#S3)

下一篇 [HORA 精讲](46_hora_paper_walkthrough.md) 按任务 → 物体编码 → PPO 特权策略 → 历史监督 → 部署 → 实验组织这些接口，并给出网络维度、符号表与 baseline 结构图。随后可以对照 [UniDexGrasp 精读](47_unidexgrasp_paper_walkthrough.md)：它用生成的目标抓姿规定任务，再将特权策略的动作蒸馏到视觉策略。两篇都包含知识迁移，但监督目标和部署模块不同。其他论文可以沿用同样的阅读顺序，同时重新确认各阶段的实际输入、目标与梯度路径。
