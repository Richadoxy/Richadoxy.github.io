# 模仿学习与 Behavior Cloning 入门

模仿学习的目标是让策略从示范中学习行为。它是理解 VLA 动作训练的直接起点。

## 1. 从示范到策略

一条机器人示范通常包含时间序列：

\[
\tau=(o_0,a_0,o_1,a_1,\ldots,o_T)
\]

其中：

- \(o_t\)：时刻 \(t\) 的观测；
- \(a_t\)：专家在该观测下执行的动作；
- \(\tau\)：一条 trajectory 或 episode。

策略学习条件分布：

\[
\pi_\theta(a_t\mid o_t)
\]

VLA 只是在观测中进一步加入视觉和语言，并常输出一段连续动作。

## 2. Observation 不等于 State

在机器人学习中可以区分：

```text
world state：环境的完整真实状态，通常不可完全获得
observation：部署时传感器能够得到的信息
proprioceptive state：关节角、夹爪状态等机器人自身观测
```

因此 VLA 的输入可以写成：

```text
o_t = images + language + proprioceptive state + history
```

这里 robot state 是 observation 的组成部分，不是 action。

## 3. Behavior Cloning

Behavior Cloning 把模仿学习转成监督学习：

```text
专家观测 -> policy -> predicted action
                         ↕ loss
                      expert action
```

连续动作直接回归时，最简单的目标是：

\[
\mathcal L_{BC}=\lVert\hat a_t-a_t\rVert^2
\]

如果动作被离散成 token，可以使用交叉熵；如果用 diffusion 或 flow matching，则监督噪声或速度场。它们仍然可以服务于 Behavior Cloning，只是动作分布的参数化方式不同。

## 4. Demonstration 从哪里来

常见数据来源包括：

- 人类遥操作；
- 手把手 kinesthetic teaching；
- 传统规划器或专家 policy；
- 仿真中的 scripted policy；
- 人类视频与机器人数据的联合预训练。

数据来源决定了观测质量、动作空间、控制频率和误差模式。人类视频没有机器人 action label，不能直接按普通 BC 使用，但可以提供视觉或动态表示监督。

## 5. Offline 训练与 Online Rollout

BC 训练通常是 offline 的：模型读取已经记录的数据，不与环境实时交互。

部署 rollout 则是闭环的：

```text
观测 -> 策略 -> 动作 -> 环境改变 -> 新观测 -> 策略
```

训练样本来自专家访问过的状态，而部署时模型自己的小错误会把机器人带到新状态。这种训练分布和策略访问分布的偏移，是经典的 covariate shift / compounding error 问题。

## 6. 为什么训练 Loss 低不等于任务成功

动作 MSE 可能低，但真实机器人仍失败，原因包括：

- 多个动作都合理，MSE 平均后反而不自然；
- 某些关键接触时刻在数据中占比很少；
- 小误差在闭环中累积；
- 相机、物体或机器人状态发生分布偏移；
- action representation、归一化或坐标系不一致；
- 推理延迟改变了动作真正开始执行的状态。

因此机器人论文需要报告 rollout success、接触质量、恢复能力和延迟，而不能只给 validation loss。

## 7. Generalist Policy 与 VLA

传统单任务策略可能只学习：

```text
固定任务图像 + state -> action
```

通用策略希望利用多任务、多机器人或多数据集：

```text
images + language instruction + state + embodiment
                      -> shared policy
                      -> action
```

语言提供任务接口，视觉提供环境语义，state 描述机器人当前配置，action head 将共享表示映射回具体控制空间。

## 8. 预训练没有取代模仿学习

VLM 能识别物体和理解语言，不代表它天然会控制机器人。VLA 通常还需要机器人示范来建立：

- 视觉语义与可执行动作之间的对应；
- 不同 embodiment 的动作接口；
- 时间连续性和接触动力学；
- 任务成功所需的闭环行为。

所以可以把 VLA 看成：

```text
视觉语言预训练提供“理解”初始化
+ 机器人模仿数据提供“怎样行动”的监督
```

## 9. 阅读模仿学习工作的检查表

1. 专家是谁，示范怎样采集？
2. observation 在部署时是否全部可得？
3. action 表示位置、增量、速度还是力矩？
4. 一次预测一步还是 action chunk？
5. loss 是 MSE、token、diffusion 还是 flow？
6. 训练数据和测试场景差异是什么？
7. 是否真实闭环 rollout，失败标准是什么？
8. 策略遇到偏离示范分布的状态能否恢复？

一句话总结：

> Behavior Cloning 用示范动作监督策略；VLA 扩展了策略的感知和任务接口，但数据分布、动作语义与闭环执行问题仍然存在。
