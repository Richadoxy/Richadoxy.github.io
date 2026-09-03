# 机器人数据：Observation、Action 与时间窗口

这篇文章把机器人数据中最容易引发“shape 正确但语义错误”的概念放到一条时间轴上。

## 1. 一条 Episode 包含什么

```text
episode
  ├─ observation.image[t]
  ├─ observation.state[t]
  ├─ instruction/task
  ├─ action[t]
  ├─ timestamp[t]
  └─ episode boundary / success metadata
```

同样是 `[T,D]` 的数组，只有知道采样频率、坐标系、关节顺序和控制语义后才是可执行数据。

## 2. Observation、State 与 Action

典型 VLA 样本：

```text
observation_t
  ├─ camera images
  ├─ language instruction
  └─ proprioceptive state_t

target
  └─ future actions a[t:t+H]
```

`state_t` 描述机器人在哪里，`action_t` 描述控制器下一步应该做什么。二者维度可能相同，但语义不能互换。

## 3. 常见 Action 表示

| 表示 | 示例 | 部署端含义 |
| --- | --- | --- |
| absolute joint position | 目标关节角 | 发送绝对目标 |
| delta joint position | 相对当前角度的变化 | 与参考 state 合成 |
| EEF pose | 末端位姿或位姿增量 | 需 IK 或笛卡尔控制器 |
| joint velocity | 关节速度 | 与控制周期相关 |
| torque/force | 力矩或力 | 对动力学和频率更敏感 |

训练和部署必须在单位、坐标系、顺序、参考点和控制频率上完全一致。

## 4. Action Chunk

许多现代策略一次预测未来 \(H\) 步：

\[
A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}]
\]

数据集从 episode 中切出：

```text
observation_t -> action[t:t+H]
```

它的优势包括减少模型调用、显式建模短期轨迹一致性；代价是 chunk 后部基于更旧的观测，且运行时需要决定执行多少步后重新规划。

## 5. Horizon、History 与 Receding Horizon

三个长度不要混淆：

- observation history：模型向过去看多少帧；
- prediction horizon：一次预测多少步；
- execution horizon：新预测到来前真正执行多少步。

例如模型可以观察最近 2 帧、预测未来 16 步，但只执行前 8 步就重新规划。这属于 receding-horizon execution。

## 6. Policy Rate 与 Control Rate

```text
policy rate：模型产生新动作或 chunk 的频率
control rate：底层控制器发送命令的频率
```

若 policy 输出 30 Hz 轨迹而控制器运行在 1 kHz，中间可能需要插值。插值改变采样密度，但不应悄悄改变 action 的绝对/增量语义。

## 7. Normalize 与 Unnormalize

神经网络通常在归一化空间训练：

```text
raw state/action -> normalize -> model -> normalized action -> unnormalize
```

统计量必须对应相同的：

- action dimension；
- 关节顺序；
- absolute/delta 规则；
- 数据分布。

改成新机器人或新动作布局后，不能继续无条件复用旧 norm stats。

## 8. Padding 与 Mask

多机器人统一模型常使用最大容量：

```text
actual action [H=16,D=56]
-> padded model tensor [H_max=40,D_max=132]
```

Mask 保证 padding 区域不进入有效 loss；推理后还要 slice 回真实 horizon 和 dimension。

内部 `[40,132]` 不表示机器人有 132 个自由度。

## 9. Episode 边界

构造 action chunk 时必须保证不跨 episode：

```text
t + H <= episode_end
```

如果还要做 temporal offset：

```text
t + max_offset + H <= episode_end
```

其他做法需要 padding 和对应 loss mask，否则下一条 episode 的动作可能被错误拼接进当前样本。

## 10. 数据合同检查表

在训练前为每个字段写清：

```text
name
shape
dtype
unit
coordinate frame
joint order
timestamp semantics
normalization
valid mask
```

数据、模型和部署三端都应遵守同一个 contract。机器人系统中，显式合同通常比只在代码里传递 shape 更重要。
