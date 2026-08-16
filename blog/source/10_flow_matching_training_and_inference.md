# 10. Flow Matching：训练、timestep 与推理

Flow matching 在 PI0.5、GR00T 和 VLA-JEPA Action Head 中承担同一类任务：学习一个速度场，把随机噪声逐步变成 action chunk。

## 1. 三个最重要的对象

设：

```text
a：数据集中的真实 action chunk，shape [B,H,D]
ε：同 shape 的高斯噪声
t：flow timestep，每个训练样本采一个标量
```

模型不直接监督“预测最终 action”，而是在噪声和 action 之间构造中间点 `x_t`，监督模型预测该位置的速度方向。

## 2. PI0.5 的约定

当前 BrainCo-IL/OpenPI 使用：

```text
x_t = t * ε + (1 - t) * a
u_t = ε - a
```

因此：

```text
t = 0：x_t 是真实 action
t = 1：x_t 是纯噪声
```

模型预测：

```text
v_θ(x_t, t | observation)
```

训练 loss：

```text
L_FM = mean((v_θ - u_t)^2)
```

## 3. GR00T 的方向约定

GR00T 常用相反参数方向：

```text
x_t = (1 - t) * ε + t * a
u_t = a - ε
```

因此：

```text
t = 0：纯噪声
t = 1：真实 action
```

两者描述的是同一条直线，只是 t 的方向和目标速度符号相反。比较论文或代码时必须同时检查：

- `x_t` 的公式；
- target velocity 的符号；
- 推理时 `dt` 的正负方向。

不能只看“t=0”或“t=1”判断实现是否相同。

## 4. timestep 为什么不是监督目标

训练监督仍然是：

```text
predicted velocity 与 target velocity 的 loss
```

t 不需要模型预测，它是告诉模型当前输入处于哪个噪声阶段的条件。

同一个模型需要同时处理：

```text
几乎纯净的动作
中等噪声的动作
接近纯噪声的动作
```

如果没有 t，模型只能看到 `x_t`，却不知道应该使用哪一个阶段的速度场。于是完整函数应写为：

```text
v_θ(x_t, t, image, language, state)
```

“条件”表示在给定 t、图像、语言和 state 的情况下预测速度；它不表示这些输入都是 loss target。

## 5. 训练时存在推理去噪步数吗

通常不存在“一次训练样本内部跑 10 次去噪”的过程。

一次训练前向通常只做：

```text
随机采一个连续 t
构造一个 x_t
预测一次 v_t
计算一次 flow loss
```

推理时才把区间离散成若干 Euler steps，例如 10 步：

```text
t_0 -> t_1 -> ... -> t_10
```

训练中的随机 t 是对连续区间的采样，不必预先被 `1/10` 分割。因为模型在训练中见过大量不同 t，推理时可以选择 4、10 或其他数量的积分步数；步数越多通常积分误差越小，但延迟越大。

有些实现会把 t 离散成 bucket 用于 embedding，例如 GR00T 的 1000 个 timestep buckets。这仍不等于推理必须执行 1000 步。

## 6. 40k training steps 对应多少次 loss

如果训练配置是：

```text
num_train_steps = 40,000
```

那么优化器通常更新 40,000 次，每个 step 产生一个 batch loss：

```text
40,000 次 batch-level loss / backward / optimizer update
```

若 batch size 为 B，每个 batch 中每个样本都独立采样噪声和 t，所以样本级 flow targets 约为：

```text
40,000 × global_batch_size
```

每个样本的 loss 又会对 `[H,D]` 上的误差求平均。不能把“一个 tensor 中有 H×D 个误差项”说成 H×D 次独立反向传播；它们汇总成一次标量 loss，再共同反向传播。

如果使用梯度累积，则 micro-batch forward 次数可能更多，但 optimizer step 仍以配置和训练循环定义为准。

## 7. 训练时反向更新什么

计算：

```text
L_FM = MSE(v_pred, velocity_target)
```

自动求导会沿预测路径更新所有允许训练的参数，例如：

- action input/output projection；
- Action Expert 或 DiT；
- timestep MLP/AdaNorm modulation；
- state encoder/condition MLP；
- 未冻结的 VLM 参数。

噪声、t、真实 action 和 target velocity 都是构造出来的训练数据，不是可训练参数。

## 8. 推理如何从噪声得到 action

推理没有真实 action，先采样：

```text
x_noise ~ Normal(0,I), shape [B,H,D]
```

以 PI0.5 为例，从噪声端向数据端做 Euler 积分：

```text
for each timestep:
    v = model(observation, x_t, t)
    x_next = x_t + dt * v
```

最终 `x` 成为 denoised action chunk。图像、语言和 state 通常在一次 action sampling 中保持不变；noisy action 和 flow timestep 每一步变化。

## 9. action chunk、模型内部 tensor 与有效输出

模型内部 tensor 不一定等于机器人真正需要的动作尺寸。

例如 GR00T N1.7 可以有：

```text
内部 capacity: [B,40,132]
有效机器人动作: [B,16,56]
```

训练时有效区域由 mask 监督；推理完成后切出：

```text
horizon 0:16
dimension 0:56
```

所以“model tensor”只是统一多机器人模型使用的 padded 内部容器，不代表机器人真的有 132 个自由度或执行 40 步。

## 10. Flow Matching 与异步执行是两个层级

Flow matching 负责单次模型调用内部如何从噪声生成 action：

```text
noise -> iterative denoising -> action chunk
```

同步、异步、VLASH 则负责不同模型调用与控制器执行之间如何调度：

```text
什么时候采 observation
推理时机器人是否继续执行旧 chunk
新 chunk 从哪个 state 开始执行
什么时候切换 active/pending chunk
```

VLASH 不需要改变 PI0.5 的 flow objective；它改变的是训练样本的时间对齐和部署调度条件。

## 11. 阅读 Flow Matching 代码的检查表

看到一个新 VLA 的 action generator 时，依次检查：

1. 真实 action 和噪声如何插值得到 `x_t`；
2. `t=0/1` 分别代表 noise 还是 action；
3. target 是 `a-ε` 还是 `ε-a`；
4. timestep 是 token fusion、AdaNorm，还是两者都有；
5. 训练时 t 的分布是什么；
6. 推理使用多少步、什么积分器；
7. action mask、padding 和有效维度如何处理；
8. 最终输出是否需要 slice、unnormalize 或坐标变换。

PI0.5 的完整结构和公式也可以配合 [pi05_structure.drawio](pi05_structure.drawio) 阅读。
