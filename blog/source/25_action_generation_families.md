# 动作生成方法：MSE、Token、Diffusion 与 Flow Matching

VLA 都要从条件信息产生动作，但“Action Head”可以对应完全不同的生成目标。本篇先建立分类，再进入具体模型。

## 1. 统一问题

给定条件：

\[
c=(\text{image},\text{language},\text{state},\text{history})
\]

策略希望生成未来 action chunk：

\[
A\in\mathbb R^{H\times D}
\]

不同方法的核心区别是：怎样表示 \(p(A\mid c)\)，训练预测什么，推理需要多少次前向。

## 2. 直接连续回归

```text
condition -> Transformer/MLP -> action chunk
```

常见 loss：

\[
\mathcal L=\lVert\hat A-A\rVert^2
\]

优点是简单、快速、通常一次前向。缺点是动作存在多种合理模式时，逐元素 MSE 可能产生平均化结果。

FuSe 的公开 Octo 主实验配置就是直接 MSE Action Head。

## 3. 离散动作 Token

先将连续 action 量化或序列化：

```text
continuous action -> discrete tokens
VLM autoregressive generation -> action tokens
-> decode -> continuous action
```

训练使用 token cross-entropy，可以直接利用 LLM 的生成接口。需要额外处理量化误差、序列长度和生成延迟。

RT-2、部分 PaliGemma VLA 路线可以从这个角度理解。

## 4. Diffusion Action Generation

Diffusion 训练先向真实动作逐步加噪，再让网络预测噪声、干净样本或其他参数化目标：

\[
A_t=\sqrt{\bar\alpha_t}A_0+sqrt{1-\bar\alpha_t}\epsilon
\]

一种常见目标是：

\[
\mathcal L=\lVert\epsilon-\epsilon_\theta(A_t,t,c)\rVert^2
\]

推理从随机噪声开始，重复调用网络去噪。它能表达多峰动作分布，但延迟取决于采样步数和 action network 大小。

原始 Octo 和 Diffusion Policy 属于这一大类，但它们的具体动作网络并不相同。

## 5. Flow Matching

Flow matching 在噪声和真实动作之间定义连续路径，并训练速度场：

\[
x_t=(1-t)\epsilon+tA
\]

\[
u_t=A-\epsilon
\]

模型学习：

\[
v_\theta(x_t,t,c)\approx u_t
\]

推理从噪声端对 ODE 做数值积分，逐步得到 action。PI0.5 可能采用相反的 \(t\) 方向和速度符号，但描述的是同一条路径。

## 6. Diffusion 与 Flow Matching 的关系

两者都可以：

- 从随机噪声生成连续动作；
- 表达比单点 MSE 更丰富的分布；
- 在推理时迭代更新 action tensor。

但训练路径、目标参数化和采样方程不同。不能只因为模型“多步去噪”就断言它使用 flow matching，也不能把所有 DiT 都等同于同一个 loss。

## 7. Action Network 与 Objective 是两个维度

“使用 Transformer”描述网络结构；“使用 diffusion/flow”描述训练和生成目标。可能的组合包括：

```text
MLP diffusion head
Transformer diffusion policy
DiT flow-matching head
multi-expert Transformer flow head
```

因此阅读论文时要分别确认：

1. noisy action 进入哪个网络；
2. timestep 怎样注入；
3. target 是 action、noise 还是 velocity；
4. 推理使用什么更新方程。

## 8. 训练与推理次数

Flow/diffusion 训练通常对每条样本随机采一个 timestep，只执行一次带噪前向：

```text
sample t -> construct x_t -> predict once -> loss
```

推理时才运行多步积分或去噪。因此“推理使用 10 步”不表示每条训练样本也在内部执行 10 次网络。

## 9. 怎样选择

| 维度 | 直接回归 | 动作 Token | Diffusion / Flow |
| --- | --- | --- | --- |
| 推理速度 | 通常最快 | 取决于 token 数 | 取决于采样步数 |
| 多模态分布 | 较弱 | 可表达 | 较强 |
| 实现复杂度 | 低 | 中 | 高 |
| 与 LLM 接口 | 需独立 head | 自然 | 通常独立 action expert |
| 连续精度 | 直接 | 受量化影响 | 直接连续生成 |

实际选择还取决于数据规模、控制频率、模型延迟和任务多解性，不能脱离 runtime 单看离线 loss。

## 10. 后续阅读坐标

- PI0.5：Gemma Action Expert + flow matching；
- GR00T：embodiment-specific adapters + DiT + flow matching；
- 原始 Octo：action readout condition + diffusion MLP；
- FuSe-Octo：MSE Action Head；
- FuSe-PaliGemma：离散动作 token。

这张分类表可以避免把“VLM”“Transformer”“DiT”“diffusion”和“flow matching”当成同一层级的名词。
