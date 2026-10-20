# PI0.5：前向结构与 Action Expert

> 建议先读：[VLA 总览](27_vla_overview.md)、[动作生成方法](25_action_generation_families.md)与[Flow Matching](10_flow_matching_training_and_inference.md)。本文先追踪连续动作路径，再接到 [FAST 与 KI：动作监督、梯度与微调](42_fast_ki_and_pi_training.md)。

可以先用一句话理解 PI0.5：

> VLM 负责理解“看到了什么、任务是什么、机器人当前在哪里”，Action Expert 负责生成“接下来怎么动”。

## 0. 这张架构图覆盖哪一部分

配套结构图描述 openpi / BrainCo-IL 中的 **flow 动作前向路径**：观测进入 Prefix，带噪动作进入 Action Expert，最后得到速度场。图里的 `SentencePiece tokenizer` 处理文字与当前 state，`action_in_proj` 把带噪动作投影为连续 hidden；它们都不承担 FAST 动作编码。

理解 π₀.₅还需要区分三个层次：

| 层次 | 阅读时要确认 |
| --- | --- |
| 原始 π₀.₅论文 | FAST 离散目标预训练，后训练加入连续 expert 并保留联合 token/flow 目标 |
| 后续 KI 与发布 checkpoint | 两种动作监督同时训练；flow 到 VLM 的梯度被隔离 |
| 公开 openpi 微调代码 | 只实现 π₀.₅的 flow head 训练与推理；不会因为加载 KI checkpoint 就自动运行 FAST 或 KI |

前两行是训练配方，第三行是公开实现范围。[原始论文](https://arxiv.org/abs/2504.16054)、[KI 论文](https://arxiv.org/abs/2505.23705)和[官方 README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)需要分别阅读。本文以三路相机、离散 state prompt 为示例；具体张量长度由配置决定。

## 1. 先纠正输入术语

Observation 包含：

```text
visual：三路相机图像
language：任务 prompt
state：机器人当前关节状态
```

监督目标是：

```text
action：未来 H 步控制目标
```

因此 `state` 是 observation 的一部分，不是 action。

## 2. 总体结构

```text
三路 RGB 图像 ─> SigLIP ─> image tokens ───────────┐
                                                    │
prompt + 当前 state ─> 离散化/tokenizer ─> tokens ─┤
                                                    ▼
                                           Gemma 2B Prefix
                                           视觉/语言/state 条件
                                                    │
                                                    │ attention
                                                    ▼
随机或带噪 action chunk ─> action_in_proj ─> Gemma 300M Action Expert
                                                    │
                                                    ▼
                                            action_out_proj
                                                    │
                                                    ▼
                                           predicted velocity
```

代码入口是[官方 `Pi0`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)。本站工程篇还链接到 [BrainCo-IL 的对应实现](../../src/openpi/models/pi0.py)，其中机器人配置与额外功能应单独核对。

## 3. Prefix 和 suffix 是什么

- **prefix（前缀）**：放在逻辑序列前面的条件信息。
- **suffix（后缀）**：放在逻辑序列后面、需要根据条件不断修正的动作信息。

在 PI0.5 中可以理解为：

```text
[ 三路图像 + prompt + 当前 state ] [ noisy action chunk ]
└────────── prefix ──────────┘ └────── suffix ──────┘
```

它们不是普通字符串的前后缀，而是 Transformer 逻辑序列中的两个区段：

```text
prefix：题目和已知条件
suffix：正在填写和修改的答案
```

具体到机器人：

```text
prefix
├─ 三路图像 tokens：现在看到了什么
├─ prompt tokens：任务要求做什么
└─ state tokens：机器人当前在哪里

suffix
└─ H 个 action tokens：未来 H 步动作目前应该如何修正
```

suffix 的原始数据是带噪 action chunk，经 `action_in_proj` 后才成为 Action Expert 使用的 action tokens。flow timestep 也会影响 Action Expert，但 PI0.5 通过 adaRMS 注入 timestep，它不是一个普通 suffix token。

两部分最重要的 attention 关系是：

```text
Prefix 看到：Prefix
Suffix 看到：Prefix + Suffix
```

也就是说，Action Expert 可以根据图像、任务和当前状态修正动作；VLM prefix 不会反过来读取 noisy action，从而避免条件信息被噪声动作干扰。

这条 attention mask 规则描述前向可见性。只要 expert 读取的 prefix K/V 没有 `stop-gradient`，flow loss 仍可沿它们更新 VLM。KI 的梯度边界需要另外画，详见 [FAST / KI 训练图](pi05_fast_ki_training.drawio)。

## 4. 图像如何进入 VLM

三路图像都先缩放到：

```text
[B, 224, 224, 3]
```

当前 SigLIP 使用 `So400m/14`。每个 patch 是 `14x14`，所以每张图像形成：

```text
224 / 14 = 16
16 x 16 = 256 个 image tokens
```

三路图像共约 768 个图像位置，每个位置被投影到 Gemma 2B 的 2048 维隐藏空间：

```text
每路 [B, 256, 2048]
三路 [B, 768, 2048]
```

## 5. prompt 和 state 如何进入 VLM

在 `discrete_state_input=True` 的常见 PI0.5 路径中，`[B, D]` state 这样进入 VLM：

```text
归一化 state
  -> 按 [-1, 1] 上的 256 个区间离散化
  -> 拼进文本
  -> SentencePiece tokenizer
  -> padding/truncation 到 max_token_len
```

文本大致为：

```text
Task: pick up the bread, State: 143 82 227 ...;
Action:
```

设文本容量为 `L = max_token_len`，最终 token IDs 的 shape 为：

```text
[B, L]
```

经过 Gemma embedding 后为：

```text
[B, L, 2048]
```

图像 tokens 与 prompt/state tokens 拼成 VLM prefix。padding token 会被 mask，不参与有效注意力。

原结构图中的 `L=256` 来自 BrainCo 示例配置；不是 π₀.₅的固定架构常数。核对的官方配置在 `pi05=True` 时默认取 200，还允许覆盖。若某个数据配置关闭 `discrete_state_input`，需要另查实际 state 入口，不能假定它自动切回 π₀的连续 `state_proj`。[官方配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)

## 6. `action_in_proj` 做什么

训练时真实 actions 的形状是：

```text
[B, H, D]
```

flow matching 先构造同形状的带噪动作 `x_t`，再执行：

```python
action_tokens = self.action_in_proj(noisy_actions)
```

当前 Action Expert 宽度为 1024：

```text
[B, H, D]
  -> Linear(D, 1024)
[B, H, 1024]
```

注意：每一个时间步的完整 D 维动作被映射成一个 action token，不是每个关节对应一个 token。

## 7. Prefix 和 suffix 如何在代码中协同

PI0.5 不是“VLM 输出一个向量，再交给另一个完全独立的网络”。Gemma 2B prefix expert 与 Gemma 300M action expert 在多专家 Transformer 中协同计算：

- prefix 使用 Gemma 2B 的 2048 维权重。
- action suffix 使用 Gemma 300M 的 1024 维权重。
- 两路各自产生 Q/K/V，再按序列位置参与联合 attention。
- Action Expert 的 query 可以关注 VLM prefix 的 K/V。
- attention mask 阻止 VLM prefix 反过来依赖 noisy action。

所以 Action Expert 能读取：

```text
图像中物体在哪里
prompt 要求做什么
机器人当前关节状态是什么
```

`action_in_proj` 不属于 VLM prefix，也不会把 noisy action 输入 Gemma 2B；它属于 Action Expert 的输入接口。

可以把信息流总结为：

```text
VLM prefix：图像、prompt、state
       │
       │ 提供 attention K/V
       ▼
Action suffix：带噪 action tokens
       │
       ▼
预测 flow velocity
```

## 8. timestep 和 adaRMS

网络还需要知道当前动作有多“嘈杂”。flow timestep 先变成正弦/余弦向量，再通过 `time_mlp_in/out` 得到 1024 维条件：

```text
timestep t
  -> sin/cos embedding
  -> time MLP
  -> adaRMS condition
```

这个条件注入 Action Expert 的 RMSNorm，帮助同一个网络处理不同噪声阶段。

## 9. 训练：学习速度场

[官方 `compute_loss()`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py) 构造：

```text
真实动作 a
随机噪声 ε
随机时间 t

x_t = t * ε + (1 - t) * a
目标速度 u_t = ε - a
```

模型预测：

```text
v_t = action_out_proj(ActionExpert(...))
```

形状为：

```text
[B, H, 1024]
  -> Linear(1024, D)
[B, H, D]
```

损失是预测速度与目标速度之间的均方误差：

```text
loss = mean((v_t - u_t)^2)
```

这是当前公开 π₀.₅微调的目标。使用全参或 LoRA 时，梯度可以继续进入可训练的 VLM 参数；冻结哪些参数由训练配置决定。不能从“输出来自 Action Expert”推断损失只更新 Action Expert。

## 10. 推理：从噪声生成 action

推理没有真实 action，先生成纯随机噪声：

```text
x_1 ~ Normal(0, I)，shape [B, H, D]
```

推理过程中，两部分的变化方式不同：

```text
prefix：图像、任务和当前 state 不变，只计算一次并缓存 K/V
suffix：从随机噪声开始，每一步 Euler 更新后都会变化
```

之后 Action Expert 重复读取相同 prefix、处理最新 suffix、预测速度，并用 Euler 方法更新动作：

```text
x_t <- x_t + dt * v_t
```

这里 `t=1` 是噪声、`t=0` 是动作，因此 `dt` 为负。默认执行 10 步，最终得到 action chunk。

因此推理可以概括为：

```text
固定的题目和条件 prefix
        +
不断修改的答案 suffix
        ↓
最终 action chunk
```

## 11. VLM 如何兼容 28D/56D state

VLM 不直接接收 `[B, 28]` 或 `[B, 56]` 连续向量。不同长度的 state 会先变成包含不同数量数字的文本，最后都 padding/truncate 为：

```text
token IDs [B, max_token_len]
```

所以 VLM 的网络宽度不需要变化，变化的是 token 内容。结构上能够接收不代表已经理解新关节语义；模型仍需通过一致的数据顺序和训练建立对应关系。

还要检查 token 是否超过 `max_token_len`，否则后部 state 可能被截断。

## 12. 从前向路径走到训练机制

同一段真实未来动作在完整训练方案中可以派生两种目标：FAST 编码用于 VLM 的 token 预测，加入噪声后用于 expert 的速度场预测。这是两条监督分支；是否先后训练、哪些梯度回到 VLM，是另外两个问题。

下一篇 [FAST 与 KI](42_fast_ki_and_pi_training.md) 按“动作表示 → 两种损失 → 训练阶段 → 梯度隔离 → 公开微调代码”展开。完成后应能在图上分别标出前向连接、训练目标和参数更新路径，再进入 [TrainState 与训练步](05_train_state_jit_checkpoint.md) 验证代码行为。
