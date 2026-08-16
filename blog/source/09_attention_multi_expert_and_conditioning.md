# 09. QKV、Attention、多专家与条件注入

这一章回答四组经常混在一起的问题：

1. Q/K/V 是什么？
2. self-attention 和 cross-attention 有什么区别？
3. PI0.5 的 multi-expert 为什么不是常见 MoE？
4. token 注入和 AdaLN/AdaRMS 条件调制有什么区别？

## 1. Q、K、V 的直觉

每个 token hidden vector 会经过三组可训练投影：

```text
Q = X W_Q  Query：我想找什么信息
K = X W_K  Key：我能用什么特征被匹配
V = X W_V  Value：匹配后真正提供什么内容
```

Attention 权重来自 Q 与 K 的相似度：

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(d_head)) V
```

直觉上：

```text
Query 提问
Key 用于检索匹配
Value 提供答案内容
```

Q/K/V 都是一次前向计算产生的 activation，不是固定数据；生成它们的 `W_Q/W_K/W_V` 才是可训练权重。

## 2. Multi-head Attention

一个 token 的隐藏宽度会被组织为多个 attention heads。

PI0.5 的 Gemma 配置中：

```text
num_query_heads = 8
head_dim = 256
8 × 256 = 2048
```

因此 2048 宽度可以拆成 8 个 256 维 query heads。但这只是该配置内部维度设计的一致关系，不表示“先决定 Q head，才产生 Gemma 2B 宽度”。模型宽度、head 数和 head_dim 是共同设计的超参数。

该配置使用 grouped/multi-query attention：

```text
Q:  8 × 256
K/V: 1 × 256
```

8 个 query heads 共享较少的 K/V heads，可以降低 KV cache 和计算开销。

## 3. Self-attention 与 Cross-attention

### 3.1 Self-attention

Q、K、V 来自同一组 token：

```text
action/state tokens --Q/K/V--> self-attention
```

它适合序列内部交互，例如：

- action chunk 中不同时间步保持轨迹一致；
- state token 告诉所有 action tokens 当前机器人姿态；
- 文本 token 结合上下文理解指令。

### 3.2 Cross-attention

Q 来自当前处理流，K/V 来自另一组 memory：

```text
Action tokens -> Q
VLM features  -> K/V
```

这表示动作网络主动从视觉语言 memory 中检索相关信息。

GR00T 的 DiT 使用这种清晰的两段式结构：先得到 VLM hidden memory，再让 state/action query stream 对它做 cross-attention。

## 4. PI0.5 的联合 Attention

PI0.5 不是简单的：

```text
Gemma 2B 完整运行结束 -> 一个向量 -> Gemma 300M
```

而是在 18 个 paired layers 中反复协同：

```text
Prefix Expert hidden [B,Lp,2048] -> 自己的 Q/K/V 投影
Action Expert hidden [B,H,1024]   -> 自己的 Q/K/V 投影

Q/K/V 在共同 head geometry 中按序列位置组合
  -> masked joint attention
  -> 输出再回到各自 expert 的 hidden width
```

Attention mask 保证：

```text
Prefix query -> Prefix K/V
Action query -> Prefix + Action K/V
```

因此 Action Expert 可以读取图像、语言和 state，Prefix 不会被 noisy action 污染。

完整图见 [pi05_structure.drawio](pi05_structure.drawio)。

## 5. Multi-expert 和常见 MoE 的区别

### 5.1 常见稀疏 MoE

典型 Mixture of Experts 有 router：

```text
token
  -> router 计算 expert scores
  -> 选择 top-k experts
  -> experts 分别计算
  -> 加权合并
```

不同 token 可能动态选择不同 expert，目的通常是在计算量增长较少的情况下扩大参数容量。

### 5.2 PI0.5 multi-expert

PI0.5 中 expert 分工是由 token 类型预先确定的：

```text
image/text/state prefix -> Gemma 2B Prefix Expert
noisy action suffix     -> Gemma 300M Action Expert
```

这里没有 router，也没有对每个 token 做 top-k 随机或动态选择。因此它是多组 expert-specific weights 的协同 Transformer，但不是最常见的 sparse routed MoE。

可以记成：

```text
稀疏 MoE：谁处理 token，由 router 动态选择
PI0.5：谁处理 token，由 prefix/action 模态身份固定决定
```

## 6. 广义条件与狭义条件调制

“condition”是一个很宽的词。任何影响输出的信息都可以叫条件：

- 图像；
- 语言；
- robot state；
- noisy action；
- flow timestep；
- embodiment ID；
- goal 或 future state。

因此“作为 token 注入”也是一种 conditioning。工程讨论中，通常把下面两种路径分开说。

## 7. 作为 token 注入

以 GR00T state 为例：

```text
state [B,D]
  -> State Encoder
  -> state token [B,1,1536]

noisy action [B,H,D]
  -> Action Encoder
  -> action tokens [B,H,1536]

concat -> [B,1+H,1536]
```

state token：

- 占一个序列位置；
- 产生自己的 Q/K/V；
- 通过 self-attention 与 action tokens 交换信息；
- 可以在不同层形成新的 hidden representation。

它像一个加入会议的参与者。

## 8. 通过 AdaLN/AdaRMS 调制

flow timestep 通常是全局噪声阶段，更适合控制整层计算方式：

```text
t -> sin/cos embedding -> time MLP -> condition c
```

对某一层 hidden states `x`：

```text
[scale, shift, gate] = Linear_layer(c)
normed = RMSNorm(x)
modulated = normed * (1 + scale) + shift
y = Attention_or_FFN(modulated)
x_next = x + gate * y
```

其中：

- `scale` 调整每个特征通道的强弱；
- `shift` 改变每个特征通道的基准；
- `gate` 控制当前子层对 residual 的贡献。

condition 没有序列位置，也不生成 Q/K/V。它更像调节整个会议室工作模式的控制面板。

BrainCo-IL 中具体实现见：

- [`Pi0.embed_suffix()`](../../src/openpi/models/pi0.py)：构造 time/state condition；
- [`RMSNorm`](../../src/openpi/models/gemma.py)：生成 scale、shift、gate；
- [`Layers.__call__()`](../../src/openpi/models/gemma.py)：在 Attention 和 FFN 前使用 condition。

## 9. PI0、PI0.5、GR00T 的 timestep 路径

| 模型 | noisy action | flow timestep |
| --- | --- | --- |
| PI0 | 投影成 action embedding | 与 action embedding 拼接后过 MLP |
| PI0.5 | `action_in_proj` 形成 action tokens | time MLP 后通过 adaRMS 调制 |
| GR00T N1.x | embodiment-specific Action Encoder | 既融合进 action encoding，又通过 AdaLayerNorm 调制 DiT |

注意 flow timestep 不是控制系统时间戳，也不是 action horizon 的位置编号。它表示当前 `x_t` 位于 noise 到 action 路径的什么位置。

## 10. VLASH future state 的条件注入

当前 BrainCo-IL 的 `state_cond=true` 路径为：

```text
c_time  = TimeMLP(t)
c_state = StateMLP(state_(t+δ))
c       = c_time + c_state
```

然后 `c` 在 Action Expert 每层产生 scale/shift/gate。

一次去噪推理中：

```text
future state s_(t+δ)：对这个请求固定
flow timestep：随每个 Euler step 改变
```

VLASH 的核心不是强制所有模型都使用 AdaRMS，而是让策略以执行时刻 state 为条件。对其他 backbone，future state 也可以作为 token、cross-attention memory 或原生 state 输入。

## 11. 为什么论文常在条件上创新

增加条件通常参数少、易复用 checkpoint，又可以让行为更可控。常见条件包括 future state、历史动作、触觉、力觉、目标图像、价值、子目标和 embodiment ID。

但有效创新不能只看“加了一个输入”，还要检查：

1. 该信息在部署时是否可获得或可靠预测；
2. 训练与推理的条件分布是否一致；
3. 是否把未来真值泄漏给了部署路径；
4. 模型是否真正使用该条件；
5. 注入位置是 token、cross-attention 还是 adaptive norm，为什么适合。

下一章将 timestep 放回 flow matching 的完整训练和推理过程，解释它为什么是必要条件。
