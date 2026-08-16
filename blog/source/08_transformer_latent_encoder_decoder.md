# 08. Transformer、Encoder/Decoder 与 Latent

这一章把后续理解 PI0.5、GR00T、VLASH 和 VLA-JEPA 所需的基础词汇放在同一条链路中：

```text
原始数据
  -> 预处理 / tokenizer
  -> encoder 或输入投影
  -> token hidden representations
  -> Transformer 计算
  -> decoder 或输出投影
  -> 任务输出
```

## 1. Transformer 是什么

Transformer 是一种神经网络架构，不是 PyTorch、JAX 那样的软件框架。

它的典型层由以下组件组成：

```text
输入 hidden states
  -> Norm
  -> Attention
  -> Residual
  -> Norm
  -> Feed-Forward Network
  -> Residual
```

其中 Attention 负责不同 token 之间的信息交换，Feed-Forward Network 负责对每个 token 的特征进行非线性变换。

Gemma 2B、Gemma 300M、Qwen3-VL、SigLIP 的 Vision Transformer，以及 GR00T 的 DiT，都使用了 Transformer 类组件，但输入类型、attention 连接方式和输出目标并不相同。

Transformer 之外还有 CNN、RNN/LSTM、状态空间模型、图神经网络和 MLP-Mixer 等架构。现代 VLA 往往不是只用一种架构，而是将视觉编码器、VLM 和动作生成网络组合起来。

## 2. 参数量和隐藏宽度不是一回事

“Gemma 2B”中的 2B 表示整个模型大约有 20 亿个可训练参数，包括：

- token embedding；
- 每层 Q/K/V/O 投影；
- Feed-Forward Network；
- normalization；
- 输出层等。

“宽度 2048”表示每一个 token 在层内由 2048 个数表示：

```text
一个 token hidden vector: [2048]
一段 L 个 token:          [L, 2048]
一个 batch:               [B, L, 2048]
```

因此：

```text
2B   = 整个模型一共有多少参数
2048 = 每个 token 的隐藏表示有多宽
```

2048 是 Gemma 2B 这个具体配置的模型宽度，不是所有 VLM 的固定输入维度。不同模型可以是 768、1024、1536、2048、4096 等宽度。

PI0.5 中，SigLIP 输出的 image tokens 需要被投影到 2048 维，才能进入 Gemma 2B prefix expert。Action Expert 则使用 1024 维 hidden representation。

## 3. Token 是位置，hidden vector 是该位置的内部表示

一个 token 不一定是一个单词。它可以表示：

- 一段文本子词；
- 一个图像 patch；
- 一个 action chunk 中的一个动作时间步；
- 当前机器人 state；
- 特殊的 `<action>` 或 `<latent_i>` 位置。

例如 PI0.5 中：

```text
3 路图像 × 每路 256 patch positions
  -> 768 个 image tokens
  -> hidden shape [B, 768, 2048]

H 个 noisy-action 时间步
  -> H 个 action tokens
  -> hidden shape [B, H, 1024]
```

这里 `[B,L,2048]` 是一次前向计算产生的 activation/hidden states，不是模型权重。模型权重是把这些 hidden states 映射成 Q/K/V、FFN 输出等结果的矩阵。

## 4. Hidden space 和 latent 的关系

`hidden representation` 强调它是网络中间层的数值张量；`latent representation` 强调它是对原始数据的压缩或抽象表达。

很多场景中，同一张量既可以叫 hidden，也可以叫 latent：

```text
图像 patch -> SigLIP -> image hidden/latent features
token IDs  -> Gemma  -> text hidden representations
noisy action -> projection/Transformer -> action hidden/latent features
```

但不能简单地说“整个 prefix 就是一个 latent”。更准确的是：

```text
prefix 是一段序列区域
prefix 内含很多 image/text/state hidden representations
这些 hidden representations 可以统称为 latent features
```

同理，suffix 是动作序列区域，其中每个位置都有自己的 action hidden representation。

## 5. Encoder 和 Decoder 的通用含义

最宽泛的抽象是：

```text
原始输入 --Encoder--> 内部表示 --Decoder--> 目标输出
```

但论文中的命名取决于模型层级，不能只根据“输入映射”和“输出映射”判断。

### 5.1 经典 Encoder–Decoder Transformer

机器翻译模型可以是：

```text
源语言 tokens
  -> Transformer Encoder
  -> source memory
  -> Transformer Decoder 自回归生成
  -> 目标语言 tokens
```

Decoder 一方面读取已经生成的目标 token，另一方面通过 cross-attention 读取 Encoder memory。

### 5.2 Decoder-only 大语言模型

Gemma、Llama 等现代大语言模型通常是 decoder-only：

```text
已有 tokens -> causal Transformer -> 预测下一个 token
```

它不需要独立的 Transformer Encoder，因为 prompt 和历史输出放在同一个 causal sequence 中。它仍然可以“理解输入并生成输出”，但架构分类是 decoder-only。

### 5.3 视觉 Encoder

SigLIP 接收像素并产生 image tokens，所以称为 Vision Encoder：

```text
pixels -> patch embedding + Vision Transformer -> image features
```

这是一个完整的、具有多层表示学习能力的编码网络。

### 5.4 Action Encoder / Decoder

GR00T 的命名层级是“动作模态适配器”：

```text
raw/noisy action -> embodiment-specific Action Encoder -> DiT tokens
DiT hidden       -> embodiment-specific Action Decoder -> action velocity
```

因此这里的 Encoder/Decoder 指动作空间和 DiT hidden space 之间的双向接口，不是语言模型意义上的 Transformer Encoder/Decoder。

### 5.5 为什么 `action_in_proj` 通常不单独叫 Encoder

PI0.5 的：

```text
action_in_proj  = Linear(D, 1024)
action_out_proj = Linear(1024, D)
```

从最宽泛功能看，它们确实做了编码和解码；但它们只是一个线性投影层，没有独立的深层表示学习结构，所以代码和论文通常称为 input/output projection，而不是完整的 Action Encoder/Decoder。

GR00T 的 embodiment-specific Action Encoder/Decoder 含有多层、非线性和不同机器人的参数选择逻辑，因此采用 Encoder/Decoder 命名更自然。

## 6. Tokenizer 不属于神经网络 Encoder

Tokenizer 做的是离散符号预处理：

```text
文本 -> token IDs
```

它通常没有 Transformer 层，也不把文本变成语义 hidden vector。真正把 token ID 变成 hidden representation 的是 embedding 层和后续 Transformer。

所以：

```text
Tokenizer：符号切分和编号
Embedding：ID 查表得到初始向量
Transformer：让向量结合上下文形成语义表示
```

## 7. 看到 Encoder/Decoder 时的判断方法

遇到一个名为 Encoder 或 Decoder 的模块，依次问：

1. 它处理的原始模态是什么：文本、图像、state 还是 action？
2. 它输出的是 token sequence、单个 latent，还是最终任务结果？
3. 它是完整多层网络，还是只有一个 projection？
4. 这里讨论的是整个模型结构，还是某个模态的输入/输出适配器？

不要仅凭名称判断它在全模型中的地位。

## 8. 与后续模型的对应关系

| 模型 | 输入表示 | 核心 Transformer | 输出表示 |
| --- | --- | --- | --- |
| PI0.5 | SigLIP image tokens、Gemma text/state tokens、action projection | Gemma Prefix Expert + Action Expert | flow velocity |
| GR00T | Qwen VLM memory、state/action tokens | AlternateVLDiT | flow velocity |
| VLASH | 沿用原 VLA 输入，但将 state/action 对齐到未来 offset | 通常不改变 backbone | temporally aligned action chunk |
| VLA-JEPA | 当前图像/语言、latent/action tokens、state/action tokens | Qwen3-VL + world predictor + DiT | future latent prediction + action velocity |

下一章继续解释这些 token 如何通过 QKV、self-attention、cross-attention、多专家 attention 和条件调制交换信息。
