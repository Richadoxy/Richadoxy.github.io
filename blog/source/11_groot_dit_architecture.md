# 11. GR00T 与 DiT Action Head

GR00T 和 PI0.5 都使用视觉语言条件生成连续 action chunk，但二者的 VLM–Action 连接方式不同。

一句话概括：

```text
PI0.5：VLM Prefix Expert 与 Action Expert 在每层做 masked joint attention
GR00T：VLM 先产生 condition memory，独立 DiT 再用 cross-attention 读取它
```

本章以 [groot_n1d7_structure.drawio](groot_n1d7_structure.drawio) 中的 GR00T N1.7 示例为准：`B=1`、3 个相机、有效 state/action 为 56D、有效 horizon 为 16。

## 1. 总体结构

```text
3 路图像 + 语言
  -> Qwen3-VL processor
  -> Cosmos-Reason2-2B / Qwen3-VL backbone
  -> VLM condition memory [1,S,2048]
                              │
                              │ Cross-Attention K/V
                              ▼
robot state -> State Encoder -> state token ─┐
                                             ├-> AlternateVLDiT
noisy action + t -> Action Encoder -> tokens ┘
                                                   ↓
                                         Action Decoder
                                                   ↓
                                          predicted velocity
```

VLM 属于 System 2 视觉语言推理，DiT action head 属于 System 1 连续动作生成。

## 2. Vision + Language processor output

Processor 负责把图像和语言转换成 Qwen3-VL 能接收的格式，常见字段包括：

```text
input_ids       [B,S]
attention_mask  [B,S]
pixel_values    动态视觉 patch 数据
image_grid_thw  每路图像/视频的网格信息
```

其中：

- `input_ids` 是文本、视觉占位符和 chat template 对应的 token IDs；
- `attention_mask` 标记序列中哪些位置有效；
- `S` 是 processor 产生的总序列长度，取决于图像网格、相机数、文本长度和视觉 merge 规则；
- `S` 不是简单固定为 `3×256`。

经过 VLM 后得到：

```text
backbone features [B,S,2048]
```

这里每个 VLM token 用 2048 维 hidden vector 表示。

## 3. State Encoder

机器人 state 先按统一容量 padding：

```text
有效 state [1,1,56]
  -> padding
模型 state [1,1,132]
```

然后根据 embodiment ID 选择对应 MLP 权重：

```text
[1,1,132]
  -> CategorySpecificMLP
[1,1,1536]
```

最终形成一个 state token。它被拼到 DiT query stream 中，通过 self-attention 影响所有 action tokens。

这里称为 State Encoder，是因为它不仅做线性投影，还包含 embodiment-specific 参数选择和多层非线性映射。

## 4. Action Encoder

训练时先构造：

```text
valid action [1,16,56]
  -> pad horizon 和 action dimensions
model action [1,40,132]
```

flow matching 产生同形状 noisy trajectory：

```text
x_t = (1-t)ε + t·a
```

Action Encoder 执行：

```text
noisy action [1,40,132]
  -> embodiment-specific projection
  -> 与 sinusoidal timestep embedding 融合
  -> MLP
  -> action tokens [1,40,1536]
```

再加 action-position embedding，区分 chunk 中第 0、1、...、39 个时间位置。

## 5. `40×132` 是模型容量，不是实际机器人尺寸

GR00T 为多种 embodiment 使用统一上限：

```text
max_action_horizon = 40
max_action_dim     = 132
```

对 16 步、56D 机器人：

```text
有效区域 [:,0:16,0:56]
padding 区域由 action_mask 排除
```

推理后也只切出：

```text
[1,40,132] -> [1,16,56]
```

所以 `model tensor` 指统一网络内部的 padded tensor。

## 6. AlternateVLDiT 是什么

DiT 是 Diffusion Transformer：使用 Transformer 对带噪数据建模，并根据 diffusion/flow timestep 预测去噪结果或速度场。

在 GR00T 中：

```text
state token [1,1,1536]
action tokens [1,40,1536]
  -> concat
DiT hidden/query stream [1,41,1536]
```

N1.7 示例配置为：

```text
num_layers          = 16
num_attention_heads = 32
attention_head_dim  = 48
hidden width        = 32 × 48 = 1536
```

`num_attention_heads=32` 表示 attention 并行使用 32 组 query subspaces；`attention_head_dim=48` 表示每个 head 的 Q/K/V 特征宽度为 48。

这两个配置与 DiT 的 attention 结构直接相关，但不是所有 DiT 都必须使用 32×48。

## 7. AlternateVLDiT 的 block 交替方式

16 个 blocks 交替执行：

```text
偶数 block：Cross-Attention
  Q   = state/action hidden stream [1536]
  K/V = VLM memory [2048]

奇数 block：Self-Attention
  state token 与 action tokens 相互交互
```

Cross-Attention block 还在文字和图像 memory 间交替：

```text
cross(text/non-image)
-> self
-> cross(image)
-> self
-> ...
```

因此 `AlternateVLDiT` 与 Gemma 300M Action Expert 处于相似功能层级：二者都负责生成动作。但它们不是同一种内部结构：

```text
Gemma 300M Action Expert：PI0.5 paired multi-expert Transformer 的一半
AlternateVLDiT：独立的 diffusion/flow Transformer，通过 cross-attention 读取 VLM memory
```

## 8. timestep 在 GR00T 中使用两次

第一条路径在 Action Encoder 内：

```text
action embedding + sinusoidal t embedding -> MLP -> action token
```

第二条路径在每个 DiT block：

```text
t -> Timestep Encoder -> 1536-D condition
  -> AdaLayerNorm scale/shift
```

GR00T AdaLayerNorm 的核心是：

```text
normalized = LayerNorm(x)
modulated  = normalized * (1 + scale(t)) + shift(t)
```

PI0.5 adaRMS 还会产生 residual gate；GR00T 这里主要是 scale/shift。二者都属于 timestep-conditioned adaptive normalization。

## 9. Action Decoder

DiT 输出先投影到 decoder hidden：

```text
[1,41,1536] -> [1,41,1024]
```

再由 embodiment-specific Action Decoder 映射到统一 action space：

```text
[1,41,1024] -> [1,41,132]
```

切出最后 40 个 action positions：

```text
predicted velocity [1,40,132]
```

训练时对有效 mask 计算 flow MSE，推理时执行 4-step Euler integration，再 slice 和 unnormalize 得到 `[1,16,56]`。

## 10. 与 PI0.5 的核心区别

| 维度 | PI0.5 | GR00T N1.7 |
| --- | --- | --- |
| VLM–Action 耦合 | 每个 paired layer 的 joint attention | VLM 先输出 memory，DiT cross-attention 读取 |
| action network | Gemma 300M Action Expert，宽度 1024 | AlternateVLDiT，宽度 1536 |
| state | 默认离散化进 VLM prompt | 连续 state 经 embodiment MLP 成 token |
| 多机器人适配 | 依赖数据/配置及 action projection | embodiment-specific encoder/decoder |
| timestep | adaRMS scale/shift/gate | action-token fusion + AdaLayerNorm scale/shift |
| 内部 action 容量 | 由当前 action_dim/horizon 配置 | 统一 padded capacity + mask/slice |

阅读 GR00T 时应把 VLM backbone、processor、state/action adapters 和 DiT 分开看，避免把 2048-D VLM memory 与 1536-D DiT hidden width 混为一谈。

参考实现：[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)。该仓库版本迭代较快，本文的具体 shape 以配套 N1.7 图所记录的配置为准。
