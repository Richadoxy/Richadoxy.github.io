# 表示学习、CLIP 与对比学习

这篇文章连接两个常被混在一起的问题：模型怎样把原始输入变成有意义的表示，以及不同模态怎样进入同一个语义空间。

建议先阅读《Token、Embedding、Encoder/Decoder 与 Latent》和《Transformer 与 Attention》。

## 1. 什么是表示学习

神经网络通常不会直接在原始像素或文字字符串上完成所有任务，而是先把输入变成内部表示：

```text
原始输入 x -> Encoder f -> representation z -> task head -> output
```

一个好的表示会保留任务所需的信息，同时压缩无关细节。例如触觉图像中的颜色可能来自传感器内部照明，而局部形变和纹理才与材料属性更相关。

`representation`、`feature`、`embedding` 和 `latent` 在论文中经常指向相近的张量，但侧重点不同：

| 名称 | 常见侧重点 |
| --- | --- |
| feature / hidden state | 网络中间层产生的数值特征 |
| embedding | 可用于比较、检索或作为下游输入的向量 |
| latent | 对原始数据的压缩或抽象表示 |
| token representation | 序列中某一个位置的 hidden vector |

## 2. 监督学习、预训练与微调

监督学习使用输入和明确目标训练模型：

```text
图像 -> 分类器 -> 物体类别
机器人观测 -> policy -> 示范动作
```

预训练则先利用规模更大、任务更宽的数据学习可复用表示，再迁移到具体任务：

```text
大规模图文数据 -> 视觉/语言表示
                  ↓
少量机器人数据 -> VLA 或触觉策略
```

迁移时常见三种方式：

- 冻结 encoder，只训练新 head；
- 使用 LoRA 等参数高效方法微调部分权重；
- 端到端更新全部或大部分网络。

这三种方式分别偏向稳定、低成本和更强适应能力。

## 3. CLIP 是什么

CLIP 是视觉—语言双编码器：

```text
图像 I -> Vision Encoder -> z_I
文本 L -> Text Encoder   -> z_L
```

两侧输出会被投影并归一化到同一维度。相似度通常用余弦相似度或归一化向量的点积：

\[
s(I,L)=\frac{z_I^T z_L}{\lVert z_I\rVert\lVert z_L\rVert}
\]

CLIP 的主要产物不是一段生成文本，而是一个可以比较图像和文本的语义坐标系。

OpenCLIP 是 CLIP 类模型的开源训练与实现生态。SigLIP 采用不同的配对损失形式，但在 VLM 中也经常承担视觉编码器和视觉语义预训练的角色。

## 4. 正样本、负样本与 InfoNCE

一个 batch 中，匹配的图文对是正样本，不匹配的组合是负样本：

```text
(image_i, text_i) -> positive
(image_i, text_j) -> negative, i != j
```

以图像检索文本为例，InfoNCE 形式可以写成：

\[
\mathcal L_i=-\log
\frac{\exp(s(z_{I_i},z_{L_i})/\tau)}
{\sum_j\exp(s(z_{I_i},z_{L_j})/\tau)}
\]

训练目标是提高正确配对的相似度，并降低 batch 内错误配对的相似度。实际实现通常还会计算文本到图像的对称方向。

温度参数 \(\tau\) 控制相似度分布的尖锐程度。batch 的构造也非常重要：错误负样本、重复描述或过强的数据捷径都可能改变模型真正学到的内容。

## 5. Global embedding 与 patch tokens

ViT 内部产生一组 patch token：

```text
image -> patches -> [N, d] token representations
```

CLIP 式对比学习通常还会把它们池化或读取为一个全局向量：

```text
[N, d] -> pooling / CLS -> [d] global embedding
```

两者用途不同：

- patch tokens 保留局部空间信息，适合 VLM、检测和细粒度融合；
- global embedding 适合检索、相似度计算和样本级对齐。

因此“CLIP 输出一个 embedding”不表示它内部没有 Transformer token；“VLM 使用 CLIP 视觉塔”也不表示它只接收一个全局向量。

## 6. 对齐不等于融合

对比学习和 cross-attention 不在同一层级：

```text
对比学习：训练目标，约束表示空间
Attention：前向算子，让一组 token 读取另一组信息
```

两个独立 encoder 可以完全不交换 token，只通过对比 loss 对齐最终 embedding。反过来，两个模态可以在 Transformer 中进行 cross-attention，却没有任何显式对比损失。

需要分别问：

1. 前向传播时，模态在哪里交换信息？
2. 训练时，什么 loss 要求它们在语义上对应？

## 7. 从图文对齐扩展到触觉

TVL 和 AnyTouch 都把 CLIP 语义空间扩展到触觉：

```text
视觉 -> z_V
文本 -> z_L
触觉 -> z_T
```

对存在配对关系的模态计算：

```text
L_TV + L_TL + L_VL
```

这样触觉 encoder 即使没有直接输出语言，也能让“粗糙地毯”的触觉表示靠近对应图像和文本。

但语义对齐可能压缩掉控制需要的细粒度信号。两个触觉帧都可以被描述为 `hard`，其中一个却可能已经出现微小滑移。因此触觉 VLA 往往还需要局部 token、时间建模和动作监督。

## 8. 如何判断对齐是否成功

常见评估包括：

- 跨模态 Top-k 检索；
- zero-shot 分类；
- linear probing；
- embedding 可视化；
- 下游任务迁移。

这些指标回答的问题不同。检索准确率高说明表示进入了较合理的语义位置，但不能直接证明：

- 模型会生成机器人动作；
- 策略会在闭环执行时使用该模态；
- 模型能感知滑移或精确控制接触力。

## 9. 阅读 CLIP 类工作时的检查表

1. 每种模态的原始输入是什么？
2. encoder 输出 patch tokens 还是 global embedding？
3. 哪些配对是正样本，负样本从哪里来？
4. 对齐的是哪些模态组合？
5. 哪些 encoder 冻结，哪些参数更新？
6. 是否存在伪标签或视觉猜测触觉属性的偏差？
7. 评估证明的是表示、生成、决策还是控制？

一句话总结：

> CLIP 类方法通过配对监督建立共享语义空间；Attention 决定一次前向中信息怎样流动；VLA 还需要把这些表示连接到动作学习和在线控制。
