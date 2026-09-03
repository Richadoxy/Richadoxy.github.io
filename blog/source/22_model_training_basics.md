# 模型训练基础：Forward、Loss、Backward 与微调

这篇文章只建立后续阅读训练代码所需的最小闭环，不绑定 PyTorch、JAX 或某个具体 VLA。

## 1. 一次训练更新

一次标准训练更新可以写成：

```text
batch
  -> forward
  -> prediction
  -> loss(prediction, target)
  -> backward / automatic differentiation
  -> gradients
  -> optimizer step
  -> new model parameters
```

Forward 更新的是当前样本的 hidden representations；optimizer step 才更新 checkpoint 中保存的模型参数。

## 2. Parameter、Activation 与 Gradient

| 对象 | 来源 | 是否长期保存 |
| --- | --- | ---: |
| parameter | 模型中的可训练矩阵和向量 | 是 |
| activation | 当前 batch 前向产生的中间表示 | 通常否 |
| gradient | loss 对参数的导数 | 通常只用于当前更新 |
| optimizer state | Adam 的动量、方差等 | 继续训练时需要 |

Attention matrix、Q/K/V 和 action hidden states 都是输入相关的 activation；生成它们的投影矩阵才是 parameter。

## 3. Loss 在监督什么

Loss 定义模型输出和训练目标之间的差异，例如：

```text
分类             -> cross entropy
连续动作回归     -> MSE
语言生成         -> token cross entropy
对比学习         -> InfoNCE
Flow matching    -> velocity MSE
Latent prediction -> representation distance
```

输入一个条件不代表它就是监督目标。图像、语言、state 和 timestep 都可以影响预测，但 loss 可能只比较动作 velocity。

## 4. Batch、Step 与 Epoch

- sample：一条训练样本；
- batch：一次更新共同使用的一组样本；
- step：通常指一次 optimizer update；
- epoch：完整遍历一次有限数据集。

如果 `batch_size=32`，一次 step 会综合 32 条样本的梯度，而不是连续更新 32 次。

机器人数据常按时间窗口动态采样，因此论文可能只报告 training steps，而不强调 epoch。

## 5. Pretraining 与 Fine-tuning

预训练学习广泛可复用的能力，微调使模型适应目标数据和任务：

```text
互联网图文 / 大规模机器人数据
             -> pretrained checkpoint
             -> 小规模目标机器人数据
             -> task-specific policy
```

加载 checkpoint 时需要区分：

- 完全相同 shape 的参数直接恢复；
- 新增 projector、sensor encoder 或 action head 随机初始化；
- action dimension 改变时，不匹配的输入/输出层需要重建。

## 6. Freeze、Full Fine-tuning 与 LoRA

### Freeze

冻结参数仍然参与 forward，但不由当前 loss 更新。适合保留大模型能力和降低训练成本。

### Full fine-tuning

更新全部主体参数，适应能力强，但显存、数据量和遗忘风险更高。

### LoRA

在原矩阵旁增加低秩增量：

\[
W' = W + BA
\]

原始 \(W\) 可以冻结，只训练较小的 \(A,B\)。LoRA 是“更新哪些参数”的选择，与动作是 28D/56D、loss 是 MSE/flow 属于不同配置维度。

## 7. Main Loss 与 Auxiliary Loss

机器人策略的主目标可能是动作模仿：

```text
L = L_action
```

也可以加入辅助监督：

```text
L = L_action + lambda_1 L_contrastive + lambda_2 L_world_model
```

辅助 loss 的意义是约束中间表示。例如 FuSe 用语言对齐要求策略真正保留触觉语义；VLA-JEPA 用未来 latent 监督动态表示。

需要检查：

- 辅助 loss 更新哪些模块；
- 它是否使用部署时不可见的信息；
- 对应 head 在推理时是否保留。

## 8. Training-only 与 Inference-time

训练框图中的模块不一定都部署：

| 模块 | 训练 | 推理 |
| --- | ---: | ---: |
| target action | 需要 | 不可用 |
| dropout/augmentation | 常用 | 关闭 |
| contrastive target encoder | 可能需要 | 常可移除 |
| JEPA target encoder/world predictor | 需要 | 标准策略中移除 |
| policy backbone/action head | 需要 | 需要 |
| optimizer state | 需要 | 不需要 |

阅读框图时，最好给每条数据边标注 `train only`、`inference only` 或 `both`。

## 9. Checkpoint 为什么不只是模型权重

部署通常只需要模型参数、归一化统计量和配置；继续训练还需要：

- 当前 step；
- optimizer state；
- learning-rate schedule 状态；
- 可选 EMA 参数；
- 数据和模型配置。

所以 `resume training` 与 `load weights for inference` 是两个不同操作。

## 10. 阅读训练代码的顺序

1. batch 中有哪些输入和 target；
2. forward 返回什么；
3. loss 在哪些维度上计算；
4. 哪些参数允许求梯度；
5. optimizer 何时更新；
6. 是否使用 gradient accumulation、clipping 或 EMA；
7. checkpoint 保存和恢复什么。

掌握这条闭环后，再进入具体 JAX 训练入口，会更容易区分通用机器学习概念和项目实现细节。
