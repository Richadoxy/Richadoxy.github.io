# 从模仿学习到 VLA：架构地图与阅读方法

VLA 不是一个单独网络层，而是一类把视觉、语言和机器人动作连接起来的策略。本篇是整套博客的入口地图。

## 1. VLA 位于哪里

```text
机器学习基础
  -> Transformer / representation learning
  -> 视觉语言模型 VLM
  -> 模仿学习与机器人数据
  -> Vision-Language-Action policy
  -> 在线闭环执行
```

VLM 主要回答“看到了什么、任务要求什么”；VLA 还必须回答“当前机器人应该怎样行动”。

## 2. 一套通用 VLA

```text
images -> Vision Encoder ------┐
                               │
language -> Tokenizer / LLM ---┼-> multimodal backbone
                               │            │
robot state -> adapter --------┘            │ condition
                                            ▼
noisy/action queries -> Action Network -> action chunk
```

不同论文主要在四个位置变化：

1. 模态怎样编码；
2. 视觉语言与动作怎样融合；
3. 动作怎样表示和生成；
4. 训练及部署时加入哪些额外约束。

## 3. 不要把不同层级放在一起比较

```text
PI0.5、GR00T：VLA architecture
Flow matching：动作生成 objective
DiT、Transformer：network architecture
VLASH：时间对齐与异步执行 framework
VLA-JEPA：加入 latent world-model supervision 的训练 framework
TVL、AnyTouch：可作为触觉感知前端的 representation model
```

“哪个更好”只有在比较对象处于同一层级、任务和运行条件一致时才有意义。

## 4. 三条主要动作路线

```text
直接回归 / MSE
离散 action tokens / autoregressive generation
连续生成 / diffusion or flow matching
```

动作生成方式决定训练 target 和推理延迟；VLM backbone 则决定视觉语言表示。二者是可以组合的两个设计维度。

## 5. 两种典型 VLM—Action 连接

### PI0.5

```text
image/language/state prefix
+ noisy action suffix
-> paired multi-expert joint attention
-> velocity
```

### GR00T

```text
image/language -> VLM memory
state/action -> DiT query stream
-> cross-attention to VLM memory
-> velocity
```

两者都由视觉语言条件生成连续动作，但内部耦合方式不同。

## 6. 表示学习和动作学习的关系

CLIP、TVL、AnyTouch 等模型学习“什么输入在语义上相近”。机器人 policy 还要学习：

- 当前任务需要关注什么；
- 哪个表示与动作选择有关；
- 机器人动作空间和 embodiment；
- 时间连续性和接触反馈。

因此一个强 tactile encoder 可以成为 VLA 前端，但不是完整动作策略。

## 7. 训练图与部署图要分开

训练中可能出现：

- target actions；
- future frames；
- frozen teacher encoder；
- contrastive/generation auxiliary heads；
- optimizer 和 data augmentation。

它们不一定在部署时运行。最终 online graph 应单独画出，只保留实时可得输入和实际调用模块。

## 8. 阅读一篇新 VLA 的六步

### 第一步：输入输出

```text
哪些相机、语言、state 和 history？
输出 position、delta、velocity 还是 torque？
action dimension 与 horizon 是多少？
```

### 第二步：表示

```text
每种模态经过哪个 encoder？
输出 global embedding 还是 token sequence？
原始维度与 hidden width 分别是多少？
```

### 第三步：融合

```text
token concat、joint attention、cross-attention、gate 还是 AdaNorm？
attention mask 允许哪些信息方向？
```

### 第四步：监督

```text
BC MSE、token CE、diffusion、flow、contrastive 或 world-model loss？
哪些参数被冻结或使用 LoRA？
```

### 第五步：推理

```text
一次前向还是多步采样？
一次输出几步，执行几步后 replan？
训练辅助模块是否移除？
```

### 第六步：系统边界

```text
future information 是否泄漏到部署路径？
observation 与 action 是否时间对齐？
真实延迟、控制频率和失败模式是什么？
```

## 9. 本博客的推荐路线

```text
F：基础概念
-> IL：模仿学习
-> VLA：模型架构
-> ENG：OpenPI/BrainCo 工程实践
-> RUN：在线执行与 VLASH
-> TOUCH：触觉表征和触觉 VLA
```

第一次阅读建议完成 F 和 IL，再选择自己关心的分支。已经熟悉 Transformer 和 BC 的读者，可以直接从 VLA00 开始。

## 10. 最终心智模型

```text
原始模态
-> token / embedding / condition
-> multimodal information fusion
-> action distribution or velocity field
-> training objective
-> inference sampling
-> runtime scheduling and closed-loop execution
```

只要能沿这条链说明每个张量、模块和时间点，通常就已经抓住了一套 VLA 的主体结构。
