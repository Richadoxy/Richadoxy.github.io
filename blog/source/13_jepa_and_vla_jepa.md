# 13. JEPA、World Model 与 VLA-JEPA

> 建议先读：[VLA 总览](27_vla_overview.md)与[模型训练基础](22_model_training_basics.md)。本文属于进阶训练框架，不是初入 VLA 时必须先掌握的动作架构。

JEPA 关注的不是重建每个未来像素，而是在 latent space 中预测目标表示。

VLA-JEPA 则将这种预测监督引入 VLA，使视觉语言表示不仅能生成动作，还能表达与状态转移相关的动态信息。

完整框图见 [vla_jepa_structure.drawio](vla_jepa_structure.drawio)。

## 1. JEPA 的通用定义

JEPA 是 Joint-Embedding Predictive Architecture。通用结构是：

```text
可见 context
  -> Context/Online Encoder
  -> context representation
  -> Predictor
  -> predicted target latent
           ↕ latent-space loss
目标区域/未来数据
  -> Target Encoder
  -> target latent
```

target 不一定是未来：

- I-JEPA 可以预测同一图像的被遮挡区域表示；
- V-JEPA 可以预测视频中未来或被遮挡时空区域表示；
- VLA-JEPA 预测未来 world-state latent。

因此更准确的定义是：

> 根据可见 context 预测不可见 target 的 latent representation，而不是直接重建原始像素。

## 2. 为什么预测 latent 而不是 pixels

像素变化可能来自：

- 光照；
- 背景；
- 相机运动；
- 纹理；
- 与操作无关的人或物体。

如果直接预测未来 pixels，loss 容易被这些高维细节支配。语义 latent 可以压缩低层噪声，更关注对象、动作和状态变化。

但 latent 并不天然保证“只含正确语义”；最终学到什么仍取决于 target encoder、数据和预测任务。

## 3. Teacher 与 Student

典型 JEPA 经常有：

```text
Student/Context Encoder：处理部署可见信息，接收梯度
Teacher/Target Encoder：产生稳定 target，不接收该 loss 的梯度
```

Teacher 参数来源有多种：

- student 参数的 EMA；
- 已预训练并冻结的 encoder；
- 更大或有 privileged information 的模型。

因此“teacher 一定是固定预训练权重”并不是 JEPA 的通用定义。

## 4. Predictor 是什么

Predictor 根据 context latent 和相关条件预测 target latent：

```text
predicted_target = Predictor(context_latent, conditions)
```

它是 JEPA 的核心组件之一，但不一定是最终部署主干。很多 JEPA 训练结束后会丢弃 predictor，只保留学到好表示的 encoder。

在 VLA-JEPA 中，world predictor 是训练辅助模块；部署 action policy 时可以移除。

## 5. World model 监督

world model 学习：

```text
当前/历史世界状态 + action-like condition
  -> 未来世界状态
```

如果在 latent space 中表示：

```text
s_t = Encoder(observation_t)
s_hat_(t+1) = WorldPredictor(s_t, z_t)
L_WM = distance(s_hat_(t+1), stopgrad(s_(t+1)))
```

这里：

- `s_t` 是抽象 world-state latent；
- `z_t` 是解释状态转移的 latent action/condition；
- `s_(t+1)` 是 target encoder 从真实未来帧产生的监督目标；
- loss 不直接比较未来 RGB pixels。

“world model 监督”就是用未来状态的真实 latent 约束模型，使中间表示包含可预测的动态结构。

## 6. VLA-JEPA 的 Teacher 路径

论文中的 target encoder 是冻结的 V-JEPA2：

```text
training video frames I_t0:tT
  -> Frozen V-JEPA2 F
  -> world-state latents s_t0:tT
```

多视角特征会组合成统一 world-state representation。该路径：

- 只在训练时使用；
- stop-gradient；
- 不由 `L_WM` 更新；
- future frames 只用于构造 target。

VLA-JEPA 不是两个同构 Qwen encoder 做 teacher/student。Qwen3-VL 是 online VLM，V-JEPA2 是异构的 frozen target encoder。

## 7. VLA-JEPA 的 Student/Online 路径

Qwen3-VL 只接收：

```text
当前多视角图像 I_t0
+ language instruction
+ learnable latent/action special tokens
```

特殊 `<latent_i>` token 的 hidden representations 记为：

```text
z_t0, z_t1, ..., z_t(T-1)
```

它们应该表达状态从一步到下一步如何变化，而不是压缩未来图像。

## 8. World Predictor 如何使用 latent action

自回归 world predictor 接收：

```text
历史/当前 teacher world states s_t0:i
+ VLM latent actions z_t0:i
```

预测：

```text
s_hat_t1:i+1 = p_WM(s_t0:i, z_t0:i)
```

并计算：

```text
L_WM = Σ distance(s_hat_tk, stopgrad(s_tk))
```

因此不是直接要求：

```text
VLM hidden z_t == teacher future state s_t+1
```

而是要求：

> `z_t` 必须提供足够的状态转移信息，使 world predictor 能从历史状态预测未来状态。

## 9. Leakage-free 为什么重要

如果 online VLM 同时看到未来帧，它可能把 `<latent_i>` 退化成“未来图像压缩包”，而不必学习动作或状态转移语义。

VLA-JEPA 的约束是：

```text
未来 frames -> 只能进入 frozen target path
未来 frames -X-> Qwen3-VL student
```

world predictor 的 time-causal mask 还保证预测某一时刻时，不能读取更未来的 target latent。

这叫 leakage-free state prediction。

## 10. Teacher forcing 不等于 Teacher network

Teacher network 指提供 target 的 V-JEPA2 encoder。

Teacher forcing 指训练自回归 predictor 时，使用真实历史 state latent 作为下一步输入：

```text
训练：用真实 s_t 预测 s_(t+1)
自由 rollout：可能用预测 s_hat_t 继续预测下一步
```

二者只是都带 teacher 一词，概念不同。

## 11. Action generation 路径

Qwen3-VL 还有一个 `<action>` token，其 hidden representation `z_a` 汇总：

```text
当前图像
语言
latent action tokens
```

`z_a` 作为 DiT Action Head 的条件：

```text
robot state -> State Encoder ------┐
                                   ├-> DiT-B -> predicted flow velocity
noisy action + flow time ----------┘
                        ↑
                    z_a condition
```

Flow loss：

```text
L_FM = MSE(v_pred, velocity_target)
```

## 12. 两类数据和联合目标

### Human video

没有机器人 action label，只使用：

```text
L_WM
```

训练 latent action 和状态转移表示。

### Robot demonstration

有图像、state 和 action，可以使用：

```text
L = L_FM + β L_WM
```

梯度关系：

```text
L_WM -> world predictor + Qwen latent-action pathway
L_FM -> DiT Action Head + Qwen action-conditioning pathway
共享 Qwen 参数同时受到两类目标影响
Frozen V-JEPA2 永远不更新
```

## 13. 推理时保留什么

机器人部署不需要未来视频 target，也不需要计算 world-model loss：

```text
保留：当前图像 + language -> Qwen3-VL -> z_a -> DiT -> action chunk

移除：Frozen V-JEPA2 Target Encoder
移除：World Predictor
移除：future training video
移除：L_WM
```

所以 world-model branch 的主要作用是训练表示，不是部署时在线预测未来画面或规划 rollout。

## 14. Teacher/Student 在 RL 中的其他含义

RL 中常见的 teacher/student 可能是：

- privileged teacher 使用仿真真值，student 只使用部署传感器；
- 大 teacher policy 蒸馏到小 student policy；
- planner/expert 产生示范，student 模仿；
- target network 提供稳定 bootstrap target。

Actor–Critic 不应简单等同于 teacher–student：Critic 评价 action，Actor 产生 action；二者有互相依赖的优化关系，但 Critic 不一定是一个固定老师。

## 15. 对 VLA-JEPA 的准确总结

可以用下面一句话概括：

> 冻结 V-JEPA2 将训练视频编码为 future world-state targets；Qwen3-VL 从当前图像和语言产生 latent-action representations；World Predictor 利用历史状态和这些 latent actions 预测 future latents，并以 `L_WM` 监督；机器人数据上再通过 `<action>` condition 和 DiT 计算 `L_FM`，两种 loss 联合训练共享 VLM，部署时只保留 VLM 与 Action Head。

参考论文：[VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model](https://arxiv.org/abs/2602.10098)。
