# 12. VLASH：Temporal Offset 与异步执行

> 建议先读：[VLA 在线执行](26_online_execution_runtime.md)。本篇聚焦 VLASH 如何修正 prediction–execution temporal misalignment。

[`07_vlash_integration.md`](07_vlash_integration.md) 记录 BrainCo-IL 当前配置和代码范围；本章专门解释 VLASH 为什么这样设计，以及部署端发生什么。

完整训练与运行时框图见 [vlash.drawio](vlash.drawio)。

## 1. 同步推理的问题

同步 action chunk 流程是：

```text
采集 observation
  -> 停止或等待模型推理
  -> 得到 action chunk
  -> 执行 chunk
  -> 再推理
```

优点是预测时 state 与执行起点一致，缺点是机器人会在推理阶段停顿，反应延迟大。

## 2. Naive async 的错位

直接异步化：

```text
时刻 t 采集 (o_t,s_t)
旧 chunk 继续执行
后台推理新 chunk
时刻 t+Δ 推理完成并切换
```

问题在于新 chunk 是按 `s_t` 规划的，但真正开始执行时机器人已经到达 `s_(t+Δ)`：

```text
预测起点：s_t
执行起点：s_(t+Δ)
```

这就是 prediction–execution temporal misalignment，可能导致 chunk 边界跳变、不连续和控制不稳定。

## 3. VLASH 的核心

VLASH 在推理开始时仍使用当前视觉：

```text
o_t = 当前图像 + 任务
```

但将 robot state 向前滚动到预计切换时刻：

```text
s_t -> estimated s_(t+Δ)
```

模型调用变成：

```text
policy(o_t, estimated s_(t+Δ))
  -> actions starting at t+Δ
```

它主要改变时间对齐和运行时调度，不是增加另一个 correction Transformer。

## 4. offset δ 为什么变化

异步推理延迟不是永远固定的。它会受以下因素影响：

- 图像预处理和网络传输；
- GPU 调度；
- prefix token 数量；
- 去噪步数；
- 系统负载和线程调度；
- actor 切换时机。

所以训练不能只让模型适应一个固定 offset。VLASH 从区间采样：

```text
δ ∈ {0,1,...,Δ_max}
```

使同一个 checkpoint 同时覆盖同步和多种异步延迟。

## 5. 为什么需要 `offset max`

`Δ_max` 定义训练覆盖的最大错位范围：

```text
δ=0       同步或近同步
δ=1...N   不同推理延迟
δ=Δ_max   训练支持的最大延迟
```

部署时应让实际 handoff delay 尽量落在训练范围内。过小会出现 OOD future state；过大会浪费数据窗口、增加 padding/边界样本损失，也可能让旧图像与未来 state 的语义差距过大。

选择时应测量：

```text
Δ = chunk_handoff_timestamp - observation_timestamp
delay_steps = ceil(Δ_seconds × policy_rate)
```

然后根据 p95/p99 延迟设 `max_delay_steps`，而不是只用一次模型 forward 的平均 90 ms。

## 6. Temporal-offset augmentation

普通样本：

```text
(o_t, s_t) -> A_t:t+H-1
```

VLASH 样本：

```text
固定 observation o_t
采样 offset δ
state  = s_(t+δ)
target = A_(t+δ):t+δ+H-1
```

state 和 action 一起平移，视觉/语言不平移。

模型因此看到：

```text
同一个 o_t
+ 不同 s_(t+δ)
-> 不同未来 action chunk
```

这会迫使策略使用 robot state，而不是只凭图像输出动作。

## 7. `N = Δ_max + 1`

如果 offset 集合是：

```text
{0,1,...,Δ_max}
```

它包含的整数数量为：

```text
N_δ = Δ_max + 1
```

例如：

```text
Δ_max = 4
offsets = {0,1,2,3,4}
N_δ = 5
```

这里的 N 是训练 offset branches 数量，不是 action horizon，也不是推理去噪步数。

## 8. Shared stale observation 的含义

`shared stale observation o_t` 表示所有 offset branches 复用同一个较早的视觉观测：

```text
branch δ=0: (o_t, s_t)   -> A_t
branch δ=1: (o_t, s_t+1) -> A_t+1
branch δ=2: (o_t, s_t+2) -> A_t+2
...
```

`fixed while offset δ changes` 指：

```text
不变：图像 o_t、语言任务
变化：state_(t+δ)、action target_(t+δ)
```

shared-observation packing 可以只编码一次昂贵的视觉 prefix，再连接多个彼此隔离的 suffix branches。它是训练效率优化，不是异步推理成立的必要条件。

当前 BrainCo-IL 只实现随机单 offset，尚未实现多 branch shared-prefix attention，详见 [`07_vlash_integration.md`](07_vlash_integration.md)。

## 9. 为什么未来图像不预测

在推理延迟期间，机器人自己的未来 state 可以由已知旧 chunk 近似推算：

```text
delta action：s_(t+Δ) ≈ s_t + Σ a_i
absolute target：使用切换时刻计划到达的 endpoint
```

但未来图像包含：

- 物体是否滑动或掉落；
- 人是否进入画面；
- 相机抖动；
- 接触结果；
- 环境中其他主体的运动。

这些不能仅由已发送的机器人 action 确定。预测 future image 需要额外 world model，会引入显著计算和误差。VLASH 选择一个务实近似：

```text
视觉仍是 o_t
robot state 更新为 estimated s_(t+Δ)
```

所以它只修正可可靠 roll forward 的 proprioceptive misalignment，不声称解决全部环境观测滞后。

## 10. 部署端如何提供 `state_(t+offset)`

Actor 已经知道正在执行的 active chunk，以及 background inference 预计在哪一步完成。

对 absolute joint targets，可取预计 handoff 时 active chunk 的目标位置：

```text
future_state = active_chunk[handoff_index]
```

对 delta actions，可从当前实测 state 累加 handoff 前仍将执行的动作：

```text
future_state = current_state + sum(remaining_delta_actions)
```

如果控制器提供轨迹前向模型，也可以用更精确的 state rollout。BrainCo-IL 部署 contract 当前约定传入未归一化的 raw absolute state，由 plugin 继续执行 normalization。

## 11. Inference node 与 Actor node 的通信

适合 VLASH 的最小状态机是：

```text
Actor:
  active_chunk + current_index
  -> 继续发送控制命令
  -> 到触发点生成 inference request

Inference request:
  observation image o_t
  estimated handoff state s_hat_(t+Δ)
  prompt / metadata

Inference node:
  后台运行 PI0.5 flow sampling
  -> 返回 pending_chunk + request identity

Actor:
  在安全边界 atomic switch
  pending_chunk -> active_chunk
```

关键不是堆很多保护参数，而是保证：

- active chunk 执行不被推理阻塞；
- request 中 future state 对应预计 handoff 时刻；
- 旧请求不会覆盖更新请求；
- chunk 切换是原子的。

## 12. Dataset inference 也可以异步

数据集模式可以把“当前 dataset frame”看作 observation source，把 action 执行进度看作虚拟 controller clock：

```text
读取 frame t 发起后台推理
虚拟 Actor 继续推进已有 chunk / dataset index
推理完成时按 handoff index 切换
```

这样可以测试相同 active/pending 状态机，但必须明确 dataset index 如何随执行步数前进，不能一边异步推理一边始终停留在同一帧。

## 13. 平滑与插值

VLASH 模式下：

- 可以保留 chunk 内 policy rate 到 control rate 的线性插值；
- 应关闭跨 chunk boundary blending，因为它会改变 future-state 对应的实际执行轨迹；
- 应关闭低通滤波，因为它使实际 state rollout 偏离已计划 chunk；
- startup transition 是否保留，应确保不会破坏首个 future-state 估计。

一句话：

```text
chunk 内重采样可以保留
会跨边界或改变已知轨迹的平滑应关闭
```

## 14. VLASH 没有改变什么

对 PI0.5：

- flow matching loss 不变；
- Action Expert 主干不变；
- 去噪循环不变；
- action output 语义不变。

主要变化是：

```text
训练数据时间对齐
+ 可选 future-state conditioning
+ 异步 action chunk scheduler
+ 部署握手语义
```

这也是它能较低成本集成到现有 VLA 的原因。

参考资料：[VLASH 论文（arXiv:2512.01031）](https://arxiv.org/abs/2512.01031)；当前仓库的实际支持范围以 [`07_vlash_integration.md`](07_vlash_integration.md) 为准。
