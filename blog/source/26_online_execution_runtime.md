# VLA 在线执行：同步、异步与 Action Chunk 调度

模型输出正确 action 只是机器人系统的一半。另一半是：观测何时采集、推理期间机器人在做什么、新 chunk 从哪个状态开始执行。

## 1. 一条端到端延迟链

```text
camera exposure
-> image decode / preprocessing
-> request serialization / network
-> server queue
-> VLM prefix computation
-> action sampling / denoising
-> response transport
-> actor receives result
-> controller handoff
```

真正影响动作时间对齐的是 observation timestamp 到 chunk handoff 的总延迟，而不只是一次模型 forward。

## 2. 同步执行

```text
采集 observation
-> 等待 inference
-> 得到 chunk
-> 执行 chunk
-> 再采集 observation
```

优点是观测和规划起点关系清晰；缺点是推理期间可能停顿，动作不连续。

## 3. Naive Async

```text
t 时刻采集 observation/state
旧 chunk 继续执行
后台计算新 chunk
t+Delta 时切换
```

如果新策略仍以 \(s_t\) 为起点规划，但真正执行时机器人已到 \(s_{t+\Delta}\)，就产生 prediction–execution misalignment。

## 4. Active 与 Pending Chunk

一个最小异步 Actor 通常维护：

```text
active_chunk：当前正在执行
active_index：已经执行到哪里
pending_chunk：后台推理得到、等待切换的新轨迹
request_id：防止旧响应覆盖新请求
```

切换应在清晰的控制边界原子完成。仅仅使用两个线程，并不自动保证时间对齐。

## 5. Replan 与 Overlap

设预测 horizon 为 \(H\)，执行 \(K\) 步后触发下一次推理：

```text
overlap = H - K
```

较大的 overlap 给后台推理更多时间，但也更频繁调用模型。若推理在 active chunk 耗尽前仍未完成，系统还需要定义继续保持、重复末动作或进入安全状态的行为。

## 6. Stale Observation

异步系统中的图像、state 和 action 可能来自不同时间。每个 request 最少应携带：

- observation timestamp；
- state timestamp 和语义；
-预计 handoff 时刻；
- active chunk identity/index；
- policy/control rate；
- request identity。

否则即使数组 shape 完全正确，也无法判断新动作对应哪个执行起点。

## 7. 插值、平滑与低通滤波

需要区分：

- chunk 内从 policy rate 到 control rate 的重采样；
- chunk 边界的 blending；
- 对控制命令做低通滤波。

后两者会改变实际执行轨迹。如果 future-state estimation 基于原 active chunk，额外平滑可能使预测 handoff state 与真实状态再次偏离。

## 8. 延迟应该怎样测

至少记录：

```text
capture -> request ready
request -> server start
prefix prefill
all denoising steps
response -> actor receive
actor receive -> actual handoff
```

报告 p50、p95、p99，而不只是平均值。异步训练覆盖范围通常要依据完整 handoff delay 的高分位数设置。

## 9. Dataset Playback 不等于真实 Runtime

离线逐帧读取数据很容易无意中得到理想化结果：

- 推理期间 dataset index 没有前进；
- 每次都使用与 target 完美对齐的 state；
- 没有控制器和网络延迟；
- 没有旧 chunk 继续执行。

若要测试异步状态机，dataset playback 也需要一套虚拟 controller clock，并明确推理期间时间如何前进。

## 10. 与 VLASH 的关系

本篇描述通用 runtime 问题。VLASH 的具体选择是：

```text
保留较早的视觉 observation o_t
+ 提供预计 handoff state s_hat_(t+Delta)
-> 生成从未来执行点开始的 action chunk
```

因此应该先理解本篇的 active/pending chunk 和时间戳，再阅读 VLASH 的 temporal-offset augmentation。
