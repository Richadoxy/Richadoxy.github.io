# 《T-Rex: Tactile-Reactive Dexterous Manipulation》精读笔记

> Dantong Niu, Zhuoyang Liu, Zekai Wang, Boning Shao 等，arXiv 2026<br>
> [arXiv](https://arxiv.org/abs/2606.17055) · [论文 PDF](https://arxiv.org/pdf/2606.17055) · [项目主页](https://tactile-reactive-dexterous.github.io/) · [官方代码](https://github.com/ZhuoyangLiu2005/T-Rex) · [公开数据](https://huggingface.co/datasets/zekaiwang/trex_dataset)

> 推荐先读：[Attention 专题](19_attention_from_embeddings_to_action_expert.md)、[Flow Matching](10_flow_matching_training_and_inference.md)、[PI0.5 架构](04_pi05_architecture.md)、[在线执行与 Action Chunk](26_online_execution_runtime.md)和[触觉 VLA 阅读地图](16_tactile_vla_reading_guide.md)。本文重点是三专家 MoT、未来视觉 latent、时空触觉编码，以及一条 flow trajectory 如何在低频规划和高频触觉修正之间分段。

## 0. 一句话定位

T-Rex 不只是“给 VLA 多加一个触觉 encoder”。它把一条从高斯噪声到动作块的 flow-matching 轨迹在 \(\tau_{\mathrm{split}}=0.4\) 处分成两段：低频 Action Expert 利用视觉、语言和未来视觉 latent 建立粗动作计划并缓存上下文，高频 Tactile Expert 使用不断刷新的触觉，从同一个中间状态多次完成后半段去噪，从而在不重复运行视觉塔的情况下修正正在执行的 action chunk。

```text
当前头部视觉 + 语言 + future queries ─→ Latent Expert ─┐
                                                        ├─ shared causal attention
腕部视觉 + state + τ + noisy action ──→ Action Expert ─┘
                                                        ↓
                                               x_1 → x_0.4
                                               缓存 latent/action KV
                                                        ↓
实时触觉 + τ + partial action ───────→ Tactile Expert
                                                        ↓
                                               x_0.4 → x_0
                                               executable action chunk
```

最值得记住的判断是：

1. 三个 expert 不是三个互不通信的完整模型，而是每层中三套按 token 类型静态路由的参数；
2. 它们分别产生 Q/K/V，随后在共享 KV 空间中做带 causal mask 的 joint self-attention；
3. Tactile Expert 不是输出一个直接相加的动作 residual，而是继续积分同一条 flow trajectory；
4. “高频/低频”指不同模块的调用频率，不是机器人同时执行两套动作；
5. 论文报告的约 5 Hz/20 Hz 是系统目标，公开代码中的绝对频率由控制配置决定，稳定不变的是约 1:4 的调用比例。

---

## 1. 论文试图解决什么问题

普通 VLA 通常按下面的循环运行：

```text
读取视觉和语言
    ↓
生成一整个 action chunk
    ↓
连续执行若干动作
    ↓
下一次视觉推理
```

这对自由空间运动有效，但在接触丰富操作中存在三个问题。

### 1.1 视觉规划和触觉反馈的自然频率不同

视觉—语言模型需要处理大量 image tokens 和 Transformer layers，单次推理较重；触觉却会在接触建立、滑移和受力突变时快速变化。若每次触觉变化都重新运行完整 VLM，计算开销太大；若只在 action chunk 开始看一次触觉，反馈又会迅速过期。

### 1.2 静态触觉不能充分表示接触动态

单帧力向量只能回答“现在受了多少力”，却很难区分：

- 力正在上升还是下降；
- 接触刚建立还是即将丢失；
- 当前变化是稳定按压还是快速冲击；
- 物体是否在逐渐滑动。

T-Rex 因此同时编码短期 F6 历史、当前 F6 和当前 deformation map。

### 1.3 小规模任务数据不足以训练高自由度灵巧策略

T-Rex 使用三阶段训练：大规模人类第一视角视频预训练提供语义和运动先验；100 小时机器人触觉数据完成 embodiment 与接触 grounding；最后每项任务约 100 条示范用于适配具体技能。

---

## 2. T-Rex Dataset：数据规模和设计逻辑

完整数据集由固定底盘的双臂 Dexmate Vega-1 采集，主要统计为：

| 项目 | 数值 |
| --- | ---: |
| 机器人数据总时长 | 100 小时 |
| episode 数 | 7,755 |
| 日常物体 | 207 |
| motor primitives | 22 |
| 可行 object–primitive 组合 | 502 |
| 每个组合平均示范 | 约 16 |
| episode 中位长度 | 29.8 秒 |
| 场景背景 | 6 种 |
| distractor | 每场景通常 0–5 个 |

官方仓库当前公开约 50 小时、5,400 多条 trajectory，而论文实验使用完整的 100 小时版本。阅读数据规模时要区分“实验数据”和“当前已公开子集”。

### 2.1 为什么围绕 motor primitive 采集

作者没有只收集十几个复杂任务的大量重复示范，而是把日常操作拆成：

```text
pick / place / press / slide / wipe / twist / pour / insert / ...
```

再与 207 个对象做可行性筛选。其目标不是让数据集平均覆盖所有笛卡尔积，而是让 mid-training 学到可组合、可迁移的接触基本单元。

### 2.2 每条 episode 的模态

每条示范同步记录：

- 1 路 head-camera RGB；
- 2 路 wrist-camera RGB；
- 双臂 2×7 关节位置和速度；
- 双手 2×22 关节状态；
- 两个手腕的 SE(3) 末端位姿；
- 10 个指尖的单通道 deformation depth map；
- 10 个指尖的 6-axis net wrench；
- action；
- 自然语言任务指令；
- 公共时间戳。

视觉、状态、动作和触觉在数据集中按 30 Hz 对齐。

### 2.3 语言怎样标注

作者从 head camera 抽取 4–6 帧，加上采集时记录的 object name 和 motor-primitive name，交给商业 VLM 生成一条命令式描述，再由人工检查幻觉和不准确描述。

因此 T-Rex 的语言主要承担任务级条件，例如“用右手食指从右向左翻一页”，并不是 taxel 级或全手接触区域的细粒度空间标注。

---

## 3. 机器人、传感器和 Action Space

### 3.1 机器人本体

平台使用：

- 两条 7-DoF 机械臂；
- 两只 22-DoF Sharpa Wave 灵巧手；
- 头部 ZED X Mini；
- 两个腕部 ZED X One S；
- 每只手五个指尖视触觉传感器。

轮子、躯干和头部关节在实验中固定，只控制双臂和双手。

### 3.2 动作表示

双臂 action dimension 为 62：

\[
9+22+9+22=62
\]

每只手臂的 9 维末端动作由：

- 3 维局部平移；
- 6D rotation representation；

组成，并相对于 action chunk 开始时的末端位姿表达。每只灵巧手输出 22 个绝对关节目标。

```text
left  : Δxyz[3] + Δrot6d[6] + hand_q[22]
right : Δxyz[3] + Δrot6d[6] + hand_q[22]
total : 62
```

机械臂末端目标通过 differential IK 转为关节命令，再经过低通滤波发送给厂商的底层 cascade PID。

---

## 4. 整体架构：不是普通 VLM + Cross-Attention Head

T-Rex 的 backbone 是基于 Qwen3-VL-2B 改造的 Mixture-of-Transformer-Experts。论文和当前代码包含三个专家流：

| Expert | 主要 token | 作用 |
| --- | --- | --- |
| Latent Expert | head image、language、future-latent queries | 全局视觉语言理解与未来视觉 latent 预测 |
| Action Expert | wrist images、robot state、flow timestep、noisy action | 低频粗动作规划与 flow 前半段 |
| Tactile Expert | force code、当前 F6、deformation、flow timestep、partial action | 高频接触反馈与 flow 后半段 |

Qwen3-VL 文本侧的重要配置为：

```text
hidden size       2048
Transformer层数   28
Q heads           16
KV heads           8
head dimension   128
```

Tactile Expert 使用更小的 FFN intermediate size，因此 fast tick 比重跑完整视觉—动作通路便宜。

### 4.1 为什么叫 MoT，而不是普通 MoE

典型 sparse MoE 是：

```text
token → learned router → top-k FFN experts → weighted merge
```

T-Rex 没有让 learned router 动态选择 expert，而是由 token 类型固定决定参数：

\[
X_L\rightarrow\theta_L,\qquad
X_A\rightarrow\theta_A,\qquad
X_T\rightarrow\theta_T
\]

并且差异不只在 FFN：三组 token 分别拥有独立的 Q/K/V/O projection、Q/K norm、LayerNorm 和 FFN。

所以它更准确地属于：

```text
modality-routed multi-expert Transformer
```

而不是 top-k sparse MoE。

---

## 5. 三个 Expert 怎样共享 Attention

对三类 hidden states：

\[
X_L\in\mathbb R^{N_L\times d},\quad
X_A\in\mathbb R^{N_A\times d},\quad
X_T\in\mathbb R^{N_T\times d}
\]

分别投影：

\[
Q_L=X_LW_Q^L,\quad K_L=X_LW_K^L,\quad V_L=X_LW_V^L
\]

\[
Q_A=X_AW_Q^A,\quad K_A=X_AW_K^A,\quad V_A=X_AW_V^A
\]

\[
Q_T=X_TW_Q^T,\quad K_T=X_TW_K^T,\quad V_T=X_TW_V^T
\]

然后沿 token 维拼接：

\[
Q=[Q_L;Q_A;Q_T],\quad
K=[K_L;K_A;K_T],\quad
V=[V_L;V_A;V_T]
\]

共同计算：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}+M_{\mathrm{causal}}
\right)V
\]

最后不同位置再经过各自的输出投影和 FFN：

\[
O_L\rightarrow W_O^L,\operatorname{FFN}_L
\]

\[
O_A\rightarrow W_O^A,\operatorname{FFN}_A
\]

\[
O_T\rightarrow W_O^T,\operatorname{FFN}_T
\]

### 5.1 它是 joint self-attention，不是显式 cross-attention

三种 token 在同一 attention geometry 中形成 Q/K/V，因此算子上仍是 self-attention；跨模态条件通过同一 attention matrix 发生，而不是在 Action Expert 后额外插一个独立 CrossAttention 模块。

### 5.2 Causal mask 决定信息方向

概念顺序为：

\[
[L;A;T]
\]

所以允许关系可以写成：

| Query \ Key | Latent | Action | Tactile |
| --- | ---: | ---: | ---: |
| Latent | ✓ | × | × |
| Action | ✓ | ✓ | × |
| Tactile | ✓ | ✓ | ✓ |

这意味着 Action token 可以读取全局视觉语言上下文，而前面的 latent token 不会依赖后面的 noisy action；fast tick 中，新的 tactile/action query 可以读取缓存的 latent/action K/V，但缓存本身不会被当前触觉反向重算。

### 5.3 Fast tick 中哪些 token 是 Query

fast tactile sequence 是：

\[
[z_{\mathrm{tactile}},\ e_\tau,\ e(x_\tau)]
\]

这些位置全部通过 Tactile Expert 产生新的 Q/K/V。最终用于输出 velocity 的是最后一组 partial-action token hidden states：

\[
\hat v_{\mathrm{tac}}
=
\operatorname{Head}_{\mathrm{tac}}
\left(H_{\mathrm{last\ action\ tokens}}\right)
\]

缓存的 latent/action 在 fast tick 中只提供过去的 K/V，不重新产生 query，也不重新更新 hidden state。

---

## 6. 为什么 Head Camera 和 Wrist Camera 进入不同 Expert

两类相机都先经过同一个 Qwen3-VL ViT。差异发生在 ViT feature 被放入 Transformer 序列以后：

```text
head image + text       → slow/latent prefix
wrist-right + wrist-left → action suffix
```

### 6.1 Head view：全局语义和预训练接口

head camera 覆盖整个 workspace，更适合提供：

- 目标物体身份；
- 全局任务阶段；
- 双手和容器关系；
- 与大规模人类 egocentric video 一致的视觉接口。

Future visual latent 也使用未来 head frames 作为训练目标。

### 6.2 Wrist view：局部几何和动作条件

wrist cameras 面向手掌，用于观察 head camera 可能被遮挡的：

- 手指—物体局部关系；
- 抓取接近和插入对准；
- 手掌遮挡区域；
- 局部接触前的视觉几何。

因此腕部视觉与 state、flow timestep 和 noisy action 一起路由给 Action Expert。

### 6.3 这是合理设计，但缺少专门的交换消融

论文没有报告“把 head/wrist expert 对调”的独立实验。因此以上解释与代码和预训练数据一致，但不能说已经由专门实验严格证明是唯一最优分配。

---

## 7. Future-Latent Query Tokens 到底是什么

T-Rex 在 latent prefix 后追加一组可学习向量：

\[
Z_{\mathrm{future}}^{(0)}
=
[z_1,z_2,\ldots,z_K]
\in\mathbb R^{K\times d}
\]

它们不是未来图像，也不是从文本 tokenizer 得到的单词，而是随机初始化、随训练更新的 learned slots。

当前官方训练配置为：

\[
8\text{ 个未来时刻}\times4\text{ tokens/frame}=32\text{ query tokens}
\]

### 7.1 训练目标

未来 head frame 经过同一个 ViT：

\[
F_{t+\Delta}=\operatorname{ViT}(I_{t+\Delta})
\]

每帧 feature 被 adaptive average pooling 成 4 个 target tokens。Latent Expert 产生的 future-query hidden states 经过 MLP：

\[
\hat F_{t+\Delta}
=
\operatorname{FlareProj}(H_{\mathrm{future}})
\]

再使用 cosine loss 对齐：

\[
\mathcal L_{\mathrm{future}}
=
1-
\frac{\hat F^\top F}{\|\hat F\|\|F\|}
\]

未来帧的 ViT target 在 `no_grad` 下计算，避免 target encoder 被这条 loss 追着移动。

### 7.2 推理不需要未来图像

推理时只保留 learned queries。它们根据当前 head image 和 language 形成预测性 latent，随后 Action Expert 可以通过 attention 读取这些 token。

因此 Future Latent 的意义是：

```text
用训练期可见的未来视觉监督
迫使当前上下文编码可预测的动作后果
```

而不是在推理时偷看未来。

### 7.3 名字叫 Query，不代表只产生 Q

这些 learned query tokens 位于 joint causal self-attention 序列中，所以每层都会产生：

\[
Q_{\mathrm{future}},K_{\mathrm{future}},V_{\mathrm{future}}
\]

“query token”描述的是它作为信息收集槽位的功能，不代表它在数学上只能作为 Q。

---

## 8. Spatial-Temporal Tactile Encoder

T-Rex 同时使用三种触觉信息：

\[
z_t^{\mathrm{tac}}
=
[\operatorname{Emb}_{\mathrm{VQ}}(E_f(f_{t-15:t}));
\operatorname{Proj}_f(f_t);
\operatorname{Proj}_d(E_d(d_t))]
\]

分别是：

1. 16 帧 F6 历史的离散动态 token；
2. 当前瞬时 F6 的连续投影；
3. 当前 deformation map 的空间 feature。

### 8.1 Temporal F6 VQ-VAE

每个指尖提供：

\[
f_t\in\mathbb R^6
\]

包含三轴力和三轴力矩。对每个手指取 16 帧窗口：

\[
f_{t-15:t}\in\mathbb R^{16\times6}
\]

1D temporal CNN 经过两次 stride downsampling 和时间平均池化，得到 256 维连续 embedding，再映射到大小为 64 的 codebook：

\[
k^*=\arg\min_k\|z_e-e_k\|_2
\]

最终每个手指产生一个离散 code/token。

为了避免大量“无接触”帧让 codebook collapse，重建 loss 对高力帧赋予更高权重；不同手指共享 temporal convolution 权重，同时注入 finger identity embedding。

### 8.2 为什么还保留当前 F6

离散 VQ token 对漂移更稳健，也能表示一段时间的动态模式，但量化和时间池化会损失当前瞬时细节。因此作者同时直接投影当前 F6：

\[
e_t^{\mathrm{instant}}=\operatorname{MLP}(f_t)
\]

形成“历史动态 + 当前数值”的互补。

### 8.3 Deformation Encoder

单通道 deformation map 经过一个由 ResNet-18 改造的轻量 CNN：

- 输入 stem 改为单通道；
- 只保留前三个 residual stages；
- 中间 feature 重投影到 128 channels；
- 展平后投影到 Transformer token width。

该 encoder 先通过自监督 convolutional autoencoder 预训练，在 policy learning 时冻结。它补充低维 wrench 中缺失的接触边缘、局部形变、滑移和剪切结构。

---

## 9. 标准 Flow Matching 回顾

设干净动作块为：

\[
x_0=A_{\mathrm{demo}}
\]

高斯噪声为：

\[
x_1=\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
\]

T-Rex 采用：

\[
x_\tau=(1-\tau)A_{\mathrm{demo}}+\tau\epsilon
\]

所以：

- \(\tau=0\)：干净动作；
- \(\tau=1\)：纯噪声。

线性路径的真实速度是：

\[
v^*=\frac{dx_\tau}{d\tau}=\epsilon-A_{\mathrm{demo}}
\]

模型训练目标：

\[
\mathcal L_{\mathrm{FM}}
=
\mathbb E
\left[
\|v_\theta(x_\tau,\tau\mid c)-v^*\|_2^2
\right]
\]

推理从 \(\tau=1\) 向 \(0\) 做 Euler 积分：

\[
x_{\tau+\Delta\tau}
=
x_\tau+\Delta\tau v_\theta(x_\tau,\tau\mid c),
\qquad \Delta\tau=-0.1
\]

---

## 10. Cascaded Flow Matching：一条 Flow 分给两个 Expert

T-Rex 将总共 10 个 Euler step 分成：

```text
τ=1.0                                              τ=0
noise ── Action Expert: 6 steps ──> x_0.4 ── Tactile Expert: 4 steps ──> action
```

### 10.1 Slow segment

Action Expert 在以下时间点预测 velocity：

\[
1.0,0.9,0.8,0.7,0.6,0.5
\]

六次更新后得到：

\[
\hat x_{0.4}
\]

高噪声区间用于确定较全局的动作模式，例如：

- 选择哪只手；
- 接近哪个物体；
- 整体运动方向；
- 抓、放、旋转或插入；
- 双手怎样分工。

### 10.2 在 split 点刷新 Action KV

最后一次 Action Expert forward 输入的是 \(x_{0.5}\)，计算 velocity 后才更新到 \(x_{0.4}\)。如果直接保留旧 cache，就会出现：

\[
\text{state}=x_{0.4},\qquad
KV_{\mathrm{action}}\approx KV(x_{0.5})
\]

代码因此删除 cache 尾部的 action K/V，并用：

\[
[\text{wrist},\text{state},e_{0.4},e(x_{0.4})]
\]

重新编码一次，得到一致的：

\[
KV_{0.4}=[KV_{\mathrm{latent}};KV_{\mathrm{action},0.4}]
\]

这是理解实现时非常容易漏掉的一步。

### 10.3 Fast segment

Tactile Expert 在：

\[
0.4,0.3,0.2,0.1
\]

四个时间点预测 velocity，最终得到：

\[
x_0=\hat A
\]

低噪声区间中的动作已有较明确结构，更适合根据触觉修正：

- 手指闭合程度；
- 接触力；
- 局部末端位置；
- 抓取保持；
- 滑移和接触丢失。

### 10.4 不是 residual addition

Tactile Expert 不是计算：

\[
A_{\mathrm{final}}=A_{\mathrm{coarse}}+\Delta A_{\mathrm{tac}}
\]

而是继续求解：

\[
x_{0.4}\rightarrow x_0
\]

因此“refinement”是功能描述，数学上是 tactile-conditioned terminal flow integration。

---

## 11. 高频与低频：代码中的真实调度

机器人最终只有一条统一的 action 执行时间轴。高低频描述的是网络模块多久重算一次。

### 11.1 一个 chunk 内的时间线

默认 action chunk 为 16，触觉更新 offset 为 4、8、12；offset 0 的第一次触觉 refinement 已包含在 `slow_and_fast` 中。

```text
action index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
               │           │           │           │
network:     slow+fast     fast        fast        fast
```

chunk 开始：

1. 发送 head/wrist/language/state/touch；
2. Action Expert 运行 6 个 flow steps；
3. 缓存 \(x_{0.4}\) 和 latent/action KV；
4. Tactile Expert 立即运行 4 个 steps；
5. 返回第一条完整 action chunk。

chunk 执行中：

1. 后台触觉线程持续更新 buffer；
2. 到 offset 4、8、12 时，只发送最新触觉；
3. 每次克隆相同 cache，并从相同 \(x_{0.4}\) 出发；
4. 重新得到完整 action chunk；
5. temporal aggregation 融合覆盖当前位置的多次预测，或直接替换剩余动作。

### 11.2 多个系统频率

当前公开默认配置是：

| 时钟 | 默认值 |
| --- | ---: |
| 低层 action streaming | 300 Hz |
| 高层 command loop | 30 Hz |
| 触觉采集 | 30 Hz |
| slow request | 每 16 个 command steps |
| fast request | 每 4 个 command steps |

如果高层 command frequency 为 \(f_c\)，执行窗口为 \(E\)，refine 间隔为 \(R\)，则：

\[
f_{\mathrm{slow}}=\frac{f_c}{E},\qquad
f_{\mathrm{fast}}=\frac{f_c}{R}
\]

当 \(E=16,R=4\) 时：

\[
f_{\mathrm{fast}}=4f_{\mathrm{slow}}
\]

公开默认 \(f_c=30\) Hz 对应名义上的 1.875 Hz 与 7.5 Hz；论文使用“约 5 Hz/20 Hz”描述实验系统。两者说明绝对频率依赖部署配置和实际延迟，架构核心是 1:4 的多速率调用关系，而不是固定写死的 5/20 Hz。

### 11.3 “Asynchronous”不是两个网络在 GPU 上真正并发

当前实现中：

- 触觉采集线程和 300 Hz 底层控制线程独立运行；
- GPU 端是单线程 ZMQ REP server；
- slow/fast 请求通过 lock 串行；
- 高层执行循环发送 fast 请求后等待返回。

所以这里的 asynchronous 更准确地表示：

```text
视觉上下文低频刷新并允许陈旧
触觉条件在同一视觉缓存周期内多次刷新
```

而不是 Action Expert 与 Tactile Expert 同时并行跑。等待 fast response 时，底层 300 Hz 线程仍会继续追踪最近一次目标。

---

## 12. 多次 Fast Tick 为什么都从同一个 Boundary 出发

设 slow stream 得到：

\[
x_{0.4},KV_{0.4}
\]

不同执行时刻获得触觉：

\[
c_{\mathrm{tac}}^{(0)},
c_{\mathrm{tac}}^{(4)},
c_{\mathrm{tac}}^{(8)},
c_{\mathrm{tac}}^{(12)}
\]

T-Rex 计算：

\[
\hat A^{(r)}
=
\operatorname{FlowTac}
(x_{0.4},KV_{0.4},c_{\mathrm{tac}}^{(r)})
\]

而不是：

\[
\hat A^{(8)}
=
\operatorname{Refine}(\hat A^{(4)},c_{\mathrm{tac}}^{(8)})
\]

这样做的优势是：

- 每次 refinement 保持同一全局视觉语言意图；
- 不会因为多次 residual 累加而持续漂移；
- 新触觉直接改变 flow 后半段的 vector field；
- 多次预测可以在 action 时间轴上做平滑 aggregation。

限制是：chunk 内出现视觉上完全新的事件时，fast stream 看不到新图像，只能依靠触觉和旧视觉 cache，直到下一次 slow tick。

---

## 13. 两个 Flow Expert 怎样训练

### 13.1 Action Expert

训练时间采样：

\[
\tau_{\mathrm{act}}\sim\operatorname{Beta}(1.5,1.0),
\qquad \tau_{\mathrm{act}}\in(0,1]
\]

损失：

\[
\mathcal L_{\mathrm{act}}
=
\left\|
f_{\theta_{\mathrm{act}}}
(x_{\tau_{\mathrm{act}}},\tau_{\mathrm{act}};c_{\mathrm{VL}})
-v^*
\right\|_2^2
\]

虽然推理只使用 Action Expert 的 \([0.4,1]\) 区间，训练却覆盖完整 \((0,1]\)。这样它能保持预训练 flow-policy 能力；触觉缺失时，也可以单独完成全部 10 步作为降级策略。

### 13.2 Tactile Expert

先采样：

\[
\tilde\tau\sim\operatorname{Beta}(1.5,1.0)
\]

再缩放到：

\[
\tau_{\mathrm{tac}}=\tau_{\mathrm{split}}\tilde\tau
\in(0,0.4]
\]

损失：

\[
\mathcal L_{\mathrm{tac}}
=
\left\|
f_{\theta_{\mathrm{tac}}}
(x_{\tau_{\mathrm{tac}}},\tau_{\mathrm{tac}};
c_{\mathrm{tac}},\operatorname{stopgrad}(KV_{0.4}))
-v^*
\right\|_2^2
\]

两个 expert 回归同一个：

\[
v^*=\epsilon-A_{\mathrm{demo}}
\]

区别只是负责不同 \(\tau\) 区间和不同条件输入。

### 13.3 Detached cache

训练 Tactile Expert 时，slow-stream cache 在 `torch.no_grad()` 下获得。因此 \(\mathcal L_{\mathrm{tac}}\) 可以训练 Tactile Expert，却不能沿 cache 反向修改 Latent/Action Expert。

这使分工更明确：

```text
Action loss  → 训练全局视觉动作规划
Tactile loss → 训练快速接触条件 completion
```

### 13.4 Delay augmentation

部署时 tactile fast tick 发生在视觉 cache 生成后的 offset 0、4、8、12，视觉和触觉天然存在 staleness。训练时随机采样：

\[
\delta\in\{0,4,8,12\}
\]

将 tactile frame 相对视觉条件平移，以匹配部署分布。否则模型只在严格同步条件下训练，却要在陈旧视觉 + 新触觉条件下推理。

### 13.5 总损失

\[
\mathcal L
=
\mathcal L_{\mathrm{act}}
+\lambda_{\mathrm{tac}}\mathcal L_{\mathrm{tac}}
+\lambda_{\mathrm{future}}\mathcal L_{\mathrm{future}}
\]

设置为：

\[
\lambda_{\mathrm{tac}}=1.0,
\qquad
\lambda_{\mathrm{future}}=0.5
\]

---

## 14. 三阶段训练 Recipe

### Stage I：人类 Egocentric Pre-training

Latent 和 Action Experts 在 22,889 小时第一视角人类视频上预训练：

- head-view 提供视觉语言和场景语义；
- 人体手腕与手部动作被重定向到统一 action space；
- 此阶段没有 Tactile Expert。

### Stage II：Tactile-Grounded Robot Mid-training

使用 100 小时 T-Rex 双臂触觉数据：

- 适配三相机机器人观测；
- grounding 到 62D 可执行动作；
- 训练 Tactile Expert 和时空触觉表示；
- 学习跨对象、跨 primitive 的接触反应。

### Stage III：Skill-Specific Post-training

复杂任务通常再使用约 100 条 task-specific demonstrations 微调。Mid-training 后模型也展示了 pick、slide、press、wipe 等较基础 primitive 的 zero-shot 能力。

---

## 15. 实验结果应该怎样读

作者在 12 个真实接触丰富任务上测试，每个任务 16 次随机化 trial。任务覆盖：

- force-sensitive contact；
- deformation-aware manipulation；
- bimanual force–deformation reasoning。

代表任务包括翻页、鸡蛋转移、擦盘子、挤牙膏、分杯、麻将排序、开锁、补药片、酸碱中和、抽卡、发牌和拧灯泡。

### 15.1 主结果

平均成功率：

| 方法 | Average |
| --- | ---: |
| ViTacFormer | 3% |
| RDP | 6% |
| Tactile-VLA | 15% |
| π0.5 | 17% |
| EgoScale | 35% |
| T-Rex | **65%** |

最值得注意的不是单独某项任务，而是两件事同时成立：

1. 大规模人类预训练使 EgoScale 明显优于从任务小数据训练的触觉策略；
2. T-Rex 在保留这种预训练收益的同时，通过 mid-training 和 fast tactile path 进一步提升接触任务。

### 15.2 直接给 π0.5 拼触觉为什么反而变差

π0.5 + tactile 的平均表现只有 6%。这说明“网络收到了触觉张量”不等于“网络学会在正确的时间使用触觉”。新模态可能带来：

- 分布偏移；
- 无接触噪声；
- 与视觉不同的采样率；
- action chunk 内反馈过期；
- 小数据下 encoder 与 policy 同时过拟合。

T-Rex 的收益来自数据、动态 encoder、专家分工和执行调度的组合，不能简单归因于多一个触觉输入。

---

## 16. 关键消融

六个代表任务上的平均结果：

| 配置 | Average | 相对完整模型 |
| --- | ---: | ---: |
| Full T-Rex | 65 | — |
| w/o tactile | 42 | -23 |
| MLP force + deformation | 58 | -7 |
| deformation only | 54 | -11 |
| MLP force + VQ-VAE force | 59 | -6 |
| w/o asynchronous refinement | 60 | -5 |

可以得到四个结论：

1. 触觉本身是最大增益来源；
2. 当前 force、temporal VQ force 和 deformation 三者互补；
3. 时序编码不只是增加参数，确实比静态 force 更有效；
4. asynchronous cascade 有额外收益，但其增益小于“有没有触觉”的差异。

作者还改变 denoising split step。中间位置最好，说明需要同时给 Action Expert 足够的高噪声规划区间，也给 Tactile Expert 留下足够的低噪声修正空间。

训练 recipe 消融的六任务平均结果为：

| Human pretrain | Tactile mid-train | Average |
| ---: | ---: | ---: |
| × | × | 18 |
| × | ✓ | 34 |
| ✓ | × | 45 |
| ✓ | ✓ | **65** |

Human pretraining 和 tactile mid-training 不是相互替代，而是分别解决“广泛运动先验”和“机器人接触 grounding”。

---

## 17. 与几种常见 VLA 架构的对照

| 模型 | 感知与动作如何连接 | 触觉使用方式 | 动作生成 |
| --- | --- | --- | --- |
| GR00T | 独立 DiT 用显式 cross-attention 读取 VLM memory | 取决于扩展模块 | Flow/Diffusion DiT |
| π0/π0.5 | Prefix 和 Action Expert 每层做 masked joint attention | 原版通常无触觉 | 单一 Action Expert 完成 flow |
| Tactile-VLA | 触觉历史 token 进入 π0 prefix | 同步、低频条件 | Action Expert 输出位置与力 |
| VLA-Touch | 原 VLA 外接 planner 和 interpolant corrector | 高层语义 + 低层外部修正 | RDT action 再运输 |
| T-Rex | Latent/Action/Tactile 三套参数做 causal joint attention | 独立高频 expert 复用缓存 | 同一 flow 在 \(\tau\) 上级联分段 |

T-Rex 最像 π0 系列的 multi-expert joint attention，但进一步增加：

- Latent Expert 的 future visual prediction；
- 第三个 Tactile Expert；
- 跨 expert 的 flow interval handoff；
- variable-rate runtime。

---

## 18. 论文的限制

### 18.1 仍然只有指尖触觉

每只手五个传感器位于指尖。论文自己明确把“缺少 dense palm sensing”列为未来方向。因此它还不能完整表示：

- 物体落在掌心哪个区域；
- 多物体如何占据整只手；
- 指节和掌面如何共同分担载荷；
- 全手接触拓扑怎样随姿态变化。

### 18.2 Fast stream 使用陈旧视觉

视觉只在 slow tick 更新。Delay augmentation 可以提高对时延的鲁棒性，但无法让 tactile expert 看见 chunk 内突然出现的新视觉事件。

### 18.3 公开实现不是完全非阻塞

GPU server 串行处理请求；客户端在 refinement offset 等待 fast response。底层控制持续运行，但 high-level waypoint 更新仍可能受推理延迟影响。

### 18.4 行为克隆仍有典型失败

论文展示的失败包括：

- 碰撞；
- 抓取后滑落；
- 位置不准；
- 多手指意外摩擦；
- 施力过大；
- 滑动对不准。

这些失败说明触觉并不会自动解决视觉定位、数据覆盖和多指协调问题。

### 18.5 贡献是组合性的

T-Rex 同时改变数据、预训练、触觉表示、专家结构、flow schedule 和 runtime。完整系统效果很强，但很难把所有提升严格归因到单一组件；`w/o async` 的提升也明显小于完整训练 recipe 的提升。

---

## 19. 对全掌触觉研究的直接启发

T-Rex 最值得迁移的不是“每个手指做一个 VQ code”这一具体表示，而是：

> 把稳定的全局视觉语言计划缓存下来，让高频触觉 token 在较轻的专家中反复改变动作分布。

若换成全掌密集 taxel，可以把 Tactile Expert 输入从：

```text
10 fingertip force codes + F6 + deformation
```

扩展成：

```text
taxel force/history
+ link ID
+ taxel position and normal under current forward kinematics
+ fingertip visuo-tactile RGB
+ hand/finger/palm-region hierarchy
```

但不能直接把上千个 taxel 全部放进 dense global attention。更合理的是：

```text
taxel local encoder
    ↓
link / finger / palm-region tokens
    ↓
lightweight tactile expert
    ↓
cached visual-language-action KV
```

另一个可能的推进点是：T-Rex 的 task-level language 没有监督“语言短语对应哪个接触区域”。全掌触觉可以增加：

\[
\mathcal L_{\mathrm{phrase-region}}
\]

让“拇指远端”“掌心尺侧”“小指侧”等语言直接对齐到 tactile region tokens，再把这种空间 grounding 接入类似 T-Rex 的 fast expert。

---

## 20. 读代码时应该看哪些文件

| 文件 | 关键内容 |
| --- | --- |
| [`modeling_qwen3vl_mot.py`](https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/qwen_vla/modeling_qwen3vl_mot.py) | 三套 Q/K/V/O、FFN、joint causal attention、position 与 cache |
| [`modeling_vla.py`](https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/qwen_vla/modeling_vla.py) | future queries、触觉 embedding、slow/fast flow continuation |
| [`train.py`](https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/scripts/train.py) | action/tactile/future 三项 loss、detach cache、训练时间采样 |
| [`test.py`](https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/scripts/test.py) | `slow`、`fast`、`slow_and_fast` 三种请求与服务端缓存 |
| [`eval_trex_async.py`](https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/hardware_code/eval/eval_trex_async.py) | chunk 执行、offset 触发、temporal aggregation 和机器人控制 |
| [`default.yaml`](https://github.com/ZhuoyangLiu2005/T-Rex/blob/main/hardware_code/config/default.yaml) | command、tactile、底层控制频率和 refinement offsets |

---

## 21. 最终记忆框架

把整篇论文压缩成五行：

```text
Head + Language + Future Queries → Latent Expert：理解任务并预测未来视觉latent
Wrist + State + Noisy Action     → Action Expert：完成 flow 的 1.0→0.4
Latent/Action KV + x_0.4         → 缓存：一个chunk内复用全局计划
Fresh Touch + Partial Action     → Tactile Expert：反复完成 flow 的 0.4→0
Refined Chunks                   → 聚合/替换剩余动作 → 统一控制时间轴
```

最后区分三个“时间”：

1. **物理时间**：机器人和传感器真实运行的秒与控制周期；
2. **chunk 时间**：\(a_t,a_{t+1},\ldots,a_{t+15}\) 的动作位置；
3. **flow time**：\(\tau:1\rightarrow0\) 的生成积分变量。

T-Rex 的核心贡献，就是在不混淆这三种时间的前提下，用缓存把低频视觉规划和高频触觉闭环连接起来。
