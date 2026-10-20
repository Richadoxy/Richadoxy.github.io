# FAST 与 KI：动作监督、梯度与微调

读懂 [π₀.₅ 的 Prefix 与 Action Expert](04_pi05_architecture.md) 以后，还需要追踪训练目标和梯度：同一段动作为什么会出现两个预测分支？FAST 在训练的哪个阶段运行？冻结 VLM、LoRA 与 Knowledge Insulation 又分别控制什么？

本文连接 [动作生成方法](25_action_generation_families.md)、[表示监督配方](32_three_consistency_supervision_paradigms.md) 与 [TrainState 和参数更新](05_train_state_jit_checkpoint.md)。这里的理解关口可以归为四类：动作表示、训练组织、梯度更新、论文与代码边界。

> 代码核对：2026-10-17，官方 `Physical-Intelligence/openpi`，commit [`215abfb`](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)。站内 BrainCo-IL 工程篇描述的是另一份项目实现，不能把它的 YAML 与 `freeze_strategy` 当成官方 openpi 的同名接口。

## 1. 先分清三个同名的 token 入口

| 入口 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| 文本 tokenizer + embedding | 指令、写入 prompt 的离散 state | token ID，再查表成为隐藏向量 | 给 VLM 提供条件 |
| FAST action tokenizer | 真实未来动作块 \(A\in\mathbb R^{H\times D}\) | 可变长度的离散动作 token 序列 | 构造自回归动作预测目标 |
| `action_in_proj` | 带噪动作块 \(x_t\) | \([H,d_{\mathrm{expert}}]\) 连续隐藏向量 | 给 flow Action Expert 提供当前生成状态 |

图中的“Action tokens”必须注明是哪一种。FAST token 是词表里的整数编号；flow action token 是连续向量，一个位置通常对应一个控制步的完整动作向量。当前 state 与未来 action 也有不同身份：前者是条件，后者是训练目标。

FAST tokenizer 本身主要做动作编码；VLA 的 next-token loss 训练的是预测这些编码的模型。这里没有通过该交叉熵对 DCT、取整或 BPE 做端到端反向传播。[FAST 官方实现](https://huggingface.co/physical-intelligence/fast/blob/main/processing_action_tokenizer.py)

## 2. FAST 怎样把动作块变成 token

FAST 是 **Frequency-space Action Sequence Tokenization**。它沿每个动作维度的时间轴处理整个 action chunk：

```text
真实动作 A [H, D]
    → 按数据集动作统计量归一化
    → 沿时间轴做离散余弦变换 DCT
    → 缩放、取整，得到整数频域系数
    → 按低频优先的顺序展开各动作维度
    → BPE 合并常见系数组合
    → 动作 token IDs [K]
```

平滑动作的主要变化往往集中在较少的频率成分中。量化后，很多很小的系数变成零，BPE 可以把重复序列和常见组合编码成更短的 token 序列。\(K\) 随动作复杂度变化，并不固定等于 \(H\) 或 \(H D\)。

解码时做 BPE 解码、恢复系数矩阵、反量化和逆 DCT，再按数据约定还原动作单位。DCT 在不量化时可逆，BPE 对量化后的系数序列进行无损编码；整个 FAST 编码的精度损失主要来自量化。它并不要求把所有高频分量一律删掉。[FAST 论文 §V](https://arxiv.org/html/2501.09747)

因此，一个 FAST token 不必对应“机器人下一个时间步”，也没有预先规定的“抓取”“放下”等语义。它编码的是动作信号中的系数或系数组合。**FAST+** 是在多种机器人动作数据上训练好的通用 tokenizer；**π₀-FAST** 则是用这类 token 训练出的策略模型，二者层级不同。[FAST 模型卡](https://huggingface.co/physical-intelligence/fast)

## 3. 同一条示范，怎样产生两种 loss

设条件为 \(c=(\text{图像、指令、state})\)，未来动作块为 \(A\)。两条分支都学习底层动作，都需要正确利用条件。

### 3.1 FAST 分支：预测离散动作编码的条件分布

先构造标签：

\[
z_{1:K}=\operatorname{FAST}(A).
\]

模型预测词表上的概率，目标为：

\[
\mathcal L_{\mathrm{FAST}}
=-\sum_{i=1}^{K}\log p_\theta(z_i\mid c,z_{<i}).
\]

训练使用 teacher forcing，输入正确的历史动作 token，预测下一个 token。因果 mask 使这些位置可以在一次前向中并行计算。若用该分支生成动作，推理时才需要依次采样 token，再交给 FAST 解码。

token ID 只是类别编号。预测 101 与标签 100 的“编号差为 1”没有动作距离意义，不能对 ID 直接算 MSE。

### 3.2 Flow 分支：预测连续生成路径上的速度

本文统一使用官方 openpi 的时间约定，\(t=1\) 为噪声，\(t=0\) 为数据：

\[
\epsilon\sim\mathcal N(0,I),\qquad
x_t=(1-t)A+t\epsilon,\qquad
u_t=\epsilon-A.
\]

训练目标为：

\[
\mathcal L_{\mathrm{flow}}
=\mathbb E_{A,c,t,\epsilon}
\left[\|v_\phi(x_t,t,c)-u_t\|_2^2\right].
\]

网络一次处理整段带噪动作，输出同形状速度。推理从噪声开始，沿时间减小的方向积分，所以 Euler 更新中的 \(\Delta t<0\)。损失中的速度是生成路径上的速度，不一定是机器人实际关节速度。虽然使用 MSE，这仍是带噪条件下的生成训练，区别于只根据观测一次回归动作均值。[openpi `compute_loss` 与 `sample_actions`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)

### 3.3 两种目标的分工

| 维度 | FAST next-token loss | Flow matching loss |
| --- | --- | --- |
| 监督对象 | 离散动作编码的条件概率 | 当前带噪动作上的连续速度场 |
| 输出空间 | 动作词表上的 logits | 与 action chunk 同形状的连续张量 |
| 训练输入 | 条件 + 正确的历史 token | 条件 + 带噪动作 + flow timestep |
| 独立用于动作生成时 | 自回归采样、FAST 解码 | 从噪声积分得到动作 |
| KI 配方中的作用 | 塑造 VLM 的机器人动作表示 | 训练连续动作 expert |

两头共享观测条件，但不把一头的预测当作另一头的监督，也没有要求两个输出直接相等的蒸馏损失。在论文的联合训练里，FAST 动作 token 与连续动作位置互相不可见，避免一条训练分支读取另一条分支包含的目标信息。[π₀.₅ §IV-B](https://arxiv.org/html/2504.16054)

## 4. 为什么额外增加 next-token 目标

第一，它给 VLM 一条直接的动作监督路径。主干必须产生足以预测动作编码的特征；Action Expert 可以从这些特征中读取控制所需信息。尤其在 KI 阻断 flow 梯度后，这条路径让主干仍能适应机器人任务。

第二，token 预测便于与文本、图像问答、目标位置及子任务预测共同训练。动作数据推动模型适应控制，视觉语言数据帮助维持语义能力。语言跟随的改善来自目标、数据混合与梯度隔离的组合，不能归因于“交叉熵天然保留语言知识”。[KI 论文 §5–6](https://arxiv.org/html/2505.23705)

第三，FAST 改善了动作 token 的信息密度。高频平滑轨迹逐步量化以后，相邻 token 很相似，模型可能仅靠重复上一时刻就降低损失。DCT 与压缩减少这种冗余。FAST 论文用不同采样频率的实验展示了这个问题；KI 的消融也发现，朴素量化的辅助监督能有帮助，但不如 FAST。[FAST §IV–V](https://arxiv.org/html/2501.09747)、[KI §6](https://arxiv.org/html/2505.23705)

这些是该训练配方的动机与实验结果，不构成“给任意模型加 CE 都会更好”的保证。新增分支有计算开销；FAST 论文报告的训练加速也不能直接解释为每个训练 step 都更便宜，或机器人推理同步变快。最终仍要比较收敛到目标性能所需的总计算量、语言跟随和闭环任务表现。

## 5. 两条训练分支与两阶段训练是两个维度

| 方案 | 训练组织 | 动作生成时运行什么 |
| --- | --- | --- |
| 原始 π₀ | 使用连续动作 expert 的 flow 目标 | flow expert |
| π₀-FAST | 预测 FAST 动作 token | 自回归动作 token + FAST 解码 |
| 原始 π₀.₅论文 | 先用 FAST 等离散目标预训练；后训练加入 action expert，保留联合 token/flow 目标 | 高层文字预测 + 连续动作 expert |
| 后续 KI 方案 | 在 VLA 适配阶段同时训练离散与连续动作，并隔离 flow 到主干的梯度 | 连续动作 expert；按任务使用文字输出 |
| 公开 openpi 的 π₀.₅微调 | 加载已用 KI 训练的权重；公开实现只运行 flow 目标 | 连续动作 expert |

“分支”描述一次训练图里有哪些输出和 loss；“阶段”描述训练进程先后如何安排。KI 的单阶段是相对于原始 π₀.₅分阶段引入 action expert 的配方而言，仍然可以使用预训练 VLM，也可以在之后继续任务微调。[π₀.₅论文](https://arxiv.org/abs/2504.16054)、[KI 论文](https://arxiv.org/abs/2505.23705)、[openpi README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)

## 6. KI 到底切断什么

**Knowledge Insulation（知识隔离）**的动机是：新初始化的连续动作 expert 带来的梯度可能干扰预训练 VLM；而完全冻结 VLM 又会限制它学习机器人表示。

KI 联合训练离散动作和连续动作，让 VLM 经离散目标继续学习，同时在 expert 读取主干 K/V 的连接上使用 stop-gradient：

```text
前向：观测 → VLM 的各层 K/V → stop_gradient → Action Expert → flow 输出
反向：flow loss → expert 参数
                    × 不穿过 stop_gradient 回到 VLM

另一路：FAST / 文字 loss → VLM 参数
```

`stop_gradient(x)` 的前向数值仍等于 `x`，只改变自动求导。以主干参数 \(\theta\) 和 expert 参数 \(\phi\) 表示，在这样的训练图中：

\[
\nabla_\theta\mathcal L_{\mathrm{flow}}=0,
\qquad
\nabla_\theta(\mathcal L_{\mathrm{FAST}}+\mathcal L_{\mathrm{text}})
\text{ 可以非零},
\qquad
\nabla_\phi\mathcal L_{\mathrm{flow}}
\text{ 可以非零}.
\]

注意力 mask 与 stop-gradient 控制的事情不同：mask 决定前向能读哪些位置；stop-gradient 决定某条计算路径上的损失能否更新上游参数。只有“Prefix 不读取 noisy action”的 mask，flow 仍能通过 expert 读取的 prefix K/V 反向更新 VLM。[KI §5.2 的 K/V 梯度定义](https://arxiv.org/html/2505.23705)

## 7. 冻结、LoRA 与 KI：先看参数，再看梯度来源

| 机制 | 控制什么 | VLM 是否继续适应任务 |
| --- | --- | --- |
| 全参数微调 | 所有选定参数可更新 | 可以；公开 flow-only 路径也能由 flow loss 更新主干 |
| 冻结整个 VLM | 把 VLM 参数排除出更新集合 | 该部分权重不再学习；expert 读取固定的表示函数 |
| LoRA | 固定部分基础权重，训练低秩增量 | 插入适配器后，VLM 的有效映射仍可变化 |
| KI | 按损失路径隔离梯度 | 继续从 FAST / 文字等目标学习，但不接受 flow 梯度 |

这些选项不在同一个轴上：LoRA 决定用什么参数承载变化，KI 决定某个 loss 的梯度可以流到哪里。从原理上可以组合，但需要明确实现；只开启 LoRA 不会自动获得 KI。

官方 `get_freeze_filter()` 的 LoRA 默认规则匹配 `.*llm.*`，并排除 `.*lora.*` 参数；Action Expert 的匹配还取决于它自己的 variant。**这套默认过滤不会顺带冻结 `PaliGemma/img` 视觉编码器。** `train=False` 也不能单独证明参数被冻结，应查看实际的 `trainable_filter` 和优化器更新集合。[`Pi0Config.get_freeze_filter`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)

KI 论文中冻结预训练主干的消融表现较差，不等于“冻结一个已经学过机器人动作的 π₀.₅ checkpoint 做小数据微调必然失败”。初始化权重、任务跨度与可训练模块已经不同，需要独立验证。[KI 的冻结消融](https://arxiv.org/html/2505.23705)

## 8. 在官方 openpi 中怎么判断实际训练了什么

不要仅凭 checkpoint 名字判断训练配方。沿下面四个入口核对：

| 代码入口 | 要核对的事实 |
| --- | --- |
| [`ModelTransformFactory`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py) | `PI05` 走 prompt/state tokenizer；`PI0_FAST` 才构造 FAST 输入 |
| [`Pi0.compute_loss`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py) | 返回速度场 MSE，没有联合 FAST CE |
| [`gemma.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py) | 检查 expert 读取主干 K/V 的路径是否有 KI 所需 stop-gradient；该公开路径没有 |
| [`trainable_filter` 与训练步](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train.py) | 哪些参数参与求导与优化器更新 |

因此，使用该版本官方 π₀.₅流程做全参、LoRA 或参数冻结微调时，优化目标仍是 flow loss。加载 KI 训练过的 checkpoint 会继承它已获得的表示，但不会自动重建 FAST 辅助分支或恢复 KI 梯度边界。若希望复现完整方案，需要同时处理动作 token 标签、因果与分支 mask、CE 输出、数据混合和 K/V 梯度隔离。

对 BrainCo-IL 的具体实验，还要继续核对本项目的 `freeze_strategy` 如何展开，不能仅凭配置文件名里的 `lora_vlm_only` 推断所有可训练参数。接着阅读 [配置系统](02_config_system.md) 和 [TrainState](05_train_state_jit_checkpoint.md)。

## 9. 从 π₀ 到 π₀.₇，FAST 出现在哪些工作中

| 工作 | FAST 的实际角色 |
| --- | --- |
| [原始 π₀](https://arxiv.org/abs/2410.24164) | 使用 flow 动作生成，未采用 FAST 训练路线 |
| [FAST / π₀-FAST](https://arxiv.org/abs/2501.09747) | 提出动作编码，并用自回归 token 生成动作 |
| [原始 π₀.₅](https://arxiv.org/abs/2504.16054) | 离散动作预训练，后训练加入连续 expert |
| [Knowledge Insulation](https://arxiv.org/abs/2505.23705) | 用 FAST 监督主干，同时训练被梯度隔离的连续 expert |
| [π₀.₆ Model Card](https://website.pi-asset.com/pi06star/PI06_model_card.pdf) | 明确沿用 FAST + KI + 连续动作 expert |
| [π*₀.₆ / RECAP](https://arxiv.org/abs/2511.14759) | 在 π₀.₆基础上学习机器人经验，模型训练描述仍包含 FAST 与 KI |
| [π₀.₇](https://arxiv.org/abs/2604.15483) | FAST 继续作为主干训练监督；附录图明确标记其动作 token 只用于训练 |

这条谱系记录的是动作表示与训练方式；版本新增的任务能力、数据和强化学习方法需要分别阅读，不能由“用了 FAST”推出。

## 10. 回到框图与代码的自测

| 问题 | 能讲清楚的判断依据 |
| --- | --- |
| 图上的 token 指什么？ | 区分文本 ID、FAST 动作 ID、连续 action hidden，并说出各自输入 |
| 两条分支是否等于两个阶段？ | 一个是训练图的结构，一个是训练进程的安排 |
| FAST token 是高层动作语义吗？ | 它来自底层动作的频域编码；高层子任务文字是另一类输出 |
| 只画 attention mask 就包含 KI 吗？ | 继续追踪 expert 读取 K/V 后的反向路径 |
| 使用 MSE 是否一定是动作均值回归？ | 检查网络是否读取噪声与 timestep，以及标签是动作还是速度 |
| LoRA 是否让 VLM 完全不变？ | 基础权重可固定，但适配器改变有效映射；视觉塔还需单独检查 |
| checkpoint 用过 KI，本次也在跑 KI 吗？ | 以本次 transforms、loss、梯度边界和 trainable 参数为准 |
| 加辅助 loss 是否一定改善控制？ | 需要固定数据与容量的消融、语言跟随和闭环 rollout 证据 |

阅读顺序可以沿 [π₀.₅前向结构](04_pi05_architecture.md) → 本文 → [参数更新](05_train_state_jit_checkpoint.md) 回到一次真实训练。若研究新增触觉模态，再把“辅助监督更新谁、部署保留什么”应用到 [监督配方](32_three_consistency_supervision_paradigms.md) 和 [触觉记忆实验设计](41_tactile_memory_experiment_design.md)。
