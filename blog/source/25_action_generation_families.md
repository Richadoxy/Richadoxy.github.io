# 动作生成方法：从连续回归到概率生成

动作生成可以看成一件很具体的事：把当前看到、摸到和记住的信息，转换成接下来要执行的命令。不同方法的区别，在于它们怎样表示多个可能的动作，以及怎样从这些可能性中生成一条动作序列。

本文先建立统一地图，再连接 [DP 与 ACT](36_diffusion_policy_and_act.md)、[Flow Matching](10_flow_matching_training_and_inference.md) 和 [生成一致性与采样蒸馏](37_generative_consistency_and_sampling_distillation.md)。网络的连接方式可以配合 [从对齐表示到 Policy 融合](19_attention_from_embeddings_to_action_expert.md) 阅读。

## 1. 数据相同，训练问题可以不同

设一次训练样本包含条件 \(c\) 和专家动作块 \(A\)：

\[
c=(\text{图像、触觉、本体状态、任务目标、允许的历史}),
\qquad A\in\mathbb R^{H\times D_a}.
\]

\(H\) 是预测多少个未来控制步，\(D_a\) 是每步命令的维数。加入 batch 维以后，动作张量是 \([B,H,D_a]\)。

这些数据既可以训练直接回归，也可以训练自回归、CVAE、扩散或 flow。训练方法会在已有样本上进一步构造 token、潜变量、噪声级或带噪动作，所以数据集里不必预先保存这些派生监督量。

~~~text
同一份 observation + 专家动作
       ├─ 直接比较预测动作与专家动作
       ├─ 将动作编码为 token，预测 token 的条件分布
       ├─ 通过潜变量编码不同的动作模式
       └─ 给动作加噪，学习把噪声分布变成动作分布
~~~

condition 是“生成动作时可参考的信息”。它可以是一个向量、一组 token，或者若干调制参数；condition 不是一种专门的数据格式。

## 2. 连续回归与多种合理动作

最直接的方法是让网络一次输出整个动作块：

\[
\hat A=f_\theta(c),\qquad
\mathcal L_{\mathrm{reg}}=
\mathbb E_{(c,A)\sim\mathcal D}\left[\|f_\theta(c)-A\|_2^2\right].
\]

公式中的期望表示在数据分布上取平均。代码通常用 minibatch 的均值近似它，实际还要对动作维度和有效时间步做归一化与 mask。

当 \(c\) 下只有一种主要的合理动作时，这种方法简单有效。但如果同一个障碍物可以从左右两边绕过，训练数据就可能包含两种路线。确定性 MSE 回归的理想解是条件均值，两条路线的平均可能穿过障碍物。

这说明动作预测还涉及分布中的多个模式。“多峰动作分布”与“多模态传感器”是两个不同含义：前者指多个合理答案，后者指视觉、触觉等不同输入来源。

L1 回归偏向条件中位数，但也不会自动产生完整的多峰分布。增加潜变量，或者直接学习分布，才能明确引入不同的生成结果。

## 3. 自回归：按顺序生成表示元素

自回归把一个联合分布拆成一串条件分布：

\[
p_\theta(y_{1:N}\mid c)
=\prod_{j=1}^{N}p_\theta(y_j\mid y_{<j},c).
\]

\(y_j\) 是序列里的第 \(j\) 个元素。对于离散动作生成，它通常是动作 tokenizer 产生的 token。一个 token 可以表示某个关节的量化值，也可以表示一段经过压缩的动作，因此 token 序号不一定对应机器人的物理时间。

训练时常用 teacher forcing：把正确的历史 token 作为输入，预测下一个 token 的概率分布。

\[
\mathcal L_{\mathrm{AR}}
=-\mathbb E_{(c,y^*)\sim\mathcal D}
\sum_{j=1}^{N}
\log p_\theta(y_j^*\mid y_{<j}^*,c).
\]

这就是 token 交叉熵。模型输出的是词表上各类别的分数，损失关注正确类别的概率；不能把 token ID 当连续数值，直接计算“预测编号减真实编号”。

~~~text
训练：正确的前缀 → 同时计算各位置的下一 token 预测 → 交叉熵
推理：已生成前缀 → 下一个 token → 更新前缀 → 继续生成
~~~

因果 attention mask 允许训练时并行计算这些位置，但普通自回归推理仍有前后依赖。Transformer 本身不等于自回归：它也可以一次处理整段带噪动作。连续变量也能按条件分布逐个生成，此时通常使用连续分布的负对数似然。

RT-2 是将动作编码为 token 并接入视觉语言模型的一个具体例子。[RT-2 原论文](https://arxiv.org/abs/2307.15818)

FAST 则先对动作块的每个维度沿时间轴做 DCT，再量化频域系数、按低频优先排列，最后用 BPE 合并常见的系数组合。它压缩的是整段动作的表示：token 顺序对应压缩后的编码顺序，不能把“预测下一个 token”理解为“预测下一个机器人控制时刻”。量化引入近似误差，BPE 对量化后的符号序列做无损合并。[FAST 原论文](https://arxiv.org/abs/2501.09747)

## 4. VAE / CVAE：用潜变量组织动作模式

VAE 是带潜变量的概率生成框架，包含 encoder、decoder 和对应的训练目标。ViT 则是一种 Transformer 网络架构。VAE 的输入可以是图像，也可以是动作或触觉，不能简单归类为“一种视觉 encoder”。[VAE 原论文](https://arxiv.org/abs/1312.6114)

用于条件动作生成时，可以写成：

\[
q_\phi(z\mid A,c),\qquad
p_\theta(A\mid z,c),\qquad
p(z\mid c).
\]

训练 encoder 看见专家动作，生成潜变量分布；decoder 根据潜变量和条件重建动作。一个常见的条件 VAE 目标是：

\[
\mathcal L_{\mathrm{CVAE}}
=\mathbb E_{z\sim q_\phi(z\mid A,c)}
[-\log p_\theta(A\mid z,c)]
+\beta D_{\mathrm{KL}}
\big(q_\phi(z\mid A,c)\|p(z\mid c)\big).
\]

第一项要求动作能重建回来。第二项让训练得到的潜变量分布靠近推理时使用的先验，常见选择是标准高斯。\(\beta=1\) 对应标准负 ELBO 的形式，其他权重改变重建与正则化的平衡。

重建损失取决于输出分布假设：固定方差高斯对应缩放后的 MSE，固定尺度 Laplace 对应缩放后的 L1。不同 VAE 不必使用同一种重建损失。

~~~text
训练：条件 c + 专家动作 A → 后验 encoder → z → decoder → 重建 A
推理：条件 c + 先验中的 z                         → decoder → 生成 A
~~~

推理时没有未来专家动作，因此不能继续使用依赖它的训练后验。ACT 是一个动作领域的 CVAE 实例；它的实际推理 latent 约定见 [DP 与 ACT](36_diffusion_policy_and_act.md)。

## 5. GAN：让判别器提供生成反馈

GAN 使用两个网络：生成器 \(G\) 根据条件和随机变量产生样本，判别器 \(D\) 学习区分真实样本与生成样本。生成器通过判别器的反馈改善输出。这种对抗训练是 [GAN 原论文](https://arxiv.org/abs/1406.2661) 的基本思路；下面把它写成条件动作生成的形式。

\[
\hat A=G_\theta(z,c),\qquad z\sim p(z).
\]

常见的判别器损失和非饱和生成器损失分别是：

\[
\mathcal L_D
=-\mathbb E_{(A,c)\sim\mathcal D}\log D_\psi(A,c)
-\mathbb E_{c,z}\log\big(1-D_\psi(G_\theta(z,c),c)\big),
\]

\[
\mathcal L_G
=-\mathbb E_{c,z}\log D_\psi(G_\theta(z,c),c).
\]

训练判别器时把生成样本视为固定输入；训练生成器时保留经过判别器的梯度，但只更新生成器参数。推理通常只保留生成器。上述公式只是一种 GAN 配方，不能直接套到 WGAN 等变体。

GAN 的训练需要处理两套网络之间的平衡，也可能遗漏部分数据模式。这里介绍它是为了补齐生成学习的地图，不能据此推断某个 VLA 已使用 GAN，也不能把一个预测操作成功率的 classifier 自动称为 GAN 判别器。

## 6. Diffusion：学习带噪动作的恢复方向

用 \(k\) 表示离散扩散步，给专家动作加上自己采样的噪声：

\[
A^k=\sqrt{\bar\alpha_k}A^0
+\sqrt{1-\bar\alpha_k}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

由于噪声是程序采样出来的，它的数值完全已知。典型噪声预测损失是：

\[
\mathcal L_\epsilon
=\mathbb E_{(c,A^0),k,\epsilon}
\left[\|\epsilon_\theta(A^k,k,c)-\epsilon\|_2^2\right].
\]

模型预测噪声以后，就能从带噪动作中减去相应成分，并交给采样器继续更新。“监督噪声”和“完成去噪”在这条链路里是相连的。[DDPM 原论文](https://arxiv.org/abs/2006.11239)

真实动作在训练中用于构造 \(A^k\)，推理则从随机噪声开始。模型每次读取当前带噪动作、噪声阶段和观测条件，逐步得到动作块。整个动作块可以并行去噪，不必按机器人时间逐步自回归。

扩散也可以直接预测干净动作 \(A^0\)。这种 \(x_0\)-prediction 仍然读取带噪输入和噪声级，并不等于只根据 observation 做一次确定性 BC 回归。不同参数化之间的关系见 [Flow Matching 与目标参数化](10_flow_matching_training_and_inference.md)。

## 7. Flow Matching：学习生成路径上的速度

以噪声到动作的线性路径为例：

\[
x_s=(1-s)\epsilon+sA,\qquad
u_s=A-\epsilon,\qquad s\in[0,1].
\]

模型学习：

\[
\mathcal L_{\mathrm{FM}}
=\mathbb E_{(c,A),s,\epsilon}
\left[\|v_\theta(x_s,s,c)-(A-\epsilon)\|_2^2\right].
\]

每个训练样本给出一对端点和一条路径上的监督速度。模型只看到当前点与条件，需要把大量样本的信息汇总成可用的速度场；它并没有在推理时提前知道目标动作。Flow Matching 也可以采用其他概率路径，线性插值只是常用示例。[Flow Matching 原论文](https://arxiv.org/abs/2210.02747)

推理时求解：

\[
\frac{dx_s}{ds}=v_\theta(x_s,s,c),\qquad x_0\sim\mathcal N(0,I).
\]

这里的速度是动作张量沿生成过程的变化率，通常不是机器人关节速度。生成时钟 \(s\) 也不是机器人实际执行的秒数。动作生成完成后，控制器才根据动作定义执行输出。

## 8. 生成一致性与采样蒸馏

扩散和 flow 常用多次网络前向得到动作。生成一致性与采样蒸馏关注另一条轴：在保留生成质量的同时，减少需要的前向次数。

一致性模型希望同一生成轨迹上不同噪声级的点映射到相同端点。一个示意目标是：

\[
\mathcal L_{\mathrm{cons}}
=\mathbb E\left[
w(s,s')\,\ell\left(
f_\theta(x_s,s,c),
\operatorname{sg}[f_{\bar\theta}(\tilde x_{s'},s',c)]
\right)\right].
\]

其中 \(\operatorname{sg}\) 表示停止梯度，两个输入必须按相应算法构造，端点还需满足边界条件。这不是把两个独立带噪样本随意拉近；仅让两边输出相同也不足以排除无意义的常量解。不同算法对配对、目标网络和加权有不同规定。

一般采样蒸馏还可以让学生直接模仿教师多步采样的结果：

\[
\mathcal L_{\mathrm{distill}}
=\mathbb E_{z,c}
\|G_\theta(z,c)-\operatorname{sg}[S_{\mathrm{teacher}}(z,c)]\|_2^2.
\]

这里假设教师采样是确定性的，或已固定教师中途的随机性，使同一份噪声与条件对应确定目标。这只是最直观的端点蒸馏示例，不代表所有蒸馏方法。教师的生成偏差也可能被学生继承。具体训练方式及其与表征一致性的区别，见 [生成一致性与采样蒸馏](37_generative_consistency_and_sampling_distillation.md)。

## 9. 把目标、网络和执行分开比较

| 方法 | 训练直接比较的量 | 推理的基本过程 | 主要代价或边界 |
| --- | --- | --- | --- |
| 确定性回归 | 预测动作与示范动作 | 一次前向输出动作块 | 多峰数据可能被平均 |
| 离散自回归 | 下一 token 的类别概率 | 逐 token 生成并解码 | 量化、token 长度和串行延迟 |
| CVAE | 重建项与 KL 正则 | 条件与 latent 经 decoder 生成 | latent 使用方式、先验与后验差异 |
| GAN | 判别真假及生成器对抗目标 | 条件与随机变量经生成器 | 对抗训练稳定性与模式覆盖 |
| Diffusion | 噪声、干净样本或其他目标 | 采样器多步恢复动作 | 采样步数、参数化与噪声调度 |
| Flow Matching | 生成路径上的速度 | 数值积分得到动作 | 路径、积分器与步数 |
| Consistency / 蒸馏 | 轨迹端点一致性或教师目标 | 一步或少步生成 | 配对与边界条件、教师误差、质量权衡 |

“Transformer”描述计算结构，“CVAE / diffusion / flow”描述学习与生成方式，“action chunk”描述输出覆盖的时间范围。三者可以组合，所以不能仅凭用了 Transformer 或输出多步动作，就判断它的 loss。

同一个 VLA 也可以同时接受两种动作监督：把专家动作编码为 FAST tokens，用 next-token 交叉熵训练 VLM 的动作表示；同时构造带噪动作，用 flow loss 训练连续 Action Expert。训练时保留离散分支，推理时仍可只由 expert 生成底层动作。KI 就采用这种联合训练，并限制 flow 梯度回到主干的路径。[KI 原论文](https://arxiv.org/abs/2505.23705)

这里不能得出“交叉熵天然优于 MSE”。两者比较的是 token 类别与连续速度等不同目标；第 2 节确定性动作回归的均值问题，也不能直接套在以噪声和生成时间为输入的 flow 模型上。判断收益要结合表示、条件输入、梯度路径与推理方式。π₀.₅ 原论文、官方 checkpoint 和公开微调代码之间的区别，见 [FAST 与 KI：动作监督、梯度与微调](42_fast_ki_and_pi_training.md)。

具体例子可以连接到 [ACT / DP](36_diffusion_policy_and_act.md)、[PI0.5](04_pi05_architecture.md)、[GR00T](11_groot_dit_architecture.md) 与 [FuSe](20_beyond_sight_fuse_paper_reading_notes.md)。这些例子说明可用组合，不构成对整个领域使用比例的统计。

## 10. 训练计算、采样计算与控制反馈

Diffusion 和 flow 的常见训练方式是随机采一个噪声级，只做一次带噪前向；推理才进行多步采样。某些蒸馏方法需要额外教师计算，不能用这个简化描述覆盖所有训练算法。

生成一个动作块后，还要选择执行多少步再接收新观测：

~~~text
训练：一段示范 → 构造目标 → 前向 → loss → 更新参数
采样：当前条件 + 随机变量 → 一次或多次前向 → 动作块
执行：动作块的短前缀 → 新传感器观测 → 再次生成
~~~

减少生成步数可能降低推理耗时，但闭环反馈还受相机、触觉、通信和控制器影响。网络输出也要经过反归一化、坐标和单位转换；学习到合理动作分布并不自动保证控制接口正确。

本文的知识链可以概括为：**数据定义需要学什么，生成目标定义怎样学，网络提供计算能力，采样器产生候选动作，执行系统决定动作何时真正影响机器人。** 对应的部署时间轴见 [在线执行与 Action Chunk 调度](26_online_execution_runtime.md)。
