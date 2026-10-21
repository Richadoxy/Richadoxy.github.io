# Diffusion Policy 与 ACT：从观测条件到动作序列

Diffusion Policy（DP）和 Action Chunking with Transformers（ACT）都能从示范中的 observation–action 对学习机器人策略。两者都会预测一段动作，但内部生成过程不同：DP 多次修正一段带噪动作，ACT 用条件变分自编码器的训练方式学习一次输出整段动作的策略。

阅读顺序可以是：[模仿学习基础](23_imitation_learning_basics.md) → [动作生成方法总览](25_action_generation_families.md) → 本文 → [Flow Matching](10_flow_matching_training_and_inference.md) 与 [生成一致性和采样蒸馏](37_generative_consistency_and_sampling_distillation.md)。

如果卡在 ACT 的概率术语，可以先读[先验与后验](#act-prior-posterior)、[KL 的数值例子](#act-kl)和[推理为什么用全零潜变量](#act-inference)，再回来看网络尺寸。本文用“同一场景下，示范者拿杯子有快有慢”的例子贯穿这些概念。

## 1. 方法、网络与控制接口分开理解

一个机器人策略可以拆成三层：

```text
传感器输入
    ↓ 感知与历史编码
决策条件 c
    ↓ 动作生成网络及其训练目标
一段动作 A
    ↓ 选择执行前缀、反归一化、控制接口
机器人执行并产生新观测
```

DP 指的是用条件扩散生成动作的策略方法。动作网络可以是 U-Net，也可以是 Transformer。它既可以独立接图像和本体状态，也可以接在更大的感知模型后面。因此，DP 没有一个由名字决定的固定参数量，原始 DP 也不是必须依赖 VLM 的 action expert。

原始 DP 研究机器人视觉运动模仿学习，将动作扩散、视觉条件和滚动执行结合起来。ACT 则来自 ALOHA 的精细双臂操作研究，将动作分块和 CVAE 训练结合起来。它们的原始任务背景分别见 [DP 项目页](https://diffusion-policy.cs.columbia.edu/) 和 [ACT 项目页](https://tonyzhaozh.github.io/aloha/)。

本文的具体维度例子用于解释张量流向，不是全部论文实验的统一配置。原论文的设计、官方实现中可选的分支，以及示意尺寸，会分别说明。

## 2. 三个时间尺度与一组张量

定义：

| 符号 | 含义 | 典型形状 |
| --- | --- | --- |
| \(t\) | 机器人真实运行的时刻 | 标量或时间戳 |
| \(T_o\) | 输入的观测历史长度 | 例如2帧 |
| \(H\) | 模型预测的动作序列长度 | 例如16步 |
| \(T_a\) | 重规划前执行的动作步数 | 例如4步或8步 |
| \(k\) | 扩散噪声级，不是机器人时间 | 每个训练样本一个整数 |
| \(D_a\) | 单步动作维度 | 由机器人接口决定 |
| \(A^0\) | 数据集中的归一化动作序列 | \([B,H,D_a]\) |
| \(A^k\) | 第 \(k\) 个噪声级的动作 | \([B,H,D_a]\) |
| \(c\) | 由观测得到的条件 | 向量或 token 序列 |

例如，一段长度16的动作序列可以由8次去噪生成，再只执行其中4步。这里的16、8、4分别描述预测长度、采样计算次数和真实执行长度，彼此不能代替。

动作维度也需要看控制器：它可能包含关节位置、末端位姿、夹爪开度等。模型输出的目标关节位置与实际测得的关节位置不是同一类数据，后者通常属于 observation。

## 3. 观测 embedding 如何成为条件

“Embedding”描述数据的表示形式；“condition”描述它在概率模型中的作用。一个观测 embedding 被用于决定生成什么动作时，它就是条件特征。

\[
Z_o=E_{\mathrm{obs}}(O),\qquad p_\theta(A\mid Z_o).
\]

以图像和本体状态为例：

```text
图像历史                   [B, To, Ncam, 3, Himg, Wimg]
    ↓ 各帧视觉编码
每时刻的视觉特征           [B, To, Dv]
    + 本体状态              [B, To, Dq]
    ↓ 拼接或投影
观测特征 Zo                [B, To, Do]

U-Net 分支：展平历史 → c    [B, To × Do]
Transformer 分支：逐步投影  [B, To, d]
```

条件可以通过拼接、FiLM 或 cross-attention 被读取，不必始终变成一个额外 token。FiLM 等通用机制见 [从表示到 Policy 融合](19_attention_from_embeddings_to_action_expert.md)。

DP 的常见视觉条件配置只给动作加噪，图像特征不参与动作去噪。一次采样中，观测编码通常只做一次，随后重复调用动作网络。训练时如果视觉编码器未冻结，动作损失仍会反传到视觉分支。

原论文主配置使用未预训练的 ResNet-18，采用空间 softmax 保留位置信息，并用 GroupNorm 替换 BatchNorm。论文也讨论了其他预训练和微调选择，因此这是一种具体实验配置，不是所有 DP 的规定。[DP 论文第3.2节](https://arxiv.org/html/2303.04137v5#S3.SS2)

连续动作通常先归一化，再经过线性层或卷积。这个过程不需要文本词表，也不必把动作量化成离散ID。“一个动作时刻对应一个 token”可以只是把该时刻的连续向量投影到 hidden width，token 在这里不等于词。[输入单元与编码器的区别](31_multimodal_encoder_alignment_and_infonce.md)

## 4. DP 的训练：人为加噪，再预测这份噪声

采用标准 DDPM 记号：

\[
\alpha_k=1-\beta_k,\qquad
\bar\alpha_k=\prod_{j=1}^{k}\alpha_j.
\]

随机采样真实动作 \(A^0\)、噪声级 \(k\) 和高斯噪声 \(\epsilon\)，直接构造：

\[
A^k=\sqrt{\bar\alpha_k}A^0+
\sqrt{1-\bar\alpha_k}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

模型输出：

\[
\hat\epsilon=\epsilon_\theta(A^k,k,c).
\]

常见简化损失为：

\[
\mathcal L_{\mathrm{DP}}
=\mathbb E_{(O,A^0),k,\epsilon}
\left[\left\|\epsilon_\theta(A^k,k,E_{\mathrm{obs}}(O))-\epsilon\right\|_2^2\right].
\]

式中的期望表示对训练样本、噪声级和噪声采样求平均。监督目标是**这一轮实际采出来的噪声张量**，不是高斯噪声的均值0，也不是传感器测得的环境噪声。

```text
专家动作 A0 ──────┐
                 ├→ 按噪声级 k 加噪 → Ak ──┐
随机噪声 ε ──────┘                         │
                                           ↓
观测 O → Encoder → c ───────────────→ 去噪网络 → ε_pred
噪声级 k ──────────────────────────→             │
                                                ↓
                                  与本次采样的 ε 做 MSE
                                                ↓
                         更新去噪网络和允许训练的观测编码器
```

加噪公式可以一步得到任意 \(k\) 的输入，不必先循环加噪100次。一次训练样本通常只抽一个 \(k\)，完成一次去噪网络前向。上述形式来自 [DDPM 训练目标](https://arxiv.org/abs/2006.11239)，DP 将其中的生成对象替换为动作，并增加观测条件。

### 4.1 噪声预测如何实现去噪

从加噪公式移项，可以构造干净动作估计：

\[
\hat A^0=
\frac{A^k-\sqrt{1-\bar\alpha_k}\hat\epsilon}
{\sqrt{\bar\alpha_k}}.
\]

因此，预测噪声是一种表达去噪结果的方式。网络先估计需要扣除的分量，采样器再根据噪声级把它换算成下一步动作。

这并不意味着模型能从一份接近纯噪声的输入中，唯一找回训练时配对的那条动作。同样的观测可能对应多条合理轨迹；模型学习的是不同噪声级下的条件分布信息，再通过采样产生其中一种结果。

### 4.2 直接回归动作与预测干净样本

下面两个网络虽然都输出动作数值，输入问题却不同：

\[
\hat A=g_\theta(c)
\quad\text{与}\quad
\hat A^0=f_\theta(A^k,k,c).
\]

第一个只有观测条件，用单点 MSE 训练时趋向于预测条件均值。例如，从障碍左边绕行和右边绕行都合理，平均轨迹可能穿过障碍。

第二个还读取当前带噪样本和噪声级，可以作为扩散的 \(x_0\)-prediction 参数化继续迭代采样。直接输出干净动作并不自动意味着普通单步 BC。

如果通过上式将噪声预测换算为 \(\hat A^0\)，可得到：

\[
\|\hat\epsilon-\epsilon\|^2
=\frac{\bar\alpha_k}{1-\bar\alpha_k}
\|\hat A^0-A^0\|^2.
\]

这个关系说明：两种输出在非退化噪声级可以转换，但未经权重调整的 MSE 对各噪声级施加的重视程度不同。进一步与 flow velocity 的关系见 [Flow Matching](10_flow_matching_training_and_inference.md)。

## 5. U-Net DP：沿动作时间轴做卷积

这里的 U-Net 是 **1D temporal U-Net**。卷积沿动作序列长度移动，动作维度作为输入通道；图像的二维卷积已经发生在视觉编码器中。

以下按官方 `ConditionalUnet1D` 的全局条件分支展示一组示意配置：\(H=16\)、\(D_a=7\)、`down_dims=[128,256,512]`。实际训练可以选择其他通道数。

```text
Ak                                           [B,16,7]
    ↓ 转置
                                             [B,7,16]
    ↓ ConditionalResBlock 7→128
    ↓ ConditionalResBlock 128→128
                                             [B,128,16]
    ↓ Downsample ×1/2
                                             [B,128,8]
    ↓ ConditionalResBlock 128→256
    ↓ ConditionalResBlock 256→256
                    保存 S1                  [B,256,8]
    ↓ Downsample ×1/2
                                             [B,256,4]
    ↓ ConditionalResBlock 256→512
    ↓ ConditionalResBlock 512→512
                    保存 S2                  [B,512,4]
    ↓ 两个中间 ConditionalResBlock
                                             [B,512,4]
    ↓ 与 S2 在通道维拼接
                                             [B,1024,4]
    ↓ ConditionalResBlock 1024→256
    ↓ ConditionalResBlock 256→256
    ↓ Upsample ×2
                                             [B,256,8]
    ↓ 与 S1 在通道维拼接
                                             [B,512,8]
    ↓ ConditionalResBlock 512→128
    ↓ ConditionalResBlock 128→128
    ↓ Upsample ×2
                                             [B,128,16]
    ↓ Conv1d block + 1×1 Conv，128→7
                                             [B,7,16]
    ↓ 转置
预测噪声                                     [B,16,7]
```

每个条件残差块都接收同一份全局条件，但有自己的条件投影参数：

\[
c_k=[\operatorname{TimeEmbed}(k);c].
\]

单个残差块的主要计算为：

```text
h                                  [B,Cin,L]
    ↓ Conv1d + GroupNorm + Mish
u                                  [B,Cout,L]
    ↓ 使用 c_k 产生 scale / bias     [B,Cout,1]
u' = scale × u + bias               [B,Cout,L]
    ↓ Conv1d + GroupNorm + Mish
    + h 的恒等或 1×1 卷积支路
输出                               [B,Cout,L]
```

长度为1的调制参数广播到全部时间位置。它改变每个通道的响应，不需要把观测复制成长度16的动作 token。官方实现也支持只加条件偏置；是否同时预测scale由 `cond_predict_scale` 控制。上图选用的是包含scale的分支。[官方 U-Net 实现](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py)

下采样扩大了时间感受野，上采样恢复动作分辨率，跳连保留不同尺度的序列特征。图中的通道宽度只是教学示例：官方某个训练配置使用更大的 `[512,1024,2048]`，不能据本文示意尺寸推断原实验参数量。[官方训练配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_hybrid_workspace.yaml)

## 6. Transformer DP：动作 token 读取观测条件

以下对应官方实现中 `time_as_cond=True` 且有观测条件的分支。设 \(H=16\)、\(D_a=7\)、\(T_o=2\)、hidden width为256：

```text
动作主路：
Ak [B,16,7]
    ↓ Linear 7→256 + action position embedding
X  [B,16,256]

条件支路：
观测 Zo [B,2,Do] → Linear Do→256       [B,2,256]
噪声级 k → sinusoidal embedding       [B,1,256]
    ↓ 拼接 + condition position embedding
                                      [B,3,256]
    ↓ MLP 或条件 Transformer Encoder
条件 memory M                         [B,3,256]

动作 X + 条件 M
    ↓ 多层 Transformer Decoder Block
动作 hidden states                    [B,16,256]
    ↓ LayerNorm + Linear 256→7
预测噪声                              [B,16,7]
```

这里的 `memory` 是 Transformer cross-attention 的条件序列，不自动表示跨控制时刻保存的长期记忆。

每个动作 decoder block 保持动作序列的形状：

| 子层 | Query来源 | Key/Value来源 | 输入输出 |
| --- | --- | --- | --- |
| Self-attention＋残差 | 动作hidden states | 动作hidden states | \([B,H,d]\rightarrow[B,H,d]\) |
| Cross-attention＋残差 | 更新后的动作hidden states | 条件memory | \([B,H,d]\rightarrow[B,H,d]\) |
| FFN＋残差 | 逐token变换 | 不适用 | \(d\rightarrow4d\rightarrow d\) |

官方这一路使用pre-norm：先LayerNorm，再子层计算和残差相加。条件编码器层数为0时使用MLP，否则使用Transformer编码器。噪声级在其他实现里也可以放到动作序列或调制支路，读架构图时应与具体配置对应。[官方 Transformer 实现](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py)

### 6.1 因果注意力不等于逐 token 生成动作

原论文使用了因果动作注意力；官方实现也提供相应开关。它规定动作位置之间的信息可见范围。

但每个扩散步仍然同时提供整段 \(A^k\)，并行输出全部位置的噪声：

```text
DP：整段带噪动作 → 整段噪声预测 → 整段更新 → 下一扩散步
AR：前缀token → 下一token → 扩展前缀 → 再预测下一token
```

因此，看见 causal mask 或 Transformer decoder，不能据此认定使用了自回归动作token交叉熵。DP的这条计算链仍然是扩散噪声预测。

## 7. 推理：采样器把噪声预测变成动作

训练时有真实动作，推理时只有观测和随机初始化：

\[
A^K\sim\mathcal N(0,I).
\]

标准DDPM的一种反向更新形式为：

\[
A^{k-1}=\frac{1}{\sqrt{\alpha_k}}
\left(A^k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}
\epsilon_\theta(A^k,k,c)\right)+\sigma_k z.
\]

其中 \(z\sim\mathcal N(0,I)\)，末步不再加入随机噪声，\(\sigma_k\) 由采样器的方差设置决定。这说明去噪不是简单地在每步执行 `A -= noise_pred`，缩放和噪声日程也是算法的一部分。

DDIM可以使用更少的噪声级。以确定性版本、从 \(k\) 跳到更低噪声级 \(s\) 为例，先计算 \(\hat A^0\)，再更新：

\[
A^s=\sqrt{\bar\alpha_s}\hat A^0+
\sqrt{1-\bar\alpha_s}\hat\epsilon.
\]

这个表达省略了某些实现中的裁剪处理；使用带随机项的DDIM时还需加入对应项。DDIM改变采样过程，可以沿用相应的扩散训练模型，并不要求先做教师—学生蒸馏。[DDIM 论文](https://arxiv.org/abs/2010.02502)

```text
新观测 → 编码得到 c ───────────────────────────┐
                                              │ 每步读取同一条件
随机初始化 AK → 去噪步 K → … → 去噪末步 → A0   │
                   ↑             ↑            │
                   └─────────────┴────────────┘
                              ↓
                 选择可执行段并反归一化
                              ↓
                 执行短前缀 → 获取新观测 → 重规划
```

观测历史长度、预测长度和可执行段的起始位置需要与数据对齐。官方某些DP policy的动作张量包含历史对齐位置，执行段从 `n_obs_steps - 1` 开始，不能在所有实现中机械地取 `A[:, :n_action_steps]`。[官方图像策略实现](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py)

训练噪声级数量也不是固定推理步数。配置里的100可能表示训练日程有100级，不代表每条训练样本执行100次前向；推理使用DDPM还是DDIM、取多少步，要看该实验的scheduler配置。

## 8. ACT：用 CVAE 训练动作分块策略

先抓住两条路径：**训练时，后验编码器和动作策略一起学习；推理时，只运行动作策略，并给它一个固定的全零潜变量。** 后验编码器帮助训练，不需要部署时再次获得专家答案。[ACT 论文第 IV-B 节](https://arxiv.org/html/2304.13705v1#S4.SS2)

本节统一用 \(s\) 表示关节状态、\(I_{\mathrm{img}}\) 表示图像、\(c=(I_{\mathrm{img}},s)\) 表示观测条件。这样，\(q_\phi\) 专门表示一个概率分布，避免与常用的关节位置符号 `q` 混淆。

### 8.1 从普通动作预测到 CVAE

想象杯子和机械臂的起始位置相同，示范者有时缓慢接近，有时较快接近。因此，同一个观测条件可以对应不同的合理动作块。只输出一个确定动作的回归网络，很难同时表达这些差异。

CVAE 给生成网络增加一个潜变量输入：

\[
\text{普通确定性预测：}\quad \hat A=f_\theta(c),
\qquad
\text{带潜变量的预测：}\quad \hat A=f_\theta(c,z).
\]

观测 \(c\) 描述当前场景；\(z\) 可以承载场景条件没有完全解释的示范差异。可以把它暂时想成“这次采用哪种动作方式”的内部代码。这个代码由模型学习，不保证某一维就对应速度、力度或路线。

数据集提供了场景和动作，却没有每条示范对应的 \(z\)。于是训练时增加一个编码器，让它读取专家动作和状态，推断合适的潜变量分布；动作策略再结合观测和潜变量，尝试重建这条动作。

| CVAE 名字中的部分 | 在 ACT 中具体指什么 |
| --- | --- |
| Conditional：条件 | 生成动作时参考当前图像和关节状态 |
| Variational：变分 | 用可训练的分布近似难以直接计算的真实后验 |
| Autoencoder：自编码器 | 训练时把专家动作编码成潜变量，再尝试重建动作 |

CVAE 描述这套概率生成与训练框架；Transformer 描述实现编码器和动作策略的网络结构。ACT 的 **CVAE decoder 就是动作策略**，其内部又包含视觉编码器、Transformer encoder 和 Transformer decoder。不同层级的“encoder/decoder”需要分别辨认。

<a id="act-prior-posterior"></a>

### 8.2 先验与后验：有没有看见这条具体示范

**先验是看到具体示范动作之前，对潜变量的默认分布约定；后验是看到这条示范之后，对哪些潜变量能解释它的概率判断。** “先后”相对于这份证据而言，与训练前后、网络前后几层无关。

ACT 采用固定先验：

\[
p(z)=\mathcal N(0,I).
\]

这里的 \(I\) 是单位矩阵，表示各维方差为 1、互不相关。先按一维理解，就是均值为 0、标准差为 1 的正态分布。它允许正负取值；这个约定不需要另一个网络来学习。

后验编码器根据当前示范，输出一个对角高斯分布：

\[
q_\phi(z\mid A,s)
=\mathcal N\left(\mu_\phi(A,s),
\operatorname{diag}(\sigma_\phi^2(A,s))\right).
\]

| 符号 | 在做什么 | 是否随当前示范变化 |
| --- | --- | --- |
| \(p(z)\) | 规定默认从什么分布取潜变量 | ACT 中固定为标准高斯 |
| \(q_\phi(z\mid A,s)\) | 看过动作后，判断对应的潜变量分布 | 是，由编码器输出 |
| \(\mu,\sigma\) | 描述这个分布的中心和宽度 | 是，它们是网络输出 |
| \(z\) | 从分布取出的一个向量，送给动作策略 | 是，它是一次前向的输入值 |
| \(\phi,\theta\) | 编码器与动作策略的网络权重 | 训练时由优化器更新 |

例如，**仅作教学假设**：快速示范的后验中心可能在 \(+0.8\)，慢速示范的中心可能在 \(-0.8\)。这个例子说明不同示范可以对应不同分布，并非 ACT 手工规定了“正数快、负数慢”。

严格说，真实后验描述的是给定动作和观测后，生成模型中的 \(z\) 有哪些可能取值。它通常难以直接计算，所以使用 \(q_\phi\) 来近似。原始 ACT 为提高训练效率，后验编码器只读取动作和本体状态，省略图像；图像仍由动作策略的视觉分支读取。[ACT 的 CVAE 设计](https://arxiv.org/html/2304.13705v1#S4.SS2)

<a id="act-kl"></a>
<a id="82-kl"></a>

### 8.3 KL：比较两个分布，而不是两条动作

KL 的全名是 **Kullback–Leibler divergence**，中文叫 KL 散度。它衡量两个概率分布之间的差异：为 0 时分布相同，存在差异时为正；交换比较顺序通常会改变结果，所以它不是通常意义上的对称距离。

ACT 中比较的是：

\[
D_{\mathrm{KL}}\!\left(
\underbrace{q_\phi(z\mid A,s)}_{\text{当前示范的后验}}
\;\middle\|\;
\underbrace{\mathcal N(0,I)}_{\text{固定先验}}
\right).
\]

先看一维例子。若后验为 \(\mathcal N(\mu,\sigma^2)\)，先验为 \(\mathcal N(0,1)\)，则：

\[
D_{\mathrm{KL}}
=\frac12\left(\mu^2+\sigma^2-\log\sigma^2-1\right).
\]

不必先背公式，观察它怎样惩罚分布变化：

| 后验分布 | 与标准高斯相比发生了什么 | KL，使用自然对数 |
| --- | --- | --- |
| \(\mathcal N(0,1)\) | 完全相同 | 0 |
| \(\mathcal N(1,1)\) | 中心移到 1，宽度不变 | 0.5 |
| \(\mathcal N(2,1)\) | 中心移到 2，宽度不变 | 2 |
| \(\mathcal N(0,0.25)\) | 中心不变，分布变窄 | 约 0.318 |
| \(\mathcal N(0,4)\) | 中心不变，分布变宽 | 约 0.807 |

括号里的第二个数是**方差**；例如方差 0.25 对应标准差 0.5。下面将表中的两个后验与共同先验画在同一坐标中，曲线高度表示概率密度：

<figure class="concept-figure">
  <img src="assets/figures/act-prior-posterior-kl.svg" width="900" height="400" loading="lazy" alt="蓝色标准高斯先验中心为零；橙色后验向右移动到二；紫色后验仍以零为中心但更窄。中心偏移和宽度变化都会产生 KL 惩罚。" />
  <figcaption>蓝线是固定先验；橙色虚线改变了中心，紫色点划线改变了宽度。曲线下的总面积都为 1。KL 比较整条分布，而不是比较一次采样的两个数。</figcaption>
</figure>

因此，“让后验接近标准高斯”同时约束均值和方差：均值别偏得太远，方差别过大或过小。即使后验已经等于标准高斯，仍然可以采出 \(-0.7\)、\(0.4\)、\(1.2\) 等不同值。**KL 不要求每个 \(z\) 都等于零，也不要求机器人动作服从标准高斯。** [VAE 原论文及高斯 KL 推导](https://arxiv.org/html/1312.6114v11)

为什么需要这项约束？训练编码器看得到答案 \(A\)，可能为每条示范编出很特殊的代码，供动作策略重建；推理却无法再次取得这些示范专属代码。KL 对偏离统一先验的程度收取代价，限制策略通过潜变量依赖示范专属信息。

但也不能只优化 KL：如果所有示范都对应完全相同的标准高斯，\(z\) 就不再携带区分示范的信息。动作重建损失与 KL 需要共同决定保留多少信息。

<a id="act-training"></a>

### 8.4 两条网络一起更新：动作误差与 KL 各自流向哪里

一次训练先从后验采样，再让动作策略重建示范：

\[
z=\mu_\phi+\sigma_\phi\odot\xi,
\qquad \xi\sim\mathcal N(0,I),
\qquad \hat A=\pi_\theta(I_{\mathrm{img}},s,z).
\]

这个写法叫**重参数化**：随机数 \(\xi\) 不需要学习，\(z\) 对 \(\mu,\sigma\) 的依赖却可以求导。所以动作误差能够经过采样操作，继续反传到后验编码器。

```text
专家动作 A + 状态 s
        ↓
后验编码器 Eφ【更新参数】
        ↓
分布参数 μ、log σ² ───────────────→ 与固定先验比较 → KL
        ↓ 重参数化采样
        z ───────────────┐
                        ↓
图像 + 状态 ───→ 动作策略 πθ【更新参数】
                        ↓
                    预测动作 Â ───→ 与专家动作 A 比较 → L1

总损失 = L1 + β × KL
```

| 部分 | 是否有参数被训练 | 在这张计算图中收到的损失梯度 |
| --- | --- | --- |
| 后验编码器、分布参数投影层 | 是 | L1 经由 \(z\) 反传；KL 直接约束其输出分布 |
| 动作策略的视觉、融合与动作输出模块 | 未冻结部分会更新 | L1 |
| 重参数化采样操作 | 没有可训练参数 | 允许梯度经过 |
| 固定先验 \(\mathcal N(0,I)\) | 没有可训练参数 | 不被更新 |

这里 KL 不直接更新独立的动作策略权重；它通过改变编码器学到的分布，影响策略在训练中收到的 \(z\)。官方优化器将两条网络中允许训练的参数一起交给 AdamW。[参数更新集合](https://github.com/tonyzhaozh/act/blob/main/detr/main.py)、[重参数化实现](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py)

按官方 `policy.py`，总损失是 masked L1 重建项加 KL 正则：

\[
\mathcal L_{\mathrm{ACT}}
=\mathbb E_{z\sim q_\phi}
\left[\operatorname{MaskedL1}(\pi_\theta(I_{\mathrm{img}},s,z),A)\right]
+\beta D_{\mathrm{KL}}
\left(q_\phi(z\mid A,s)\|\mathcal N(0,I)\right).
\]

L1 比较动作数值，KL 比较潜变量分布。\(\beta\) 太小，后验可能过度编码示范信息；太大，后验可能被压到几乎与先验相同，失去携带动作差异的作用。

在多维对角高斯情况下，将各维的 KL 相加：

\[
D_{\mathrm{KL}}
=\frac12\sum_j
\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right).
\]

代码通常输出 `logvar = log σ²`，再用 `exp(logvar)` 得到方差。官方实现对各潜变量维度求和，再对 batch 求均值。尾部 padding 动作不应形成有效重建监督，L1 的具体归约方式也应与实现一致。[ACT 官方损失](https://github.com/tonyzhaozh/act/blob/main/policy.py)

<a id="act-inference"></a>

### 8.5 推理为什么可以舍弃编码器，并令 z = 0

训练数据保存了未来专家动作，因此可以利用它推断 \(z\)。机器人真正执行时只有当前观测，没有未来正确动作，无法再运行同样的后验分支。

原版 ACT 采用固定潜变量：

\[
z=[0,0,\ldots,0],\qquad
\hat A=\pi_\theta(I_{\mathrm{img}},s,0).
\]

```text
训练：专家动作 + 状态 → 后验编码器 → 采样 z → 动作策略 → 重建动作
                                                ↑
                                          图像 + 状态

推理：                              固定 z = 0 → 动作策略 → 新动作
                                                ↑
                                          图像 + 状态
```

**“取先验均值”就是把全零向量当作一个输入值，训练好的网络权重仍然保留。** 杯子移动后，图像和状态发生变化，输出动作也会变化。它既不表示动作全部为零，也不表示把策略权重清零。

舍弃后验编码器，只是停止执行这条训练分支；不会抹掉它参与训练时，动作误差已经写入策略参数的学习结果。KL 有助于减小后验与先验的差异，但不能数学保证 \(z=0\) 对所有任务都最优。

一般 CVAE 可以从先验采样，生成不同结果；原版 ACT 为得到确定性输出，选择固定先验均值。非线性策略在 \(z=0\) 的输出也不等于所有示范动作的算术平均。[原版 ACT 推理代码](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py)、[论文训练与推理说明](https://arxiv.org/html/2304.13705v1#S4.SS2)

<a id="81"></a>

### 8.6 把概念对应回网络张量

理解上述分工后，再看实际结构。\(N_v\) 是所有相机视觉特征展开后的 token 数，\(D_s\) 是状态维度，\(D_z\) 是潜变量维度：

```text
训练专用后验分支：
专家动作 A              [B,H,Da] → Linear → [B,H,d]
本体状态 s              [B,Ds]   → Linear → [B,1,d]
CLS                                              [B,1,d]
    ↓ 拼接 + 位置编码
                                                 [B,H+2,d]
    ↓ Transformer Encoder
CLS 输出                                        [B,d]
    ↓ 参数投影，拆成两组
μ、log σ²                                       各 [B,Dz]
    ↓ 重参数化采样
z                                               [B,Dz]

动作策略分支（训练、推理都保留）：
多相机图像 → ResNet → 空间展开 + 位置编码         [B,Nv,d]
s → Linear                                      [B,1,d]
z → Linear                                      [B,1,d]
    ↓ 拼接后经过 Transformer Encoder
融合后的 memory                                 [B,Nv+2,d]
    ↓ 由 H 个动作 query 通过 Transformer Decoder 读取
动作 hidden states                              [B,H,d]
    ↓ 动作投影
预测动作 Â                                      [B,H,Da]
```

动作 query 代表各个待预测的时间位置，因此 ACT 在一次策略调用中并行输出整个 chunk。推理舍弃的是上面的后验分支；下面策略内部的视觉 encoder 和 Transformer encoder 仍然需要运行。

### 8.7 ACT 的后验编码器与 DP 的观测编码器有什么不同

| 问题 | 原版 ACT | 常见的视觉条件 DP |
| --- | --- | --- |
| 编码器读取什么 | 后验分支读取专家动作和状态；另有视觉分支 | 观测编码器读取图像、状态或其历史 |
| 是否有训练专用的动作后验网络 | 有，输出潜变量分布 | 通常没有 ACT 这种分支 |
| 专家动作怎样参与训练 | 输入后验编码器，同时作为重建目标 | 用来构造带噪动作，并提供对应训练信息 |
| 动作网络接收什么 | 观测和 \(z\) | 观测条件、当前带噪动作和噪声级 |
| 原版推理怎样得到动作 | 固定 \(z=0\)，一次前向输出动作块 | 从随机动作噪声出发，迭代去噪 |
| 观测编码器是否保留到部署 | 是 | 是 |

所以，“ACT 后验编码器只在训练时使用”不能推广为“所有 encoder 都只在训练时使用”。DP 的观测编码器提供当前场景条件，部署时仍需运行。[DP 论文第 2.3 节](https://arxiv.org/html/2303.04137v5#S2.SS3)

扩散推导中也会出现“后验”，例如 \(q(A^{k-1}\mid A^k,A^0)\)。在标准 DDPM 中，它由已知的加噪过程和干净样本确定，可以解析计算，不能仅凭这个词就认为存在 ACT 那样的可训练后验编码器。[DDPM 后验公式](https://arxiv.org/html/2006.11239v2#S2)

最后区分三个常见的随机操作：CVAE 的采样用来产生潜变量；DP 的加噪用来构造去噪学习问题；dropout 用来临时屏蔽网络特征。这三者的用途不同，不能因为都包含随机数就混为一谈。Dropout 的具体定义与模型配置见[训练与泛化方法](27_vla_overview.md)。

## 9. ACT 的 temporal ensembling 与 DP 的滚动执行

动作分块只是输出格式，不决定多久读取一次新观测。

DP常见做法是生成一个chunk，执行短前缀后重新规划。ACT的temporal ensembling则允许频繁生成重叠chunk，收集不同调用对**同一个执行时刻**的预测：

\[
\hat a_t^{(t)},\quad
\hat a_t^{(t-1)},\quad
\hat a_t^{(t-2)},\ldots
\]

再加权融合：

\[
a_t=\frac{\sum_{s\in\mathcal S_t}w_{t,s}\hat a_t^{(s)}}
{\sum_{s\in\mathcal S_t}w_{t,s}}.
\]

它融合的是对同一目标时刻的多个估计，不是把未来不同时间的动作直接平均。原论文的指数权重按其定义的预测顺序索引；不能看到指数衰减就自动断言越新的预测权重越大。[ACT 论文第IV-A节](https://arxiv.org/html/2304.13705v1#S4.SS1)

平滑和快速反应之间仍有取舍。接触发生突变时，历史预测可能已经过时；更频繁地调用策略也需要真实算力支持。chunk长度、实际更新频率和传感器到命令的延迟应一起报告。

## 10. 参数量与预训练需要分层记录

模型“小”与是否需要预训练没有必然关系。应分别记录：

| 层级 | 需要确认的内容 |
| --- | --- |
| 视觉/触觉encoder | 随机初始化，还是加载ImageNet、触觉或其他权重 |
| 动作网络 | 从任务示范训练，还是已有机器人策略初始化 |
| 历史/融合模块 | 新增随机参数，还是从已有模型迁移 |
| 训练阶段 | 哪些参数更新，哪些参数冻结 |
| 参数统计 | 全模型、可训练参数、部署保留参数分别多少 |

DP论文主视觉配置从零训练，不意味着所有DP都必须如此。ACT论文写到每个任务独立训练策略，但官方视觉backbone代码使用torchvision预训练权重；`FrozenBatchNorm2d`也不等于冻结整个ResNet。描述复现实验时，应明确具体初始化与 `requires_grad` 设置。[ACT 官方 backbone](https://github.com/tonyzhaozh/act/blob/main/detr/models/backbone.py)

参数量应在实际实例化后统计：

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

统计时还要说明是否包含多相机encoder、ACT训练专用后验，以及是否有共享权重。扩散网络重复执行10次并不会变成10套参数，却会增加计算量。反过来，参数更少的模型也不一定比一次前向的较大模型更快。

## 11. 开源数据与“必须预训练”的关系

DP官方项目提供代码、实验数据及复现入口，涉及Push-T、Robomimic等机器人任务。ACT项目也提供代码、仿真和任务数据入口。这些是示范学习资源，不等于存在一个所有DP或ACT必须使用的统一机器人预训练数据集。[DP 代码与数据入口](https://diffusion-policy.cs.columbia.edu/)、[ACT 官方仓库](https://github.com/tonyzhaozh/act)

一个任务策略可以只用该任务的有效示范训练，也可以先做视觉、触觉或多任务动作预训练。是否采用预训练，取决于数据量、域差异、模型容量和迁移收益，而不是仅由DP/ACT这个方法名决定。

对触觉扩展而言，最直接的做法是增加触觉encoder，把输出加入观测条件。若要记住遮挡前的对象身份或接触变化，还需要时间模块：有限观测窗口和跨时刻递归记忆是不同机制。新增触觉或记忆分支后，应通过对应的序列数据继续训练，不能只在推理时拼接随机特征。

## 12. 知识关系

| 概念 | 在本文中的作用 | 与其他概念的联系 |
| --- | --- | --- |
| Behavior Cloning | 使用专家示范训练策略 | DP和ACT是不同的条件动作模型 |
| Condition | 决定当前该生成什么动作 | 可以是embedding，通过FiLM或attention读取 |
| U-Net / Transformer | 实现动作生成网络 | 不单独决定loss或采样方法 |
| 噪声预测 | DP的一种训练参数化 | 可以换算干净动作估计，需配合噪声日程 |
| CVAE | ACT的潜变量训练框架 | 后验训练时看动作，推理采用先验约定 |
| Action chunk | 一次预测多个时刻 | 与是否自回归、是否扩散是不同维度 |
| 滚动执行 / temporal ensemble | 将预测接入真实控制 | 影响平滑、反应速度和闭环延迟 |
| 预训练 | 参数初始化与学习阶段 | 不等于必须冻结，也不由模型大小决定 |

继续理解生成目标，可回到 [动作生成方法](25_action_generation_families.md)；理解预测速度，可接着读 [Flow Matching](10_flow_matching_training_and_inference.md)；理解如何减少重复网络调用，则进入 [生成一致性与采样蒸馏](37_generative_consistency_and_sampling_distillation.md)。
