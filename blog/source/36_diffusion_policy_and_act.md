# Diffusion Policy 与 ACT：从观测条件到动作序列

Diffusion Policy（DP）和 Action Chunking with Transformers（ACT）都能从示范中的 observation–action 对学习机器人策略。两者都会预测一段动作，但内部生成过程不同：DP 多次修正一段带噪动作，ACT 用条件变分自编码器的训练方式学习一次输出整段动作的策略。

阅读顺序可以是：[模仿学习基础](23_imitation_learning_basics.md) → [动作生成方法总览](25_action_generation_families.md) → 本文 → [Flow Matching](10_flow_matching_training_and_inference.md) 与 [生成一致性和采样蒸馏](37_generative_consistency_and_sampling_distillation.md)。

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

ACT训练时有两条分支：

1. **后验编码器**读取专家动作和本体状态，得到潜变量分布。
2. **动作策略**读取图像、本体状态和潜变量，预测动作chunk。

\[
q_\phi(z\mid A,q)
=\mathcal N\left(\mu_\phi(A,q),
\operatorname{diag}(\sigma_\phi^2(A,q))\right).
\]

通过重参数化采样：

\[
z=\mu_\phi+\sigma_\phi\odot\xi,
\qquad \xi\sim\mathcal N(0,I),
\qquad \hat A=\pi_\theta(I,q,z).
\]

这里的VAE encoder不是视觉encoder。它编码的是动作示范及本体状态；视觉另有ResNet分支。CVAE描述的是包含潜变量、后验和解码器的生成学习框架，Transformer则描述其内部网络结构。[ACT 后验与策略实现](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py)

### 8.1 训练网络的输入输出

以下采用一般符号，\(N_v\) 是所有相机视觉特征展开后的token数：

```text
训练专用后验分支：
专家动作 A              [B,H,Da] → Linear → [B,H,d]
本体状态 q              [B,Dq]   → Linear → [B,1,d]
CLS                                              [B,1,d]
    ↓ 拼接 + 位置编码
                                                 [B,H+2,d]
    ↓ Transformer Encoder
CLS输出                                         [B,d]
    ↓ 两个参数投影
μ、log σ²                                       [B,Dz]
    ↓ 重参数化采样
z                                               [B,Dz]

动作策略分支：
多相机图像 → ResNet → 空间展开 + 位置编码         [B,Nv,d]
q → Linear                                      [B,1,d]
z → Linear                                      [B,1,d]
    ↓ 拼接后经过 Transformer Encoder
融合后的 memory                                 [B,Nv+2,d]
    ↓ 由 H 个动作query通过 Transformer Decoder读取
动作hidden states                               [B,H,d]
    ↓ 动作投影
预测动作 Â                                      [B,H,Da]
```

动作query代表各个待预测的时间位置；它们不需要输入此前生成的动作token。因此，ACT在一次策略调用中并行输出整个chunk。

### 8.2 重建项与 KL 的分工

按官方 `policy.py`，主要损失是masked L1重建项加KL正则：

\[
\mathcal L_{\mathrm{ACT}}
=\mathbb E_{z\sim q_\phi}
\left[\operatorname{MaskedL1}(\pi_\theta(I,q,z),A)\right]
+\beta D_{\mathrm{KL}}
\left(q_\phi(z\mid A,q)\|\mathcal N(0,I)\right).
\]

对于对角高斯，KL可以写成：

\[
D_{\mathrm{KL}}
=\frac12\sum_j
\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right).
\]

L1要求动作接近示范；KL约束后验分布不要任意偏离推理使用的先验。尾部padding动作不应形成普通有效监督，具体损失归约方式需要与实现一致。[ACT 官方损失](https://github.com/tonyzhaozh/act/blob/main/policy.py)

训练时后验看到真实动作是CVAE构造的一部分，但这个分支不能在推理时继续要求真实动作。原版ACT推理舍弃后验编码器，令 \(z=0\)，即标准高斯先验的均值：

```text
训练：专家动作 + q → 后验 → 采样 z ─┐
                                   ├→ 策略 → Â → L1 + KL
              图像 + q ────────────┘

推理：                  z = 0 ─────┐
                                   ├→ 策略 → 动作chunk
              图像 + q ────────────┘
```

一般CVAE可以从先验采样，但“原版ACT默认推理随机采样style”并不准确。使用先验均值也不等于输出示范动作的算术平均，因为解码器是非线性的。[ACT 训练与推理说明](https://tonyzhaozh.github.io/aloha/)

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
