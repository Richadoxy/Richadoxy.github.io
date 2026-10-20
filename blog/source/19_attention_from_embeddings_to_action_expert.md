# 从对齐表示到 Policy 融合：Transformer、Attention、Readout 与 Gate

> 本文把必要的 Transformer 基础放进多模态 Policy 的真实数据流中理解：Attention 在前向传播中改变什么，self、cross 与 joint attention 怎样交换信息，CLS/readout 怎样汇总，gate、projector、AdaNorm 和 Action Head 又分别位于哪一层。

> 阅读位置：[统一研究范式](27_vla_overview.md) → [公共语义空间](31_multimodal_encoder_alignment_and_infonce.md) → [表示监督配方](32_three_consistency_supervision_paradigms.md) → 本文。本文不再把 Transformer 当作孤立基础知识，而是围绕“对齐后的表示怎样真正影响动作”展开。

## 0. 先给出核心结论

此前的理解已经抓住了主要方向：

- self-attention的Q、K、V来自同一组token；
- cross-attention的Q来自一组token，K、V来自另一组memory；
- QK相似度决定每个query应该从哪些value读取信息。

但“通过attention更新权重”这句话容易把两种完全不同的weight混在一起：

| 名称 | 是什么 | 什么时候变化 | 是否保存在checkpoint中 |
| --- | --- | --- | ---: |
| Attention weights | `softmax(QK^T)`得到的动态信息分配系数 | 每次前向、每个输入都重新计算 | 否 |
| Model weights | `W_Q`、`W_K`、`W_V`、MLP等可训练参数 | 反向传播后由optimizer更新 | 是 |

因此更准确的说法是：

> **前向传播时，attention根据当前输入临时算出attention weights，并用它们更新token的hidden representations；训练时，loss再通过反向传播更新生成Q/K/V的模型参数。**

---

## 1. 一个token是“2048个向量”吗

不是。以GR00T的VLM宽度2048为例：

```text
一个token = 一个2048维向量
          = [h_1, h_2, ..., h_2048]
```

其中2048个元素是标量特征，不是2048个独立向量。

如果一批数据中有`B`个样本，每个样本有`N`个token，Transformer hidden state通常写成：

\[
H \in \mathbb{R}^{B\times N\times d}
\]

例如：

```text
GR00T VLM memory: [B, S, 2048]

B    batch size
S    图像和文本组成的token序列长度
2048 每个token的hidden width
```

`2048`描述的是每个位置能携带多少通道的信息；`S`才表示有多少个token位置。

---

## 2. Attention的一次前向传播到底做什么

设输入hidden states为：

\[
H=[h_1,h_2,\ldots,h_N]
\]

模型使用四组已经训练好的参数做投影：

\[
Q=HW_Q,\qquad K=HW_K,\qquad V=HW_V
\]

然后计算：

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
\]

\[
O=AV
\]

### 2.1 Multi-head里的单头维度是什么

设每个token的hidden width为\(d\)，一个attention layer有\(h\)个query heads。对第\(r\)个head：

\[
W_Q^{(r)},W_K^{(r)},W_V^{(r)}\in\mathbb R^{d\times d_h}
\]

因此：

\[
Q^{(r)},K^{(r)},V^{(r)}\in\mathbb R^{N\times d_h}
\]

\(d_h\)是**单个attention head内部的特征维度**，也就是一次query-key点积包含多少个通道。标准Multi-Head Attention通常满足：

\[
d=h\,d_h
\]

例如Qwen3-VL-2B的文本Transformer使用：

```text
hidden size d       = 2048
query heads h       = 16
head dimension d_h  = 128
KV heads            = 8
```

所以每个query head在128维子空间中计算相似度。它使用Grouped-Query Attention：16个Q heads共享8组K/V heads，即每两组query heads共享一组K/V。实现中通常把所有head的投影合并为一个大矩阵，而不是显式保存16个Python层：

\[
W_Q\in\mathbb R^{d\times(hd_h)}
\]

“self/cross/joint”描述Q/K/V来自哪里和哪些位置可连接；“16 heads”描述有多少组并行子空间，两者不是同一种分类。

### 2.2 沿Key维度做Softmax

对固定query token \(i\)，先计算它与每个key token \(j\)的score：

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}+M_{ij}
\]

然后沿所有key位置\(j\)归一化：

\[
a_{ij}=\frac{\exp(s_{ij})}{\sum_{m=1}^{N_K}\exp(s_{im})}
\]

数值实现通常先减去该行最大值：

\[
a_{ij}=\frac{\exp(s_{ij}-\max_m s_{im})}{\sum_m\exp(s_{im}-\max_m s_{im})}
\]

这样既不改变结果，又能避免指数溢出。若一行score为：

\[
[2,1,0]
\]

softmax后约为：

\[
[0.665,0.245,0.090]
\]

每项非负且一行和为1。被mask的位置加上\(-\infty\)，经过softmax后权重为0。

这里“沿Key维度”指query对不同key token的分数归一化，不是对一个key向量内部的\(d_h\)个特征做softmax。

### 2.3 Attention Matrix的形状与含义

单个head的attention matrix为：

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)
\in\mathbb R^{N_Q\times N_K}
\]

其中：

- 第\(i\)行对应query token \(i\)；
- 第\(j\)列对应key/value token \(j\)；
- \(a_{ij}\)表示query \(i\)从value \(j\)读取多少信息。

多头和batch维都保留时：

\[
A\in\mathbb R^{B\times h\times N_Q\times N_K}
\]

最终：

\[
o_i=\sum_j a_{ij}v_j
\]

Attention matrix是由当前输入临时计算的activation，不是checkpoint中的可训练参数。同一个模型输入不同图像、语言或触觉时会产生不同的\(A\)。

这里：

- `Q`表示每个token当前想检索什么；
- `K`表示每个token可以通过什么特征被检索；
- `V`表示被选中以后真正提供的内容；
- `A`是attention matrix；
- `O`是聚合其他token信息后的新表示。

经过输出投影和residual connection，可简化写成：

\[
H' = H + OW_O
\]

于是前向传播中发生的是：

```text
旧token表示 H
  -> 计算当前输入专属的检索关系 A
  -> 汇总其他token的V
  -> 得到新的token表示 H'
```

### 2.4 一个极简例子

假设一句话只有三个token：

```text
[拿起] [红色] [杯子]
```

当“杯子”token作为query时，它可能对“红色”token产生较大的attention coefficient。于是更新后的“杯子”hidden state不再只表示杯子类别，也携带“要拿的是红色那个”的上下文。

下次输入变成“拿起蓝色杯子”时，模型参数没变，但Q、K和attention matrix会随输入变化。

---

## 3. 前向“更新表示”与训练“更新参数”

把一个训练step拆开，就很清楚了。

### 3.1 Forward

```text
输入tokens
  -> 用当前W_Q/W_K/W_V计算Q/K/V
  -> 计算attention matrix A
  -> 生成新的hidden states
  -> 经过后续层得到预测
  -> 计算loss
```

这一步只是在内存中生成新的activation。`W_Q/W_K/W_V`被读取，但没有在attention算子内部自行改变。

### 3.2 Backward与optimizer step

```text
loss
  -> 反向传播得到 dLoss/dW_Q、dLoss/dW_K、dLoss/dW_V
  -> optimizer根据梯度更新参数
```

例如：

\[
W_Q \leftarrow W_Q-\eta\frac{\partial \mathcal L}{\partial W_Q}
\]

模型学到的不是某一张固定attention map，而是**如何从不同输入生成有用attention map的投影规则**。

### 3.3 为什么不只更新产生Query的投影参数

设cross-attention为：

\[
Y=\operatorname{softmax}\left(
\frac{(X_qW_Q)(X_mW_K)^\top}{\sqrt{d_h}}
\right)(X_mW_V)W_O
\]

Loss对\(Y\)反向传播时：

- attention score依赖Q和K，因此\(W_Q\)与\(W_K\)都有梯度；
- 加权读取的内容直接依赖V，因此\(W_V\)有梯度；
- 最终输出依赖\(W_O\)，所以\(W_O\)也有梯度；
- 如果memory encoder没有freeze或detach，梯度还会进入产生\(X_m\)的网络。

“Cross-attention只更新query stream”描述的是前向输出写回哪个token位置，不代表训练时只有query一侧参数得到梯度。

例如Action tokens以Q读取VLM memory的K/V：

```text
forward：新的hidden states写回Action token位置
backward：W_Q、W_K、W_V、W_O和未冻结的上游网络都可能获得梯度
```

只有在memory被显式freeze、`detach()`或在`no_grad()`中缓存时，来自该loss的梯度才不会继续穿过memory分支。T-Rex训练Tactile Expert时就使用了detached latent/action KV cache，因此tactile loss可以读取slow-stream知识，但不会反向更新slow stream。

### 3.3 推理时会发生什么

推理没有backward和optimizer，因此模型参数被冻结；但attention matrix仍会根据每次输入重新计算。于是模型仍能动态关注不同图像区域、文字或VLM memory。

---

## 4. Transformer是否就是hidden state进、hidden state出

如果只看一个Transformer block，答案基本是“对”。典型Pre-Norm block可以写成：

\[
\widetilde H_l=H_l+\operatorname{Attention}(\operatorname{Norm}(H_l))
\]

\[
H_{l+1}=\widetilde H_l+\operatorname{MLP}(\operatorname{Norm}(\widetilde H_l))
\]

输入和输出常常具有相同shape：

```text
H_l     [B, N, d]
  ↓ Transformer block
H_{l+1} [B, N, d]
```

两个子层分工不同：

- Attention：让不同token位置之间交换信息；
- MLP/FFN：对每个token内部的特征通道做非线性变换。

但“完整Transformer模型”通常还包括：

```text
原始输入
  -> tokenizer / patch embedding / action encoder
  -> 一叠Transformer blocks
  -> pooling / language head / action decoder
  -> 最终输出
```

因此，block是hidden-state-to-hidden-state；完整模型则可以是文本到文本、图像到embedding或噪声动作到velocity。

---

## 5. Self-attention与Cross-attention的严格区别

命名依据是Q、K、V来自哪组hidden states，而不是“里面有几种语义模态”。

### 5.1 Self-attention

\[
Q=H W_Q,\quad K=H W_K,\quad V=H W_V
\]

Q、K、V来自同一序列。输出更新的也是这组序列。

### 5.2 Cross-attention

设query stream为`H_q`，外部memory为`M`：

\[
Q=H_q W_Q,\quad K=M W_K,\quad V=M W_V
\]

输出更新query stream，而memory主要被读取：

```text
Action tokens ──Q────┐
                     ├─> attention -> 更新Action tokens
VLM memory ───K/V────┘
```

### 5.3 拼接多模态token的特殊情况

如果先把触觉token和文本token拼成一个序列，再在整个序列上做self-attention，那么从算子定义看它仍是self-attention；从功能上看，它已经发生了跨模态交流。

Octopi中的Vicuna就是这种情况。不要仅凭“没有单独叫CrossAttention的模块”断言模型没有跨模态交互。

### 5.4 Joint Causal Attention：共同计算不等于完全双向

假设三类token按顺序拼接：

\[
H=[H_L;H_A;H_T]
\]

分别代表latent、action与tactile。三类token可以使用不同的projection：

\[
Q=[H_LW_Q^L;H_AW_Q^A;H_TW_Q^T]
\]

\[
K=[H_LW_K^L;H_AW_K^A;H_TW_K^T],\qquad
V=[H_LW_V^L;H_AW_V^A;H_TW_V^T]
\]

它们仍然在同一个attention matrix中计算，但causal mask使信息流具有方向：

| Query \ Key | Latent | Action | Tactile |
| --- | ---: | ---: | ---: |
| Latent | ✓ | × | × |
| Action | ✓ | ✓ | × |
| Tactile | ✓ | ✓ | ✓ |

所以“joint attention”只说明token共享attention geometry，不能据此推断所有模态双向互读。真正的信息流必须结合token顺序和mask判断。

T-Rex就是这个例子：Latent、Action、Tactile Expert分别拥有独立Q/K/V/O和FFN，再把三组Q/K/V拼起来做joint causal self-attention。Action query可以读取前面的视觉语言latent；fast tick中的tactile query可以读取缓存的latent/action K/V，但缓存不会被触觉反向更新。详见[T-Rex精读](34_trex_paper_reading_notes.md)。

### 5.5 Learned Query Token不代表它只产生Q

CLS、action readout、Q-Former queries或future-latent queries常被称为“query tokens”。这通常是功能命名：它们是一组可学习槽位，用于在attention中收集特定任务需要的信息。

如果learned query和其他token一起进入self-attention，它通常仍会产生自己的：

\[
Q_{\mathrm{query}},K_{\mathrm{query}},V_{\mathrm{query}}
\]

只有在一个明确的cross-attention模块中，架构才可能规定query stream只提供Q，而另一个memory stream提供K/V。

因此需要区分：

| 说法 | 含义 |
| --- | --- |
| query在attention公式中的角色 | 本次计算中提供Q的hidden states |
| learned query token | 可训练的信息收集槽位，不保证只产生Q |

T-Rex的future-latent query tokens属于第二种：它们位于causal self-attention序列中，每层都会通过Latent Expert生成Q/K/V；训练目标则让这些位置的最终hidden states对齐未来head-camera的ViT features。

---

## 6. OpenCLIP视觉编码器中的Self-attention

以常见的ViT-L/14、224×224输入作为说明性例子：

```text
图像 224×224
  -> 每14×14像素形成一个patch
  -> 16×16 = 256个patch tokens
  -> 加1个CLS token
  -> 共257个tokens
```

每个Transformer层内，Q、K、V都来自这257个视觉token：

```text
[CLS, patch_1, ..., patch_256]
               ↓ self-attention
每个patch读取其他图像区域
CLS逐层汇聚全图信息
```

对某个attention head而言，attention matrix的空间尺寸为：

\[
A\in\mathbb{R}^{257\times257}
\]

例如，描述杯子把手的patch可以关注杯身边缘；CLS token可以汇总形状、纹理和物体类别。最后，CLS或pooled feature经过projection进入CLIP共享embedding空间。

需要注意：具体hidden width、head数量和是否采用其他pooling取决于OpenCLIP checkpoint。这里的257仅来自`224/14`这个示例，不是所有CLIP视觉塔的固定长度。

### 6.1 OpenCLIP文本塔

文本token也在自己的Transformer中做self-attention：

```text
“a rough red cup”
  -> text tokens
  -> causal/self-attention
  -> pooled text embedding
```

视觉塔和文本塔在encoder前向中各自工作，二者不做cross-attention。训练末端通过CLIP contrastive loss，使匹配图文的全局embedding接近。

---

## 7. GR00T Action Expert中的Cross-attention

GR00T提供了与OpenCLIP完全不同的例子。VLM先处理图像和指令，产生memory：

```text
VLM memory [B, S, 2048]
```

这里是`S`个token，每个token宽2048。Action Expert不是直接把每个VLM token解释成动作，而是先形成自己的query stream：

```text
robot state -> 1个state token  [B, 1, 1536]
noisy action -> 40个action tokens [B, 40, 1536]
concat -> [B, 41, 1536]
```

所以要纠正一个常见说法：**进入GR00T Action Expert的VLM memory宽度是2048，但Action Expert自身的hidden width是1536。**

### 7.1 两边宽度不同，如何做QK相似度

Cross-attention使用不同的线性投影把二者映射到相同的head space。以32个head、每个head 48维为例：

```text
Action hidden [B, 41, 1536]
  -> W_Q
Q [B, 32, 41, 48]

VLM memory [B, S, 2048]
  -> W_K, W_V
K,V [B, 32, S, 48]
```

此时可以计算：

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{48}}\right)
\in\mathbb{R}^{B\times32\times41\times S}
\]

再用`A`对VLM的`V`加权，输出经过投影后回到：

```text
更新后的state/action hidden [B, 41, 1536]
```

### 7.2 这个Cross-attention在做什么

假设指令是“把红色杯子放进蓝色碗中”。某个action token可以产生一个query，从VLM memory中重点读取：

- “红色杯子”的文字和视觉位置；
- “蓝色碗”的文字和视觉位置；
- 当前任务阶段需要关注的目标。

不同时间位置的action token可能得到不同的attention map：靠近杯子时关注杯子，搬运末端则更多关注碗。

GR00T的AlternateVLDiT还穿插self-attention：

```text
Cross-attention：action/state从VLM memory读取任务信息
Self-attention：state token和不同时间的action tokens彼此协调
```

完整结构可对照[GR00T与DiT Action Head](11_groot_dit_architecture.md)。

---

## 8. 对比学习与Cross-attention不是同一层级

这是理解TVL和AnyTouch的关键。

### 8.1 对比学习是一种训练目标

```text
Touch Encoder  -> x_T
Vision Encoder -> x_V
Text Encoder   -> x_L

loss约束：x_T ≈ x_V ≈ x_L
```

每个encoder可以独立前向。InfoNCE根据配对关系计算标量loss，再通过backward更新encoder参数。

它直接塑造的是embedding space的几何结构：

- 匹配样本更接近；
- 不匹配样本更远；
- 单一模态在推理时也能独立得到可比较的embedding。

### 8.2 Cross-attention是一种前向信息融合算子

```text
Touch tokens -> Q
Vision tokens -> K/V
        ↓
更新后的touch-conditioned-on-vision tokens
```

它告诉模型“如何从另一模态读取信息”，却没有自动规定正确答案是什么。要让cross-attention变得有用，仍需分类loss、生成loss、动作loss等监督。

### 8.3 核心对照

| 维度 | 对比学习 | Cross-attention |
| --- | --- | --- |
| 类型 | loss / 训练目标 | 网络层 / 前向算子 |
| 输入 | 通常是各模态独立得到的全局embedding | 两组token hidden states |
| 输出 | 标量loss | 融合后的query hidden states |
| 主要作用 | 建立共享可度量语义空间 | 按当前任务动态读取另一模态细节 |
| 推理时是否要求两种模态同时存在 | 通常不要求 | 通常要求memory存在 |
| 粒度 | 常见为样本级/global | token级 |
| 是否天然支持检索和缓存 | 是 | 较弱，因表示依赖另一模态 |
| 是否自动保证模态对齐 | 是其直接目标 | 否，仍由最终loss决定 |

所以两者不互斥。常见组合是：

```text
第一步：用对比学习预训练独立、可迁移的encoder
第二步：在LLM或VLA中用joint/cross-attention完成任务相关融合
```

---

## 9. 为什么TVL和AnyTouch的模态encoder之间不直接用Cross-attention

因为它们当前阶段的目标是“对齐并获得独立可用的encoder”，而不是“给定所有模态完成一个特定下游任务”。

### 9.1 需要单模态独立推理

开放词汇触觉分类时，系统可能只有触觉输入；检索时，也希望提前缓存视觉或文本库的embedding。如果触觉表示必须cross-attend视觉才能生成，它就不再是独立表征。

### 9.2 训练数据经常缺模态

AnyTouch整合的九个数据集并非每条样本都有触觉、视觉和文本。独立encoder加pairwise loss可以自然跳过缺失模态；cross-attention通常要求另一组token真实存在。

### 9.3 配对监督往往只有样本级

如果只知道“一段触觉对应一张物体图”，却不知道每个触觉patch对应图像中的哪个patch，直接做细粒度cross-attention容易学习外观捷径或噪声关系。

### 9.4 检索和迁移效率

独立embedding可以离线编码、建立向量索引，并被不同分类器、语言模型和机器人策略复用。Cross-attention每次任务都要同时运行两个模态流，计算和耦合更强。

这并不表示cross-attention不适合触觉。到了“根据当前视觉、触觉和语言生成动作”的VLA阶段，它通常非常合适；只是它解决的是下一层的**条件融合**问题。

---

## 10. Attention在TVL、Octopi、AnyTouch中的具体位置

### 10.1 TVL

TVL encoder阶段：

```text
Tactile ViT  --self-attention--> x_T
OpenCLIP ViT --self-attention--> x_V
Text Transformer --self-attention--> x_L

x_T, x_V, x_L --contrastive loss--> 共享语义空间
```

三个encoder之间没有细粒度cross-attention。随后TVL-LLaMA把视觉和触觉全局特征经过projector送入LLaMA-Adapter。LLaMA文本query会读取adapter中的多模态K/V，并由零初始化的`tanh` gate控制注入强度。它可以理解为带门控的cross-attention式adapter路径，但不是触觉patch与视觉patch之间的直接cross-attention。

更多背景见[TVL精读：从触觉表征到LLaMA](15_tvl_paper_reading_notes.md)。

### 10.2 Octopi

Octopi的CLIP ViT-L/14触觉编码器内部使用self-attention，Visual Prompt Tuning加入的prompt tokens也参与同一视觉序列的attention。

触觉特征经过projector以后，被放入`<tact_start>`与`<tact_end>`之间，再和文字embedding一起送入Vicuna：

```text
[文本token, tact_start, tactile embeddings, tact_end, 文本token]
                         ↓
                 Vicuna causal self-attention
```

从模块命名看是self-attention；从语义功能看，文本与触觉token已经能在同一序列中交互。Octopi没有TVL那样正常的外部视觉输入分支，也没有单独的cross-attention模块或gate。

更多背景见[Octopi精读：从触觉属性到大模型物理推理](17_octopi_paper_reading_notes.md)。

### 10.3 AnyTouch

AnyTouch的Touch Encoder、Vision Encoder和Text Encoder各自在内部使用self-attention，模态之间通过全局对比loss对齐。

Stage 1的Touch Decoder也用Transformer处理重建任务；Stage 2的跨传感器匹配则是两个embedding逐元素相乘后进入MLP Task Head，并不是cross-attention。

更多背景见[AnyTouch精读：统一静态—动态多传感器触觉表征](18_anytouch_paper_reading_notes.md)。

---

## 11. 四类模型放在同一张表中

| 模型/阶段 | Attention发生在哪里 | 模态如何相遇 | 输出目的 |
| --- | --- | --- | --- |
| OpenCLIP | 图像塔和文本塔分别self-attention | 全局embedding上做contrastive loss | 图文对齐与检索 |
| TVL encoder | 触觉、视觉、文本塔分别self-attention | 全局embedding上做contrastive loss | 触觉进入视觉—语言语义空间 |
| TVL-LLaMA | LLaMA self-attention + gated adapter交互 | 文本query读取多模态adapter K/V | 触觉/视觉语言生成 |
| Octopi | 触觉ViT self-attention；Vicuna联合序列causal self-attention | 投影后的触觉embedding插入文本序列 | 属性描述与物理推理 |
| AnyTouch | 各模态encoder分别self-attention | contrastive loss；跨传感器MLP matching | 通用触觉embedding |
| GR00T Action Expert | action/state self-attention与对VLM memory的cross-attention交替 | Action query读取视觉语言K/V | 连续action chunk生成 |
| T-Rex | Latent/Action/Tactile分别投影Q/K/V后做joint causal self-attention | 后置token读取前置token；fast tactile读取缓存KV | 分段flow-matching action chunk |

这张表揭示了一个很实用的分层：

```text
Encoder表征层：常用self-attention + contrastive alignment
LLM理解层：常用联合序列self-attention或adapter/cross-attention
VLA动作层：常用action query对VLM/tactile memory做cross-attention
```

不是每篇论文都必须使用全部三层。

---

## 12. Attention进入触觉VLA时可以怎么设计

未来把AnyTouch或TVL接入VLA，可以有几种常见方式。

### 12.1 全部token早期拼接

```text
[vision tokens, tactile tokens, language tokens, action tokens]
                       ↓
               joint self-attention
```

优点是模态充分交互；缺点是序列很长，计算量随token数平方增长，而且训练更耦合。

### 12.2 Action Expert做Cross-attention

```text
vision/language/tactile encoders -> multimodal memory
action tokens -> Q
memory -> K/V
```

这与GR00T思路接近：感知表示先形成memory，动作query按时间步读取相关信息。触觉只在发生接触以后出现，也容易通过mask或异步更新memory。

### 12.3 分层或门控注入

```text
视觉语言作为主memory
触觉adapter/residual在接触时门控增强
```

优点是可以从已有VLA初始化，避免触觉噪声在无接触时干扰全部动作。gate应由接触状态或学习到的可靠度控制，而不只是固定相加。

无论采用哪种attention结构，仍需要动作监督、flow matching或行为克隆loss。Cross-attention本身不会凭空教会机器人“触到滑移以后应该加力”。

### 12.4 Gate：控制新信息写入多少

新增触觉、音频或外部 memory 时，常见 residual injection 是：

\[
H'=H+g\cdot F(H,M)
\]

其中：

- \(H\)：原 backbone 的 hidden states；
- \(F(H,M)\)：adapter 或 cross-attention 读出的新模态信息；
- \(g\)：可学习 gate。

如果 \(g\) 采用零或小值初始化，训练开始时模型接近原策略，随后逐渐学会使用新模态。这能降低随机初始化分支突然破坏预训练能力的风险。

Gate 不是“决定某个 attention token 看谁”的 attention matrix；它控制的是整条新增信息路径对 residual stream 的贡献强度。

### 12.5 FiLM、AdaLN 与 AdaRMS：Condition 不一定成为 Token

有些条件适合通过 token 参与 attention，例如语言、图像和触觉 patch。另一些全局条件更适合调节整层计算，例如 diffusion/flow timestep、机器人 embodiment 或全局任务向量。

FiLM 的一般形式是：

\[
H'=\gamma(c)\odot H+\beta(c)
\]

Adaptive Normalization 则先归一化，再由 condition 产生 scale、shift 或 gate：

```text
timestep / embodiment / pooled condition
                 ↓ MLP
          scale, shift, gate
                 ↓
        调制Transformer block
```

- AdaLN 使用 LayerNorm；
- AdaRMS 使用 RMSNorm；
- condition 本身没有序列位置，也不需要产生独立 Q/K/V。

可以把 token condition 理解成“加入会议的参与者”，把 AdaNorm 理解成“调整整间会议室的工作模式”。

### 12.6 CLS、Readout 与 Action Head：汇总和输出不是一回事

CLS 是最经典的 readout token：

```text
[CLS, patch₁, ..., patchₙ]
          ↓ self-attention
hCLS汇总一张图的信息
```

机器人 Transformer 中的 readout 通常更任务化：

```text
observation/task tokens
+ action readout tokens
          ↓ Transformer
action readout hidden states
          ↓ Action Head
robot action chunk
```

需要严格区分：

| 组件 | 职责 |
| --- | --- |
| CLS/readout token | 在 Transformer 中承载汇总后的 hidden representation |
| Projector | 转换维度或接口空间 |
| Action Head/Decoder | 把 hidden states 解码成动作、动作 token、噪声或 velocity |

因此 readout 不是最终动作，Action Head 也不一定只是单层线性 projector。复杂策略中它可能是 CVAE decoder、Diffusion Transformer 或 Flow-Matching Action Expert。

### 12.7 Multi-expert 不一定是稀疏 MoE

典型 sparse MoE 是：

```text
token → router → top-k experts → weighted merge
```

PI0/PI0.5 一类 multi-expert Transformer 则更接近：

```text
vision/language prefix → Prefix Expert参数
state/noisy action     → Action Expert参数
                    ↓
       通过联合attention geometry协同
```

它由 token 区段固定选择参数，不一定有 router 或 top-k。阅读论文时不能看到 `expert` 就自动理解成稀疏路由 MoE。

### 12.8 一条完整的 Policy Fusion 链路

把本章概念串起来：

```text
各模态原始输入
   ↓ modality encoders
vision / language / tactile tokens
   ↓ projector到兼容宽度
joint attention或形成perception memory
   ↓
action/readout tokens作为Query读取memory
   ↓ optional gate / AdaNorm condition
action hidden states
   ↓ Action Head / Expert
action chunk
```

对齐预训练回答“每种模态怎样被表示”；这一链路回答“这些表示在当前任务下怎样真正改变动作”。

---

## 13. 常见误区

### 误区一：Attention weight就是模型参数

不对。attention matrix是当前样本临时计算的activation；`W_Q/W_K/W_V`才是训练参数。

### 误区二：一个2048宽token包含2048个token或向量

不对。它是一个包含2048个标量分量的向量。

### 误区三：不同hidden width无法做Cross-attention

不对。分别使用`W_Q`和`W_K/W_V`投影到共同head dimension即可。GR00T的1536宽action stream读取2048宽VLM memory就是实例。

### 误区四：有Cross-attention就完成了模态对齐

不对。Cross-attention只提供信息读取通道，是否学到正确关系取决于loss和数据。

### 误区五：没有名为CrossAttention的层，就没有跨模态交流

不对。多模态token拼进同一序列后，普通self-attention也可以完成跨模态交互，Octopi/Vicuna就是例子。

### 误区六：Transformer只等于Attention

不对。典型block至少还有normalization、MLP/FFN、residual connection；完整模型还包含输入embedding和输出head。

### 误区七：只有Query投影会被动作Loss更新

不对。只要计算图没有被freeze或detach，loss会同时沿attention score和value aggregation更新\(W_Q/W_K/W_V/W_O\)以及相关上游encoder。前向中“更新Q位置的hidden state”和反向中“哪些参数有梯度”是两个问题。

### 误区八：Joint Attention表示所有模态完全双向可见

不对。Joint只表示共享一次attention计算；causal、prefix、blockwise或padding mask仍可禁止大量连接。读VLA论文时必须同时写出token顺序和attention mask。

---

## 14. 最终记忆框架

先记住四句话：

1. **Token**：一个位置对应一个`d`维hidden vector。
2. **Attention forward**：用固定的当前模型参数生成Q/K/V，动态计算attention matrix，再更新token representation。
3. **Training backward**：loss通过梯度更新`W_Q/W_K/W_V`等模型参数。
4. **对比学习与cross-attention**：前者塑造共享空间，后者执行条件信息融合；两者经常前后衔接，而不是相互替代。

把两篇具体模型压缩成最后一幅数据流：

```text
OpenCLIP / TVL / AnyTouch
  输入 -> encoder self-attention -> global embedding
                              -> contrastive loss建立共享空间

GR00T Action Expert
  action/state hidden -> Q ───────┐
                                  ├-> cross-attention -> 新action hidden
  VLM memory -> K/V ──────────────┘
                                  -> action decoder -> 连续动作
```

这也解释了为什么触觉VLA通常可以先学一个可独立使用的tactile encoder，再让action expert用cross-attention或joint attention读取它：**先把触觉“表示清楚”，再让动作网络“按任务使用”。**
