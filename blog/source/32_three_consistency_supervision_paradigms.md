# 多模态表征学习的三类一致性监督：跨模态、时间与增强不变性

> 本文是学习多模态 VLM/VLA、触觉表示学习与多模态对齐的前置内容，重点解释三类反复出现的训练范式：Cross-Modal Alignment、Temporal Coherence 和 Augmentation/Mask Consistency。

> 建议先读：[多模态 Encoder 对齐](31_multimodal_encoder_alignment_and_infonce.md)、[CLIP 与对比学习](21_clip_and_contrastive_learning.md)和[Attention 专题](19_attention_from_embeddings_to_action_expert.md)。

## 0. 先修正一个术语

这三类方法经常被笼统地称为“监督信号”，但不一定属于传统 supervised learning。

传统监督学习依赖人工标签：

```text
输入：一张触觉图像
标签：hard / soft
Loss：分类Cross-Entropy
```

本文讨论的三类方法通常不需要人工逐条标注，而是从数据自身构造监督关系：

```text
同步采集     → 跨模态正样本
时间相邻     → 时间正样本
同一输入增强 → 一致性正样本
```

因此更准确的名称是：

- self-supervised representation learning；
- weakly supervised alignment；
- consistency regularization；
- contrastive learning。

它们共同回答一个问题：

> 哪些输入虽然表面形式不同，但在 latent space 中应该被看作“同一个状态”或“相关状态”？

---

## 1. 统一视角：先定义什么应该相同

三类监督都可以写成：

```text
输入A → Encoder → zA
输入B → Encoder → zB

已知A和B表达同一个或相关状态
               ↓
要求zA与zB接近
```

它们的区别只在于 A 和 B 如何构造：

| 范式 | 输入 A | 输入 B | 希望学习的不变性 |
| --- | --- | --- | --- |
| Cross-Modal Alignment | 同一状态的视觉 | 同一状态的触觉/语言 | 模态变化不改变核心语义 |
| Temporal Coherence | 时刻 \(t\) 的状态 | 相邻时刻 \(t+1\) 的状态 | 状态随时间连续演化 |
| Augmentation/Mask Consistency | 输入的一种增强或遮挡 | 同一输入的另一种增强或遮挡 | 非关键扰动不改变核心表示 |

可以将它们记成三条等价关系：

```text
同一事件，不同模态       → 应该相似
同一过程，相邻时间       → 应该相关
同一输入，不同扰动       → 应该稳定
```

---

## 2. 范式一：Cross-Modal Alignment

### 2.1 它解决什么问题

同一个物理状态可以由不同模态观测：

```text
Vision：看到海绵被压缩
Touch：感受到柔软形变
Text：“soft and deformable sponge”
```

三个 encoder 原本可能产生完全不同的坐标系：

```text
Vision Encoder → zV
Touch Encoder  → zT
Text Encoder   → zL
```

跨模态对齐希望：

\[
z_V\approx z_T\approx z_L
\]

这里的“约等于”不是逐维完全一致，而是：

- 正确配对相似度更高；
- 可以跨模态检索；
- 表达相同语义的样本具有相近邻域；
- 不匹配样本相互远离。

### 2.2 正负样本怎样构造

假设 batch 中有 \(N\) 组同步视觉—触觉样本：

\[
(V_1,T_1),(V_2,T_2),\ldots,(V_N,T_N)
\]

对于 \(T_i\)：

- \(V_i\)：同一对象或同一时刻的视觉，是正样本；
- \(V_j,j\ne i\)：batch 中其他视觉，通常作为负样本。

### 2.3 CLIP/InfoNCE 公式

先将 encoder 输出投影并归一化：

\[
z_V=\operatorname{norm}(P_V(E_V(V))),\qquad
z_T=\operatorname{norm}(P_T(E_T(T)))
\]

相似度为：

\[
s_{ij}=\frac{z_{T_i}^{\top}z_{V_j}}{\tau}
\]

触觉检索视觉的 InfoNCE：

\[
\mathcal L_{T\rightarrow V}
=
-\frac{1}{N}\sum_{i=1}^{N}
\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{N}\exp(s_{ij})}
\]

通常还会反向计算：

\[
\mathcal L_{VT}
=
\frac{1}{2}
\left(
\mathcal L_{T\rightarrow V}
+
\mathcal L_{V\rightarrow T}
\right)
\]

三模态时可以组合：

\[
\mathcal L_{\mathrm{cross}}
=
\alpha_{VT}\mathcal L_{VT}
+
\alpha_{TL}\mathcal L_{TL}
+
\alpha_{VL}\mathcal L_{VL}
\]

### 2.4 它学到什么

跨模态对齐更容易保留：

- 对象身份；
- 材料和物理属性；
- 可以被语言描述的语义；
- 不同传感器之间的共同信息。

但它不一定保留：

- 高频接触变化；
- 微小滑移；
- 瞬时力波动；
- 对具体机器人动作最重要、但难以语言化的细节。

### 2.5 常见陷阱

#### 假负样本

batch 中两条数据可能来自不同实例，但实际语义相同。例如两块相似海绵可能被当作负样本。

#### 只对齐对象身份

模型可能主要识别“这是海绵”，却没有理解当前接触是轻压、重压还是滑动。

#### 配对不精确

不同传感器依次触摸同一个位置时，真实接触位置和力度可能存在误差。

#### 全局对齐忽略局部信息

只比较 CLS/global embedding，不能保证视觉 patch 和触觉局部区域具有精细对应关系。

### 2.6 常见程度

这是成熟、广泛使用的多模态学习范式，不是某篇触觉论文首次提出。CLIP、ImageBind、TVL、AnyTouch、OmniVTLA 和 Beyond Sight 都采用了这一思想的不同版本。

---

## 3. 范式二：Temporal Coherence

### 3.1 它解决什么问题

接触不是静态属性，而是一个连续过程：

```text
接近物体
  ↓
刚刚接触
  ↓
逐渐施力
  ↓
发生形变
  ↓
出现滑移或卡住
```

如果逐帧独立编码，模型可能看不出：

- 力正在增大还是减小；
- 当前是刚接触还是即将脱离；
- 滑移是突然发生还是持续加剧；
- 插接件是正常进入还是逐渐卡住。

Temporal Coherence 假设：

> 在合理采样频率下，同一轨迹的相邻状态通常共享大量信息，并应在 latent space 中形成连续轨迹。

### 3.2 基本构造

对于一段历史：

\[
s_1,s_2,\ldots,s_L
\]

将相邻状态定义为正样本：

\[
(s_i,s_{i+1})
\]

不同轨迹、不同样本或不对应的状态作为负样本。

### 3.3 Temporal InfoNCE

常见形式是：

\[
\mathcal L_{\mathrm{temp}}
=
\frac{1}{L-1}
\sum_{i=1}^{L-1}
\mathcal L_{\mathrm{NCE}}(s_i,s_{i+1})
\]

其中每个 \(s_i\) 可以是：

- 单一视觉表示；
- 单一触觉表示；
- 视觉和触觉的平均或融合表示；
- robot state、action 和 sensor 的联合表示。

例如 VT-MUSE Stage I 先构造同步多模态状态：

\[
s_i
=
\operatorname{Normalize}
\left(
\frac{z_{V_i}+z_{T_i}}{2}
\right)
\]

再对相邻的 \(s_i,s_{i+1}\) 做 InfoNCE。

### 3.4 它在 latent space 中产生什么

理想状态下：

```text
s₁ —— s₂ —— s₃ —— s₄ —— s₅
```

而不是把同一段轨迹的状态随机散落：

```text
s₁             s₃

        s₅

   s₂                 s₄
```

它强调的是相对连续性，不要求相邻状态完全相同。

### 3.5 与下一帧预测的区别

| 方法 | 训练要求 |
| --- | --- |
| Temporal contrast | 相邻状态的 latent 应该相关 |
| Next-frame prediction | 根据当前表示生成下一帧像素或特征 |
| Forward dynamics | 根据当前状态和 action 预测未来状态 |
| World model | 对较长未来观测、状态或奖励建模 |

Temporal contrast 的监督更轻，不要求精确恢复未来；下一帧预测和 dynamics model 对表示的信息量要求更高。

### 3.6 常见陷阱

#### 过度平滑

接触、碰撞和滑移可能突然发生。机械地拉近所有相邻状态，可能模糊关键事件边界。

#### 采样间隔不合适

- stride 太小：相邻帧几乎相同，任务过于简单；
- stride 太大：相邻帧已经发生状态跳变，不再适合作为正样本。

#### 时间负样本定义不合理

不同轨迹可能处于相同接触阶段，却被错误当成负样本。

#### 忽略 action

相同状态在不同 action 下可能产生不同未来。只学习状态连续性，不一定足以学习可控动力学。

### 3.7 常见程度

Temporal Coherence、Temporal Contrastive Learning 和 Predictive Coding 都是成熟范式，广泛用于视频、语音、传感器、机器人和强化学习。VT-MUSE 没有首次发明这种思想，其特点是把它用于同步视觉—触觉的多模态状态。

---

## 4. 范式三：Augmentation/Mask Consistency

### 4.1 它解决什么问题

同一输入经过不改变核心语义的扰动后，encoder 不应该产生完全不同的表示。

```text
同一个输入x
├─ augmentation/mask A → z¹
└─ augmentation/mask B → z²

要求：z¹ ≈ z²
```

常见扰动包括：

- 随机 crop；
- 颜色变化；
- 遮挡部分 patch；
- 传感器噪声；
- 不同 dropout；
- 缺失部分模态；
- 不同相机视角。

### 4.2 Cosine Consistency

最简单的表示一致性损失为：

\[
\mathcal L_{\mathrm{cons}}
=
1-cos(z^{(1)},z^{(2)})
\]

对于 batch 和时间窗口：

\[
\mathcal L_{\mathrm{cons}}
=
1-
\frac{1}{BL}
\sum_{b=1}^{B}
\sum_{i=1}^{L}
\cos
\left(
s_{b,i}^{(1)},s_{b,i}^{(2)}
\right)
\]

也可以使用：

- MSE consistency；
- KL divergence；
- teacher–student targets；
- stop-gradient prediction loss；
- contrastive loss。

### 4.3 Mask Consistency 的特殊意义

在视觉—触觉任务中，可以对同一视觉历史生成两种遮挡：

```text
原始视觉：[V₁,V₂,V₃,V₄,V₅]

Mask A： [V₁,MASK,V₃,V₄,MASK]
Mask B： [MASK,V₂,V₃,MASK,V₅]

触觉：   [T₁,T₂,T₃,T₄,T₅]，保持完整
```

要求两种输入得到相似的交互状态表示，会迫使模型：

- 不过度依赖某一张视觉帧；
- 利用完整触觉补偿视觉缺失；
- 从时间历史中恢复稳定状态；
- 对遮挡和视觉损坏更稳健。

### 4.4 Consistency 与 Reconstruction 不一样

Consistency：

```text
Mask A → representation A
Mask B → representation B

要求A ≈ B
```

它不要求模型知道被遮挡区域的具体像素。

Reconstruction：

```text
masked input → Encoder/Decoder → 重建原图
```

要求输出被遮挡内容：

\[
\mathcal L_{\mathrm{rec}}=\|\hat X-X\|_2^2
\]

因此：

| 目标 | 更强调什么 |
| --- | --- |
| Consistency | 扰动前后语义稳定 |
| Reconstruction | 保留足以恢复细节的信息 |

两者经常一起使用，但作用不同。

### 4.5 Collapse 风险

如果只要求：

\[
z^{(1)}=z^{(2)}
\]

最简单的作弊方式是所有输入都输出同一个常数。因此 consistency 方法通常需要至少一种防坍塌机制：

- batch negatives；
- contrastive objective；
- teacher–student；
- stop-gradient；
- variance/covariance regularization；
- 重建或分类等其他目标；
- 冻结的预训练 encoder。

VT-MUSE 将 \(L_{\mathrm{cons}}\) 与 \(L_{\mathrm{cross}}\)、\(L_{\mathrm{temp}}\) 和预训练 ViT 一起使用，因此不是只靠单独的一致性项训练。

### 4.6 常见程度

这是成熟的 consistency regularization / multi-view self-supervision 范式。BYOL、SimSiam、DINO 等方法都使用“同一输入的不同视图应有一致表示”这一思想，但网络、teacher/student、stop-gradient 和具体 loss 与 VT-MUSE 并不相同。

VT-MUSE 的具体特点是：

> 随机移除视觉证据、保留触觉，让模型学习视觉缺失下的多模态状态稳定性。

---

## 5. 三类范式之间的关系

三者并不是互相替代，而是从不同方向组织 latent space。

```text
                         一个真实交互状态
                                  │
         ┌────────────────────────┼────────────────────────┐
         ↓                        ↓                        ↓
   不同模态观察它           相邻时刻延续它           不同扰动保留它
 Vision / Touch / Text       State_t / State_t+1       Mask_A / Mask_B
         ↓                        ↓                        ↓
  Cross-Modal Loss          Temporal Loss          Consistency Loss
```

### Cross-Modal Alignment

回答：

> 同一状态换一种传感器观察，表示是否仍然对应？

### Temporal Coherence

回答：

> 同一交互过程向前发展一点，表示是否保持连续？

### Augmentation/Mask Consistency

回答：

> 输入缺失或受到无关扰动，表示是否仍然稳定？

---

## 6. VT-MUSE Stage I 是怎样组合三者的

VT-MUSE 使用独立的 Visual ViT 和 Tactile ViT：

```text
RGB image          → Visual ViT  → visual CLS vᵢ
bilateral tactile  → Tactile ViT → tactile CLS hᵢ
```

然后优化：

\[
\mathcal L_{\mathrm{Stage1}}
=
\lambda_{\mathrm{cross}}\mathcal L_{\mathrm{cross}}
+
\lambda_{\mathrm{temp}}\mathcal L_{\mathrm{temp}}
+
\lambda_{\mathrm{cons}}\mathcal L_{\mathrm{cons}}
\]

### \(L_{\mathrm{cross}}\)

```text
visual CLS vᵢ ↔ tactile CLS hᵢ
```

同步跨模态 InfoNCE，使两个 encoder 对同一接触状态形成相近表示。

### \(L_{\mathrm{temp}}\)

先构造多模态状态：

\[
s_i=\operatorname{Normalize}\left(\frac{v_i+h_i}{2}\right)
\]

再对相邻状态：

```text
sᵢ ↔ sᵢ₊₁
```

进行 temporal InfoNCE，使接触过程在 latent space 中连续。

### \(L_{\mathrm{cons}}\)

```text
同一窗口 + visual mask A → s⁽¹⁾
同一窗口 + visual mask B → s⁽²⁾
```

通过 cosine consistency 提高视觉遮挡下的稳健性，并迫使表示利用触觉。

所以 Stage I 可以压缩成：

```text
Lcross：不要被模态变化影响
Ltemp：不要忽略时间连续性
Lcons：不要被非关键遮挡影响
```

---

## 7. 三种范式分别适合什么场景

| 需求 | 优先监督 |
| --- | --- |
| 跨模态检索、开放词汇分类 | Cross-Modal Alignment |
| 接触阶段识别、动作趋势、滑移演化 | Temporal Coherence / Prediction |
| 遮挡、缺帧、传感器损坏或噪声 | Mask/Augmentation Consistency |
| 需要恢复具体缺失信息 | Reconstruction |
| 需要直接输出机器人动作 | Action Supervision |

在接触丰富机器人任务中，通常希望组合：

```text
跨模态对齐
+ 时间建模
+ 缺失模态稳健性
+ 最终动作监督
```

---

## 8. 它们为什么属于通用范式，而不是某篇论文的发明

三类思想都早已广泛存在：

- Cross-modal contrastive learning：图文、视听、视觉—深度、视觉—触觉；
- Temporal contrastive learning：视频、语音、机器人状态、强化学习；
- Consistency learning：图像增强、teacher–student、半监督学习、多视图自监督。

论文的创新一般不在“第一次写出余弦损失或 InfoNCE”，而在：

1. 如何定义正负样本；
2. 哪些模态参与；
3. 对齐 global embedding 还是局部 tokens；
4. 如何处理缺失模态和不同采样率；
5. 是否加入时序、动作和任务条件；
6. 多个 loss 如何组合；
7. 学到的 representation 如何接入下游 policy；
8. 是否通过严格消融证明每个目标有用。

VT-MUSE 的特点是把三种成熟监督组合为 Stage I，并在 Stage II 进一步训练 sequential memory，最后通过 gated cross-attention 接入动作策略。

---

## 9. 阅读相关论文时的检查清单

### Cross-Modal Alignment

- 哪些模态互相对齐？
- 正样本按对象、时刻还是接触位置定义？
- 负样本来自 batch 还是专门采样？
- encoder 哪些冻结、哪些更新？
- 使用 CLS、mean pooling 还是局部 token？
- 对齐到已有 CLIP 空间，还是共同训练新空间？

### Temporal Coherence

- 相邻的时间跨度是多少？
- stride 是否固定？
- 正样本是相邻帧还是未来帧？
- 是否加入 action 条件？
- 会不会把接触突变过度平滑？
- 负样本会不会实际处于相同接触阶段？

### Consistency Learning

- 使用了什么 augmentation 或 mask？
- 这些扰动真的不改变任务语义吗？
- 是比较 global embedding 还是每个 token？
- 有没有防止 representation collapse？
- consistency 与 reconstruction 是否同时使用？
- 测试环境中是否真的存在相应缺失或噪声？

### 实验是否充分

- 是否分别移除每一个 loss？
- 是否保持模型容量和数据量一致？
- 表征指标提升是否转化为动作成功率？
- 结果证明的是对齐、稳健性，还是整个系统的综合收益？

---

## 10. 最终记忆框架

把多模态表征学习看成在定义三种“不应该改变核心状态”的变换：

\[
\underbrace{\text{换模态}}_{\mathcal L_{\mathrm{cross}}}
,\qquad
\underbrace{\text{向前移动一个时间步}}_{\mathcal L_{\mathrm{temp}}}
,\qquad
\underbrace{\text{施加非关键扰动或遮挡}}_{\mathcal L_{\mathrm{cons}}}
\]

最终希望 encoder 满足：

```text
同一状态换一种传感器 → 仍能识别
同一过程向前发展一点 → 表示连续
视觉缺一部分或受到扰动 → 表示稳定
```

但这三类 representation loss 仍然不等于动作学习。要让机器人真正完成任务，通常还需要：

```text
learned representation
        ↓
attention / fusion
        ↓
policy / action head
        ↓
action supervision
```

最简洁的总结是：

> **Cross-modal loss 规定不同传感器如何描述同一个状态；temporal loss 规定状态如何沿时间组织；consistency loss 规定哪些缺失和扰动不应改变状态。三者共同塑造 representation，而下游 action loss 决定这些表示是否真正能用于机器人控制。**
