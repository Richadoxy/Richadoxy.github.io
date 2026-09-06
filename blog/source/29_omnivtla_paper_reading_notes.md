# 《OmniVTLA: Vision-Tactile-Language-Action Models with Semantic-Aligned Tactile Sensing》精读笔记

> Zhengxue Cheng, Yiqian Zhang, Anni Tang, Keyu Wang, Wenkang Zhang, Haoyu Li, Hengdi Zhang, Li Song<br>
> arXiv 2025；本文依据2026年6月更新的v3版本整理，目前未注明正式会议或期刊<br>
> [arXiv](https://arxiv.org/abs/2508.08706) · [论文PDF](https://arxiv.org/pdf/2508.08706) · [Hugging Face论文页](https://huggingface.co/papers/2508.08706) · [ObjTac项目主页](https://readerek.github.io/Objtac.github.io/)

> 推荐先读：[AnyTouch](18_anytouch_paper_reading_notes.md)、[Attention专题](19_attention_from_embeddings_to_action_expert.md)、[Flow Matching](10_flow_matching_training_and_inference.md)、[VLA架构地图](27_vla_overview.md)与[VLA-Touch](28_vla_touch_paper_reading_notes.md)。本文代表的是“将触觉token直接接入π0内部”的路线。

## 0. 一句话定位

OmniVTLA没有重新设计VLM或Action Expert，而是在π0基础上增加一条**双编码器触觉路径**：同一份触觉力场同时经过视觉预训练ViT和语义对齐SA-ViT，得到两组tactile tokens，与语言、第三人称视觉和腕部视觉token共同进入VLM prefix，再条件化π0的flow-matching Action Expert生成动作。

```text
Language ── tokenizer ─────────────────────────┐
                                              │
RGB images ── SigLIP ─────────────────────────┤
                                              ├─> Gemma/PaliGemma VLM prefix
Tactile force image ── pretrained ViT ────────┤
                                              │
Tactile force image ── SA-ViT ────────────────┘
                                                        ⇅ joint attention
State + noisy action + timestep ───────> π0 Action Expert
                                                ↓
                                     flow-matching action chunk
```

这篇论文最值得关注的不是新的VLA主干，而是：

1. 力触觉怎样变成可输入VLA的token；
2. 触觉encoder是否应该先与视觉、语言做语义对齐；
3. 单路与双路触觉encoder，哪个更适合真实闭环控制。

---

## 1. 论文试图解决什么问题

视觉—语言基础模型经过互联网规模预训练，vision encoder与language embedding天然具有一定语义联系。触觉则不同：

- 不同传感器输出可能是RGB形变图、深度图、marker位移或力阵列；
- 机器人动作数据中的触觉样本远少于视觉—语言数据；
- 将原始触觉张量直接输入VLA，不代表模型理解材料、硬度和接触状态；
- 在少量遥操作数据上从头训练触觉encoder容易过拟合。

作者提出的核心假设是：

> 如果先让触觉表示与视觉和语言语义对齐，再接入VLA，模型会更容易利用少量机器人示范学习稳定抓取、柔性抓取和接触调整。

这里的“semantic-aligned”主要指物体层面的语义：

- object identity；
- material；
- roughness；
- hardness。

它不是把触觉与机器人动作预先对齐，也不是预训练一个触觉动作模型。

---

## 2. 主要贡献与创新边界

### 2.1 Dual-Encoder Tactile Path

OmniVTLA最直接的架构创新不是“增加一个触觉encoder”，而是为同一份触觉输入设置两条并行路径：

```text
                            ┌→ Pretrained ViT ─→ tactile tokens A
force tactile pseudo-image ┤
                            └→ SA-ViT ─────────→ tactile tokens B
```

两条路径承担不同角色。

#### Pretrained ViT路径

该路径从大规模自然图像预训练继承：

- 通用空间结构；
- 局部形状和边缘；
- 类图像纹理特征；
- 已有视觉语义先验。

虽然触觉力场不是真实RGB图像，但二维排列后仍然具有接触位置、轮廓和空间分布。

#### SA-ViT路径

SA-ViT使用触觉—视觉—文本配对数据训练，重点表示：

- 触觉专用的物理属性；
- Paxini力场的信号结构；
- 不同触觉传感器之间的共性；
- 与视觉和语言对应的对象语义。

作者希望两条路径形成互补：

```text
Pretrained ViT：通用视觉和空间先验
SA-ViT：触觉专用且语义对齐的属性先验
```

### 2.2 ObjTac三模态数据集

作者构建ObjTac：

- 56个物体；
- 10个材料类别；
- 135K视觉—触觉配对样本；
- 270K条力数据记录；
- 252段第一视角视频；
- 触觉采样60 Hz，视频30 FPS；
- object name、material、roughness、hardness和文本描述。

ObjTac使用Paxini Gen2力触觉传感器。其输出不是GelSight图像，而是经传感器内部反演得到的多维力分布。

ObjTac没有机器人action，因此只用于预训练SA-ViT，不能直接监督Action Expert。

### 2.3 将多传感器力场统一为ViT输入

作者将多个触觉传感器输出：

1. 做max-min force normalization；
2. 映射为int8三通道张量；
3. 按固定空间布局拼接成一张图；
4. resize为\(224\times224\)；
5. 用ViT编码成256个tokens。

这使两指夹爪的两个传感器，以及四指灵巧手的11个传感器，都能使用统一的图像式输入接口。

### 2.4 触觉Encoder的受控比较

论文不只比较“有触觉”和“无触觉”，还控制参数量，对从头训练、视觉预训练、触觉预训练、语义对齐和双encoder组合进行消融。

### 2.5 没有重新发明的部分

以下组件主要继承π0：

- PaliGemma/Gemma VLM backbone；
- SigLIP视觉encoder；
- Action Expert；
- joint-attention式条件交互；
- flow-matching action generation。

SA-ViT的跨模态训练也明确沿用AnyTouch第二阶段。更准确的评价是：

> OmniVTLA是一篇以触觉encoder设计和实证比较为核心的VLA论文，不是一种全新的VLA主干范式。

---

## 3. Figure 2：每个模块是什么

### 3.1 语言路径

```text
“Pick up the short can and move it to the plate.”
                         ↓
                PaliGemma tokenizer
                         ↓
                   language tokens
```

语言没有单独的T5或CLIP text encoder，而是沿用PaliGemma/Gemma的tokenizer与embedding。

### 3.2 视觉路径

机器人同时输入：

- third-view RGB；
- wrist-camera RGB。

每张图resize为\(224\times224\)，由SigLIP-400M编码为256个vision tokens：

```text
third-view RGB ─┐
                ├→ SigLIP-400M → 2 × 256 vision tokens
wrist RGB ──────┘
```

### 3.3 触觉路径

```text
多个Paxini传感器的force arrays
               ↓
normalize + reshape + stitch
               ↓
      224×224×3 pseudo-image
          ┌────┴────┐
          ↓         ↓
 pretrained ViT   SA-ViT
          ↓         ↓
 256 tokens A    256 tokens B
```

“dual encoder”不是指vision encoder和tactile encoder各一个，而是**触觉本身有两个encoder**。

### 3.4 VLM backbone

文本、视觉和两组触觉token在序列维度拼接：

\[
C=
[L,V_{\mathrm{third}},V_{\mathrm{wrist}},T_{\mathrm{pre}},T_{\mathrm{SA}}]
\]

论文将主干写作Gemma 2.6B；结合π0架构，可以理解为PaliGemma派生的VLM prefix部分。

### 3.5 Action Expert

Figure 2为了突出触觉encoder，省略了π0 Action Expert的标准输入：

- robot proprioceptive state；
- noisy action chunk；
- flow timestep。

Action Expert以VLM prefix为条件，预测flow velocity并生成动作序列。

---

## 4. Attention怎样发生

可以用“跨模态交互”描述，但不要机械地理解为：

```text
VLM先self-attention
        ↓
Action Expert再插入一个标准cross-attention模块
```

π0更接近joint attention或共享attention context：

```text
VLM prefix tokens：language + vision + tactile
Action tokens：state + noisy action + timestep
```

两侧可以使用不同的QKV和MLP参数，但Action Expert token能读取VLM prefix的K/V。动作token因此能够综合：

- 指令要求；
- 场景位置；
- 腕部视角；
- 当前接触力分布；
- 物体软硬语义。

它具有cross-modal conditioning的效果，但不一定存在一个单独命名为Cross-Attention的模块。

---

## 5. VLM没有触觉预训练，为什么能work

Gemma/PaliGemma原始预训练没有见过Paxini force field，因此它不可能天然理解原始触觉。OmniVTLA依靠三个层次解决。

### 5.1 接口兼容

力阵列先被转换成三通道伪图像，再通过ViT变成与VLM hidden width兼容的token。Transformer本身不要求token一定来自文字或自然RGB，只要求：

- embedding维度相同；
- position编码和attention mask合法；
- token经过合理投影；
- 下游存在训练监督。

### 5.2 语义初始化

SA-ViT预训练使触觉表示大致满足：

\[
x_T\approx x_V\approx x_L
\]

例如软海绵的触觉表示会靠近对应视觉样本，以及“soft”“sponge”“fabric”等语言表示。VLM收到的不再是随机力数组，而是带有视觉—语言语义的token。

### 5.3 Action Loss继续教模型使用触觉

OmniVTLA不是把SA-ViT接到完全冻结的π0后直接zero-shot控制。它还会在遥操作机器人数据上fine-tune：

```text
vision + language + tactile + state
                    ↓
                 OmniVTLA
                    ↓
              predicted action
                    ↓ compare
          teleoperation action
```

Flow-matching action loss让模型学习：

- 未接触时继续接近；
- 接触稳定后停止闭合夹爪；
- 插孔遇阻时调整位置；
- 柔软物体不应持续加大夹持。

因此真正使它工作的机制是：

\[
\boxed{
\text{语义对齐提供良好初始化}
+
\text{机器人动作监督学习怎样使用触觉}
}
\]

VTLA-FS从头训练触觉encoder也能超过无触觉π0，说明语义对齐不是使用触觉的必要条件；它主要改善初始化、数据效率、闭环性能和泛化。

---

## 6. 训练阶段一：SA-ViT预训练

### 6.1 输入数据

```text
触觉力场T
对应视觉帧V
对象文本与属性L
跨传感器正负样本
```

作者将ObjTac加入已有触觉数据，并采用AnyTouch第二阶段的多模态对齐与跨传感器匹配范式。

### 6.2 三模态对比对齐

\[
\mathcal L_{\mathrm{align}}
=
\alpha_{VL}
\frac{\mathcal L_{V\rightarrow L}+\mathcal L_{L\rightarrow V}}{2}
+
\alpha_{VT}
\frac{\mathcal L_{V\rightarrow T}+\mathcal L_{T\rightarrow V}}{2}
+
\alpha_{TL}
\frac{\mathcal L_{T\rightarrow L}+\mathcal L_{L\rightarrow T}}{2}
\]

它同时对齐：

```text
vision ↔ language
vision ↔ tactile
tactile ↔ language
```

每个方向使用CLIP/InfoNCE式batch内对比：正确配对靠近，错误配对远离。

### 6.3 Cross-Sensor Matching

模型还要判断两份触觉信号是否来自同一物体、同一接触位置：

```text
touch sample A + touch sample B
               ↓
         matching head
               ↓
             0 / 1
```

使用Binary Cross Entropy：

\[
\mathcal L_{\mathrm{match}}
=
\operatorname{BCE}(\hat y,y)
\]

所以总目标可概括为：

\[
\mathcal L_{\mathrm{SA}}
=
\mathcal L_{\mathrm{align}}
+
\lambda\mathcal L_{\mathrm{match}}
\]

### 6.4 哪些参数更新

本文可以确认更新：

- SA-ViT encoder；
- SA-ViT decoder/projection；
- cross-sensor matching head。

如果完整沿用其明确引用的AnyTouch Stage 2设置：

- text encoder冻结，作为anchor；
- vision encoder通过LoRA调整；
- tactile encoder是主要训练对象。

但OmniVTLA没有重新列出这一阶段全部模块的freeze mask，因此vision/text部分属于依据AnyTouch训练方案进行的推断。

---

## 7. 训练阶段二：机器人示范上的VTLA Fine-tuning

### 7.1 机器人示范数据

Pick-and-place：

- 两指夹爪使用4种物体；
- 四指灵巧手使用2种训练物体；
- 每个物体40条teleoperated demonstrations；
- 采集频率30 Hz。

Peg insertion：

- square与triangle用于训练；
- 共采集54条遥操作示范；
- round peg作为unseen shape；
- 采集频率30 Hz。

每个训练时刻大致包括：

```text
language instruction
third-view RGB
wrist RGB
multi-sensor tactile force
robot EEF state
ground-truth teleoperation action
```

### 7.2 动作表示

两指夹爪动作维度为：

\[
3\ \text{relative position}
+6\ \text{rotation}
+1\ \text{gripper}
=10
\]

四指灵巧手动作维度为：

\[
3+6+16=25
\]

其中16维是手指关节绝对位置。VTLA的action chunk长度为50。

### 7.3 Flow-Matching Loss

给定专家动作块\(A\)、高斯噪声\(\epsilon\)和随机时间\(\tau\)，一种常见等价写法是：

\[
A_\tau=(1-\tau)\epsilon+\tau A
\]

目标速度为：

\[
u_\tau=A-\epsilon
\]

Action Expert预测：

\[
v_\theta
\left(
A_\tau,\tau
\mid
V,T,L,s
\right)
\]

训练目标：

\[
\mathcal L_{\mathrm{FM}}
=
\mathbb E
\left[
\left\|
v_\theta-u_\tau
\right\|^2
\right]
\]

不同实现可能采用相反时间方向，速度target符号随之改变，但本质相同：学习从噪声动作分布流向专家动作分布的速度场。

### 7.4 这一阶段没有哪些loss

机器人policy fine-tuning阶段没有报告：

- 触觉属性分类loss；
- language generation loss；
- 触觉重建loss；
- 继续计算三模态对比loss；
- 独立contact prediction loss。

两个阶段不是一个联合总loss：

```text
ObjTac和已有触觉数据
        ↓ Lalign + Lmatch
      SA-ViT初始化
        ↓
机器人遥操作数据
        ↓ LFM
      OmniVTLA policy
```

### 7.5 训练超参数

Appendix Table 10给出的VTLA/π0设置：

| 设置 | 数值 |
| --- | --- |
| GPU | NVIDIA A100 80GB |
| 方法 | fine-tune |
| Peak LR | \(2.5\times10^{-5}\) |
| Warmup | 1K steps |
| Cosine decay | 29K steps，降至\(2.5\times10^{-6}\) |
| Total steps | 30K |
| Batch size | 32 |
| Action chunk | 50 |
| Input size | \(224\times224\) |
| Observation history | 无 |
| Robot state | EEF state |
| Image augmentation | 有 |

### 7.6 哪些参数会更新

可以确定或强烈支持更新：

- tactile encoder path；
- Action Expert/action head。

因为各变体被定义为“初始化后在少量teleoperation data上fine-tune”，触觉encoder不只是固定特征提取器。

但论文没有给出逐模块freeze table，也没有说明：

- Gemma 2.6B是否全量更新；
- SigLIP-400M是否冻结；
- 是否使用LoRA；
- 是否只训练部分π0 block。

项目页在本文整理时仍显示代码Coming Soon，无法从官方实现核对。最稳妥的表述是：

> OmniVTLA按照π0 fine-tuning recipe训练，至少触觉路径和Action Expert更新；Gemma与视觉SigLIP的确切训练状态，论文披露不足。

---

## 8. 推理链路

### 8.1 构造VLM Prefix

```text
instruction → tokenizer → language tokens

two RGB views → SigLIP → vision tokens

tactile arrays → normalize/stitch → pseudo-image
                                 ├→ pretrained ViT tokens
                                 └→ SA-ViT tokens
```

四类token在sequence维度拼接后进入Gemma/PaliGemma prefix。

### 8.2 Flow推理

```text
Gaussian action noise
robot state
flow timestep
VLM prefix context
        ↓
π0-style Action Expert
        ↓ 多步ODE/flow integration
50-step action chunk
```

Action Expert在积分过程中持续以包含触觉token的VLM context为条件。

### 8.3 闭环执行

机器人执行动作块中的一部分，再重新读取RGB、触觉和state，生成下一段动作。论文不使用Octopi或GPT-4o重新生成指令，也没有在π0后增加独立动作修正器。

---

## 9. Tactile Encoder感知实验

Table 2比较不同encoder在material、roughness和hardness分类上的能力。

### 9.1 作者采集数据

| Encoder | Material | Roughness | Hardness |
| --- | ---: | ---: | ---: |
| CLIP | 54.64 | 70.16 | 91.57 |
| SigLIP | 49.15 | 70.02 | 89.38 |
| Tac-ViT w/o Align | 32.78 | 70.82 | 90.44 |
| AnyTouch | 40.21 | 68.01 | 90.11 |
| SA-ViT | **70.44** | **82.21** | **93.91** |

SA-ViT在ObjTac/Paxini domain上明显最好。

### 9.2 Touch and Go外部数据

| Encoder | Material | Roughness | Hardness |
| --- | ---: | ---: | ---: |
| AnyTouch | **79.39** | **86.32** | **95.16** |
| SA-ViT | 74.90 | 85.46 | 92.10 |

SA-ViT没有在Touch and Go上超过AnyTouch。更准确的结论是：

> SA-ViT显著适配了作者的Paxini力触觉domain，同时在外部视触觉数据上维持较强能力，但不是所有触觉数据集上的统一最佳encoder。

---

## 10. Encoder架构消融

作者构造六种触觉版本：

| 模型 | 触觉路径 |
| --- | --- |
| VTLA-FS | 单encoder，从头训练 |
| VTLA-Pre | 单encoder，视觉预训练ViT |
| VTLA-SA | 单encoder，SA-ViT |
| VTLA-Pre-Pre | 双encoder，两个视觉预训练encoder |
| VTLA-Pre-Tac | 视觉预训练encoder + 未语义对齐Tac-ViT |
| OmniVTLA | 视觉预训练encoder + SA-ViT |

Pre-Pre和Pre-Tac是parameter-matched controls，用于排除“OmniVTLA只是因为参数更多”的解释。

```text
FS vs Pre：大规模视觉预训练是否有用？
Pre vs SA：触觉语义预训练是否更好？
Pre-Pre vs Omni：只是多一个encoder，还是SA对齐真正有用？
Pre-Tac vs Omni：触觉预训练本身有用，还是semantic alignment有用？
单路SA vs Omni：双路是否具有互补性？
```

---

## 11. 两指夹爪Pick-and-Place

作者使用Short Can、Square Coffee Bottle、Gum Tin和Milk Carton等物体。每种条件采用网格化初始位姿，共评估32个rollouts。

| 模型 | Success Rate | Completion Time/step |
| --- | ---: | ---: |
| π0 | 75.0% | 657 |
| VTLA-FS | 81.2% | 537 |
| VTLA-Pre | 84.4% | 586 |
| VTLA-SA | 87.5% | **484** |
| VTLA-Pre-Pre | 81.3% | 501 |
| VTLA-Pre-Tac | 84.4% | 550 |
| OmniVTLA | **96.9%** | 498 |

结果支持：

1. 加入触觉通常优于无触觉π0；
2. 单路SA-ViT优于从头训练和视觉预训练；
3. 仅增加第二个视觉encoder不能解释提升；
4. Pretrained ViT与SA-ViT组合取得最高成功率；
5. 单路VTLA-SA的平均完成时间略优于OmniVTLA。

双encoder在成功率上最好，但不是每个指标都最好。

---

## 12. 四指灵巧手实验（重点）

论文在以下配置的四指灵巧手上验证：

- 16个手指关节动作维度；
- 11个触觉传感器；
- 一个腕部相机；
- 一个外部场景相机。

| 模型 | Average SR | CT/step |
| --- | ---: | ---: |
| π0 | 93.8% | 343 |
| OmniVTLA | **100%** | **322** |

对于训练中未出现的Plastic bottle和Square bottle：

- π0：87.5%；
- OmniVTLA：100%。

这说明多传感器stitching和双触觉encoder可以从两指夹爪扩展到11触觉传感器的灵巧手。

但任务仍是pick-and-place，没有验证：

- in-hand rotation；
- regrasp；
- 指尖步态；
- 双手协同；
- 灵巧工具操作。

所以这是“使用灵巧手硬件的触觉VLA验证”，还不能等同于复杂dexterous manipulation。

---

## 13. Peg Insertion

Peg包含：

- triangle，间隙0.9 mm；
- square，间隙1.2 mm；
- round，间隙1.8 mm，作为unseen shape。

每个形状测试10次。

| 模型 | Average SR | CT/step |
| --- | ---: | ---: |
| π0 | 50.0% | 816 |
| VTLA-FS | 46.7% | 730 |
| VTLA-Pre | 70.0% | 906 |
| VTLA-SA | 70.0% | **678** |
| VTLA-Pre-Pre | 53.3% | 799 |
| VTLA-Pre-Tac | 43.3% | 910 |
| OmniVTLA | **83.3%** | 802 |

对于未见过的round peg：

- π0：30%；
- OmniVTLA：90%。

插孔时peg会遮挡孔洞，视觉信息逐渐退化。触觉能告诉策略是否发生单侧接触、卡住以及应该向哪个方向调整，所以该任务比普通抓放更能体现触觉价值。

但OmniVTLA的平均完成时间只略好于π0，明显慢于单路VTLA-SA，说明双encoder提高了稳健成功率，却没有同时带来最快执行。

---

## 14. Diffusion Policy上的触觉对比

作者还从头训练Diffusion Policy，比较vision-only与vision-touch：

| 模型 | 输入 | SR | CT/step |
| --- | --- | ---: | ---: |
| DP | Vision | 59.4% | 851 |
| VTA | Vision + Touch | 78.1% | 682 |

这个实验说明触觉不只对π0有效，也能改善普通visuomotor policy。但它不能单独证明SA-ViT或双encoder更好，因为这里没有使用完整OmniVTLA架构。

---

## 15. 轨迹平滑度与柔软物体

### 15.1 轨迹平滑度

作者测量：

- 末端轨迹的加速度变化；
- gripper action的平均绝对变化。

| 模型 | End-effector ↓ | Gripper ↓ |
| --- | ---: | ---: |
| π0 | 2.70 | 2.52 |
| VTLA-SA | **2.44** | **2.17** |
| OmniVTLA | 2.65 | 2.26 |

OmniVTLA比π0平滑，但两个指标的最佳模型都是单路VTLA-SA。

### 15.2 未见柔软物体

作者测试训练中没有出现的Towel和Sponge：

| 模型 | Average SR |
| --- | ---: |
| π0 | 50.0% |
| VTLA-FS | **93.8%** |
| VTLA-Pre | **93.8%** |
| VTLA-SA | 62.5% |
| OmniVTLA | 81.3% |

单看成功率，OmniVTLA不是最好。但作者观察到VTLA-FS和VTLA-Pre常通过非常紧的夹爪闭合抓起物体，造成明显变形；VTLA-SA和OmniVTLA的夹持更温和。

该实验支持的是：

> Semantic alignment可能使模型理解soft material，从而避免为了抓取成功而过度夹持。

它没有证明semantic alignment在柔软物体上带来最高成功率。

---

## 16. Offline MSE与闭环成功率

论文还在遥操作validation data上计算动作MSE。VTLA-Pre-Tac的平均MSE最低，OmniVTLA只排第二，但真实机器人闭环成功率由OmniVTLA领先。

这说明：

\[
\text{低offline action MSE}
\not\Rightarrow
\text{高closed-loop task success}
\]

可能原因包括：

- 专家动作具有多模态性；
- 小幅动作误差不一定导致任务失败；
- 接触后的及时反馈比整段轨迹平均MSE更重要；
- 离线数据没有覆盖policy闭环产生的状态分布。

---

## 17. 实验最终支持了什么

### 17.1 支持较充分

1. 触觉能够改善π0和Diffusion Policy的接触任务表现；
2. 大规模触觉预训练通常优于只靠少量机器人示范从头学习；
3. SA-ViT在作者的Paxini force-tactile domain上学到了更好的属性表示；
4. `Pretrained ViT + SA-ViT`在真实闭环成功率上优于参数量匹配的双encoder baseline；
5. 触觉在稳定抓取、柔软物体、视觉遮挡和插孔调整中尤其重要；
6. 图像式触觉接口可以用于两指夹爪和带11个传感器的四指灵巧手。

### 17.2 尚未充分证明

1. VLM已经像理解语言一样理解触觉；
2. SA-ViT可以zero-shot接入完全冻结的π0；
3. 双encoder在所有指标上都最好；
4. 模型具有大范围开放词汇触觉泛化；
5. 模型能完成复杂in-hand dexterity；
6. 模型能在杂乱场景中改善视觉导航；
7. 三模态语义对齐在policy fine-tuning后仍被完整保留。

特别是最后一点：SA-ViT在机器人数据上继续fine-tune后，其embedding空间可能发生漂移。论文将semantic alignment作为好的初始化，但没有测量fine-tuning前后的对齐保持程度。

---

## 18. 与此前论文的架构区别

| 工作 | 触觉接入位置 | 动作输出 | 统一policy内部融合 |
| --- | --- | --- | ---: |
| TVL | CLIP空间与LLaMA | 无 | 否 |
| Octopi | CLIP-based encoder接Vicuna | 无 | 否 |
| AnyTouch | 统一触觉representation | 无 | 否 |
| Beyond Sight / FuSe | 触觉token进入Octo/PaliGemma policy | BC/动作token | 是 |
| VLA-Touch | GPT-4o规划器与RDT后置Controller | RDT + 动作修正 | 否 |
| OmniVTLA | 双触觉encoder进入π0 VLM prefix | Flow Matching | 是 |

### 18.1 与AnyTouch

AnyTouch主要回答：

```text
怎样跨传感器学习统一的静态—动态触觉表示？
```

OmniVTLA复用其语义对齐阶段，并进一步回答：

```text
这种语义对齐触觉表示能否作为π0的输入并改善机器人动作？
```

### 18.2 与VLA-Touch

VLA-Touch：

```text
触觉在基础VLA外部
Octopi/GPT-4o负责高层规划
Interpolant Controller负责低层修正
```

OmniVTLA：

```text
触觉token进入基础VLA内部
与视觉和语言共同形成VLM prefix
直接条件化Action Expert
```

所以OmniVTLA更接近“原生触觉VLA”，而VLA-Touch更接近“给已有VLA安装触觉外挂”。

### 18.3 与FuSe

两者都让触觉进入policy内部，但训练方式不同：

- FuSe使用action loss、language contrastive loss和sensory-grounded generation loss共同迫使策略理解新增传感器；
- OmniVTLA先离线预训练SA-ViT，再使用flow-matching action loss fine-tuneπ0。

FuSe强调多任务辅助监督，OmniVTLA强调触觉encoder初始化与架构选择。

---

## 19. 优点与局限

### 19.1 优点

1. 在π0这一强VLA基线上直接验证触觉内部融合；
2. 对触觉encoder进行了系统、参数量匹配的受控实验；
3. 同时覆盖力触觉、两指夹爪和多传感器灵巧手；
4. 使用成功率、完成时间、平滑度和柔软物体变形等多种指标；
5. Peg insertion较好体现视觉遮挡下的接触反馈价值；
6. 明确展示offline MSE与closed-loop success不一致。

### 19.2 局限

1. 核心对齐方法较大程度复用AnyTouch；
2. VLM和Action Expert没有新结构，创新集中在输入侧；
3. 机器人任务种类有限，主要是pick-and-place与peg insertion；
4. 灵巧手只验证抓放，没有复杂手内操作；
5. 触觉被压成单张伪图像，没有显式建模时间动态；
6. VTLA没有observation history，短时滑移等动态可能丢失；
7. 多传感器直接拼图依赖固定布局，跨机器人迁移仍需验证；
8. SA-ViT在外部Touch and Go上不是最佳；
9. OmniVTLA在完成时间和平滑度上不是最佳；
10. 模块冻结、projector和完整训练配置披露不足；
11. 项目代码尚未开放，暂时无法核对实际attention mask和参数更新范围；
12. 作者承认模型在cluttered scenes中没有明显改善，因为失败主要来自视觉导航。

---

## 20. 最终理解

OmniVTLA不是在VLA外面附加Octopi，也不是让LLM先生成触觉描述。它直接把触觉力场变成token，放进π0的VLM prefix：

```text
Force array
    ↓ pseudo-image
    ├─ pretrained visual prior
    └─ tactile semantic prior
             ↓
      tactile token sequence
             ↓
language + vision + tactile prefix
             ↓
       π0 Action Expert
             ↓
     flow-matching actions
```

最重要的研究结论不是“只要有触觉就更好”，而是：

> 在少量机器人示范条件下，触觉encoder的预训练方式和组合方式会显著影响闭环控制；通用视觉先验与触觉语义先验具有一定互补性。

它把此前的“触觉—视觉—语言表征对齐”真正接进了π0式VLA，但仍然缺少大规模任务、动态触觉建模和完全透明的训练实现。

---

## 21. 精读时最值得记住的六点

1. Dual encoder指同一份触觉输入经过两个encoder，不是vision和touch各一个；
2. SA-ViT对齐touch、vision和language，但ObjTac本身没有action；
3. tactile tokens确实进入VLM prefix，并通过π0 joint attention影响Action Expert；
4. VLM并非天然理解触觉，语义初始化之后仍需机器人flow-matching action loss；
5. OmniVTLA真实成功率最好，但VTLA-SA在完成时间和平滑度上更好；
6. 四指灵巧手实验值得标注，但只属于灵巧手抓放，不是复杂手内操作。

## 22. 原文定位

| 内容 | 原文位置 |
| --- | --- |
| 核心问题与贡献 | Introduction、Figure 1 |
| 总体网络架构 | Figure 2、Section 3.1–3.2 |
| ObjTac与SA-ViT训练 | Figure 3、Section 3.3 |
| 三模态对齐loss | Equation 3 |
| 触觉encoder感知能力 | Table 2 |
| 机器人平台与任务 | Figure 4、Section 4.1 |
| Encoder消融定义 | Section 4.1 Evaluation Variants |
| 两指夹爪抓放 | Table 3 |
| 四指灵巧手抓放 | Table 4 |
| Diffusion Policy触觉实验 | Table 5 |
| Peg insertion | Table 6、Figure 7 |
| 轨迹平滑度 | Table 7 |
| 未见柔软物体 | Table 8、Figure 8 |
| 数据与训练超参数 | Appendix 6.1、Tables 9–10 |
