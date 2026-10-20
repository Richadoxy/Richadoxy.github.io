# 触觉 VLA 与模仿学习推荐阅读顺序

> 目标读者：刚进入触觉机器人学习、希望最终研究触觉VLA、接触丰富操作或灵巧手的新人。<br>
> 文中“论文”链接均指向公开的PMLR或arXiv页面。<br>
> 状态说明：ICML 2024和ICLR 2025按正式版本标注；其余若公开版本首页只显示arXiv，则统一写作“arXiv”，不猜测后续录用状态。<br>

## 0. 先建立整个领域的地图

触觉VLA并不是一个单一问题，可以拆成五层：

```text
① 触觉传感器与数据表示
          ↓
② 视觉—触觉—语言表征对齐
          ↓
③ 将触觉接入VLA/模仿学习策略
          ↓
④ 执行时闭环反馈、力控和动态预测
          ↓
⑤ 灵巧手、多指和双臂高自由度操作
```

推荐按这个依赖关系阅读，而不是单纯按论文年份阅读。

每篇论文按六个维度整理：传感器与原始输入、encoder 的初始化与更新范围、模态融合位置、动作表示与生成目标、示范和标签来源，以及闭环实验与接触质量。它们串起了“数据进入模型，再由动作影响环境”的完整链条。

### 从小模型到触觉记忆的学习分支

先理解可独立训练的小型策略，有助于把触觉的作用从大模型规模、语言能力和预训练数据中分开。下面这条分支连接动作生成、触觉输入、历史状态和研究验证。

| 顺序 | 知识笔记 | 与下一步的联系 |
| --- | --- | --- |
| 1 | [动作生成方法](25_action_generation_families.md) | 区分数据、网络、训练目标和采样方式 |
| 2 | [Diffusion Policy 与 ACT](36_diffusion_policy_and_act.md) | 把抽象 loss 放进完整观测—动作网络 |
| 3 | [Flow Matching](10_flow_matching_training_and_inference.md) | 理解噪声、干净动作、速度参数化与生成时间 |
| 4 | [生成一致性与采样蒸馏](37_generative_consistency_and_sampling_distillation.md) | 理解少步生成怎样改变延迟与训练成本 |
| 5 | [从原始触觉到策略条件](39_tactile_input_representation.md) | 将真实传感器读数构造成有空间和时间含义的输入 |
| 6 | [CGP 精读](38_cgp_paper_reading_notes.md) | 区分未来物理状态、触觉表示与实际控制命令 |
| 7 | [策略记忆与对象级触觉状态](40_policy_memory_and_object_centric_tactile_state.md) | 从一帧接触信息走向遮挡下的历史状态估计 |
| 8 | [触觉记忆的实验设计](41_tactile_memory_experiment_design.md) | 把模型设计变成能够被验证或推翻的研究假设 |

这条分支也与已有基础文章相连：[表示与编码](31_multimodal_encoder_alignment_and_infonce.md) 解释输入怎样变成特征，[监督配方](32_three_consistency_supervision_paradigms.md) 解释特征学什么，[Policy 融合](19_attention_from_embeddings_to_action_expert.md) 解释这些特征怎样影响动作。

### 知识之间的四条连接

- **表示到动作：** 触觉重建和对齐塑造输入特征；动作监督决定特征是否服务于控制。
- **单帧到历史：** 当前接触可能不足以确定状态；历史、已执行动作和新观测共同帮助更新估计。
- **生成到反馈：** 采样步数影响推理耗时，传感器刷新和动作前缀决定新触觉何时改变行为。
- **模型到证据：** 离线误差、对象归属准确率和闭环成功率分别描述不同能力，需要各自的测试。

---

## 第一阶段：触觉—视觉—语言表征基础

这一阶段先回答“触觉怎样变成模型可以使用的语义表示”，暂时不急着看动作策略。

### 1. TVL：从这里开始

**Fu et al., A Touch, Vision, and Language Dataset for Multimodal Alignment — ICML 2024**<br>
[论文](<Fu et al. - 2024 - A Touch Vision and Language Dataset for Multimodal Alignment.pdf>)

重点：

- DIGIT原始RGB触觉图像；
- 视觉—触觉—语言三模态InfoNCE；
- CLIP语义空间；
- 开放词汇跨模态检索；
- projector、gate以及接入LLaMA的方法；
- GPT-4V伪标签的优势与偏差。

读完应能回答：embedding、token、对齐和生成分别发生在哪一层。

### 2. Touch100k：看数据规模如何扩大

**Cheng et al., Touch100k: A Large-Scale Touch-Language-Vision Dataset — arXiv 2024**<br>
[论文](<Cheng et al. - 2024 - Touch100k A Large-Scale Touch-Language-Vision Dataset.pdf>)

与TVL对照阅读：

- 数据规模和覆盖对象如何变化；
- 语言标签如何获得；
- 数据多样性是否真正转化为跨对象、跨环境泛化；
- 是否仍然过度依赖视觉来猜触感。

### 3. Octopi：从对齐进入触觉属性推理

**Yu et al., Octopi: Object Property Reasoning with Large Tactile-Language Models — arXiv 2024**<br>
[论文](<Yu et al. - 2024 - Octopi Object Property Reasoning with Large Tactile-Language Models.pdf>)

重点看：

- 触觉语言模型如何从“检索形容词”发展到“物体属性推理”；
- 触觉信息是否真的带来视觉无法提供的增量；
- 生成式评估是否可靠。

### 4. AnyTouch：跨传感器、静态与动态表征

**Feng et al., AnyTouch: Unified Static-Dynamic Representation across Visuo-tactile Sensors — ICLR 2025**<br>
[论文](<Feng et al. - 2025 - AnyTouch Unified Static-Dynamic Representation across Visuo-tactile Sensors.pdf>)

重点看：

- 不同视觉式触觉传感器之间的域差异；
- 静态触觉图像与动态触觉序列如何统一；
- TVL只用单帧全局embedding的局限如何被推进；
- 表征能否迁移到真正的下游机器人任务。

完成第一阶段后，应该能画出：

```text
raw tactile signal
      ↓
tactile encoder
      ↓
aligned embedding / tactile tokens
      ↓
LLM或robot policy
```

---

## 第二阶段：从触觉表征进入Language-Action策略

这一阶段重点关注：触觉信息如何从“能被描述”变成“能影响动作”。

### 5. Beyond Sight：理解异构传感器如何接入通用策略

**Jones et al., Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding — arXiv 2025**<br>
[论文](<Jones 等 - 2025 - Beyond Sight Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding.pdf>)

推荐先读它，因为它在表征学习和策略学习之间起桥梁作用。重点：

- 为什么用语言作为不同传感器之间的中介；
- 如何改造已经预训练的通用机器人策略；
- 新传感器接入时冻结和微调哪些模块；
- 是否需要重新训练整个VLA。

### 6. TLA：直接进入Tactile-Language-Action

**Hao et al., TLA: Tactile-Language-Action Model for Contact-Rich Manipulation — arXiv 2025**<br>
[论文](<Hao et al. - 2025 - TLA Tactile-Language-Action Model for Contact-Rich Manipulation.pdf>)

重点：

- 触觉、语言和动作怎样形成统一模型；
- 与传统行为克隆/视觉策略相比，触觉在哪些接触阶段有用；
- action是连续回归、离散token还是chunk；
- 训练和推理时是否都使用触觉。

### 7. VTLA：学习偏好优化如何用于连续机器人动作

**Zhang et al., VTLA: Vision-Tactile-Language-Action Model with Preference Learning — arXiv 2025**<br>
[论文](<Zhang et al. - 2025 - VTLA Vision-Tactile-Language-Action Model with Preference Learning.pdf>)

重点：

- 视觉—触觉—动作—指令数据的组织；
- DPO怎样用于连续操作或动作token；
- 与Diffusion Policy、TLA和视觉VLA的比较；
- 仿真到真实peg-in-hole是否依赖特定触觉传感器。

### 8. VLA-Touch：区分高层语义触觉与低层反馈触觉

**Bi et al., VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback — arXiv 2025**<br>
[论文](<Bi et al. - 2025 - VLA-Touch Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback.pdf>)

重点：

- dual-level tactile feedback具体对应哪些网络层和控制层；
- 触觉用于高层规划还是低层动作修正；
- 执行时反馈频率是否足够；
- 与把触觉简单拼接进VLA相比有何收益。

### 9. OmniVTLA：语义对齐的触觉VLA

**Cheng et al., OmniVTLA: Vision-Tactile-Language-Action Models with Semantic-Aligned Tactile Sensing — arXiv，本地版本更新至2026**<br>
[论文](<Cheng et al. - 2025 - OmniVTLA Vision-Tactile-Language-Action Models with Semantic-Aligned Tactile Sensing.pdf>)

把它和TVL放在一起比较：

- TVL的语义对齐怎样被真正用于action模型；
- 对齐损失是在预训练阶段还是策略训练阶段使用；
- 语义对齐是否改善了下游操作泛化。

### 10. Tactile-VLA：物理知识与触觉泛化

**Huang et al., Tactile-VLA: Unlocking Physical Knowledge for Tactile Generalization — arXiv 2025**<br>
[论文](<Huang et al. - 2025 - Tactile-VLA Unlocking Physical Knowledge for Tactile Generalization.pdf>)

重点看泛化对象：

- 未见物体；
- 未见材质；
- 未见任务；
- 未见接触状态；
- 更换触觉传感器。

不要只看平均成功率，要确认论文所说的“generalization”究竟是哪一种。

---

## 第三阶段：力觉、闭环反馈和未来预测

前两阶段关注“触觉能不能进入模型”，这一阶段关注“执行时能不能及时改变动作”。

### 11. ForceVLA：先理解力觉如何进入VLA

**Yu et al., ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation — arXiv 2025**<br>
[论文](<Yu et al. - 2025 - ForceVLA Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation.pdf>)

重点：

- force与vision-based tactile的区别；
- MoE专家如何根据接触状态分工；
- 力觉影响的是表征、动作头还是控制器；
- 模型频率是否满足接触控制需要。

### 12. DreamTacVLA：从当前触觉转向未来触觉预测

**Ye et al., Learning to Feel the Future: DreamTacVLA — arXiv 2025/2026版**<br>
[论文](<Ye et al. - 2025 - Learning to Feel the Future DreamTacVLA.pdf>)

重点：

- 为什么只编码当前触觉还不够；
- 预测未来触觉是否改善动作决策；
- 预测误差会不会反过来误导策略；
- 它更接近world model还是普通VLA辅助头。

### 13. TaF-VLA、FD-VLA与ForceVLA2：成组阅读

1. **TaF-VLA: Tactile-Force Alignment in VLA Models — arXiv 2026**<br>
   [论文](<Huang et al. - 2026 - TaF-VLA Tactile-Force Alignment in VLA Models.pdf>)
2. **FD-VLA: Force-Distilled Vision-Language-Action Model — arXiv 2026**<br>
   [论文](<Zhao et al. - 2026 - FD-VLA Force-Distilled Vision-Language-Action Model.pdf>)
3. **ForceVLA2: Hybrid Force-Position Control with Force Awareness — arXiv 2026**<br>
   [论文](<Li et al. - 2026 - ForceVLA2 Hybrid Force-Position Control with Force Awareness.pdf>)

比较表建议这样填写：

| 问题 | TaF-VLA | FD-VLA | ForceVLA2 |
|---|---|---|---|
| 输入是触觉图像还是力/力矩 |  |  |  |
| 是否需要推理时力传感器 |  |  |  |
| action表示 |  |  |  |
| 是否使用独立低层控制器 |  |  |  |
| 控制频率 |  |  |  |
| 评价是否包含峰值力/平均力 |  |  |  |

### 14. TacVLA、AT-VLA、ResTacVLA与ViTaR：比较融合机制

建议按下面顺序读：

1. **TacVLA: Contact-Aware Tactile Fusion for Robust VLA Manipulation — arXiv 2026**<br>
   [论文](<Zhang et al. - 2026 - TacVLA Contact-Aware Tactile Fusion for Robust VLA Manipulation.pdf>)
2. **AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction — arXiv 2026**<br>
   [论文](<Li et al. - 2026 - AT-VLA Adaptive Tactile Injection for Enhanced Feedback Reaction.pdf>)
3. **Feeling the Unexpected: ResTacVLA via Residual Tactile Representation — arXiv 2026**<br>
   [论文](<Zhang et al. - 2026 - Feeling the Unexpected ResTacVLA via Residual Tactile Representation.pdf>)
4. **ViTaR: Visuo-Tactile Residual Adaptation for Foundation VLA Manipulation — arXiv 2026**<br>
   [论文](<Wang et al. - 2026 - ViTaR Visuo-Tactile Residual Adaptation for Foundation VLA Manipulation.pdf>)

这组论文共同回答：触觉应该直接拼接、门控注入、接触时激活，还是只作为视觉VLA的残差修正？

重点比较：

- 是否保留原有VLA能力；
- 无接触阶段是否抑制触觉；
- 触觉是修正hidden state还是直接修正action；
- 视觉和触觉冲突时模型信任谁；
- residual方法是否真的比完全融合更稳定。

### 15. TacForcing：执行时流式触觉反馈

**Zhou et al., TacForcing: Streaming Action Generation with Execution-Time Tactile Feedback — arXiv 2026**<br>
[论文](<Zhou et al. - 2026 - TacForcing Streaming Action Generation with Execution-Time Tactile Feedback.pdf>)

放在后面读，因为它要求先理解action chunk、推理延迟和闭环反馈。重点：

- action生成期间如何接收新触觉；
- 是否需要中断或修正已经生成的action chunk；
- 实际感知—决策—执行频率；
- 与开环VLA的关键差别。

### 补充阅读：RDP 的慢速扩散与快速触觉闭环

**Xue et al., Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation — RSS 2025**<br>
[精读笔记](35_rdp_paper_reading_notes.md) · [arXiv](https://arxiv.org/abs/2503.02881v3) · [官方代码](https://github.com/xiaoxiaoxh/reactive_diffusion_policy)

RDP 是任务特定的视觉—触觉模仿学习策略，可在阅读触觉 VLA 的执行时反馈机制时作为基础对照。重点理解 Fig. 5 中动作块、temporal ensembling 与慢快策略的反馈时序差异，以及 Fig. 6 中非对称 tokenizer 和潜在扩散策略的两阶段训练。论文同时展示 TactAR 如何改善示范质量，但快策略仍从记录下来的专家动作学习，不能凭触觉自动纠正系统性的差示范。

### 16. T-Rex：把一条Flow拆成低频规划与高频触觉修正

**Niu et al., T-Rex: Tactile-Reactive Dexterous Manipulation — arXiv 2026**<br>
[精读笔记](34_trex_paper_reading_notes.md) · [arXiv](https://arxiv.org/abs/2606.17055) · [官方代码](https://github.com/ZhuoyangLiu2005/T-Rex)

建议在理解Flow Matching、Action Chunk和TacForcing以后阅读。重点：

- Latent、Action、Tactile三套Transformer Expert怎样共享causal joint attention；
- future-latent query tokens怎样用未来head-camera feature监督；
- 16帧F6历史怎样通过VQ-VAE形成离散动态token；
- 为什么在\(\tau_{\mathrm{split}}=0.4\)处分割同一条action flow；
- slow stream怎样缓存\(x_{0.4}\)和latent/action KV；
- fast stream怎样用新触觉多次从同一个boundary重新生成action chunk；
- 论文约5/20 Hz表述与公开代码可配置调度之间的区别。

T-Rex与TacForcing都关注execution-time touch，但T-Rex的核心接口是“缓存上下文 + 级联flow后半段”，不是逐token强制或简单action residual。

---

## 第四阶段：灵巧手与高自由度操作

这一阶段需要同时理解遥操作数据、手臂—手指动作分解、触觉闭环和高维action建模。

### 17. 【灵巧手重点】End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy

**Cui et al. — arXiv 2025**<br>
[论文](<Cui et al. - 2025 - End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy.pdf>)

推荐作为灵巧手第一篇，重点：

- human负责宏观手臂运动、自动策略负责微观手指控制的shared autonomy；
- tactile和local vision如何用于DexGrasp-VLA Copilot；
- 如何收集协调的arm-hand demonstrations；
- macro arm与micro hand特征怎样分开又共享；
- corrective teleoperation如何追加失败恢复数据；
- 约90%成功率和50多个物体的实验是否有严格未见物体划分。

### 18. 【灵巧手重点】MoDE-VLA

**Tang et al., Towards Human-Like Manipulation with MoDE-VLA — arXiv 2026**<br>
[论文](<Tang et al. - 2026 - Towards Human-Like Manipulation with MoDE-VLA.pdf>)

这篇进一步进入双臂、多指、接触丰富的in-hand manipulation：

- IMCopilot如何用强化学习原子技能辅助遥操作；
- 63-DoF双臂灵巧操作的数据瓶颈；
- Mixture-of-Dexterous-Experts如何组织不同技能；
- force与tactile如何残差注入预训练VLA；
- VLA高层决策与低层技能primitive如何分工。

建议把这两篇做成对照：

| 维度 | Dexterous Arm-Hand VLA | MoDE-VLA |
|---|---|---|
| 单臂/双臂 |  |  |
| 灵巧手自由度 |  |  |
| 人类和自动模块分工 |  |  |
| 触觉传感器 |  |  |
| action层级 |  |  |
| 原子技能/端到端策略 |  |  |
| 失败恢复方式 |  |  |

### 19. Tabero：温柔操作和物理交互质量

**Wu et al., Tabero: Learning Gentle Manipulation with Vision, Touch and Language — arXiv 2026**<br>
[论文](<Wu et al. - 2026 - Tabero Learning Gentle Manipulation with Vision Touch and Language（复件）.pdf>)

虽然不应仅按“灵巧手论文”理解，但它对多指抓取和易损物体操作很重要。重点：

- 成功抓起并不等于操作质量好；
- force-position解耦命令；
- 固定hybrid controller如何执行VLA输出；
- 平均抓力、峰值力、物体损伤等指标；
- “gentle”语言指令如何真正改变物理交互。

---

## 第五阶段：前沿扩展，按研究兴趣选择

这些论文建议在完成前四阶段后阅读，因为它们通常默认读者已经理解基础VLA和触觉融合。

### A. 多传感器与物理世界建模

**MLA: A Multisensory Language-Action Model — arXiv，本地版本更新至2026**<br>
[论文](<Liu et al. - 2025 - MLA A Multisensory Language-Action Model.pdf>)

- 同时处理2D图像、3D点云和触觉token；
- 直接让LLM承担多模态感知；
- 通过预测未来多传感器目标学习物理动态。

**VT-WAM: Visual-Tactile World Action Model — arXiv 2026**<br>
[论文](<Tian et al. - 2026 - VT-WAM Visual-Tactile World Action Model.pdf>)

- 重点理解world model、未来状态预测和action生成之间的关系。

### B. 触觉token规模化

**N0-VTLA: Scaling with Latent Tactile Tokens — arXiv 2026**<br>
[论文](<NeoteAI and Fudan TEAI - 2026 - N0-VTLA Scaling with Latent Tactile Tokens.pdf>)

- 对照TVL“一个全局触觉embedding”的信息瓶颈；
- 关注latent tactile tokens的数量、压缩方式和计算成本；
- 检查规模扩大是否带来跨任务、跨物体和跨传感器收益。

### C. 标注、提示与触觉监督

**TAP-VLA: Tactile Annotation Prompting for VLA Models — arXiv 2026**<br>
[论文](<Van der Merwe et al. - 2026 - TAP-VLA Tactile Annotation Prompting for VLA Models.pdf>)

- 触觉语言标注怎样作为提示或辅助监督影响VLA；
- 与TVL的GPT-4V伪标签路线比较；
- 目录中的“复件”是同一论文，无需重复阅读。

### D. 训练时有触觉、推理时没有触觉

**HapticVLA: Contact-Rich Manipulation without Inference-Time Tactile Sensing — arXiv 2026**<br>
[论文](<Gubernatorov et al. - 2026 - HapticVLA Contact-Rich Manipulation without Inference-Time Tactile Sensing.pdf>)

- 触觉是否仅作为训练期privileged information；
- 没有推理时触觉后，模型如何应对真实的意外滑移和接触扰动；
- 这种路线与真正的闭环触觉策略适用场景有何不同。

---

## 推荐的最短主线

如果暂时不想读完全部论文，可以先按这14篇走：

1. TVL：三模态语义对齐；
2. Touch100k：触觉语言数据规模化；
3. AnyTouch：跨传感器、静态—动态表征；
4. Octopi：触觉属性推理；
5. Beyond Sight：异构传感器接入通用策略；
6. TLA：触觉—语言—动作模型；
7. VTLA：接触任务与偏好学习；
8. VLA-Touch：高层/低层触觉反馈；
9. ForceVLA：力觉、MoE和接触操作；
10. DreamTacVLA：未来触觉预测；
11. TacForcing：执行期闭环触觉；
12. RDP：潜在动作块与高频触觉解码；
13. T-Rex：低频视觉规划与高频触觉flow refinement；
14. **Dexterous Arm-Hand VLA → MoDE-VLA：灵巧手与双臂高自由度操作。**

这条主线对应：

```text
静态表征
  → 开放语言语义
  → 跨传感器/动态表征
  → action生成
  → 力觉与闭环反馈
  → 灵巧手高维控制
```

---

## 每个阶段的产出建议

### 第一阶段完成后

画一张encoder结构图，说明每种输入、embedding维度、对齐损失和评估任务。

### 第二阶段完成后

建立VLA对比表：backbone、触觉注入层、action表示、训练数据和真实机器人任务。

### 第三阶段完成后

建立控制对比表：传感器频率、VLA推理频率、低层控制器、动作chunk长度、反馈延迟和力指标。

### 第四阶段完成后

画出灵巧手系统的数据闭环：遥操作 → shared autonomy → 示范数据 → 策略训练 → 失败恢复 → 数据追加。

### 从知识点到研究判断

| 设计选择 | 需要理解的联系 |
| --- | --- |
| 语义预训练或端到端动作学习 | 预训练标签保留的信息，应与下游操作需要的信息相匹配 |
| 全局向量、区域 tokens 或历史序列 | 空间分辨率决定接触能否区分，时间范围决定过去事件能否被利用 |
| 注入 backbone、动作头或残差控制器 | 注入位置影响哪些计算能够读取触觉，也影响反馈延迟 |
| 低频规划与高频修正 | 长时任务意图与短时接触变化需要在执行时间轴上配合 |
| 视触觉与力传感器组合 | 局部形变、法向压力和方向性受力提供不同测量信息 |
| 臂、腕与手指控制 | 宏观接近、掌部支撑和局部接触调整具有不同的动作作用 |
| 感知指标与操作指标 | 更好的表示还需要转化为目标完成、保持稳定和可接受的交互质量 |

沿着这些联系整理知识，可以从“记住某篇论文用了什么模块”，逐步走向“根据任务信息和可用数据选择模块，并用实验检验选择”。
