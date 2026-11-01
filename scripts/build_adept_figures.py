#!/usr/bin/env python3
"""Build original ADEPT training-stage and perceptive-student diagrams."""

from build_simtoolreal_figures import Figure


def training_stages():
    f = Figure(
        1430,
        "ADEPT：沿 Fig. 2 看谁学习、谁被冻结",
        "以 Kuka–Allegro 插入任务为例；箭头表示训练中的知识迁移与阶段顺序。",
        "通用任务先用 PPO 训练预训练 Actor 与 Critic。后训练依次进行 Actor 行为克隆、"
        "冻结 Actor 的新 Critic 预热，以及小学习率 PPO。状态 Teacher 再监督感知 Student，"
        "部署时仅保留感知 Student 和底层控制。",
    )

    f.panel(130, 250, "① Pre-training：先获得抓取、转正和搬运能力")
    f.box(55, 210, 375, 110, ["随机物体＋目标位姿", "仿真交互产生奖励"], "env", 23)
    f.box(510, 210, 375, 110, ["PPO：非对称 Actor–Critic", "更新 Actor 与 Critic"], "network", 22)
    f.box(965, 210, 380, 110, ["预训练策略 π_pre", "提供可复用的操作行为"], "network", 23)
    f.arrow([(430, 265), (510, 265)])
    f.arrow([(885, 265), (965, 265)])
    f.text(55, 355, "阶段结束：已有基础操作能力；具体的精确插入还需要后训练。", 22)

    f.panel(405, 355, "② Post-training：先迁移行为，再适应新奖励，最后学习插入")
    f.box(55, 490, 375, 150, ["a. BC：新观测接口", "冻结 π_pre，提供动作标签", "只训练 π_post"], "network", 22)
    f.box(510, 490, 375, 150, ["b. Critic warm-up", "冻结 π_post，继续采集轨迹", "按新奖励训练新的 V_post"], "signal", 22)
    f.box(965, 490, 380, 150, ["c. Conservative PPO", "Actor 使用较低学习率", "更新 π_post 与 V_post"], "network", 22)
    f.arrow([(430, 565), (510, 565)])
    f.arrow([(885, 565), (965, 565)])
    f.text(55, 689, "a 先继承能完成的搬运／重定向；b 与 c 使用最终插入目标。", 22)
    f.text(55, 735, "π_post 的角色变化：先向旧策略学习，再经 RL 学会具体任务，随后成为感知 Teacher。", 22)

    f.panel(785, 315, "③ Teacher–Student：把状态策略迁移为感知策略")
    f.box(55, 865, 375, 120, ["固定状态 Teacher", "π_post 读取仿真状态", "提供动作分布标签"], "data", 22)
    f.box(510, 865, 375, 120, ["DAgger＋监督学习", "Student 执行，Teacher 标注", "动作模仿＋8 角点辅助损失"], "signal", 21)
    f.box(965, 865, 380, 120, ["更新感知 Student", "双 RGB＋机器人数值状态", "学习从真实可得输入行动"], "network", 22)
    f.arrow([(430, 925), (510, 925)])
    f.arrow([(885, 925), (965, 925)])
    f.text(55, 1034, "感知课程：Student 先跟预训练 Teacher 学较易任务，再跟下游 Teacher 学插入。", 22)
    f.text(55, 1076, "两段都有动作与角点监督；物体角点真值是训练标签，不是部署观测。", 21)

    f.panel(1125, 275, "④ Real-world deployment：最终 Student 前向控制真机")
    f.box(55, 1205, 375, 110, ["真实相机与机器人读数", "加上控制器内部数值状态"], "data", 22)
    f.box(510, 1205, 375, 110, ["固定感知 Student", "权重不更新，记忆持续更新"], "network", 22)
    f.box(965, 1205, 380, 110, ["23 维关节动作", "底层控制 → 真实机器人"], "env", 22)
    f.arrow([(430, 1260), (510, 1260)])
    f.arrow([(885, 1260), (965, 1260)])
    f.text(55, 1365, "训练时迁移知识；真机运行最终 Student，不将三个策略串联执行。", 22)
    f.save("training-stages", prefix="adept")


def student_network():
    f = Figure(
        1235,
        "ADEPT：感知 Student 的网络结构",
        "依据论文附录 H；上方为 Kuka 视觉版本，下方列出 Sharpa 触觉扩展。",
        "两路 RGB 经共享 ResNet 和 cross-attention 得到 256 维视觉特征，与机器人本体特征"
        "和控制器状态融合，经两层 512 维 MLP、1024 维 LSTM 后输出动作。视觉特征另接训练用"
        "八角点辅助头。Sharpa 版本额外编码五指触觉并以指尖位置进行 FiLM 调节。",
    )

    f.panel(130, 405, "① 视觉与数值输入：图像特征和机器人状态共同参与决策")
    f.box(55, 215, 340, 110, ["两张 RGB 图像", "共享 ResNet 提取特征"], "data", 22)
    f.box(470, 215, 400, 110, ["Cross-attention 融合", "256 维视觉特征"], "network", 23)
    f.box(945, 215, 400, 110, ["8 个三维角点辅助头", "24 维预测＋仿真真值监督", "提供训练辅助损失"], "signal", 21)
    f.arrow([(395, 270), (470, 270)])
    f.arrow([(870, 270), (945, 270)], dashed=True)
    f.box(55, 390, 340, 100, ["本体感知特征", "控制器相关数值状态"], "data", 22)
    f.box(470, 390, 400, 100, ["拼接与融合", "视觉＋数值状态"], "network", 23)
    f.arrow([(395, 440), (470, 440)])
    f.arrow([(670, 325), (670, 390)])
    f.text(1145, 400, "辅助头教网络看清朝向", 22, anchor="middle")
    f.text(1145, 444, "部署无需输入角点真值", 21, anchor="middle")

    f.panel(560, 280, "② 控制主干：先融合当前特征，再结合近期记忆")
    f.box(55, 645, 290, 105, ["融合特征", "来自上方输入分支"], "data", 22)
    f.box(395, 645, 290, 105, ["两层 MLP", "512 → 512"], "network", 24)
    f.box(735, 645, 290, 105, ["LSTM", "1024 维记忆"], "network", 24)
    f.box(1075, 645, 270, 105, ["动作输出头", "Kuka：23 维"], "signal", 23)
    for x1, x2 in [(345, 395), (685, 735), (1025, 1075)]:
        f.arrow([(x1, 698), (x2, 698)])
    f.text(55, 804, "动作交给底层控制器执行。部署权重固定，LSTM 记忆随每次前向计算更新。", 22)

    f.panel(865, 340, "③ Sharpa 扩展：下方触觉支路仅用于视触觉版本")
    f.box(55, 950, 280, 120, ["五指触觉输入", "每指：深度图＋接触图", "两通道"], "data", 21)
    f.box(390, 950, 280, 120, ["共享小 CNN", "每个指尖输出 32 维", "五指共用网络权重"], "network", 21)
    f.box(725, 950, 280, 120, ["指尖位置 → FiLM", "按空间位置", "调节触觉特征"], "network", 21)
    f.box(1060, 950, 285, 120, ["160 维触觉特征", "加入上方融合", "动作头输出 29 维"], "signal", 21)
    for x1, x2 in [(335, 390), (670, 725), (1005, 1060)]:
        f.arrow([(x1, 1010), (x2, 1010)])
    f.text(55, 1124, "FiLM：利用指尖位置，对对应触觉特征做缩放与偏移。", 22)
    f.text(55, 1170, "触觉深度图采用统一表示；真实传感器输出需先经过校准与转换。", 21)
    f.save("student-network", prefix="adept")


if __name__ == "__main__":
    training_stages()
    student_network()
