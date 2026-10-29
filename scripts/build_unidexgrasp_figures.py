#!/usr/bin/env python3
"""Build six self-contained SVG diagrams for the UniDexGrasp reading guide.

Run: python3 scripts/build_unidexgrasp_figures.py
The output is limited to blog/assets/figures/unidexgrasp-*.svg.
"""

from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "blog" / "assets" / "figures"
C = {
    "network": ("#eff6ff", "#2563eb"),
    "data": ("#f1f5f9", "#64748b"),
    "change": ("#fff7ed", "#ea580c"),
    "env": ("#ecfdf5", "#059669"),
    "removed": ("#f8fafc", "#94a3b8"),
}


class Fig:
    def __init__(self, h, title, desc, sub):
        self.h = h
        self.p = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{h}" viewBox="0 0 1400 {h}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
            '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10z" fill="#475569"/></marker></defs>',
            '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif;fill:#172033}.bold{font-weight:700}.muted{fill:#475569}</style>',
            f'<rect width="1400" height="{h}" fill="white"/>',
        ]
        self.txt(40, 55, title, 32, True)
        self.txt(40, 94, sub, 21)
        for x, k, t in [
            (40, "network", "学习模块"),
            (270, "data", "输入／表示"),
            (500, "change", "相对完整方案的改动"),
            (880, "removed", "移除／跳过"),
            (1130, "env", "仿真／评估"),
        ]:
            fill, stroke = C[k]
            self.p.append(
                f'<rect x="{x}" y="120" width="24" height="24" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            )
            self.txt(x + 34, 140, t, 20)

    def txt(self, x, y, s, size=22, bold=False, anchor="start", muted=False):
        self.p.append(
            f'''<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" class="{("bold" if bold else "muted" if muted else "")}">{escape(s)}</text>'''
        )

    def panel(self, y, h, title):
        self.p.append(
            f'<rect x="24" y="{y}" width="1352" height="{h}" rx="16" fill="#fbfcfe" stroke="#e2e8f0" stroke-width="2"/>'
        )
        self.txt(55, y + 37, title, 25, True)

    def node(self, x, y, w, h, lines, kind="data", size=22):
        fill, stroke = C[kind]
        dash = 'stroke-dasharray="8 6"' if kind == "removed" else ""
        self.p.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2" {dash}/>'
        )
        lh = size + 8
        top = y + h / 2 - (len(lines) - 1) * lh / 2 + size * 0.35
        for i, line in enumerate(lines):
            self.txt(x + w / 2, top + i * lh, line, size, i == 0, "middle")

    def edge(self, pts, dash=False):
        st = 'stroke-dasharray="7 6"' if dash else ""
        d = "M" + " L".join((f"{x} {y}" for x, y in pts))
        self.p.append(
            f'<path d="{d}" fill="none" stroke="#475569" stroke-width="2.5" marker-end="url(#arr)" {st}/>'
        )

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / name
        path.write_text("\n".join(self.p + ["</svg>"]), encoding="utf8")
        ET.parse(path)
        print(path)


def generation():
    f = Fig(
        2070,
        "UniDexGrasp：抓姿生成方法与消融（Table 1／2）",
        "比较生成抓姿的模型、分解方式和测试时优化。所有框图是功能示意，不代表全部网络层。",
        "R：旋转；t：位置；q：关节配置。TTA 调整抓姿数值；图中省略采样批次与坐标恢复。",
    )
    rows = [
        (
            "DDG：直接回归抓姿",
            [["多视角深度"], ["DDG 回归网络"], ["R、t、q"]],
            ["data", "network", "data"],
            "保留其原始无桌面设定；论文对此方法不施加桌面穿透过滤。",
        ),
        (
            "GraspTTA（C + T）：联合 CVAE + 测试时修正",
            [["点云"], ["CVAE", "联合生成 R、t、q"], ["ContactNet + TTA"]],
            ["data", "network", "network"],
            "C：CVAE；T：测试时适应／优化（TTA）。T 不表示 Teacher。",
        ),
        (
            "R + C + T：固定旋转生成路线，将 Glow 换为 CVAE",
            None,
            None,
            "与完整方案比较：第二个生成模型从 Glow 换成 CVAE。",
        ),
        (
            "ReLie + T：旋转与其余抓姿参数联合建模",
            [["点云"], ["联合 flow", "轴角／Lie 代数旋转表示"], ["ContactNet + TTA"]],
            ["data", "change", "network"],
            "联合生成 R、t、q；与旋转单独使用 IPDF 的分解方案比较。",
        ),
        (
            "ProHMR + T：另一种旋转表示的联合 flow",
            [["点云"], ["联合 flow", "6D 旋转表示 → SO(3)"], ["ContactNet + TTA"]],
            ["data", "change", "network"],
            "这是一条替换建模方式的 baseline，不是直接拔掉某个层。",
        ),
        (
            "完整方案（R + GL + T）：旋转分解 + Glow + TTA",
            None,
            None,
            "GL：GraspGlow。位置与关节配置仍然联合生成。",
        ),
    ]
    y = 180
    for i, (label, nodes, kinds, note) in enumerate(rows):
        f.panel(y, 205, label)
        if nodes:
            xs = [65, 435, 1015]
            ws = [290, 500, 305]
            for j, (x, w, lines, k) in enumerate(zip(xs, ws, nodes, kinds)):
                f.node(x, y + 65, w, 88, lines, k)
                if j < 2:
                    f.edge([(x + w, y + 109), (xs[j + 1], y + 109)])
        else:
            ns = [
                (60, 130, ["点云"], "data"),
                (230, 190, ["GraspIPDF", "采样旋转 R"], "network"),
                (460, 215, ["按 R⁻¹", "规范化点云"], "data"),
                (
                    715,
                    265,
                    ["CVAE" if i == 2 else "GraspGlow", "生成位置 t、关节 q"],
                    "change" if i == 2 else "network",
                ),
                (1020, 300, ["ContactNet + TTA", "优化当前抓姿"], "network"),
            ]
            for j, (x, w, lines, k) in enumerate(ns):
                f.node(x, y + 65, w, 88, lines, k)
                if j < 4:
                    f.edge([(x + w, y + 109), (ns[j + 1][0], y + 109)])
        f.txt(60, y + 185, note, 20, muted=True)
        y += 222
    f.txt(40, y + 35, "Table 2：进一步检查“怎样分解”和“是否保留测试时优化”", 26, True)
    y += 65
    for label, mid, last, note in [
        (
            "decoup.T：进一步拆开位置与关节生成",
            ["分开生成 t 与 q", "概念示意"],
            ["ContactNet + TTA"],
            "论文未给出精确条件顺序；此图不假设 t 与 q 独立，也不指定自回归网络。",
        ),
        (
            "w/o TTA：跳过测试时抓姿修正",
            ["GraspGlow", "联合生成 t、q"],
            ["跳过 TTA", "直接输出抓姿"],
            "移除的是测试时优化；不能据此说 ContactNet 相关训练损失也被移除。",
        ),
    ]:
        f.panel(y, 205, label)
        ns = [
            (60, 270, ["点云 → GraspIPDF", "采样旋转 R"], "network"),
            (370, 215, ["按 R⁻¹", "规范化点云"], "data"),
            (625, 315, mid, "change" if "decoup" in label else "network"),
            (980, 340, last, "network" if "decoup" in label else "removed"),
        ]
        for j, (x, w, lines, k) in enumerate(ns):
            f.node(x, y + 65, w, 88, lines, k)
            if j < 3:
                f.edge([(x + w, y + 109), (ns[j + 1][0], y + 109)])
        f.txt(60, y + 185, note, 20, muted=True)
        y += 222
    f.txt(
        40,
        2040,
        "阅读边界：主表没有单独的“去掉端到端接触训练”一行；不额外补造消融。",
        21,
    )
    f.save("unidexgrasp-generation-comparisons.svg")


def teacher():
    f = Fig(
        1540,
        "UniDexGrasp：Teacher 的结构与监督消融（Table 3）",
        "完整教师、去掉坐标规范化、去掉类别辅助监督的对照。PPO更新策略；分类损失更新几何表示。",
        "蓝色：保留的学习模块；橙色／灰色：改动。三行均需按对应设置训练和评估。",
    )
    y = 180
    for title, sc, cls in [
        ("完整 Teacher", True, True),
        ("w/o SC：移除状态规范化与动作坐标恢复", False, True),
        ("w/o cls：移除类别辅助监督，保留 PointNet", True, False),
    ]:
        f.panel(y, 395, title)
        f.node(55, y + 75, 215, 75, ["完整物体点云", "Xᴼ"])
        f.node(310, y + 75, 210, 75, ["PointNet", "物体几何特征"], "network")
        f.edge([(270, y + 112), (310, y + 112)])
        f.node(
            620,
            y + 65,
            260,
            80,
            [
                "分类 MLP" if cls else "移除分类分支",
                "类别监督 CE" if cls else "不计算类别 loss",
            ],
            "network" if cls else "removed",
        )
        f.edge([(415, y + 75), (415, y + 57), (750, y + 57), (750, y + 65)], not cls)
        f.node(55, y + 215, 465, 80, ["物体真值状态 + 本体状态 + 目标", "sᵒ、sʳ、g"])
        f.node(
            570,
            y + 215,
            200,
            80,
            ["SC 输入变换" if sc else "跳过 SC", "规范坐标系" if sc else "原坐标输入"],
            "data" if sc else "change",
        )
        f.edge([(520, y + 255), (570, y + 255)])
        f.node(
            825,
            y + 190,
            260,
            125,
            ["Teacher 策略", "Actor / Critic MLP", "PPO 训练"],
            "network",
        )
        f.edge([(770, y + 255), (825, y + 255)])
        f.edge(
            [
                (520, y + 112),
                (560, y + 112),
                (560, y + 170),
                (955, y + 170),
                (955, y + 190),
            ]
        )
        f.node(
            1130,
            y + 205,
            205,
            95,
            ["Actor 动作", "SC 恢复 → 执行" if sc else "原坐标 → 执行"],
            "data" if sc else "change",
            size=21,
        )
        f.edge([(1085, y + 255), (1130, y + 255)])
        f.txt(
            60,
            y + 357,
            "拼接后供控制器读取；分类头只提供辅助训练目标，不是 Critic。",
            21,
        )
        y += 420
    f.txt(40, 1480, "这是论文的模块与监督关系图，不是完整自动微分图。", 22, True)
    f.txt(
        40,
        1515,
        "论文的 PointNet 涉及 BC／分类训练；公开 state-based 代码读取预计算点云特征。",
        21,
    )
    f.save("unidexgrasp-teacher-ablations.svg")


def curriculum():
    f = Fig(
        1180,
        "UniDexGrasp：物体课程消融（Table 3）",
        "改变物体任务分布的顺序，保持策略模块结构。比较无课程、一阶段、二阶段和完整三阶段课程。",
        "a：单物体；b：单类别多个物体；c：三个代表类别；d：所有类别。箭头表示继续训练。",
    )
    y = 180
    rows = [
        ("w/o OCL：直接训练所有类别", ["d"], "从初始化直接面对全部物体分布。"),
        ("1-stage OCL", ["a", "d"], "先学一个物体，再扩展到全部类别。"),
        ("2-stage OCL", ["a", "b", "d"], "增加同类别内的物体变化。"),
        (
            "完整 3-stage OCL",
            ["a", "b", "c", "d"],
            "在全部类别前，增加代表类别之间的几何变化。",
        ),
    ]
    names = {
        "a": ["a：单物体"],
        "b": ["b：单类别", "多个物体"],
        "c": ["c：三个", "代表类别"],
        "d": ["d：所有类别"],
    }
    xs = [70, 395, 720, 1045]
    for title, seq, note in rows:
        f.panel(y, 205, title)
        for i, key in enumerate(seq):
            x = xs[i]
            f.node(
                x,
                y + 65,
                270,
                85,
                names[key],
                "change" if key == "d" and len(seq) == 1 else "network",
            )
            if i < len(seq) - 1:
                f.edge([(x + 270, y + 107), (xs[i + 1], y + 107)])
        f.txt(65, y + 183, note, 21, muted=True)
        y += 225
    f.txt(
        40,
        1125,
        "四组不靠删掉 Actor、Critic 或 Encoder 区分；改变的是训练任务分布与训练顺序。",
        22,
        True,
    )
    f.txt(
        40,
        1160,
        "这里的阶段不是“靠近 → 闭指 → 抬升”的动作阶段，也不是 UniDexGrasp++ 的专家整合流程。",
        20,
    )
    f.save("unidexgrasp-object-curriculum.svg")


def goals():
    f = Fig(
        1430,
        "UniDexGrasp：测试目标扰动与目标条件消融",
        "Table4固定视觉Student，只切换目标抓姿。Table8移除目标条件及目标奖励后重新训练，与Table4不同。",
        "Table 4：同一个训练好的 Student；只改变送入的目标抓姿，网络权重不变。",
    )
    f.panel(180, 770, "Table 4：目标来源与扰动测试")
    labels = [
        ["GT", "合成数据中的参考抓姿"],
        ["pred", "生成器预测的抓姿"],
        ["(a) 小穿透", "扰动预测抓姿"],
        ["(b) 大穿透", "扰动预测抓姿"],
        ["(c) 无接触", "扰动预测抓姿"],
    ]
    for i, lines in enumerate(labels):
        y = 250 + i * 120
        f.node(60, y, 370, 90, lines, "data" if i < 2 else "change")
        f.edge([(430, y + 45), (490, y + 45), (490, 545), (565, 545)])
    f.node(565, 495, 270, 100, ["目标输入 g", "每次选择一种设置"])
    f.node(565, 300, 270, 110, ["当前场景点云 X_t", "机器人本体状态 sʳ"])
    f.edge([(835, 355), (885, 355), (885, 520), (930, 520)])
    f.edge([(835, 545), (930, 545)])
    f.node(
        930,
        460,
        390,
        145,
        ["同一个视觉 Student", "PointNet + 动作 MLP", "权重固定"],
        "network",
    )
    f.node(930, 690, 390, 110, ["输出动作 → 仿真执行", "统计抓取成功率"], "env")
    f.edge([(1125, 605), (1125, 690)])
    f.txt(60, 916, "这些行不是网络消融；无需为每种扰动重新训练 Student。", 23, True)
    f.panel(980, 360, "Table 8：是否保留目标条件（需要重新训练）")
    f.node(60, 1050, 250, 90, ["Goal-conditioned", "有目标输入 g"], "network")
    f.node(380, 1050, 470, 90, ["策略与 RL 奖励", "包含目标抓姿项 r_goal"], "network")
    f.node(940, 1050, 380, 90, ["按目标抓取", "评价成功率与目标误差"], "env")
    f.edge([(310, 1095), (380, 1095)])
    f.edge([(850, 1095), (940, 1095)])
    f.node(60, 1190, 250, 90, ["Non-goal-conditioned", "移除目标输入 g"], "change")
    f.node(
        380, 1190, 470, 90, ["重新训练策略", "同时移除目标抓姿奖励 r_goal"], "change"
    )
    f.node(940, 1190, 380, 90, ["只要求抓起物体", "主要比较成功率"], "env")
    f.edge([(310, 1235), (380, 1235)])
    f.edge([(850, 1235), (940, 1235)])
    f.txt(
        40,
        1390,
        "符号 g 表示目标抓姿；r_goal 表示接近该目标的奖励项。未在图中展开具体观测编码。",
        21,
    )
    f.save("unidexgrasp-goal-conditioning.svg")


def gradient(f, points):
    """Draw an update arrow, distinct from data flow."""
    if not getattr(f, "has_gradient_marker", False):
        f.p.append(
            '<defs><marker id="grad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10z" fill="#ea580c"/></marker></defs>'
        )
        f.has_gradient_marker = True
    path = "M" + " L".join((f"{x} {y}" for x, y in points))
    f.p.append(
        f'<path d="{path}" fill="none" stroke="#ea580c" stroke-width="2.8" stroke-dasharray="8 7" stroke-linejoin="round" marker-end="url(#grad)"/>'
    )


def pipeline():
    f = Fig(
        1480,
        "UniDexGrasp：从目标抓姿生成到闭环执行（Fig. 2 解读）",
        "先由初始点云生成抓姿目标，再由策略依据每步观测执行。生成阶段和策略训练不共用一个端到端 PPO 更新。",
        "R：手根部旋转；t：位置；q：关节角；g：目标抓姿。下标 0 为初始时刻，下标 t 为控制时刻。",
    )
    f.panel(180, 640, "A. 生成一个目标：概率采样 → 接触修正 → 恢复坐标")
    row = [
        (60, 220, ["初始场景点云", "X₀"], "data"),
        (330, 230, ["GraspIPDF", "采样旋转 R"], "network"),
        (610, 230, ["按 R⁻¹ 旋转", "规范化点云 X̃₀"], "data"),
        (890, 430, ["GraspGlow", "联合采样位置 t̃、关节 q"], "network"),
    ]
    for i, (x, w, lines, kind) in enumerate(row):
        f.node(x, 270, w, 110, lines, kind)
        if i + 1 < len(row):
            f.edge([(x + w, 325), (row[i + 1][0], 325)])
    f.node(
        1040, 500, 280, 145, ["初始候选抓姿", "手的规范姿态", "与采样旋转 R"], "data"
    )
    f.edge([(1180, 380), (1180, 500)])
    f.node(
        610,
        500,
        380,
        145,
        ["ContactNet + TTA", "预测目标接触分布", "微调当前抓姿参数"],
        "network",
    )
    f.edge([(1040, 572), (990, 572)])
    f.edge([(725, 380), (725, 500)])
    f.txt(745, 447, "场景几何", 20)
    f.node(330, 500, 230, 145, ["按 R 恢复坐标", "得到世界系抓姿"], "data")
    f.edge([(610, 572), (560, 572)])
    f.node(60, 500, 220, 145, ["目标 g", "(R, t, q)"], "data")
    f.edge([(330, 572), (280, 572)])
    f.txt(
        60,
        704,
        "TTA：固定网络权重，优化当前抓姿；ContactNet 同时读取场景与候选手的几何。",
        22,
    )
    f.txt(
        60,
        746,
        "生成网络先训练好；后续控制策略的 PPO 训练不会回传更新这条生成链。",
        22,
        True,
    )
    f.txt(
        60,
        785,
        "图中省略手部正向运动学与点云采样；t̃ 表示规范坐标系中的位置。",
        21,
        muted=True,
    )
    f.panel(850, 540, "B. 执行一个目标：g 在本次抓取中保持为条件，观测每步更新")
    f.node(60, 940, 310, 100, ["当前场景点云 Xₜ", "每步视觉观测"], "data")
    f.node(440, 940, 310, 100, ["本体状态 sʳₜ", "关节与机器人状态"], "data")
    f.node(820, 940, 310, 100, ["目标 g", "来自上面的生成器"], "data")
    f.node(
        450,
        1130,
        340,
        135,
        ["视觉 Student 策略", "SC + PointNet + MLP", "读取观测与目标"],
        "network",
    )
    f.edge([(215, 1040), (215, 1080), (530, 1080), (530, 1130)])
    f.edge([(595, 1040), (595, 1130)])
    f.edge([(975, 1040), (975, 1080), (710, 1080), (710, 1130)])
    f.node(850, 1150, 180, 95, ["动作 aₜ", "坐标恢复"], "data")
    f.node(1090, 1150, 240, 95, ["仿真环境", "接触与运动"], "env")
    f.edge([(790, 1197), (850, 1197)])
    f.edge([(1030, 1197), (1090, 1197)])
    f.edge([(1210, 1245), (1210, 1320), (215, 1320), (215, 1040)])
    f.txt(520, 1356, "新观测反馈：下一步重新决定动作", 22)
    f.txt(
        40,
        1432,
        "目标条件策略输出控制动作；生成器输出目标姿态，两者的输出语义不同。",
        23,
        True,
    )
    f.save("unidexgrasp-pipeline.svg")


def training_distillation():
    f = Fig(
        1800,
        "UniDexGrasp：Teacher 训练与 Student 蒸馏（Fig. 3 解读）",
        "Teacher 通过 PPO 从环境回报学习；Student 在自己访问的状态上向冻结 Teacher 请求动作标签。橙色虚线表示参数更新。",
        "实线：数据／监督目标流；橙色虚线：参数更新。阶段 B 冻结 Teacher，只训练 Student。",
    )
    f.panel(180, 690, "A. 训练 Teacher：控制学习与几何表示辅助学习各有监督")
    f.node(60, 280, 300, 100, ["物体真值 + 本体 + 目标", "sᵒₜ、sʳₜ、g"], "data")
    f.node(410, 280, 230, 100, ["状态规范化 SC", "变换到参考坐标系"], "data")
    f.node(700, 280, 170, 100, ["拼接输入", "状态 + 几何"], "data")
    f.edge([(360, 330), (410, 330)])
    f.edge([(640, 330), (700, 330)])
    f.node(950, 250, 190, 90, ["Actor MLP", "动作分布 πᴱ"], "network")
    f.node(950, 380, 190, 90, ["Critic MLP", "价值预测 V"], "network")
    f.edge([(870, 315), (910, 315), (910, 295), (950, 295)])
    f.edge([(870, 345), (910, 345), (910, 425), (950, 425)])
    f.node(1190, 250, 140, 90, ["动作 aᴱ", "SC 恢复"], "data", size=20)
    f.edge([(1140, 295), (1190, 295)])
    f.node(60, 450, 280, 90, ["完整物体点云 Xᴼ", "预采样物体几何"], "data")
    f.node(390, 450, 250, 90, ["PointNet", "几何特征"], "network")
    f.edge([(340, 495), (390, 495)])
    f.edge([(640, 495), (780, 495), (780, 380)])
    f.node(390, 630, 320, 100, ["表示学习辅助目标", "BC + 类别分类 CE"], "change")
    f.edge([(470, 540), (470, 630)])
    gradient(f, [(610, 630), (610, 540)])
    f.node(1130, 570, 200, 100, ["仿真环境", "交互轨迹与奖励"], "env")
    f.edge([(1260, 340), (1260, 570)])
    f.node(
        800,
        630,
        280,
        125,
        ["PPO 更新", "回报目标 / 优势", "Actor 与 Critic loss"],
        "change",
    )
    f.edge([(1130, 620), (1100, 620), (1100, 690), (1080, 690)])
    f.edge([(1000, 470), (1000, 560), (940, 560), (940, 630)])
    gradient(f, [(890, 630), (890, 230), (1045, 230), (1045, 250)])
    gradient(f, [(1030, 630), (1030, 520), (1080, 520), (1080, 470)])
    f.txt(
        60,
        800,
        "PPO 训练控制网络；BC／分类辅助训练 PointNet，不把分类头当作 Critic。",
        22,
        True,
    )
    f.txt(
        60,
        840,
        "这是模块关系图：发布的 state-based Teacher 代码读取预计算点云特征。",
        21,
    )
    f.panel(900, 840, "B. 动作蒸馏：在 Student 访问的状态上获得 Teacher 标签")
    f.node(60, 1000, 340, 90, ["Student 观测与目标", "Xₜ、sʳₜ、g"], "data")
    f.node(450, 1000, 270, 90, ["SC + 点云编码", "输入表示"], "network")
    f.node(800, 1000, 270, 90, ["Student 策略 πˢ", "可训练参数 θˢ"], "network")
    f.node(1140, 1000, 190, 90, ["预测动作 aˢ", "SC 恢复后执行"], "data", size=20)
    f.edge([(400, 1045), (450, 1045)])
    f.edge([(720, 1045), (800, 1045)])
    f.edge([(1070, 1045), (1140, 1045)])
    f.node(1120, 1180, 220, 90, ["仿真环境", "Student rollout"], "env")
    f.edge([(1235, 1090), (1235, 1180)])
    f.edge([(1340, 1225), (1360, 1225), (1360, 975), (230, 975), (230, 1000)])
    f.node(60, 1180, 340, 90, ["同一个访问状态", "对应特权输入与目标"], "data")
    f.node(450, 1180, 320, 90, ["冻结 Teacher πᴱ", "提供动作标签"], "data")
    f.node(820, 1180, 190, 90, ["标签动作 aᴱ", "无梯度"], "data")
    f.edge([(1120, 1235), (1090, 1235), (1090, 1310), (230, 1310), (230, 1270)])
    f.edge([(400, 1225), (450, 1225)])
    f.edge([(770, 1225), (820, 1225)])
    f.node(
        60,
        1430,
        360,
        120,
        ["收集／聚合标注样本", "(Student 输入, aᴱ)", "反复交互并重新拟合"],
        "data",
    )
    f.edge([(915, 1270), (915, 1380), (240, 1380), (240, 1430)])
    f.node(
        550,
        1430,
        430,
        120,
        ["动作模仿损失", "L = ‖aˢ − aᴱ‖²", "更新 Student 编码器与策略"],
        "change",
    )
    f.edge([(420, 1490), (550, 1490)])
    f.edge([(1330, 1045), (1350, 1045), (1350, 1490), (980, 1490)])
    gradient(f, [(980, 1520), (1040, 1520), (1040, 1090)])
    gradient(f, [(550, 1520), (430, 1520), (430, 1115), (585, 1115), (585, 1090)])
    f.txt(
        60,
        1624,
        "DAgger 的关键是访问状态与标签采集方式；动作 MSE 本身并不等于 DAgger。",
        22,
        True,
    )
    f.txt(
        60,
        1662,
        "E / S 分别表示 Teacher / Student；图中 θˢ 包含 Student 的感知编码与动作网络参数。",
        21,
    )
    f.txt(
        60,
        1700,
        "g 是共享任务条件；目标特征的构造是否读取真值位姿，仍需检查代码的数据来源。",
        21,
    )
    f.txt(
        40,
        1775,
        "SC 为固定坐标变换；橙色更新仅作用于神经网络参数。图中省略批次与回合终止细节。",
        21,
    )
    f.save("unidexgrasp-training-distillation.svg")


if __name__ == "__main__":
    generation()
    teacher()
    curriculum()
    goals()
    pipeline()
    training_distillation()
