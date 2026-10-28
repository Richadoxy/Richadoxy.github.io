#!/usr/bin/env python3
"""Build the self-contained SVG diagrams used by the RL learning path.

Run from any directory: python3 scripts/build_rl_figures.py
Only the four named SVG files below are generated. No external assets are used.
"""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "blog" / "assets" / "figures"
COLORS = {
    "network": ("#eff6ff", "#2563eb"),
    "fixed": ("#f1f5f9", "#64748b"),
    "environment": ("#ecfdf5", "#059669"),
    "loss": ("#fff7ed", "#ea580c"),
    "removed": ("#f8fafc", "#94a3b8"),
}


class Figure:
    def __init__(self, height: int, title: str, description: str):
        self.height = height
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" '
            f'viewBox="0 0 1280 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            '<defs>',
        ]
        for name, color in (("data", "#475569"), ("fixed", "#64748b"), ("grad", "#ea580c")):
            self.parts.append(
                f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
            )
        self.parts.extend([
            '</defs>',
            '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif;'
            'fill:#172033} .title{font-weight:750} .muted{fill:#475569}</style>',
            f'<rect width="1280" height="{height}" fill="#ffffff"/>',
        ])
        self.text(40, 57, title, 32, weight=True)

    def text(self, x: float, y: float, value: str, size: int = 22,
             anchor: str = "start", weight: bool = False, muted: bool = False):
        cls = "title" if weight else "muted" if muted else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'class="{cls}">{escape(value)}</text>'
        )

    def subtitle(self, value: str):
        self.text(40, 99, value, muted=True)

    def legend(self, fixed_label: str = "固定数据／冻结模块", network_label: str = "网络／可导计算"):
        for x, label, kind in ((40, network_label, "network"), (260, fixed_label, "fixed"),
                               (585, "物理环境", "environment"), (795, "损失／改动", "loss")):
            fill, stroke = COLORS[kind]
            self.parts.append(f'<rect x="{x}" y="126" width="25" height="25" rx="5" '
                              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            self.text(x + 36, 147, label)
        self.edge([(1020, 135), (1070, 135)], "grad")
        self.text(1083, 143, "梯度更新", 22)

    def panel(self, y: int, height: int, label: str):
        self.parts.append(f'<rect x="24" y="{y}" width="1232" height="{height}" rx="18" '
                          'fill="#fbfcfe" stroke="#e2e8f0" stroke-width="2"/>')
        self.text(75, y + 42, label, 25, weight=True)

    def node(self, x: int, y: int, width: int, height: int, lines: list[str],
             kind: str = "fixed", dashed: bool = False, size: int = 24):
        fill, stroke = COLORS[kind]
        dash = ' stroke-dasharray="8 6"' if dashed or kind == "removed" else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="13" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
        line_height = size + 9
        top = y + height / 2 - (len(lines) - 1) * line_height / 2 + size * .35
        for i, line in enumerate(lines):
            self.text(x + width / 2, top + i * line_height, line, size,
                      anchor="middle", weight=i == 0)

    def edge(self, points: list[tuple[float, float]], kind: str = "data", arrow: bool = True):
        path = "M " + " L ".join(f"{x} {y}" for x, y in points)
        color = "#ea580c" if kind == "grad" else "#64748b" if kind == "fixed" else "#475569"
        dash = ' stroke-dasharray="8 7"' if kind == "grad" else ""
        marker = f' marker-end="url(#{kind})"' if arrow else ""
        self.parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" '
                          f'stroke-linejoin="round" stroke-linecap="round"{dash}{marker}/>')

    def save(self, name: str):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


def architecture():
    f = Figure(1010, "Actor–Critic：一次前向计算，两个不同职责",
               "通用共享骨干示意。原始输入经过编码器和共享MLP后，Actor预测动作均值，结合可学习的log标准差形成高斯分布；Critic预测一个状态价值。PPO是外部训练算法，不是前向层。")
    f.subtitle("这张图只画前向计算；网络结构可以替换，Actor／Critic 的职责保持明确。")
    f.legend(fixed_label="数值／数据", network_label="可学习模块")
    f.node(60, 215, 245, 115, ["原始输入 x_t", "观测、历史、任务"])
    f.node(380, 215, 280, 115, ["Encoder", "生成特征 z_t"], "network")
    f.node(735, 215, 470, 115, ["共享网络", "示例：MLP，输出特征 f_t"], "network")
    f.edge([(305, 272), (380, 272)])
    f.edge([(660, 272), (735, 272)])
    f.edge([(970, 330), (970, 370), (290, 370), (290, 425)])
    f.edge([(970, 370), (695, 370), (695, 425)])
    f.node(80, 425, 420, 135, ["Critic 输出头", "f_t → V_ψ(x_t)", "状态价值：一个标量"], "network")
    f.node(540, 425, 310, 135, ["Actor 输出头", "f_t → 均值 m_t", "m_t ∈ ℝᵈ"], "network")
    f.node(900, 425, 305, 135, ["可学习参数 log σ", "σ = exp(log σ)", "标准差 σ ∈ ℝᵈ"], "network")
    f.node(80, 640, 420, 120, ["预测未来累计回报", "例如：V_ψ(x_t) = 5", "由训练学出，不是奖励公式"])
    f.edge([(290, 560), (290, 640)])
    f.node(700, 640, 505, 120, ["动作分布 π_θ(· | x_t)", "Normal(m_t, diag(σ²))", "连续动作使用概率密度"])
    f.edge([(695, 560), (695, 605), (850, 605), (850, 640)])
    f.edge([(1052, 560), (1052, 640)])
    f.node(700, 825, 505, 95, ["训练时采样 a_t ∈ ℝᵈ", "保存该动作的 log π_old(a_t | x_t)"])
    f.edge([(952, 760), (952, 825)])
    f.node(80, 825, 420, 95, ["Critic 帮助训练", "普通策略部署通常只保留 Actor"])
    f.text(40, 970, "d：动作维度；θ、ψ：Actor 与 Critic 参数，可包含共享部分。PPO 不在推理路径中。", 22)
    f.save("rl-actor-critic-architecture.svg")


def ppo_loop():
    f = Figure(2090, "PPO：采样 → 构造固定目标 → 重新前向与更新",
               "PPO三个阶段。先固定参数采集轨迹和旧策略概率及旧价值，再用奖励与旧价值计算GAE和固定return目标。更新时重新编码旧观测，计算旧动作的新概率和新的价值预测。策略损失及价值损失反向更新模型。固定数据不接收梯度。")
    f.subtitle("实线传递数据；橙色虚线更新参数。old 指采样快照，new 指正在更新的网络。")
    f.legend()
    f.panel(190, 365, "① A · 采样：参数暂时不变，不构建梯度图")
    f.node(60, 270, 350, 120, ["行为策略与价值网络", "π_old、V_old", "输入原始 x_t"], "network")
    f.node(475, 275, 280, 110, ["采样动作 a_t", "计算 old_logp"])
    f.node(860, 270, 350, 120, ["物理／仿真环境", "执行动作，产生后果", "不通过环境反向传播"], "environment")
    f.edge([(410, 330), (475, 330)])
    f.edge([(755, 330), (860, 330)])
    f.node(60, 440, 1150, 80, ["轨迹缓存：原始 x、动作 a、奖励 r、old_logp、old_V、next_V、终止／截断标记"] , size=22)
    f.edge([(235, 390), (235, 440)])
    f.edge([(615, 385), (615, 440)])
    f.edge([(1035, 390), (1035, 440)])

    f.panel(585, 400, "② B · 计算训练信号：全部 stop-gradient")
    f.node(100, 660, 1080, 95, ["GAE：δ_t = r_t + γ b_t V_old,next − V_old,t", "Â_t = δ_t + γ λ c_t Â_(t+1)"])
    f.edge([(635, 520), (635, 660)])
    f.node(100, 815, 440, 95, ["优势估计 Â_t", "Actor 的固定样本权重"])
    f.node(740, 815, 440, 95, ["回报目标 R̂_t = Â_t + V_old,t", "Critic 的固定回归标签"], size=23)
    f.edge([(420, 755), (420, 785), (320, 785), (320, 815)])
    f.edge([(860, 755), (860, 785), (960, 785), (960, 815)])
    f.text(100, 955, "b_t：是否允许 bootstrap；c_t：是否属于同一段连续轨迹。两者在外部截断时不同。", 22)

    f.panel(1020, 950, "③ C · 更新：同一批旧经历，多轮重新前向计算")
    f.node(300, 1110, 680, 115, ["当前网络读取缓存里的原始 x", "重新经过 Encoder、共享骨干、Actor 与 Critic", "此处建立计算图"], "network", size=23)
    f.node(100, 1290, 440, 100, ["new_logp = log π_θ(a_t | x_t)", "评价旧动作，不重新采样"], "network", size=23)
    f.node(740, 1290, 440, 100, ["V_new = V_ψ(x_t)", "当前网络对旧输入的新预测"], "network", size=23)
    f.edge([(500, 1225), (500, 1260), (320, 1260), (320, 1290)])
    f.edge([(780, 1225), (780, 1260), (960, 1260), (960, 1290)])
    f.node(100, 1475, 440, 170, ["Actor loss：PPO-Clip", "ρ = exp(new_logp − old_logp)", "−mean min(ρÂ, clip(ρ)Â)", "Â、old_logp 固定，new 可导"], "loss", size=22)
    f.node(740, 1475, 440, 170, ["Critic loss：回归", "mean (V_new − R̂)²", "拟合这批经历给出的目标", "R̂ 固定，只有 V_new 可导"], "loss", size=22)
    f.edge([(320, 1390), (320, 1475)])
    f.edge([(960, 1390), (960, 1475)])
    f.edge([(100, 860), (45, 860), (45, 1555), (100, 1555)], "fixed")
    f.edge([(1180, 860), (1235, 860), (1235, 1555), (1180, 1555)], "fixed")
    f.node(350, 1715, 580, 95, ["总损失", "L = L_actor + c_V L_value − c_H entropy"], "loss", size=22)
    f.edge([(320, 1645), (320, 1762), (350, 1762)])
    f.edge([(960, 1645), (960, 1762), (930, 1762)])
    f.node(300, 1870, 680, 75, ["梯度更新：Encoder、共享骨干、Actor、Critic"], "network", size=23)
    f.edge([(640, 1810), (640, 1870)], "grad")
    f.text(40, 2020, "下一轮重新采样：当前网络成为新的行为策略；固定目标不是跨所有迭代永远不变。", 22)
    f.text(40, 2057, "γ：奖励折扣；λ：GAE 系数；ρ：新旧密度比；c_V、c_H：损失权重；符号和边界处理见正文。", 22)
    f.save("rl-ppo-training-loop.svg")


def hora_stages():
    f = Figure(1780, "HORA：特权训练、历史适应与真实部署",
               "HORA第一阶段用PPO训练物体信息编码器和共享ActorCritic；第二阶段冻结编码器和策略，仅以latent均方误差训练历史适应模块，同时用预测latent运行冻结策略收集历史；部署只保留历史适应和动作策略，输出经裁剪和累积转换为PD位置目标。")
    f.subtitle("同一条控制策略 π，训练与部署时从不同路径获得环境表征。")
    f.legend()
    f.panel(190, 430, "阶段 1 · Base policy：特权信息支持的 PPO 训练")
    f.node(50, 270, 220, 100, ["特权信息 e_t", "位置与属性：9 维"] , size=22)
    f.node(330, 270, 260, 100, ["物体编码器 μ", "9 → 256 → 128 → 8"], "network", size=22)
    f.node(650, 270, 170, 100, ["z_t", "8 维，可导"], "network", size=22)
    f.node(650, 410, 170, 90, ["短历史 o_t", "96 维"], size=22)
    f.node(900, 270, 310, 230, ["控制网络", "共享 MLP", "输入：96 + 8", "Actor π：动作分布", "Critic V：状态价值"], "network", size=23)
    f.edge([(270, 320), (330, 320)])
    f.edge([(590, 320), (650, 320)])
    f.edge([(820, 320), (900, 320)])
    f.edge([(820, 455), (850, 455), (850, 385), (900, 385)])
    f.node(50, 430, 540, 110, ["仿真交互与 PPO 损失", "更新 μ、共享骨干、Actor 与 Critic"], "loss", size=23)
    f.edge([(460, 430), (460, 370)], "grad")
    f.edge([(590, 505), (835, 505), (835, 485), (900, 485)], "grad")
    f.edge([(1055, 500), (1055, 575), (320, 575), (320, 540)])
    f.text(650, 602, "动作 → 仿真；奖励与轨迹 → PPO", 22, anchor="middle", muted=True)

    f.panel(650, 510, "阶段 2 · Adaptation：固定控制器，只训练历史估计器")
    f.node(50, 740, 180, 100, ["特权信息 e_t", "仅训练可得"], size=22)
    f.node(290, 740, 220, 100, ["冻结 μ", "不更新权重"])
    f.node(590, 740, 160, 100, ["标签 z_t", "stop-grad"], size=22)
    f.node(850, 740, 350, 100, ["latent MSE", "mean ‖ẑ_t − z_t‖²"], "loss")
    f.node(50, 920, 180, 100, ["历史 H_t", "30 × 32"], size=22)
    f.node(290, 920, 220, 100, ["适应模块 φ", "MLP + 时间卷积"], "network", size=22)
    f.node(590, 920, 160, 100, ["预测 ẑ_t", "8 维，可导"], "network", size=22)
    f.node(850, 920, 350, 100, ["冻结 π(o_t, ẑ_t)", "在仿真中继续采集历史"], size=23)
    f.edge([(230, 790), (290, 790)])
    f.edge([(510, 790), (590, 790)])
    f.edge([(750, 790), (850, 790)])
    f.edge([(230, 970), (290, 970)])
    f.edge([(510, 970), (590, 970)])
    f.edge([(750, 970), (850, 970)])
    f.edge([(750, 950), (790, 950), (790, 820), (850, 820)])
    f.edge([(1200, 790), (1230, 790), (1230, 1070), (400, 1070), (400, 1020)], "grad")
    f.text(75, 1125, "MSE 仅更新 φ；策略执行产生新历史，但这里不对动作或环境做反向传播。", 22)

    f.panel(1190, 500, "部署 · 固定全部权重，持续更新历史与 latent")
    f.node(50, 1270, 210, 90, ["真实历史 H_t", "关节＋历史目标"], size=22)
    f.node(310, 1270, 240, 90, ["固定权重的 φ", "每步重新前向"], size=23)
    f.node(610, 1270, 170, 90, ["当前 ẑ_t", "随历史变化"], size=22)
    f.node(850, 1270, 350, 145, ["固定权重的 Actor π", "输入：o_t 与 ẑ_t", "取动作均值"], size=23)
    f.node(610, 1380, 170, 80, ["短历史 o_t", "仍每步更新"], size=22)
    f.edge([(260, 1315), (310, 1315)])
    f.edge([(550, 1315), (610, 1315)])
    f.edge([(780, 1315), (850, 1315)])
    f.edge([(780, 1420), (815, 1420), (815, 1375), (850, 1375)])
    f.node(850, 1480, 350, 105, ["原始动作 a_t", "clamp 到 [−1, 1]"], size=23)
    f.node(420, 1480, 360, 105, ["累积位置目标 u_t", "u_prev + a_clip / 24", "再限制关节范围"], size=22)
    f.node(50, 1480, 300, 105, ["PD 控制器与真实手", "位置目标 → 关节力矩"], "environment", size=22)
    f.edge([(1025, 1415), (1025, 1480)])
    f.edge([(850, 1532), (780, 1532)])
    f.edge([(420, 1532), (350, 1532)])
    f.text(75, 1640, "新的关节读数与已下发目标进入下一步历史；部署不需要 μ、Critic 或 PPO 更新。", 22)
    f.text(40, 1738, "μ 是论文的物体编码器；π 是 Actor。原始动作 a_t 与累积 PD 位置目标 u_t 必须分开。", 22)
    f.save("hora-training-deployment.svg")


def hora_baselines():
    f = Figure(1870, "HORA 对照实验：改变的是哪一条信息路径",
               "六行对照Ours、DR、SysID、NoAdapt、Periodic、Expert。DR重新训练无latent分支策略；SysID替换估计目标为实际参数但其后端连接未完整公开；NoAdapt固定初始latent但保留观测反馈；Periodic回放动作；Expert使用仿真真实信息经编码器得到latent。")
    f.subtitle("以下比较测试时的功能信息流；DR 是重新训练的对照，不是把成品 HORA 的分支拔掉。")
    f.legend(fixed_label="输入／中间量", network_label="网络模块")
    methods = [
        ("Ours", "完整 HORA", ["长历史 H_t", "30 步"], ["适应模块 φ", "每步前向"], ["ẑ_t", "8 维"], ["控制策略 π", "接收当前 o_t"], "保留历史推断与持续 latent 更新。"),
        ("DR", "删除表征分支", ["移除", "长历史分支"], ["移除 φ", "无 latent 估计"], ["移除", "ẑ_t 输入"], ["策略 π_DR", "仅用 o_t，重新训练"], "保留短历史反馈；训练中仍有域随机化。"),
        ("SysID", "替换估计目标", ["运动历史", "H_t"], ["系统辨识网络", "预测实际信息"], ["ê_t", "9 维"], ["参数条件控制器", "内部接法未完整公开"], "功能示意：不据此推断是否删除 μ，或是否直接拼接 9 维。"),
        ("NoAdapt", "停止更新 latent", ["初始历史", "H_0"], ["适应模块 φ", "仅初始调用"], ["缓存 ẑ_0", "全程不变"], ["控制策略 π", "o_t 仍持续更新"], "固定的是环境表征；没有固定动作，也没有取消策略反馈。"),
        ("Periodic", "改为动作回放", [], [], [], ["按时间回放", "实时 π 不参与"], "高层不根据当前观测重新选动作；底层 PD 仍可有关节反馈。"),
        ("Expert", "替换信息来源", ["仿真真值 e_t", "9 维"], ["物体编码器 μ", "真实信息编码"], ["z_t", "8 维"], ["控制策略 π", "同一基础策略"], "以特权信息分支替代 φ 的推断，作为仿真性能参考。"),
    ]
    for idx, (name, change, inp, enc, latent, policy, note) in enumerate(methods):
        y = 190 + idx * 265
        f.parts.append(f'<rect x="24" y="{y}" width="1232" height="245" rx="16" '
                       'fill="#fbfcfe" stroke="#e2e8f0" stroke-width="2"/>')
        f.text(45, y + 47, name, 27, weight=True)
        f.text(45, y + 83, change, 22, muted=True)
        top = y + 30
        if name == "Periodic":
            f.node(230, top, 605, 85, ["预先记录的 Expert 动作序列", "测试时直接按顺序取出"], size=23)
            f.edge([(835, top + 42), (875, top + 42)])
        else:
            removed = name == "DR"
            kind = "removed" if removed else "fixed"
            f.node(230, top, 165, 85, inp, kind, size=22)
            f.node(435, top, 220, 85, enc, "removed" if removed else "network", size=22)
            f.node(695, top, 140, 85, latent, "removed" if removed else "loss" if name in ("SysID", "NoAdapt") else "fixed", size=22)
            if not removed:
                f.edge([(395, top + 42), (435, top + 42)])
                f.edge([(655, top + 42), (695, top + 42)])
                f.edge([(835, top + 42), (875, top + 42)])
        p_kind = "fixed" if name == "SysID" else "loss" if name == "Periodic" else "network"
        f.node(875, top, 260, 115, policy, p_kind, dashed=name == "SysID", size=22)
        f.node(1175, top + 10, 65, 65, ["a_t"], size=22)
        f.edge([(1135, top + 42), (1175, top + 42)])
        if name != "Periodic":
            f.node(695, top + 105, 140, 60, ["o_t：96 维"], size=22)
            f.edge([(835, top + 135), (855, top + 135), (855, top + 95), (875, top + 95)])
        else:
            f.text(695, top + 158, "当前 o_t 不参与高层动作选择", 22, anchor="middle", muted=True)
        f.text(230, y + 228, note, 22)
    f.text(40, 1835, "DR、SysID 是方案对照；NoAdapt 是机制消融；Periodic 是回放对照；Expert 是特权参考。", 22)
    f.save("hora-baseline-comparison.svg")


def main():
    architecture()
    ppo_loop()
    hora_stages()
    hora_baselines()
    print(f"Built 4 RL figures in {OUT}")


if __name__ == "__main__":
    main()
