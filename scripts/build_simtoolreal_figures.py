#!/usr/bin/env python3
"""Build original SVGs for SimToolReal v2 and its visual-perception primer."""

from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET


OUT = Path(__file__).resolve().parents[1] / "blog" / "assets" / "figures"
COLORS = {
    "data": ("#f1f5f9", "#64748b"),
    "network": ("#eff6ff", "#2563eb"),
    "env": ("#ecfdf5", "#059669"),
    "signal": ("#fff7ed", "#ea580c"),
}


class Figure:
    def __init__(self, height, title, subtitle, description):
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" viewBox="0 0 1400 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10z" fill="#475569"/></marker></defs>',
            '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif;fill:#172033}.bold{font-weight:700}</style>',
            f'<rect width="1400" height="{height}" fill="white"/>',
        ]
        self.text(40, 55, title, 32, bold=True)
        self.text(40, 96, subtitle, 21)

    def text(self, x, y, value, size=22, bold=False, anchor="start"):
        cls = ' class="bold"' if bold else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{cls}>{escape(value)}</text>'
        )

    def panel(self, y, height, title):
        self.parts.append(
            f'<rect x="24" y="{y}" width="1352" height="{height}" rx="16" fill="#fbfcfe" stroke="#e2e8f0" stroke-width="2"/>'
        )
        self.text(50, y + 38, title, 25, bold=True)

    def box(self, x, y, width, height, lines, kind="data", size=22):
        fill, stroke = COLORS[kind]
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        top = y + height / 2 - (len(lines) - 1) * (size + 10) / 2 + size * 0.35
        for i, line in enumerate(lines):
            self.text(x + width / 2, top + i * (size + 10), line, size, i == 0, "middle")

    def arrow(self, points, dashed=False):
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        path = "M" + " L".join(f"{x} {y}" for x, y in points)
        self.parts.append(
            f'<path d="{path}" fill="none" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"{dash}/>'
        )

    def line(self, x1, y1, x2, y2, color="#cbd5e1"):
        self.parts.append(f'<path d="M{x1} {y1}L{x2} {y2}" stroke="{color}" stroke-width="2" fill="none"/>')

    def save(self, suffix, prefix="simtoolreal"):
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / f"{prefix}-{suffix}.svg"
        target.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")
        ET.parse(target)
        print(target)


def pipeline():
    f = Figure(1170, "SimToolReal：训练、视频处理与真机控制", "按 v2 Fig. 2／3／12 重组；三行分别说明策略学习、任务准备和在线执行。", "仿真通过随机工具与随机目标训练策略；人类视频在训练后提供工具模型和目标；真实执行使用位姿反馈闭环控制。")
    f.panel(130, 260, "① 仿真 RL：学习一个跨工具的目标控制器")
    f.box(55, 210, 290, 115, ["程序生成工具", "柄＋头；几何与密度变化"], "env")
    f.box(395, 210, 260, 115, ["随机目标与交互", "状态 → 动作 → reward"], "env")
    f.box(710, 210, 280, 115, ["SAPG 训练", "LSTM Actor＋特权 Critic"], "network", 21)
    f.box(1045, 210, 300, 115, ["通用 Actor 权重", "部署时固定"], "network")
    for x1, x2 in [(345, 395), (655, 710), (990, 1045)]:
        f.arrow([(x1, 267), (x2, 267)])
    f.text(55, 365, "这一步使用仿真数值状态；测试工具视频不提供 Actor 的动作训练标签。", 21)

    f.panel(415, 280, "② 新任务准备：人类 RGB-D 视频提供物体轨迹")
    f.box(55, 500, 270, 120, ["RGB-D＋用户点提示", "SAM 2／SAM 3D", "米制几何与部件分割"], size=21)
    f.box(380, 500, 300, 120, ["工具 mesh＋抓握区域", "原点在柄；+x 指向头", "与训练坐标约定一致"], size=21)
    f.box(735, 500, 260, 120, ["FoundationPose", "提取物体位姿轨迹"], "network")
    f.box(1050, 500, 295, 120, ["目标序列 g¹…gᴷ", "30 → 3 Hz", "裁掉桌面静止前缀"], size=21)
    for x1, x2 in [(325, 380), (680, 735), (995, 1050)]:
        f.arrow([(x1, 560), (x2, 560)])
    f.text(55, 665, "预处理得到任务条件；mesh 服务于跟踪，粗抓握几何服务于策略观测。", 21)

    f.panel(720, 415, "③ 在线执行：追踪当前目标，达到后再推进")
    f.box(55, 810, 290, 120, ["本体感知＋视觉跟踪", "工具 pose：30 Hz", "抓握框与当前目标 gᵏ"], size=21)
    f.box(405, 810, 270, 120, ["固定 LSTM Actor", "60 Hz 前向控制", "递归状态持续更新"], "network", 21)
    f.box(735, 810, 260, 120, ["29 维关节位置目标", "7 维手臂增量", "22 维手绝对目标"], "signal", 21)
    f.box(1055, 810, 290, 120, ["真实机器人＋工具", "运动产生新的观测"], "env", 21)
    for x1, x2 in [(345, 405), (675, 735), (995, 1055)]:
        f.arrow([(x1, 870), (x2, 870)])
    f.arrow([(1200, 930), (1200, 985), (200, 985), (200, 930)])
    f.text(700, 978, "闭环反馈", 21, anchor="middle")
    f.box(405, 1020, 590, 75, ["当前目标达到阈值 → k 加一；不按视频时钟强制跳转"], "signal", 20)
    f.save("pipeline")


def actor_critic():
    f = Figure(905, "Actor 选择动作，Critic 学习预测回报", "两条前向路径使用不同的信息；底部说明训练算法如何将它们联系起来。", "Actor将140维部署观测输入LSTM和MLP产生29维动作。Critic读取更完整的仿真状态预测累计回报，训练算法结合奖励与预测计算优势并更新Actor。没有额外的Student蒸馏。")
    f.panel(135, 265, "① Actor：训练后直接部署的控制网络")
    f.box(55, 220, 280, 110, ["140 维物理观测", "机器人自身状态", "工具、目标与抓握尺寸"], size=21)
    f.box(390, 220, 255, 110, ["LSTM", "当前输入＋上一步记忆", "更新内部记忆"], "network", 21)
    f.box(700, 220, 305, 110, ["MLP", "将特征映射为", "29 维动作均值"], "network", 22)
    f.box(1060, 220, 285, 110, ["选择动作并转换", "7 维手臂＋22 维手", "发送关节目标"], "signal", 22)
    for x1, x2 in [(335, 390), (645, 700), (1005, 1060)]:
        f.arrow([(x1, 275), (x2, 275)])
    f.text(55, 373, "训练时从动作分布采样进行探索；真机前向中继续更新 LSTM 记忆。", 22)

    f.panel(425, 270, "② Critic：利用额外仿真信息，辅助训练")
    f.box(55, 510, 390, 110, ["较完整的仿真状态", "干净的工具状态与速度", "历史进展、是否已举起等"], size=22)
    f.box(510, 510, 350, 110, ["独立 Critic MLP", "预测未来累计回报", "输出一个价值标量"], "network", 22)
    f.box(925, 510, 420, 110, ["与回报目标比较", "目标由交互奖励等构造", "用误差更新 Critic"], "signal", 22)
    f.arrow([(445, 565), (510, 565)])
    f.arrow([(860, 565), (925, 565)])
    f.text(55, 668, "Critic 只帮助估值；它不提供“正确的手指动作”，也不是这里的 Teacher。", 22)

    f.text(55, 752, "训练算法：环境奖励＋价值预测 → 估计 advantage → 更新 Actor 的动作分布", 24, bold=True)
    f.text(55, 808, "Advantage 由训练算法计算，不是 Actor 的一个输出。", 23)
    f.text(55, 858, "部署：保留 Actor，权重固定、记忆更新；没有额外训练 Student 的阶段。", 23)
    f.save("actor-critic")


def keypoints():
    f = Figure(1080, "两套关键点：几何条件与成功度量分开", "两路都从局部点经位姿变换得到世界点；尺度的来源不同。对应 v2 Fig. 9、附录 A／C。", "观测关键点使用工具抓握区域尺寸，奖励关键点使用固定参考尺寸；最大对应点距离衡量平移和旋转的联合误差。")
    f.panel(135, 530, "同一个工具位姿，构造两种不同用途的点")
    f.box(65, 210, 590, 90, ["观测分支：s = 当前工具抓握区域尺寸"], "network", 24)
    f.box(745, 210, 590, 90, ["奖励分支：sʳᵉʷ = [0.14, 0.03, 0.03] m"], "signal", 23)
    for x, color, label in [(110, "#2563eb", "尺寸随工具变化"), (790, "#ea580c", "跨工具固定尺度")]:
        pts = [(x, 365), (x + 385, 400), (x + 440, 470), (x + 55, 435)]
        f.parts.append('<polygon points="' + " ".join(f"{a},{b}" for a, b in pts) + f'" fill="#f8fafc" stroke="{color}" stroke-width="2"/>')
        for i, (px, py) in zip([3, 1, 2, 4], pts):
            f.parts.append(f'<circle cx="{px}" cy="{py}" r="7" fill="{color}"/>')
            f.text(px, py - 16, f"k{i}", 20, anchor="middle")
        f.text(x + 220, 513, label, 22, anchor="middle")
    f.text(360, 555, "局部四点示意，非八角点／语义检测点", 20, anchor="middle")
    f.text(1040, 555, "长 x 轴对横向旋转更敏感", 21, anchor="middle")
    f.box(65, 580, 590, 55, ["进入 Actor：相对手掌位置＋目标点误差＋s"], "network", 21)
    f.box(745, 580, 590, 55, ["进入 reward／评估：最差对应点的距离"], "signal", 21)

    f.panel(690, 350, "世界坐标中的同一距离计算，含平移与旋转误差")
    f.box(60, 775, 390, 100, ["当前点", "pₜ,ᵢ = Rₜ kᵢ + pₜ"], size=24)
    f.box(510, 775, 390, 100, ["目标点", "pɢ,ᵢ = Rɢ kᵢ + pɢ"], size=24)
    f.box(960, 775, 380, 100, ["距离 d", "maxᵢ ‖pₜ,ᵢ − pɢ,ᵢ‖₂"], "signal", 24)
    f.arrow([(255, 775), (255, 745), (1150, 745), (1150, 775)])
    f.arrow([(900, 825), (960, 825)])
    f.text(60, 930, "v2：训练阈值 1 cm；评估阈值 2 cm。它们不是独立的平移与角度阈值。", 23)
    f.text(60, 980, "中心到达目标仍可能朝向错误；对应点误差会保留这部分旋转差异。", 22)
    f.save("keypoints")


def reward():
    f = Figure(1100, "Reward：阶段切换与刷新最好距离", "按 v2 附录 A 解释机制；保持步数、权重和课程设置需另查对应版本代码。", "举起前奖励指尖靠近与抬升，举起后以目标进展和到达奖励为主；历史最小距离防止反复进退刷奖励。")
    f.panel(135, 285, "① 举起工具，触发训练中的阶段标志")
    f.box(55, 220, 350, 125, ["从桌面开始", "指尖接近进展", "抬升高度＋首次举起 bonus"], "env", 21)
    f.box(480, 220, 400, 125, ["达到举起门槛", "I_grasped 锁存为 1", "不是触觉 force-closure 检测"], "signal", 21)
    f.box(955, 220, 390, 125, ["转向目标控制", "最好距离进展＋到达 bonus", "允许手内换指和重定向"], "network", 21)
    f.arrow([(405, 282), (480, 282)])
    f.arrow([(880, 282), (955, 282)])
    f.text(55, 390, "整个过程另有关节速度 L1 惩罚；手臂与手采用不同权重。", 22)

    f.panel(445, 400, "② 对同一个目标：先算进展，再更新最好值")
    f.text(60, 530, "正进展 = max(d* − d, 0)       更新：d* ← min(d*, d)", 28, bold=True)
    xs = [110, 420, 740, 1070]
    for x, label in zip(xs, ["当前距离 d", "更新前最好 d*", "本步正进展", "更新后最好 d*"]):
        f.text(x, 590, label, 23, bold=True)
    f.line(60, 611, 1340, 611)
    rows = [("0.08 m", "0.10 m", "0.02 m", "0.08 m"), ("0.09 m", "0.08 m", "0", "0.08 m"), ("0.08 m", "0.08 m", "0", "0.08 m"), ("0.06 m", "0.08 m", "0.02 m", "0.06 m")]
    for i, row in enumerate(rows):
        for x, item in zip(xs, row):
            f.text(x, 650 + 48 * i, item, 24)
    f.panel(870, 190, "③ 达到目标：奖励成功，切换目标，重建距离基准")
    f.text(60, 957, "旧目标 gᵏ → 新目标 gᵏ⁺¹；用当前工具到新目标的距离初始化 d*。", 25)
    f.text(60, 1010, "同一目标下退远再靠近不会重复获得旧进展；换目标后也不能沿用旧记录。", 23)
    f.save("reward")


def evidence():
    f = Figure(1370, "实验地图：不同对照支持不同结论", "所有图表编号与协议均指 arXiv v2；Task Progress 是轨迹进度，不是功能成功率。", "分别展示120次真机评估、刷子基线比较、仿真专用策略比较、训练相关性和训练消融，明确分布与证据边界。")
    rows = [
        (135, "Fig. 4／Table II：真实未见工具与轨迹", ["6 类 × 2 实例 × 2 轨迹", "每组合 5 次，共 120 次", "测量：Task Progress"], ["通用策略的真实迁移", "分工具、分轨迹差异", "不能等同功能任务完成"]),
        (375, "Fig. 5：真机 red brush，两个初始条件", ["完整方法／Fixed Grasp／Retargeting", "不需旋转：98.0／61.0／8.1%", "需要旋转：82.7／10.8／0.0%"], ["联合手臂与手的闭环控制有价值", "同时涉及换指与反馈差异", "不是 24 组合上的全面对照"]),
        (615, "Fig. 6／16：仿真中的 Specialist 泛化", ["训练 Obj A／Traj A", "测试 A/A、A/B、B/A", "每类、每条件 10 次"], ["比较训练分布与跨物体／轨迹迁移", "通用策略在变化条件下保持性能", "未报告 B/B 第四栏"]),
        (855, "Fig. 7：不同 checkpoint 的训练与测试表现", ["左：程序物体随机目标 reward", "右：仿真 DexToolBench 进度", "累计环境步到约 120B"], ["两种目标的表现总体相关", "测试进度并非逐点单调", "不是因果消融或真机训练曲线"]),
        (1095, "Fig. 8：SAPG 与不对称 Critic 消融", ["SAPG＋特权 Critic：完整配置", "PPO＋特权／SAPG＋受限 Critic", "5 个种子；训练 reward；约 9B 步"], ["优化／探索与价值信息均影响学习", "没有完整 2×2 组合", "未单独证明 LSTM 或触觉收益"]),
    ]
    for y, heading, setup, claim in rows:
        f.panel(y, 215, heading)
        f.box(55, y + 65, 600, 125, setup, "data", 22)
        f.box(745, y + 65, 600, 125, claim, "signal", 22)
        f.arrow([(655, y + 127), (745, y + 127)])
    f.save("evidence")


def foundationpose():
    f = Figure(1040, "FoundationPose：先找到物体，再持续跟踪", "以已提供工具 mesh 的用法为例。位姿估计负责感知；后续控制器负责选择动作。", "首帧生成候选位姿，通过渲染比较、Refiner修正和Scorer选择得到初始位姿。后续帧使用前一结果与新图像继续修正，再将位姿转换到机器人坐标系供控制使用。")
    f.panel(135, 285, "① 首帧：mesh＋RGB-D＋相机内参＋目标 mask")
    f.box(55, 220, 280, 120, ["提出候选位姿", "粗略位置", "多种可能的朝向"], size=22)
    f.box(390, 220, 315, 120, ["渲染比较＋Refiner", "对比实拍与假设画面", "修正位置和朝向"], "network", 22)
    f.box(760, 220, 265, 120, ["Scorer", "评价修正后的候选", "选出最匹配者"], "network", 22)
    f.box(1080, 220, 265, 120, ["初始位姿", "物体相对相机", "4×4 变换矩阵"], "signal", 22)
    for x1, x2 in [(335, 390), (705, 760), (1025, 1080)]:
        f.arrow([(x1, 280), (x2, 280)])
    f.text(55, 385, "Refiner 与 Scorer 是预训练的视觉网络；Scorer 的分数不是 RL 累计回报。", 22)

    f.panel(445, 250, "② 后续帧：利用上一步结果，减少重新搜索")
    f.box(55, 525, 365, 100, ["上一帧位姿＋新 RGB-D", "mesh 与相机内参沿用"], size=23)
    f.box(510, 525, 380, 100, ["Refiner 继续修正", "从已有位姿开始"], "network", 23)
    f.box(980, 525, 365, 100, ["当前位姿", "保存给下一帧使用"], "signal", 23)
    f.arrow([(420, 575), (510, 575)])
    f.arrow([(890, 575), (980, 575)])
    f.text(55, 665, "官方跟踪接口沿用前一位姿；保留这个结果不等于使用 LSTM。", 22)

    f.panel(720, 285, "③ 接入机器人：让状态、目标与手掌处于共同坐标系")
    f.box(55, 805, 280, 110, ["相机中的工具位姿", "FoundationPose 输出"], size=22)
    f.box(390, 805, 315, 110, ["相机 → 机器人坐标", "使用标定变换"], "signal", 22)
    f.box(760, 805, 265, 110, ["构造策略观测", "加入目标、本体状态"], size=21)
    f.box(1080, 805, 265, 110, ["Actor → 动作", "执行后继续观测"], "network", 22)
    for x1, x2 in [(335, 390), (705, 760), (1025, 1080)]:
        f.arrow([(x1, 860), (x2, 860)])
    f.text(55, 967, "物体几何与任务目标需另外提供；位姿估计本身不决定机器人应该完成什么任务。", 22)
    f.save("pipeline", prefix="foundationpose")


if __name__ == "__main__":
    pipeline()
    actor_critic()
    keypoints()
    reward()
    evidence()
    foundationpose()
