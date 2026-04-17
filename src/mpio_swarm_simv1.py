from __future__ import annotations

import math
import csv
import dataclasses
import sys
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
import time

import numpy as np

# 可选依赖：matplotlib（更稳的 backend 选择 + 交互绘图）
try:
    import os
    import matplotlib
    # 若外部要求无 GUI（例如 headless 绘图），尊重 MPLBACKEND=Agg
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        matplotlib.use("Agg", force=True)
    else:
        # 在不同环境里尽量选一个“能弹窗且稳定”的 backend
        for _bk in ("QtAgg", "TkAgg"):
            try:
                matplotlib.use(_bk, force=True)
                break
            except Exception:
                pass

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False



# ---------------------------- 数值工具 ----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def norm2(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _seg_point_dists(p0: np.ndarray, p1: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return distances from a 2D segment (p0->p1) to multiple 2D points."""
    p0 = np.asarray(p0, dtype=float).reshape(2)
    p1 = np.asarray(p1, dtype=float).reshape(2)
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0,), dtype=float)

    d = p1 - p0
    l2 = float(np.dot(d, d))
    if l2 < 1e-12:
        return np.linalg.norm(pts - p0[None, :], axis=1)

    t = ((pts - p0[None, :]) @ d) / l2
    t = np.clip(t, 0.0, 1.0)
    proj = p0[None, :] + t[:, None] * d[None, :]
    return np.linalg.norm(pts - proj, axis=1)


# ---------------------------- 参数配置（与 MATLAB 保持一致） ----------------------------

@dataclass
class SimParams:
    # 模型常量（Eq.(3)(4)）
    T_v: float = 0.8
    T_psi: float = 0.6
    T_h: float = 0.8
    T_lambda: float = 0.25
    Vxy_min: float = 4.0
    Vxy_max: float = 12.0
    lambda_min: float = -5.0
    lambda_max: float = 5.0
    n_max: float = 3.0  # 侧向过载上限（g）
    g: float = 9.8

    # 自推进/编队/避障增益（与 MATLAB 相同）
    Kf: float = 0.25
    Kc: float = 100000.0
    Ka_vn: float = 0.1
    Ka_he: float = 3.0
    Kve: float = 1.0

    R1_comm: float = 40.0
    R_desire: float = 10.0
    R1_lim: float = 2.0
    R2_comm: float = 40.0
    R2_lim: float = 5.0
    # cost1 平滑切换参数（与 R2_comm 无关，只看安全边界附近）
    cost1_buffer: float = 3.0
    cost1_sigma: float = 8.0
    # 安全判定的数值容差（只用于统计，不改变几何障碍定义）
    collision_tol_obs: float = 1.0
    collision_tol_nbr: float = 0.0
    # 安全裕度判定带（用于抑制离散采样下的数值抖动）
    safe_soft_band: float = 1.0
    # safe_ok 净距阈值：distance(center, UAV) >= r_obs + safe_clearance_m
    safe_clearance_m: float = 0.5

    # Gap selection / avoidance tuning
    gap_k_speed: float = 0.7  # r_eff = r_obs + R2_lim + k * v * dt
    gap_margin_deg: float = 8.0
    gap_samples: int = 7
    gap_blocked_samples: int = 31
    gap_w_clear: float = 3.0
    gap_w_width: float = 1.0
    gap_w_align: float = 0.6  # used as progress weight
    gap_w_turn: float = 0.2
    gap_boundary_enable: bool = True
    gap_clear_m: float = 15.0
    gap_clear_min: float = 0.4
    gap_prog_min: float = 0.4
    gap_w_min_deg: float = 10.0
    gap_edge_margin_deg: float = 8.0
    gap_edge_penalty: float = 0.3
    gap_boundary_clear_adv: float = 0.2
    gap_boundary_bias: float = 0.2
    gap_lookahead: float = 0.0

    # yaw rate / blocked handling
    yaw_rate_max: float = math.radians(60.0)  # rad/s
    blocked_speed_scale: float = 0.6

    # Swarm-level yaw (front-k average) to avoid split decisions
    swarm_yaw_enable: bool = True
    swarm_yaw_k: int = 3

    he: float = 50.0
    ve3: float = 0.0
    ve_xy: Tuple[float, float] = (8.0, 0.0)  # 期望水平速度

    theta_lim: float = math.pi / 2.0
    Rc: float = 10.0

    # MPIO 参数（与 MATLAB 相同）
    N: int = 58
    R: float = 0.3
    tr: float = 3.0
    Ncmax: int = 20
    Nd: int = 2
    p1: float = 0.9
    e_learn: float = 0.01
    sl: int = 2

    # 采样
    dt: float = 0.5
    sim_time: float = 59.5

    # 控制死区（Step 11，Eq.(25)(26)）
    u_lim: float = 0.25

    # Cost2 权重
    f1: float = 1.0
    f2: float = 1.0

    # 巡航模式 cost1 权重（速度误差/方向误差）
    w_speed: float = 1.0
    w_direction: float = 1.0

    # 允许误差吸附阈值（论文 Step 11）
    Vxy_c_lim: float = 0.25
    psi_c_lim: float = 0.10

    # 障碍定义（与 MATLAB 相同）：(x, y, r)
    obstacles: np.ndarray = field(default_factory=lambda: np.array([
        [120.0, 120.0, 5.0],
        [240.0, 75.0, 5.0],
        [350.0, 40.0, 5.0],
        [240.0, 155.0, 5.0],
        [360.0, 110.0, 5.0],
        [350.0, 180.0, 5.0],
    ], dtype=float))
    # 方块障碍（方案一：画成方块；计算用外接圆）
    square_enable: bool = True
    square_center_xy: Tuple[float, float] = (320.0, 140.0)
    square_side: float = 20.0  # 边长 a
    square_vxy: Tuple[float, float] = (0.6, 0.0)  # 横向速度：+x/-x 来回
    # 障碍速度（vx, vy）；默认不动，可按需设置
    obstacles_vxy: np.ndarray = field(default_factory=lambda: np.zeros((6, 2), dtype=float))
    # 障碍物运动边界（用于反弹），xmin/xmax/ymin/ymax
    obstacles_bounds: Tuple[float, float, float, float] = (0.0, 400.0, 0.0, 200.0)
    # 是否用一步预测作为规划障碍
    obstacles_use_prediction: bool = False

    # 是否对邻居位置做“一拍预测”用于 Cost4（队内碰撞硬约束）与 Cost2（编队距离项）评估。
    # 说明：这不会改变“分布式”属性——仍只使用邻居的当前位置+速度（本就可通信/可观测），
    # 仅用于在离散同步更新时避免“双方都在动但约束只看对方当前点”的漏检。
    neighbors_use_prediction: bool = True

    # 初始状态（与 MATLAB 表 1 一致）（x, y, h）
    init_P: np.ndarray = field(default_factory=lambda: np.array([
        [14.6929, 107.3676, 68.1682],
        [21.2809, 116.6406, 34.8423],
        [20.3911, 113.6529, 24.6351],
        [3.5699, 108.9509, 96.3770],
        [10.2116, 111.5580, 30.1431],
    ], dtype=float))

    # 初始水平速度：全部 8 m/s 朝 +X
    init_Vxy: np.ndarray = field(default_factory=lambda: np.tile(np.array([8.0, 0.0], dtype=float), (5, 1)))

    # 其它
    rng_seed: Optional[int] = 42  # 方便复现实验

    # 调试：是否打印 UAV1 的每拍状态
    verbose_uav1: bool = True
    # 调试：是否打印 MPIO 有效样本统计
    verbose_mpio: bool = False

    # 是否输出 CSV 调试日志
    csv_path: Optional[str] = "uav_debug_log.csv"

    # 鏈€灏忛€傜敱锛氬妯″潡娴嬭瘯涓紝淇濆瓨涓婂噯鐨勮惤瀹炰笌鍖哄埆淇℃伅
    last_w: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float))
    last_w_valid: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))


# ---------------------------- 自推进/避障/成本函数 ----------------------------

def neighbors_within(P: np.ndarray, i: int, R: float) -> List[int]:
    idxs = []
    for j in range(P.shape[0]):
        if j == i:
            continue
        if np.linalg.norm(P[j, :2] - P[i, :2]) <= R:
            idxs.append(j)
    return idxs


def compute_raw_forces(P: np.ndarray, V_xy: np.ndarray, i: int, sp: SimParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 ff_raw, fa_raw, fc_raw （Eq.(6)(7)(8)）"""
    ff_raw = np.zeros(2, dtype=float)
    fa_raw = np.zeros(2, dtype=float)
    fc_raw = np.zeros(2, dtype=float)

    nbrs = neighbors_within(P, i, sp.R1_comm)
    for j in range(P.shape[0]):
        if j == i:
            continue

        rel = P[j, :2] - P[i, :2]
        dij = np.linalg.norm(rel)

        # flocking geometry (within R1_comm)
        if dij <= sp.R1_comm and dij > 1e-6:
            ff_raw += sp.Kf * (1.0 - (sp.R_desire / dij) ** 2) * rel
            fa_raw += sp.Ka_vn * (V_xy[j, :] - V_xy[i, :])

        # collision avoidance (within R1_lim)
        if dij <= sp.R1_lim and dij > 1e-6:
            rep_mag = sp.Kc * (1.0 / dij - 1.0 / sp.R1_lim) ** 2
            fc_raw -= rep_mag * (rel / dij)

    return ff_raw, fa_raw, fc_raw


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _angle_diff(a: float, b: float) -> float:
    return _wrap_angle(a - b)


def _ray_clearance(drone: np.ndarray,
                   obstacles: np.ndarray,
                   yaw: float,
                   r_eff: np.ndarray,
                   r_comm: float) -> float:
    """Distance along ray to first intersection with expanded obstacles; returns r_comm if none."""
    dir_vec = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
    best = r_comm
    for (cx, cy, _), re in zip(obstacles, r_eff):
        rel = np.array([cx - drone[0], cy - drone[1]], dtype=float)
        t = float(np.dot(rel, dir_vec))
        if t <= 0.0:
            continue
        d2 = float(np.dot(rel, rel))
        perp2 = d2 - t * t
        if perp2 < 0.0:
            perp2 = 0.0
        if perp2 >= re * re:
            continue
        dt = math.sqrt(max(re * re - perp2, 0.0))
        hit = t - dt
        if hit < best:
            best = hit
    return best


def _rate_limit_angle(desired: float, current: float, max_rate: float, dt: float) -> float:
    if max_rate is None or max_rate <= 0.0:
        return desired
    max_delta = max_rate * dt
    diff = _wrap_angle(desired - current)
    diff = clamp(diff, -max_delta, max_delta)
    return _wrap_angle(current + diff)


def obstacle_avoidance_uav(P_i: np.ndarray,
                           theta_e: float,
                           obstacles: np.ndarray,
                           sp: SimParams,
                           return_debug: bool = False,
                           v_mag: Optional[float] = None,
                           v_xy: Optional[np.ndarray] = None) -> Tuple[float, List[int], Optional[dict]]:
    """
    统一候选角评分的避障：
    - 用扩展圆生成遮挡角区间；
    - 仅保留“可能碰撞”的前方障碍；
    - interior gap 优先，必要时才退到 boundary gap；
    - 单/多障碍不分支，统一评分框架。
    """
    drone = P_i[:2]
    R_comm = sp.R2_comm
    theta_v = sp.theta_lim
    R2_lim = sp.R2_lim
    Rc = sp.Rc  # 你在 SimParams 里有这个

    # reference axis: current velocity direction (fallback to ve)
    v_min = 0.5
    if v_xy is not None:
        v_vec = np.array(v_xy, dtype=float).reshape(2)
        v_norm = float(np.linalg.norm(v_vec))
    else:
        v_vec = None
        v_norm = float(v_mag) if v_mag is not None else 0.0
    if v_vec is None or v_norm < v_min:
        v_vec = np.array(sp.ve_xy, dtype=float)
        v_norm = float(np.linalg.norm(v_vec))
    if v_norm < 1e-6:
        v_vec = np.array([1.0, 0.0], dtype=float)
        v_norm = 1.0
    f_hat = v_vec / v_norm
    ref_yaw = math.atan2(f_hat[1], f_hat[0])

    blocked: List[Tuple[float, float]] = []
    sensed_idx: List[int] = []
    lookahead = float(getattr(sp, "gap_lookahead", 0.0))
    if lookahead <= 0.0:
        lookahead = min(0.7 * R_comm, 80.0)

    # -------- 1. 计算扩展圆阻塞角区间（和你之前一样） --------
    v_for_eff = float(v_norm) if v_norm > 0.0 else float(np.linalg.norm(sp.ve_xy))
    r_eff_all = obstacles[:, 2] + R2_lim + float(getattr(sp, "gap_k_speed", 0.7)) * v_for_eff * sp.dt
    for j in range(obstacles.shape[0]):
        cx, cy, r_obs = obstacles[j]
        rel = np.array([cx - drone[0], cy - drone[1]], dtype=float)
        d = float(np.linalg.norm(rel))
        if d < 1e-6:
            continue

        r_eff = float(r_eff_all[j])
        # collision-related filter: forward + within lateral tube
        t = float(np.dot(rel, f_hat))
        if t <= 0.0:
            continue
        if t > lookahead:
            continue
        d2 = float(np.dot(rel, rel))
        perp2 = d2 - t * t
        if perp2 < 0.0:
            perp2 = 0.0
        d_perp = math.sqrt(perp2)
        if d_perp > r_eff:
            continue

        # 扩展圆和关注圆 (R_comm) 没交集就忽略
        if d - r_eff > R_comm:
            continue

        alpha = math.atan2(rel[1], rel[0])          # 障碍中心全局角
        delta = _wrap_angle(alpha - ref_yaw)        # 参考方向局部坐标系下的角

        if d <= r_eff:
            beta = theta_v                          # 已经在扩展圆里，视场全被挡
        else:
            beta = math.asin(min(1.0, r_eff / d))   # 扩展圆半角宽度

        left = delta - beta
        right = delta + beta

        # 与视场 [-theta_v, theta_v] 没重叠就跳过
        if right < -theta_v or left > theta_v:
            continue

        left = max(left, -theta_v)
        right = min(right, theta_v)
        if left < right:
            blocked.append((left, right))
            sensed_idx.append(j)

    # 视场中没有被任何扩展圆遮挡：沿期望方向飞
    if not blocked:
        if return_debug:
            debug = {
                "gap_type": "none",
                "blocked_cnt": 0,
                "gap_cnt": 0,
                "gap_left": math.nan,
                "gap_right": math.nan,
                "gap_width": math.nan,
                "local_angle": _wrap_angle(theta_e - ref_yaw),
                "blocked": False,
                "gap_score": math.nan,
                "gap_clear": math.nan,
            }
            return theta_e, [], debug
        return theta_e, []

    # -------- 2. 合并阻塞区间 --------
    blocked.sort(key=lambda seg: seg[0])
    merged = [blocked[0]]
    for a, b in blocked[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))

    # -------- 3. 统一 gap 选择（内部 + 边界） --------
    interior_gaps: List[Tuple[float, float]] = []
    for k in range(len(merged) - 1):
        left = merged[k][1]
        right = merged[k + 1][0]
        if right > left:
            interior_gaps.append((left, right))

    boundary_gaps: List[Tuple[float, float]] = []
    first_left, _ = merged[0]
    _, last_right = merged[-1]
    if first_left > -theta_v:
        boundary_gaps.append((-theta_v, first_left))
    if last_right < theta_v:
        boundary_gaps.append((last_right, theta_v))

    # --- scoring params ---
    obs_use = obstacles[sensed_idx] if sensed_idx else obstacles
    r_eff_use = r_eff_all[sensed_idx] if sensed_idx else r_eff_all

    margin = math.radians(float(getattr(sp, "gap_margin_deg", 8.0)))
    edge_margin = math.radians(float(getattr(sp, "gap_edge_margin_deg", 8.0)))
    w_min = math.radians(float(getattr(sp, "gap_w_min_deg", 10.0)))
    n_cand = int(getattr(sp, "gap_samples", 5))
    m_safe = float(getattr(sp, "gap_clear_m", 15.0))
    clear_min = float(getattr(sp, "gap_clear_min", 0.4))
    prog_min = float(getattr(sp, "gap_prog_min", 0.4))
    edge_pen = float(getattr(sp, "gap_edge_penalty", 0.3))
    boundary_adv = float(getattr(sp, "gap_boundary_clear_adv", 0.2))
    boundary_bias = float(getattr(sp, "gap_boundary_bias", boundary_adv))

    k1 = float(getattr(sp, "gap_w_clear", 3.0))
    k2 = float(getattr(sp, "gap_w_width", 1.0))
    k3 = float(getattr(sp, "gap_w_align", 0.6))  # progress weight
    k4 = float(getattr(sp, "gap_w_turn", 0.2))

    def _candidate_fracs(n: int) -> List[float]:
        if n <= 2:
            return [0.5]
        if n == 3:
            return [0.25, 0.5, 0.75]
        if n == 4:
            return [0.2, 0.4, 0.6, 0.8]
        return [0.2, 0.35, 0.5, 0.65, 0.8]

    def _eval_gaps(gaps: List[Tuple[float, float]],
                   stage: str) -> Tuple[Optional[Tuple[float, float]], float, float, float, float]:
        best_score = -1e9
        best_angle = 0.0
        best_gap = None
        best_clear = 0.0
        best_clear_norm = 0.0
        for (a, b) in gaps:
            if b <= a:
                continue
            width = b - a
            if width < w_min:
                continue
            aa = a + margin
            bb = b - margin
            if bb <= aa:
                aa = a
                bb = b
            candidates = [aa + (bb - aa) * f for f in _candidate_fracs(n_cand)]
            if a <= 0.0 <= b:
                candidates.append(0.0)
            alpha_e = _wrap_angle(theta_e - ref_yaw)
            if a <= alpha_e <= b:
                candidates.append(alpha_e)
            for alpha in sorted(set(candidates)):
                if abs(alpha) > (theta_v - edge_margin):
                    continue
                prog = math.cos(alpha)
                if prog < prog_min:
                    continue
                yaw = _wrap_angle(ref_yaw + alpha)
                clear = _ray_clearance(drone, obs_use, yaw, r_eff_use, sp.R2_comm)
                clear_norm = clear / m_safe if m_safe > 1e-6 else 1.0
                clear_norm = clamp(clear_norm, 0.0, 1.0)
                if clear_norm < clear_min:
                    continue
                width_norm = width / (2.0 * theta_v)
                s_turn = -abs(alpha) / max(theta_v, 1e-6)
                score = k1 * clear_norm + k2 * width_norm + k3 * prog + k4 * s_turn
                if stage == "boundary":
                    score -= edge_pen * (abs(alpha) / max(theta_v, 1e-6))
                if score > best_score:
                    best_score = score
                    best_angle = alpha
                    best_gap = (a, b)
                    best_clear = clear
                    best_clear_norm = clear_norm
        return best_gap, best_angle, best_score, best_clear, best_clear_norm

    # Evaluate interior + boundary, then soft-prefer interior
    best_gap_i, best_angle_i, best_score_i, best_clear_i, _ = _eval_gaps(
        interior_gaps, "interior"
    )
    best_gap_b, best_angle_b, best_score_b, best_clear_b, _ = _eval_gaps(
        boundary_gaps, "boundary"
    )

    # detect one-sided obstacles (all blocked intervals on same side)
    side_signs = []
    for a, b in merged:
        c = 0.5 * (a + b)
        if c > 1e-6:
            side_signs.append(1)
        elif c < -1e-6:
            side_signs.append(-1)
    one_side = False
    if side_signs:
        one_side = all(s > 0 for s in side_signs) or all(s < 0 for s in side_signs)

    best_gap = None
    best_angle = 0.0
    best_score = -1e9
    best_clear = 0.0
    gap_type = "none"
    gap_cnt = 0

    if best_gap_i is not None:
        best_gap = best_gap_i
        best_angle = best_angle_i
        best_score = best_score_i
        best_clear = best_clear_i
        gap_type = "interior"
        gap_cnt = len(interior_gaps)

    if bool(getattr(sp, "gap_boundary_enable", True)) and best_gap_b is not None:
        bias = 0.0 if one_side else boundary_bias
        b_score_adj = best_score_b - bias
        if best_gap is None or b_score_adj > best_score:
            best_gap = best_gap_b
            best_angle = best_angle_b
            best_score = b_score_adj
            best_clear = best_clear_b
            gap_type = "boundary"
            gap_cnt = len(boundary_gaps)

    if best_gap is None:
        # blocked: no qualified gap -> maximize clearance in full FOV
        m = int(getattr(sp, "gap_blocked_samples", 31))
        angles = [(-theta_v + 2.0 * theta_v * i / (m - 1)) for i in range(m)]
        best_c = -1.0
        best_a = 0.0
        for ang in angles:
            if abs(ang) > (theta_v - edge_margin):
                continue
            yaw = _wrap_angle(ref_yaw + ang)
            clear = _ray_clearance(drone, obs_use, yaw, r_eff_use, sp.R2_comm)
            if clear > best_c:
                best_c = clear
                best_a = ang
        local_angle = best_a
        yaw_desired = _wrap_angle(ref_yaw + local_angle)
        if return_debug:
            debug = {
                "gap_type": "blocked",
                "blocked_cnt": len(merged),
                "gap_cnt": 0,
                "gap_left": math.nan,
                "gap_right": math.nan,
                "gap_width": math.nan,
                "local_angle": local_angle,
                "blocked": True,
                "gap_score": math.nan,
                "gap_clear": best_c,
            }
            return yaw_desired, sensed_idx, debug
        return yaw_desired, sensed_idx

    gap_left, gap_right = best_gap
    gap_width = gap_right - gap_left
    local_angle = best_angle
    yaw_desired = _wrap_angle(ref_yaw + local_angle)
    if return_debug:
        debug = {
            "gap_type": gap_type,
            "blocked_cnt": len(merged),
            "gap_cnt": gap_cnt,
            "gap_left": gap_left,
            "gap_right": gap_right,
            "gap_width": gap_width,
            "local_angle": local_angle,
            "blocked": False,
            "gap_score": best_score,
            "gap_clear": best_clear,
        }
        return yaw_desired, sensed_idx, debug
    return yaw_desired, sensed_idx


# ---------------------------- MPIO（多目标鸽群优化） ----------------------------

@dataclass
class Pigeon:
    Position: np.ndarray
    V: np.ndarray
    Cost12: np.ndarray = field(default_factory=lambda: np.array([np.inf, np.inf], dtype=float))
    Rank: float = float("inf")
    CrowdingDistance: float = 0.0


def initializePigeons(N: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.random((N, 2))  # w=[w1,w2] in [0,1]
    V = 0.05 * (2.0 * rng.random((N, 2)) - 1.0)
    return X, V


def _resolve_rng(sp: SimParams, rng: Optional[np.random.Generator]) -> np.random.Generator:
    """Return a generator without re-seeding on every call."""
    if rng is not None:
        return rng
    existing = getattr(sp, "_rng", None)
    if isinstance(existing, np.random.Generator):
        return existing
    new_rng = np.random.default_rng(sp.rng_seed)
    setattr(sp, "_rng", new_rng)
    return new_rng


def _pick_from_archive(A: List[Pigeon]) -> Optional[np.ndarray]:
    if not A:
        return None
    A_costs = np.vstack([p.Cost12 for p in A])
    ok = np.all(np.isfinite(A_costs), axis=1)
    if not np.any(ok):
        return None
    A_ok = [A[i] for i in np.where(ok)[0]]
    best = min(A_ok, key=lambda p: (p.Cost12[1], p.Cost12[0]))
    return best.Position.copy()


def nondominated_sort_safe(Costs: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    # 基本的 Pareto 非支配排序（小心 inf/NaN）
    N = Costs.shape[0]
    dom_count = np.zeros(N, dtype=int)
    S: List[List[int]] = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            # i dominates j ?
            if np.all(Costs[i] <= Costs[j]) and np.any(Costs[i] < Costs[j]):
                S[i].append(j)
                dom_count[j] += 1
            elif np.all(Costs[j] <= Costs[i]) and np.any(Costs[j] < Costs[i]):
                S[j].append(i)
                dom_count[i] += 1

    rank = np.full(N, np.inf)
    current = [idx for idx in range(N) if dom_count[idx] == 0]
    fronts: List[List[int]] = []
    r = 1
    while current:
        fronts.append(current)
        for idx in current:
            rank[idx] = r
        next_front: List[int] = []
        for idx in current:
            for j in S[idx]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        current = sorted(set(next_front))
        r += 1

    # 对未分配 rank 的设置为最后一层
    rank[np.isinf(rank)] = r
    return rank, fronts


def CalcCrowdingDistance(pop: List[Pigeon], fronts: List[List[int]]) -> List[Pigeon]:
    for F in fronts:
        if not F:
            continue
        Costs = np.array([pop[i].Cost12 for i in F], dtype=float)
        n, nObj = Costs.shape
        if n == 0:
            continue
        d = np.zeros((n, nObj), dtype=float)
        for j in range(nObj):
            cj = Costs[:, j].copy()
            so = np.argsort(cj)
            d[so[0], j] = np.inf
            if n > 2:
                denom = cj[so[-1]] - cj[so[0]]
                if abs(denom) < 1e-9:
                    denom = 1.0
                for ii in range(1, n - 1):
                    d[so[ii], j] = (cj[so[ii + 1]] - cj[so[ii - 1]]) / denom
            d[so[-1], j] = np.inf

        # 写回 crowding distance
        for local_idx, pigeon_idx in enumerate(F):
            pop[pigeon_idx].CrowdingDistance = float(np.sum(d[local_idx, :]))
    return pop


def calculateCosts(P_i_next: np.ndarray, V_xy_i_next: np.ndarray, P_swarm: np.ndarray, i_uav: int,
                   V_xy_swarm: np.ndarray, obstacles: np.ndarray, sp: SimParams) -> Tuple[float, float, int, int]:
    """
    成本函数：
      - cost3：与障碍物的硬约束（碰撞，R2_lim 内为 1）
      - cost4：与邻居的硬约束（碰撞，R1_lim 内为 1）
      - cost1：软约束（避障区：最大化投影；巡航区：速度&方向对齐）
      - cost2：软约束（编队质量与速度对齐）
    """
    pos_i_next = P_i_next[:2]
    vel_i_next = V_xy_i_next
    pos_i_cur = np.asarray(P_swarm[i_uav, :2], dtype=float)

    # 规划视角下的“邻居位置”（用于避免同步更新漏检）
    # 仍然是分布式：只用邻居当前位置+速度做常速度一步预测。
    nbr_xy = P_swarm[:, :2]
    if bool(getattr(sp, "neighbors_use_prediction", False)):
        nbr_xy = nbr_xy + V_xy_swarm * sp.dt

    # --- 硬约束：障碍距离（R2_lim）- 用线段检测避免隧穿 ---
    cost3 = 0
    if obstacles.size > 0:
        p0 = pos_i_cur
        p1 = pos_i_next
        centers = obstacles[:, :2]
        radii = obstacles[:, 2] + sp.R2_lim
        dists = _seg_point_dists(p0, p1, centers)
        if dists.size > 0 and np.any(dists < radii):
            cost3 = 1

    # --- 硬约束：队友距离（R1_lim） ---
    cost4 = 0
    for j in range(P_swarm.shape[0]):
        if i_uav != j and np.linalg.norm(nbr_xy[j] - pos_i_next) < sp.R1_lim:
            cost4 = 1
            break

    ve = np.array(sp.ve_xy, dtype=float)
    ve_mag = float(np.linalg.norm(ve))
    ve_unit = ve / (ve_mag if ve_mag > 1e-6 else 1.0)

    # ---- cost1: smooth mix (based on CURRENT clearance to safety boundary) ----
    buffer = float(getattr(sp, "cost1_buffer", 3.0))
    sigma = float(getattr(sp, "cost1_sigma", 8.0))

    d_min = 1e9
    for j in range(obstacles.shape[0]):
        r_obs = obstacles[j, 2]
        d = np.linalg.norm(obstacles[j, :2] - pos_i_cur) - (r_obs + sp.R2_lim + buffer)
        if d < d_min:
            d_min = d

    alpha = float(np.clip(1.0 - d_min / sigma, 0.0, 1.0))

    # 近障碍：更关心“向前推进分量”别掉速（允许侧向绕）
    forward = float(np.dot(vel_i_next, ve_unit))
    progress_cost = ve_mag - forward

    # 远障碍：更关心速度贴近 ve（稳）
    vel_match = abs(ve[0] - vel_i_next[0]) + abs(ve[1] - vel_i_next[1])

    cost1 = alpha * progress_cost + (1.0 - alpha) * vel_match

    # --- cost2（编队几何&速度对齐） ---
    cost2 = 0.0
    nbr_cnt = 0
    for j in range(P_swarm.shape[0]):
        if i_uav == j:
            continue
        pos_j_current = nbr_xy[j]
        vel_j_current = V_xy_swarm[j, :]
        dist_ij_future = float(np.linalg.norm(pos_j_current - pos_i_next))
        if dist_ij_future <= sp.R1_comm:
            nbr_cnt += 1
            geom_error = abs(sp.R_desire - dist_ij_future)
            vel_diff_align = vel_i_next - vel_j_current
            align_error = abs(vel_diff_align[0]) + abs(vel_diff_align[1])
            cost2 += sp.f1 * geom_error + sp.f2 * align_error
    if nbr_cnt > 0:
        cost2 /= float(nbr_cnt)

    return float(cost1), float(cost2), int(cost3), int(cost4)


def _init_obstacle_state(sp: SimParams) -> np.ndarray:
    obs_state = np.zeros((sp.obstacles.shape[0], 5), dtype=float)
    obs_state[:, :3] = sp.obstacles
    vxy = np.array(sp.obstacles_vxy, dtype=float)
    if vxy.ndim == 1 and vxy.size == 2:
        vxy = np.tile(vxy, (obs_state.shape[0], 1))
    if vxy.shape != (obs_state.shape[0], 2):
        vxy = np.zeros((obs_state.shape[0], 2), dtype=float)
    obs_state[:, 3:5] = vxy
    return obs_state


def _append_square_as_circle(sp: SimParams) -> Optional[int]:
    """把方块用外接圆近似加入 sp.obstacles，返回它在障碍列表中的索引。"""
    if not getattr(sp, "square_enable", False):
        sp.square_idx = None
        return None
    # 防止重复追加（数据集循环里很关键）
    if getattr(sp, "_square_appended", False):
        return sp.square_idx

    a = float(sp.square_side)
    cx, cy = sp.square_center_xy
    r = (math.sqrt(2) / 2.0) * a

    sp.obstacles = np.vstack([sp.obstacles, np.array([[cx, cy, r]], dtype=float)])

    vxy = np.asarray(sp.obstacles_vxy, dtype=float)
    if vxy.ndim == 1 and vxy.size == 2:
        vxy = np.tile(vxy, (sp.obstacles.shape[0] - 1, 1))
    if vxy.shape[0] < sp.obstacles.shape[0] - 1:
        pad = np.zeros((sp.obstacles.shape[0] - 1 - vxy.shape[0], 2), dtype=float)
        vxy = np.vstack([vxy, pad])
    elif vxy.shape[0] > sp.obstacles.shape[0] - 1:
        vxy = vxy[:sp.obstacles.shape[0] - 1, :]
    sp.obstacles_vxy = np.vstack([vxy, np.array([list(sp.square_vxy)], dtype=float)])

    sp.square_idx = int(sp.obstacles.shape[0] - 1)
    sp._square_appended = True
    return sp.square_idx


def _update_obstacle_state(obs_state: np.ndarray, sp: SimParams) -> None:
    xmin, xmax, ymin, ymax = sp.obstacles_bounds
    for j in range(obs_state.shape[0]):
        x, y, r, vx, vy = obs_state[j]
        x += vx * sp.dt
        y += vy * sp.dt

        min_x = xmin + r
        max_x = xmax - r
        min_y = ymin + r
        max_y = ymax - r

        if x < min_x:
            x = min_x + (min_x - x)
            vx = -vx
        elif x > max_x:
            x = max_x - (x - max_x)
            vx = -vx

        if y < min_y:
            y = min_y + (min_y - y)
            vy = -vy
        elif y > max_y:
            y = max_y - (y - max_y)
            vy = -vy

        obs_state[j] = np.array([x, y, r, vx, vy], dtype=float)


def extract_features(pos_i: np.ndarray,
                     vel_i: np.ndarray,
                     psi_i: float,
                     lamb_i: float,
                     neighbors: np.ndarray,
                     obstacles: np.ndarray,
                     ve_xy: Tuple[float, float],
                     k_n: int = 3,
                     k_o: int = 3) -> np.ndarray:
    """
    固定维度特征：self + desired + Top-K 邻居 + Top-K 障碍。
    邻居特征：[dx, dy, dvx, dvy, dist]，障碍特征：[dx, dy, r, dvx, dvy, d_clear]。
    """
    pos_i = np.asarray(pos_i, dtype=float).reshape(-1)
    vel_i = np.asarray(vel_i, dtype=float).reshape(-1)
    ve = np.asarray(ve_xy, dtype=float).reshape(-1)
    if pos_i.size < 2:
        pos_i = np.pad(pos_i, (0, 2 - pos_i.size), mode="constant")

    vx_i = float(vel_i[0]) if vel_i.size > 0 else 0.0
    vy_i = float(vel_i[1]) if vel_i.size > 1 else 0.0
    speed_i = float(math.hypot(vx_i, vy_i))
    h_i = float(pos_i[2]) if pos_i.size >= 3 else 0.0
    self_feat = np.array([vx_i, vy_i, speed_i, float(psi_i), float(lamb_i), h_i], dtype=float)

    desired_feat = np.array([float(ve[0]), float(ve[1])], dtype=float)

    neighbors = np.asarray(neighbors, dtype=float)
    if neighbors.size == 0:
        neighbors = np.zeros((0, 4), dtype=float)
    if neighbors.ndim == 1:
        neighbors = neighbors.reshape(1, -1)
    if neighbors.shape[1] < 4:
        pad = np.zeros((neighbors.shape[0], 4 - neighbors.shape[1]), dtype=float)
        neighbors = np.hstack([neighbors, pad])
    n_pos = neighbors[:, :2]
    n_vel = neighbors[:, 2:4]
    n_dxdy = n_pos - pos_i[:2]
    n_dv = n_vel - np.array([vx_i, vy_i], dtype=float)
    n_dist = np.linalg.norm(n_dxdy, axis=1)
    n_feat = np.zeros((k_n, 5), dtype=float)
    if n_dist.size > 0:
        order = np.argsort(n_dist)
        for out_i, idx in enumerate(order[:k_n]):
            n_feat[out_i, :] = np.array([
                n_dxdy[idx, 0], n_dxdy[idx, 1],
                n_dv[idx, 0], n_dv[idx, 1],
                n_dist[idx],
            ], dtype=float)

    obstacles = np.asarray(obstacles, dtype=float)
    if obstacles.size == 0:
        obstacles = np.zeros((0, 3), dtype=float)
    if obstacles.ndim == 1:
        obstacles = obstacles.reshape(1, -1)
    if obstacles.shape[1] < 3:
        pad = np.zeros((obstacles.shape[0], 3 - obstacles.shape[1]), dtype=float)
        obstacles = np.hstack([obstacles, pad])
    o_pos = obstacles[:, :2]
    o_r = obstacles[:, 2]
    if obstacles.shape[1] >= 5:
        o_vxy = obstacles[:, 3:5]
    else:
        o_vxy = np.zeros((obstacles.shape[0], 2), dtype=float)
    o_dxdy = o_pos - pos_i[:2]
    o_dist = np.linalg.norm(o_dxdy, axis=1)
    o_clear = o_dist - o_r
    o_feat = np.zeros((k_o, 6), dtype=float)
    if o_dist.size > 0:
        order = np.argsort(o_dist)
        for out_i, idx in enumerate(order[:k_o]):
            o_feat[out_i, :] = np.array([
                o_dxdy[idx, 0], o_dxdy[idx, 1],
                o_r[idx],
                o_vxy[idx, 0], o_vxy[idx, 1],
                o_clear[idx],
            ], dtype=float)

    return np.concatenate([self_feat, desired_feat, n_feat.reshape(-1), o_feat.reshape(-1)], axis=0)


def run_episode(sp: Optional[SimParams],
                scene_cfg: Optional[Dict[str, object]],
                collect: bool = True,
                stride: int = 2,
                policy_fn: Optional[Callable[..., Optional[np.ndarray]]] = None) -> Dict[str, object]:
    """
    单场景运行：返回轨迹列表、成本/编队指标和布尔验收标志。
    scene_cfg 支持对 sp 的简单覆盖（key=属性名）。
    """
    if sp is None:
        sp = SimParams()
    sp_local = copy.deepcopy(sp)
    if scene_cfg:
        for key, value in scene_cfg.items():
            setattr(sp_local, key, value)

    rng = np.random.default_rng(sp_local.rng_seed)
    results = run_simulation(sp_local, enable_plot=False, live_plot=False, csv_path=None, policy_fn=policy_fn, rng=rng)
    X = results["X"]
    Y = results["Y"]
    Z = results["Z"]
    Vx = results["Vx"]
    Vy = results["Vy"]
    obs_hist = results.get("obs_hist", None)
    decision_latency_ms = np.asarray(results.get("decision_latency_ms", np.zeros((0,), dtype=np.float32)), dtype=float)
    step_decision_latency_ms = np.asarray(results.get("step_decision_latency_ms", np.zeros((0,), dtype=np.float32)), dtype=float)

    budget_ms = float(sp_local.dt) * 1000.0
    if decision_latency_ms.size > 0:
        decision_mean_ms = float(np.mean(decision_latency_ms))
        decision_std_ms = float(np.std(decision_latency_ms))
        decision_p95_ms = float(np.percentile(decision_latency_ms, 95))
    else:
        decision_mean_ms = decision_std_ms = decision_p95_ms = 0.0
    if step_decision_latency_ms.size > 0:
        step_mean_ms = float(np.mean(step_decision_latency_ms))
        step_std_ms = float(np.std(step_decision_latency_ms))
        step_p95_ms = float(np.percentile(step_decision_latency_ms, 95))
        step_overrun_count = int(np.sum(step_decision_latency_ms > budget_ms))
    else:
        step_mean_ms = step_std_ms = step_p95_ms = 0.0
        step_overrun_count = 0
    latency_summary = {
        "budget_ms": budget_ms,
        "decision_count": int(decision_latency_ms.size),
        "decision_mean_ms": decision_mean_ms,
        "decision_std_ms": decision_std_ms,
        "decision_p95_ms": decision_p95_ms,
        "step_count": int(step_decision_latency_ms.size),
        "step_mean_ms": step_mean_ms,
        "step_std_ms": step_std_ms,
        "step_p95_ms": step_p95_ms,
        "step_overrun_count": step_overrun_count,
        "step_overrun_ratio": float(step_overrun_count / max(int(step_decision_latency_ms.size), 1)),
    }

    num_drones, total_steps = X.shape
    if collect:
        X_list = [X[:, k].copy() for k in range(0, total_steps, stride)]
        Y_list = [Y[:, k].copy() for k in range(0, total_steps, stride)]
    else:
        X_list, Y_list = [], []

    traj = {}
    if collect:
        traj = {
            "X": X,
            "Y": Y,
            "Z": Z,
            "Vx": Vx,
            "Vy": Vy,
        }
        if obs_hist is not None:
            traj["obs_hist"] = obs_hist
        if "t" in results:
            traj["t"] = results["t"]

    # 验收：打印固定维度 feature 的形状（只打一次）
    if collect and num_drones > 0 and total_steps > 0:
        P0 = np.stack([X[:, 0], Y[:, 0], Z[:, 0]], axis=1)
        neighbors_idx = neighbors_within(P0, 0, sp_local.R1_comm)
        neighbors = np.array([
            [X[j, 0], Y[j, 0], Vx[j, 0], Vy[j, 0]] for j in neighbors_idx
        ], dtype=float)
        if obs_hist is not None and obs_hist.size > 0:
            obs0 = obs_hist[:, 0, :]
        else:
            obs0 = np.asarray(sp_local.obstacles, dtype=float)
        vxy = np.asarray(sp_local.obstacles_vxy, dtype=float)
        if vxy.ndim == 1 and vxy.size == 2:
            vxy = np.tile(vxy, (obs0.shape[0], 1))
        if vxy.shape == (obs0.shape[0], 2):
            obs0 = np.hstack([obs0, vxy])
        feat = extract_features(
            pos_i=P0[0, :],
            vel_i=np.array([Vx[0, 0], Vy[0, 0]], dtype=float),
            psi_i=0.0,
            lamb_i=0.0,
            neighbors=neighbors,
            obstacles=obs0,
            ve_xy=sp_local.ve_xy,
        )
        print(f"[feature] shape={feat.shape}", flush=True)

    # safety check
    collision_obs = False
    collision_obs_hard = False
    collision_nbr = False
    obs_count_total = 0
    obs_count_steps = 0
    # safety-margin check with a small numerical tolerance band
    soft_band = float(getattr(sp_local, "safe_soft_band", 1.0))
    safe_clearance = float(getattr(sp_local, "safe_clearance_m", 0.0))
    for k in range(max(0, total_steps - 1)):
        pos0 = np.stack([X[:, k], Y[:, k]], axis=1)
        pos1 = np.stack([X[:, k + 1], Y[:, k + 1]], axis=1)
        if obs_hist is not None and obs_hist.size > 0:
            obs_k = obs_hist[:, k, :]
        else:
            obs_k = np.asarray(sp_local.obstacles, dtype=float)
        if obs_k.size > 0:
            band = max(0.0, min(soft_band, float(sp_local.R2_lim) - 1e-6))
            hard_lim = obs_k[:, 2]
            if safe_clearance > 0.0:
                soft_lim = obs_k[:, 2] + safe_clearance
            else:
                soft_lim = obs_k[:, 2] + (float(sp_local.R2_lim) - band)
            centers = obs_k[:, :2]
            for i in range(num_drones):
                dist_seg = _seg_point_dists(pos0[i, :], pos1[i, :], centers)
                hard_hit = bool(np.any(dist_seg < hard_lim))
                soft_hit = bool(np.any(dist_seg < soft_lim))
                if hard_hit:
                    collision_obs_hard = True
                if hard_hit or soft_hit:
                    collision_obs = True
                    break
                dist0 = np.hypot(centers[:, 0] - pos0[i, 0], centers[:, 1] - pos0[i, 1])
                obs_count_total += int(np.sum(dist0 < (obs_k[:, 2] + sp_local.R2_comm)))
                obs_count_steps += 1
            if collision_obs:
                break
        for i in range(num_drones):
            for j in range(i + 1, num_drones):
                if math.hypot(pos0[i, 0] - pos0[j, 0], pos0[i, 1] - pos0[j, 1]) < sp_local.R1_lim:
                    collision_nbr = True
                    break
            if collision_nbr:
                break
        if collision_nbr:
            break

    obs_count_mean = float(obs_count_total / obs_count_steps) if obs_count_steps > 0 else 0.0
    safe_ok = not (collision_obs or collision_nbr)

    # formation metrics: dynamic nearest-neighbor distance keeping
    k_n = int(getattr(sp_local, "formation_k_neighbors", 3))
    err_mean_list = []
    err_max_list = []
    for k in range(total_steps):
        Pk = np.stack([X[:, k], Y[:, k]], axis=1)
        e_list = []
        for i in range(num_drones):
            dists = np.linalg.norm(Pk - Pk[i, :], axis=1)
            dists[i] = np.inf
            order = np.argsort(dists)
            for j in order[:k_n]:
                if not np.isfinite(dists[j]):
                    continue
                e_list.append(abs(dists[j] - sp_local.R_desire))
        if e_list:
            err_mean_list.append(float(np.mean(e_list)))
            err_max_list.append(float(np.max(e_list)))

    if err_mean_list:
        err_mean = float(np.mean(err_mean_list))
        err_max = float(np.max(err_max_list))
        mean_thresh = float(getattr(sp_local, "formation_err_mean_thresh", 2.0))
        max_thresh = float(getattr(sp_local, "formation_err_max_thresh", 4.0))
        over_limit_ratio = float(np.mean(
            (np.asarray(err_mean_list) > mean_thresh) | (np.asarray(err_max_list) > max_thresh)
        ))
    else:
        err_mean = err_max = over_limit_ratio = 0.0
        mean_thresh = float(getattr(sp_local, "formation_err_mean_thresh", 2.0))
        max_thresh = float(getattr(sp_local, "formation_err_max_thresh", 4.0))

    formation_metrics = {
        "mean_err": err_mean,
        "max_err": err_max,
        "over_limit_ratio": over_limit_ratio,
        "mean_thresh": mean_thresh,
        "max_thresh": max_thresh,
    }

    # reach check (可选)
    reach_ok = True
    if scene_cfg and "reach_point" in scene_cfg:
        reach_point = np.asarray(scene_cfg["reach_point"], dtype=float)
        reach_radius = float(scene_cfg.get("reach_radius", 5.0))
        final_mean = np.array([np.mean(X[:, -1]), np.mean(Y[:, -1])], dtype=float)
        reach_ok = bool(np.linalg.norm(final_mean - reach_point[:2]) <= reach_radius)

    formation_over_limit = float(scene_cfg.get("formation_over_limit", 0.20)) if scene_cfg else 0.20
    formation_ok = bool(
        (err_mean <= mean_thresh)
        and (err_max <= max_thresh)
        and (over_limit_ratio <= formation_over_limit)
    )

    return {
        "X_list": X_list,
        "Y_list": Y_list,
        "traj": traj,
        "cost_flags": {
            "c3_hard_any": collision_obs_hard,
            "c3_margin_any": collision_obs,
            "c3_any": collision_obs,
            "c4_any": collision_nbr,
            "obs_count_mean": obs_count_mean,
        },
        "formation_metrics": formation_metrics,
        "safe_ok": safe_ok,
        "reach_ok": reach_ok,
        "formation_ok": formation_ok,
        "latency": latency_summary,
    }


def update_drone_state(P: np.ndarray, V_xy: np.ndarray, psi: float, lamb: float,
                       u_prime: np.ndarray, sp: SimParams) -> Tuple[np.ndarray, np.ndarray, float, float, Dict[str, int]]:
    """
    Eq.(25)(26)(3)：控制输入 + 自动驾驶仪动力学的一拍推进。
    返回 flags 指示 dead-zone/saturation 是否触发，便于调试。
    """
    flags = {"u_dead": 0, "clipL": 0, "clipH": 0, "rate_clamp": 0, "snap_v": 0, "snap_psi": 0}

    u_xy = u_prime[:2].copy()
    if norm2(u_xy) < sp.u_lim:
        u_xy[:] = 0.0
        flags["u_dead"] = 1

    V_xy_mag = float(np.linalg.norm(V_xy))
    # 参考（Eq.(26)）
    V_xy_c = V_xy_mag + sp.T_v * (u_xy[0] * math.cos(psi) + u_xy[1] * math.sin(psi))
    if V_xy_mag < 0.1:
        psi_c = psi
    else:
        psi_c = psi + sp.T_psi / V_xy_mag * (-u_xy[0] * math.sin(psi) + u_xy[1] * math.cos(psi))

    h_c = P[2] + (sp.T_h / sp.T_lambda) * lamb + sp.T_h * u_prime[2]

    # 允许误差吸附（Step 11）
    ve_norm = float(np.linalg.norm(sp.ve_xy))
    if abs(V_xy_c - ve_norm) < sp.Vxy_c_lim:
        V_xy_c = ve_norm
        flags["snap_v"] = 1
    psi_m = math.atan2(sp.ve_xy[1], sp.ve_xy[0])
    if abs(psi_c - psi_m) < sp.psi_c_lim:
        psi_c = psi_m
        flags["snap_psi"] = 1

    # 动力学（Eq.(3) 离散化）
    V_xy_dot = (V_xy_c - V_xy_mag) / sp.T_v
    psi_dot = (psi_c - psi) / sp.T_psi
    lambda_dot = ((h_c - P[2]) / sp.T_h - lamb / sp.T_lambda)

    # 航向速率限（Eq.(4)）
    # |psi_dot| <= nmax*g/Vxy（避免除零）
    v_for_limit = max(V_xy_mag, 0.1)
    psi_dot_lim = sp.n_max * sp.g / v_for_limit
    if abs(psi_dot) > psi_dot_lim:
        psi_dot = math.copysign(psi_dot_lim, psi_dot)
        flags["rate_clamp"] = 1

    V_xy_mag_next = V_xy_mag + V_xy_dot * sp.dt
    psi_next = psi + psi_dot * sp.dt
    lambda_next = lamb + lambda_dot * sp.dt
    lambda_next = clamp(lambda_next, sp.lambda_min, sp.lambda_max)

    # 位置积分
    h_next = P[2] + lambda_next * sp.dt
    x_next = P[0] + V_xy_mag_next * math.cos(psi_next) * sp.dt
    y_next = P[1] + V_xy_mag_next * math.sin(psi_next) * sp.dt

    # 速度向量
    V_xy_next = np.array([V_xy_mag_next * math.cos(psi_next), V_xy_mag_next * math.sin(psi_next)], dtype=float)

    # 速度上下限（Eq.(4)）——仅用于调试标记，不强制裁剪状态（与 MATLAB 保持一致）
    if V_xy_mag_next < sp.Vxy_min:
        flags["clipL"] = 1
    if V_xy_mag_next > sp.Vxy_max:
        flags["clipH"] = 1

    P_next = np.array([x_next, y_next, h_next], dtype=float)
    return P_next, V_xy_next, psi_next, lambda_next, flags


def optimizePigeons_wrapper(sp: SimParams, P_swarm: np.ndarray, i_uav: int, V_xy_swarm: np.ndarray,
                            obstacles: np.ndarray, ff_raw: np.ndarray, fa_raw: np.ndarray, fc_raw: np.ndarray,
                            vf_z: float, vo_raw: np.ndarray, lamb_i: float, psi_i: float,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """初始化种群 → 计算首轮 cost → 进入核心 MPIO"""
    rng = _resolve_rng(sp, rng)

    X, V = initializePigeons(sp.N, rng)
    pop: List[Pigeon] = []

    if hasattr(sp, "last_w_valid") and hasattr(sp, "last_w"):
        if i_uav < sp.last_w_valid.shape[0] and sp.last_w_valid[i_uav]:
            w_prev = sp.last_w[i_uav].copy()
            vf_prev_xy = w_prev[0] * (ff_raw + fa_raw) + fc_raw
            vo_prev_xy = w_prev[1] * vo_raw
            u_total_prev = vf_prev_xy + vo_prev_xy
            u_prime_prev = np.array([u_total_prev[0] - V_xy_swarm[i_uav, 0],
                                     u_total_prev[1] - V_xy_swarm[i_uav, 1],
                                     vf_z], dtype=float)
            P_prev, V_prev, _, _, _ = update_drone_state(P_swarm[i_uav, :], V_xy_swarm[i_uav, :], psi_i, lamb_i,
                                                         u_prime_prev, sp)
            c1p, c2p, c3p, c4p = calculateCosts(P_prev, V_prev, P_swarm, i_uav, V_xy_swarm, obstacles, sp)
            infeasible_prev = (c3p == 1 or c4p == 1)
            if not infeasible_prev:
                pop.append(Pigeon(Position=w_prev, V=np.zeros(2, dtype=float),
                                  Cost12=np.array([c1p, c2p], dtype=float), Rank=0.0))

    for i in range(sp.N):
        w_p = X[i, :].copy()
        vf_prime_xy = w_p[0] * (ff_raw + fa_raw) + fc_raw
        vo_prime_xy = w_p[1] * vo_raw
        u_total_xy = vf_prime_xy + vo_prime_xy
        u_prime_p = np.array([u_total_xy[0] - V_xy_swarm[i_uav, 0],
                              u_total_xy[1] - V_xy_swarm[i_uav, 1],
                              vf_z], dtype=float)

        P_next, V_xy_next, _, _, _ = update_drone_state(P_swarm[i_uav, :], V_xy_swarm[i_uav, :], psi_i, lamb_i,
                                                        u_prime_p, sp)
        c1, c2, c3, c4 = calculateCosts(P_next, V_xy_next, P_swarm, i_uav, V_xy_swarm, obstacles, sp)
        infeasible = (c3 == 1 or c4 == 1)
        if infeasible:
            cost12 = np.array([np.inf, np.inf], dtype=float)
            rank = float("inf")
        else:
            cost12 = np.array([c1, c2], dtype=float)
            rank = 0.0
        pop.append(Pigeon(Position=w_p, V=V[i, :].copy(), Cost12=cost12, Rank=rank))

    if pop:
        valid0 = int(np.sum(np.all(np.isfinite(np.vstack([p.Cost12 for p in pop])), axis=1)))
    else:
        valid0 = 0
    if valid0 == 0:
        print(f"[MPIO] step infeasible: no valid candidates for uav={i_uav}", flush=True)

    return optimizePigeons_core(pop, sp, P_swarm, i_uav, V_xy_swarm, obstacles,
                                ff_raw, fa_raw, fc_raw, vf_z, vo_raw, lamb_i, psi_i, rng=rng)


def optimizePigeons_core(pop: List[Pigeon], sp: SimParams, P_swarm: np.ndarray, i_uav: int,
                         V_xy_swarm: np.ndarray, obstacles: np.ndarray, ff_raw: np.ndarray, fa_raw: np.ndarray,
                         fc_raw: np.ndarray, vf_z: float, vo_raw: np.ndarray, lamb_i: float, psi_i: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    修改版 MPIO（带层级学习/拥挤度/精英保留/逐步减员），与 MATLAB 实现保持一致。
    最终从 Pareto 前沿中选 cost2 最小的个体（Eq.(24)）。
    """
    rng = _resolve_rng(sp, rng)
    R_map = sp.R
    ft = sp.tr
    historical_A: List[Pigeon] = []
    archive_size = 50
    initial_valid0 = int(np.sum(np.all(np.isfinite(np.vstack([p.Cost12 for p in pop])),
                                       axis=1))) if pop else 0

    # 统一数组形式，便于广播
    ff_raw = np.array(ff_raw, dtype=float)
    fa_raw = np.array(fa_raw, dtype=float)
    fc_raw = np.array(fc_raw, dtype=float)
    vo_raw = np.array(vo_raw, dtype=float)

    for Nc in range(1, max(1, sp.Ncmax) + 1):
        # --- 非支配排序 & 拥挤度 ---
        pop = [p for p in pop if np.all(np.isfinite(p.Cost12))]
        if not pop:
            break
        Costs = np.vstack([p.Cost12 for p in pop])
        rank, fronts = nondominated_sort_safe(Costs)
        if not fronts or not fronts[0]:
            break
        for ii, p in enumerate(pop):
            p.Rank = float(rank[ii])
        pop = CalcCrowdingDistance(pop, fronts)

        current_S1_pop = [pop[idx] for idx in fronts[0]]

        # --- 历史存档 A 更新 ---
        if not historical_A:
            combined_A = current_S1_pop
        else:
            combined_A = list(historical_A) + list(current_S1_pop)

        if combined_A:
            A_Costs = np.vstack([p.Cost12 for p in combined_A])
            _, A_fronts = nondominated_sort_safe(A_Costs)
            historical_A = [combined_A[idx] for idx in (A_fronts[0] if A_fronts and A_fronts[0] else [])]
            if len(historical_A) > archive_size:

                # 重新计算拥挤度
                historical_A = CalcCrowdingDistance(historical_A, [list(range(len(historical_A)))])
                historical_A.sort(key=lambda pp: pp.CrowdingDistance)  # 升序
                historical_A = historical_A[-archive_size:]  # 保留拥挤度大的
        else:
            historical_A = []

        # --- 选择 Xg / Xc ---
        if historical_A:
            Xg = historical_A[rng.integers(0, len(historical_A))].Position.copy()
        else:
            Xg = current_S1_pop[rng.integers(0, len(current_S1_pop))].Position.copy()
        Xc = np.mean(np.vstack([p.Position for p in current_S1_pop]), axis=0)

        # --- 领导者数量 ---
        nPop = len(pop)
        nLeaders = max(1, min(nPop, int(math.ceil(sp.p1 * nPop))))

        # --- 逐个体更新 ---
        # 先排序：rank 小者在前，保证“上层可学”
        order = np.argsort([p.Rank for p in pop])
        pop = [pop[idx] for idx in order]

        for i in range(nPop):
            old = dataclasses.replace(pop[i])  # 备份
            # 领导者：地图-罗盘 + 地标（Eq.(19)(20)）
            if pop[i].Rank <= nLeaders:
                rand1 = rng.random()
                rand2 = rng.random()
                term1 = math.exp(-R_map * Nc) * pop[i].V
                distXg = np.linalg.norm(Xg - pop[i].Position)
                distXc = np.linalg.norm(Xc - pop[i].Position)
                lgXg = math.log(distXg + 1.0)
                lgXc = math.log(distXc + 1.0)
                Vi_new = term1 + rand1 * ft * (1.0 - lgXg) * (Xg - pop[i].Position) + rand2 * ft * lgXc * (Xc - pop[i].Position)
                Vi_new = np.clip(Vi_new, -0.2, 0.2)
                pop[i].V = Vi_new
                pop[i].Position = np.clip(pop[i].Position + Vi_new, 0.0, 1.0)
            else:
                # 跟随者：层级学习（Eq.(21)）
                upper_indices = [idx for idx in range(nPop) if pop[idx].Rank < pop[i].Rank]
                if not upper_indices:
                    pop[i].Position = np.clip(pop[i].Position + 0.01 * (2.0 * rng.random(2) - 1.0), 0.0, 1.0)
                    pop[i].V = 0.01 * (2.0 * rng.random(2) - 1.0)
                else:
                    j_idx = upper_indices[rng.integers(0, len(upper_indices))]
                    for _ in range(sp.sl):
                        d_star = rng.integers(0, pop[i].Position.size)
                        pop[i].Position[d_star] = pop[j_idx].Position[d_star] + sp.e_learn * (2.0 * rng.random() - 1.0)
                        pop[i].Position[d_star] = clamp(pop[i].Position[d_star], 0.0, 1.0)
                    pop[i].V = np.clip(pop[i].Position - old.Position, -0.2, 0.2)

            # --- 重新评估 ---
            w_p = pop[i].Position
            vf_prime_xy = w_p[0] * (ff_raw + fa_raw) + fc_raw
            vo_prime_xy = w_p[1] * vo_raw
            u_total_xy = vf_prime_xy + vo_prime_xy
            u_prime_p = np.array([u_total_xy[0] - V_xy_swarm[i_uav, 0],
                                  u_total_xy[1] - V_xy_swarm[i_uav, 1],
                                  vf_z], dtype=float)
            P_next, V_xy_next, _, _, _ = update_drone_state(P_swarm[i_uav, :], V_xy_swarm[i_uav, :], psi_i, lamb_i,
                                                            u_prime_p, sp)
            c1, c2, c3, c4 = calculateCosts(P_next, V_xy_next, P_swarm, i_uav, V_xy_swarm, obstacles, sp)
            if c3 == 1 or c4 == 1:
                new_cost = np.array([np.inf, np.inf], dtype=float)
            else:
                new_cost = np.array([c1, c2], dtype=float)

            # 精英保留：若新解被旧解支配，则回滚
            is_new_dominated = (np.all(old.Cost12 <= new_cost) and np.any(old.Cost12 < new_cost))
            if is_new_dominated:
                pop[i] = old
            else:
                pop[i].Cost12 = new_cost

        # --- 减员 ---
        if Nc <= sp.Ncmax and len(pop) > sp.Nd:
            Costs_after = np.vstack([p.Cost12 for p in pop])
            rank_after, fronts_after = nondominated_sort_safe(Costs_after)
            for ii, p in enumerate(pop):
                p.Rank = float(rank_after[ii])
            pop = CalcCrowdingDistance(pop, fronts_after)

            # 按 [Rank 升序, -CrowdingDistance 降序] 排序，保留前 K
            sort_keys = np.lexsort((-np.array([p.CrowdingDistance for p in pop]), np.array([p.Rank for p in pop])))
            current_N = len(pop)
            keep = max(2, current_N - sp.Nd)
            pop = [pop[idx] for idx in sort_keys[:keep]]

    validA = int(np.sum(np.all(np.isfinite(np.vstack([p.Cost12 for p in historical_A])),
                               axis=1))) if historical_A else 0
    if bool(getattr(sp, "verbose_mpio", False)):
        print(f"[mpio] uav={i_uav} valid0={initial_valid0} validA={validA}", flush=True)

    if not pop:
        w_hist = _pick_from_archive(historical_A)
        if w_hist is not None:
            return w_hist
        return np.array([0.2, 0.8], dtype=float)

    Costs = np.vstack([p.Cost12 for p in pop])
    _, frontsF = nondominated_sort_safe(Costs)
    if not frontsF or not frontsF[0]:
        w_hist = _pick_from_archive(historical_A)
        if w_hist is not None:
            return w_hist
        return np.array([0.2, 0.8], dtype=float)

    s1_pop = [pop[idx] for idx in frontsF[0]]
    s1_costs = np.vstack([p.Cost12 for p in s1_pop])
    valid_rows = np.all(np.isfinite(s1_costs), axis=1)
    if not np.any(valid_rows):
        w_hist = _pick_from_archive(historical_A)
        if w_hist is not None:
            return w_hist
        return np.array([0.2, 0.8], dtype=float)

    # Eq.(24)：在前沿上选 cost2 最小
    valid_costs = s1_costs[valid_rows, :]
    valid_pop = [p for (p, ok) in zip(s1_pop, valid_rows) if ok]
    best_idx = int(np.argmin(valid_costs[:, 1]))
    w = valid_pop[best_idx].Position.copy()
    return np.clip(w.astype(float), 0.0, 1.0)


# ---------------------------- 主仿真 ----------------------------

def run_simulation(sp: Optional[SimParams] = None,
                   enable_plot: bool = True,
                   live_plot: bool = False,
                   csv_path: Optional[str] = None,
                   policy_fn: Optional[Callable[..., Optional[np.ndarray]]] = None,
                   rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """
    运行完整仿真（不依赖 AirSim），结果与 MATLAB 仿真保持一致。
    返回：包含轨迹/速度/航向等数组；若 enable_plot=True 则绘图。
    """
    if sp is None:
        sp = SimParams()
    if csv_path is None:
        csv_path = sp.csv_path

    rng = _resolve_rng(sp, rng)
    setattr(sp, "_rng", rng)

    # 初值
    P = sp.init_P.copy()
    V_xy = sp.init_Vxy.copy()
    psi = np.zeros(P.shape[0], dtype=float)
    lamb = np.zeros(P.shape[0], dtype=float)
    _append_square_as_circle(sp)
    obs_state = _init_obstacle_state(sp)

    num_drones = P.shape[0]
    sp.last_w = np.full((num_drones, 2), [0.2, 0.8], dtype=float)
    sp.last_w_valid = np.zeros((num_drones,), dtype=bool)
    time_vector = np.arange(0.0, sp.sim_time + 1e-9, sp.dt)
    total_steps = time_vector.size
    n_obs = obs_state.shape[0]
    obs_hist = np.zeros((n_obs, total_steps, 3), dtype=float)
    obs_hist[:, 0, :] = obs_state[:, :3]
    decision_latency_ms: List[float] = []
    step_decision_latency_ms: List[float] = []

    # 轨迹缓存
    X_uav = np.zeros((num_drones, total_steps), dtype=float)
    Y_uav = np.zeros((num_drones, total_steps), dtype=float)
    Z_uav = np.zeros((num_drones, total_steps), dtype=float)
    Vx_uav = np.zeros((num_drones, total_steps), dtype=float)
    Vy_uav = np.zeros((num_drones, total_steps), dtype=float)
    X_uav[:, 0] = P[:, 0]
    Y_uav[:, 0] = P[:, 1]
    Z_uav[:, 0] = P[:, 2]
    Vx_uav[:, 0] = V_xy[:, 0]
    Vy_uav[:, 0] = V_xy[:, 1]
    # 动画窗口（可选）
    live_state = None
    if enable_plot and live_plot and _HAS_MPL:
        live_state = _init_live_figure(num_drones, sp)


    # CSV 调试日志
    csv_file = None
    csv_writer = None
    if csv_path:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "t", "uav", "x", "y", "vx", "vy", "psi_i", "psi_m", "psi_c", "u_dead", "rate_clamp",
            "vxy_mag", "clipL", "clipH", "w1", "w2", "vo_x", "vo_y", "nbr", "obs",
            "c1", "c2", "c3", "c4", "theta_e", "yaw", "d_min", "alpha", "sensed_r2",
            "blocked", "gap_score", "gap_clear",
            "gap_type", "blocked_cnt", "gap_cnt", "gap_left", "gap_right", "gap_width", "local_angle"
        ])
    debug_snake = bool(getattr(sp, "debug_snake", False))
    debug_uav = int(getattr(sp, "debug_uav", 0))
    debug_steps = int(getattr(sp, "debug_steps", 10))
    debug_scene = str(getattr(sp, "scene_type", "")) == "snake_corridor"
    debug_active = debug_snake and debug_scene
    need_debug = (csv_writer is not None) or debug_active

    # 主循环
    for k in range(total_steps - 1):
        step_t0 = time.perf_counter()
        if (k % 20) == 0:
            print(f"[progress] k={k + 1}/{total_steps - 1}", flush=True)
        step_decision_ms = 0.0

        t = time_vector[k]

        # --- 同步更新（避免在同一拍内“先更新的 UAV 影响后更新的 UAV”） ---
        # 在分布式控制中，每个 UAV 的决策应基于同一时刻的观测/通信信息。
        # 因此这里对状态做快照：所有 UAV 都用 (P_k, V_k, psi_k, lamb_k) 计算下一拍，
        # 再统一写回到 (P, V_xy, psi, lamb)。
        P_k = P.copy()
        V_k = V_xy.copy()
        psi_k = psi.copy()
        lamb_k = lamb.copy()
        if sp.obstacles_use_prediction:
            obs_pred = obs_state.copy()
            obs_pred[:, 0] += obs_pred[:, 3] * sp.dt
            obs_pred[:, 1] += obs_pred[:, 4] * sp.dt
            obstacles_plan = obs_pred[:, :3]
        else:
            obstacles_plan = obs_state[:, :3]

        shared_yaw = None
        shared_sensed = []
        shared_dbg = None
        if bool(getattr(sp, "swarm_yaw_enable", False)):
            fwd = unit(np.array(sp.ve_xy, dtype=float))
            if norm2(fwd) < 1e-6:
                fwd = np.array([1.0, 0.0], dtype=float)
            proj = P_k[:, 0] * fwd[0] + P_k[:, 1] * fwd[1]
            k_front = int(getattr(sp, "swarm_yaw_k", 3))
            k_front = max(1, min(k_front, num_drones))
            idx = np.argsort(proj)[-k_front:]
            rep_pos = np.mean(P_k[idx, :2], axis=0)
            rep_z = float(np.mean(P_k[idx, 2])) if P_k.shape[1] > 2 else 0.0
            rep_vel = np.mean(V_k[idx, :], axis=0)
            rep_v_mag = norm2(rep_vel)
            if rep_v_mag < 0.5:
                rep_v_mag = norm2(np.array(sp.ve_xy, dtype=float))
            P_rep = np.array([rep_pos[0], rep_pos[1], rep_z], dtype=float)
            theta_e_swarm = math.atan2(sp.ve_xy[1], sp.ve_xy[0])
            if need_debug:
                shared_yaw, shared_sensed, shared_dbg = obstacle_avoidance_uav(
                    P_rep, theta_e_swarm, obstacles_plan, sp, return_debug=True, v_mag=rep_v_mag, v_xy=rep_vel
                )
            else:
                shared_yaw, shared_sensed = obstacle_avoidance_uav(
                    P_rep, theta_e_swarm, obstacles_plan, sp, v_mag=rep_v_mag, v_xy=rep_vel
                )
        P_next = np.zeros_like(P_k)
        V_next = np.zeros_like(V_k)
        psi_next = np.zeros_like(psi_k)
        lamb_next = np.zeros_like(lamb_k)

        for i in range(num_drones):
            pos_i_cur = P_k[i, :2].copy()
            # --- 原始力 ---
            ff_raw, fa_raw, fc_raw = compute_raw_forces(P_k, V_k, i, sp)
            vf_z = sp.Ka_he * (sp.he - P_k[i, 2]) + sp.Kve * (sp.ve3 - lamb_k[i])

            # --- 避障期望速度（Eq.(11)）---
            theta_e = math.atan2(sp.ve_xy[1], sp.ve_xy[0])
            if shared_yaw is not None:
                yaw_desired = shared_yaw
                sensed_idx = shared_sensed
                avoid_dbg = shared_dbg if need_debug else None
            else:
                v_mag = norm2(V_k[i, :])
                if need_debug:
                    yaw_desired, sensed_idx, avoid_dbg = obstacle_avoidance_uav(
                        P_k[i, :], theta_e, obstacles_plan, sp, return_debug=True, v_mag=v_mag, v_xy=V_k[i, :]
                    )
                else:
                    yaw_desired, sensed_idx = obstacle_avoidance_uav(
                        P_k[i, :], theta_e, obstacles_plan, sp, v_mag=v_mag, v_xy=V_k[i, :]
                    )
                    avoid_dbg = None
            yaw_desired = _rate_limit_angle(yaw_desired, psi_k[i], sp.yaw_rate_max, sp.dt)

            vo_raw = np.array([np.linalg.norm(sp.ve_xy) * math.cos(yaw_desired),
                               np.linalg.norm(sp.ve_xy) * math.sin(yaw_desired)], dtype=float)
            if avoid_dbg is not None and bool(avoid_dbg.get("blocked", False)):
                scale = float(getattr(sp, "blocked_speed_scale", 0.6))
                vo_raw *= scale

            # --- MPIO / Policy 搜索权重 w=[w1,w2] ---
            t_decision0 = time.perf_counter()
            if policy_fn is None:
                w = optimizePigeons_wrapper(sp, P_k, i, V_k, obstacles_plan,
                                            ff_raw, fa_raw, fc_raw, vf_z, vo_raw, lamb_k[i], psi_k[i], rng=rng)
            else:
                w = policy_fn(i, P_k, V_k, psi_k, lamb_k, obs_state, obstacles_plan, sp,
                              ff_raw, fa_raw, fc_raw, vf_z, vo_raw, k, t)
                if w is None:
                    w = optimizePigeons_wrapper(sp, P_k, i, V_k, obstacles_plan,
                                                ff_raw, fa_raw, fc_raw, vf_z, vo_raw, lamb_k[i], psi_k[i], rng=rng)
            decision_ms = (time.perf_counter() - t_decision0) * 1000.0
            decision_latency_ms.append(decision_ms)
            step_decision_ms += decision_ms

            # --- 合成控制输入 ---
            vf_prime_xy = w[0] * (ff_raw + fa_raw) + fc_raw
            vo_prime_xy = w[1] * vo_raw
            u_total_xy = vf_prime_xy + vo_prime_xy

            # 速度增量限（最大加速度：n_max * g）（与 MATLAB 主循环一致）
            max_delta = sp.n_max * sp.g * sp.dt
            cur_norm = norm2(V_k[i, :])
            des_norm = norm2(u_total_xy)
            if des_norm > cur_norm + max_delta:
                u_total_xy = u_total_xy / (des_norm + 1e-12) * (cur_norm + max_delta)
            elif des_norm < cur_norm - max_delta:
                u_total_xy = u_total_xy / (des_norm + 1e-12) * (cur_norm - max_delta)

            # Step 11 → u' = [u_total_xy - V_xy_i, vf_z]
            u_prime = np.array([u_total_xy[0] - V_k[i, 0],
                                u_total_xy[1] - V_k[i, 1],
                                vf_z], dtype=float)

            # 推进一拍（Eq.(3)）——基于快照状态
            P_i_next, V_xy_i_next, psi_i_next, lamb_i_next, flags = update_drone_state(
                P_k[i, :], V_k[i, :], psi_k[i], lamb_k[i], u_prime, sp
            )
            P_next[i, :] = P_i_next
            V_next[i, :] = V_xy_i_next
            psi_next[i] = psi_i_next
            lamb_next[i] = lamb_i_next


            # 调试：成本评估（用下一拍位置/速度，参照 MATLAB 在 MPIO 内部的做法）
            # 调试：成本评估（用下一拍位置/速度；邻居/障碍使用同一拍的快照/规划视角）
            c1, c2, c3, c4 = calculateCosts(P_i_next, V_xy_i_next, P_k, i, V_k, obstacles_plan, sp)
            if (c3 == 0) and (c4 == 0):
                sp.last_w[i] = w.copy()
                sp.last_w_valid[i] = True

            if debug_active and i == debug_uav and k < debug_steps:
                buffer = float(getattr(sp, "cost1_buffer", 3.0))
                sigma = float(getattr(sp, "cost1_sigma", 8.0))
                d_min = float("inf")
                if obstacles_plan.size > 0:
                    d_vec = obstacles_plan[:, :2] - pos_i_cur[None, :]
                    d_center = np.linalg.norm(d_vec, axis=1)
                    d = d_center - (obstacles_plan[:, 2] + sp.R2_lim + buffer)
                    d_min = float(np.min(d))
                alpha = float(np.clip(1.0 - d_min / sigma, 0.0, 1.0)) if math.isfinite(d_min) else 0.0
                sensed_r2 = 0
                if obstacles_plan.size > 0:
                    d_next = np.linalg.norm(obstacles_plan[:, :2] - P_i_next[:2], axis=1)
                    sensed_r2 = int(np.any(d_next < (obstacles_plan[:, 2] + sp.R2_comm)))
                print(
                    "[snake-debug] k={k} t={t:.2f} uav={i} sensed={sensed} sensed_r2={sensed_r2} "
                    "d_min={d_min:.2f} alpha={alpha:.2f} theta_e={theta_e:.2f} yaw={yaw:.2f} "
                    "w=({w0:.2f},{w1:.2f}) c1={c1:.2f} c2={c2:.2f} c3={c3} c4={c4}".format(
                        k=k,
                        t=t,
                        i=i,
                        sensed=len(sensed_idx),
                        sensed_r2=sensed_r2,
                        d_min=d_min,
                        alpha=alpha,
                        theta_e=theta_e,
                        yaw=yaw_desired,
                        w0=w[0],
                        w1=w[1],
                        c1=c1,
                        c2=c2,
                        c3=c3,
                        c4=c4,
                    ),
                    flush=True,
                )

            # 记录轨迹
            # 轨迹统一在本拍结束后整体写入（同步更新）

            # UAV1 日志（与用户提供的格式对应）
            if sp.verbose_uav1 and i == 0:
                vxy_mag = norm2(V_xy_i_next)
                psi_m = theta_e
                print(f"[t={t:5.1f}] UAV1 psi_i={psi_i_next:+6.3f}, psi_m={psi_m:+6.3f}, psi_c={'N/A' if True else '':s} "
                      f"(dead={'Y' if flags['u_dead'] else 'n'}, rate={'Y' if flags['rate_clamp'] else 'n'}); "
                      f"vxy={vxy_mag:4.2f} (dead={'n'}, clipL={'Y' if flags['clipL'] else 'n'}, clipH={'Y' if flags['clipH'] else 'n'}); "
                      f"w=[{w[0]:.2f},{w[1]:.2f}] vo=({vo_raw[0]:+5.2f},{vo_raw[1]:+5.2f}); "
                      f"nbr={len(neighbors_within(P_k, i, sp.R1_comm))}, obs={len(sensed_idx)}")

            # CSV
            if csv_writer is not None:
                d_min = float("inf")
                alpha = 0.0
                sensed_r2 = 0
                if obstacles_plan.size > 0:
                    buffer = float(getattr(sp, "cost1_buffer", 3.0))
                    sigma = float(getattr(sp, "cost1_sigma", 8.0))
                    d_center = np.linalg.norm(obstacles_plan[:, :2] - P_i_next[:2], axis=1)
                    d = d_center - (obstacles_plan[:, 2] + sp.R2_lim + buffer)
                    d_min = float(np.min(d))
                    if sigma > 1e-9:
                        alpha = float(np.clip(1.0 - d_min / sigma, 0.0, 1.0))
                    sensed_r2 = int(np.any(d_center < (obstacles_plan[:, 2] + sp.R2_comm)))
                gap_type = ""
                gap_cnt = ""
                blocked_cnt = ""
                gap_left = ""
                gap_right = ""
                gap_width = ""
                local_angle = ""
                if avoid_dbg is not None:
                    gap_type = avoid_dbg.get("gap_type", "")
                    blocked_cnt = avoid_dbg.get("blocked_cnt", "")
                    gap_cnt = avoid_dbg.get("gap_cnt", "")
                    gap_left = avoid_dbg.get("gap_left", "")
                    gap_right = avoid_dbg.get("gap_right", "")
                    gap_width = avoid_dbg.get("gap_width", "")
                    local_angle = avoid_dbg.get("local_angle", "")
                    blocked_flag = int(bool(avoid_dbg.get("blocked", False)))
                    gap_score = avoid_dbg.get("gap_score", "")
                    gap_clear = avoid_dbg.get("gap_clear", "")
                else:
                    blocked_flag = 0
                    gap_score = ""
                    gap_clear = ""
                csv_writer.writerow([
                    f"{t:.2f}", i + 1, f"{P_i_next[0]:.6f}", f"{P_i_next[1]:.6f}",
                    f"{V_xy_i_next[0]:.6f}", f"{V_xy_i_next[1]:.6f}",
                    f"{psi_i_next:.6f}", f"{theta_e:.6f}", "",  # psi_c 在 update 内部，不单独存
                    flags["u_dead"], flags["rate_clamp"],
                    f"{norm2(V_xy_i_next):.6f}", flags["clipL"], flags["clipH"],
                    f"{w[0]:.6f}", f"{w[1]:.6f}", f"{vo_raw[0]:.6f}", f"{vo_raw[1]:.6f}",
                    len(neighbors_within(P_k, i, sp.R1_comm)), len(sensed_idx),
                    f"{c1:.6f}", f"{c2:.6f}", c3, c4,
                    f"{theta_e:.6f}", f"{yaw_desired:.6f}", f"{d_min:.6f}",
                    f"{alpha:.6f}", sensed_r2,
                    blocked_flag, gap_score, gap_clear,
                    gap_type, blocked_cnt, gap_cnt,
                    f"{gap_left}" if gap_left != "" else "",
                    f"{gap_right}" if gap_right != "" else "",
                    f"{gap_width}" if gap_width != "" else "",
                    f"{local_angle}" if local_angle != "" else "",
                ])

        # 同步写回本拍所有 UAV 的状态
        P = P_next
        V_xy = V_next
        psi = psi_next
        lamb = lamb_next

        # 记录轨迹
        X_uav[:, k + 1] = P[:, 0]
        Y_uav[:, k + 1] = P[:, 1]
        Z_uav[:, k + 1] = P[:, 2]
        Vx_uav[:, k + 1] = V_xy[:, 0]
        Vy_uav[:, k + 1] = V_xy[:, 1]
        _update_obstacle_state(obs_state, sp)
        obs_hist[:, k + 1, :] = obs_state[:, :3]
        step_decision_latency_ms.append(step_decision_ms)

        # ✅ 实时刷新（放在 for k 循环内！）
        if live_state is not None and ((k % 5) == 0 or k == total_steps - 2):
            _update_live_figure(
                live_state,
                time_vector[k + 1],
                X_uav[:, : (k + 2)],
                Y_uav[:, : (k + 2)],
                obs_state,
            )
        if live_state is not None:
            plt.pause(0.001)

    if csv_file is not None:
        csv_file.close()

    results = {
        "t": time_vector,
        "X": X_uav,
        "Y": Y_uav,
        "Z": Z_uav,
        "Vx": Vx_uav,
        "Vy": Vy_uav,
        "psi": psi,
        "lambda": lamb,
        "obs_hist": obs_hist,
        "decision_latency_ms": np.asarray(decision_latency_ms, dtype=np.float32),
        "step_decision_latency_ms": np.asarray(step_decision_latency_ms, dtype=np.float32),
    }

    if enable_plot and _HAS_MPL:
        plt.ioff()
        _plot_results(results, sp, obs_hist=obs_hist)

    return results


def _plot_results(results: Dict[str, np.ndarray], sp, obs_hist: Optional[np.ndarray] = None) -> None:
    """
    优化后的可视化布局：
    上方：宽屏显示 2D 轨迹（解决比例压缩问题）。
    下方：并排显示 4 个状态曲线。
    """
    t = results["t"]
    X, Y, Z = results["X"], results["Y"], results["Z"]
    Vx, Vy = results["Vx"], results["Vy"]
    v_mag = np.hypot(Vx, Vy)
    yaw = np.arctan2(Vy, Vx)

    # 1. 调整画布大小，使其更宽
    fig = plt.figure(figsize=(15, 9))

    # 2. 使用 subplot2grid 进行更灵活的布局
    # 网格定义为：3行 x 4列

    # 轨迹图：占据前两行(rowspan=2)，横跨所有列(colspan=4) -> 这是一个宽屏区域
    ax_traj = plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=4)

    # 下方四个小图：占据最后一行，各占1列
    ax_spd = plt.subplot2grid((3, 4), (2, 0))
    ax_alt = plt.subplot2grid((3, 4), (2, 1))
    ax_dzdt = plt.subplot2grid((3, 4), (2, 2))
    ax_yaw = plt.subplot2grid((3, 4), (2, 3))

    # --- 绘制轨迹 ---
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(X.shape[0]):
        color = colors[i % len(colors)]
        ax_traj.plot(X[i, :], Y[i, :], linewidth=2, label=f"UAV {i + 1}", color=color)
        # 起点
        ax_traj.scatter(X[i, 0], Y[i, 0], s=30, marker='o', color=color)
        # 终点
        ax_traj.scatter(X[i, -1], Y[i, -1], s=40, marker='s', color=color)

    # 绘制障碍物（轨迹 + 终点圆；可选初始圆）
    if obs_hist is not None and obs_hist.size > 0:
        square_j = getattr(sp, "square_idx", None)
        for j in range(obs_hist.shape[0]):
            ax_traj.plot(obs_hist[j, :, 0], obs_hist[j, :, 1], linestyle="--", linewidth=1)
            if (square_j is not None) and (j == square_j):
                continue
            x_end, y_end, r_end = obs_hist[j, -1, :]
            circle_end = plt.Circle((x_end, y_end), r_end, alpha=0.4, edgecolor='k', facecolor='gray')
            ax_traj.add_patch(circle_end)
            x0, y0, r0 = obs_hist[j, 0, :]
            circle_start = plt.Circle((x0, y0), r0, alpha=0.15, edgecolor='k', facecolor='gray')
            ax_traj.add_patch(circle_start)
        if square_j is not None:
            a = float(sp.square_side)
            cx, cy = obs_hist[square_j, -1, 0], obs_hist[square_j, -1, 1]
            rect_end = Rectangle((cx - a / 2.0, cy - a / 2.0), a, a, fill=False, linewidth=2)
            ax_traj.add_patch(rect_end)
            cx0, cy0 = obs_hist[square_j, 0, 0], obs_hist[square_j, 0, 1]
            rect_start = Rectangle((cx0 - a / 2.0, cy0 - a / 2.0), a, a, fill=False, linewidth=1,
                                   linestyle="--", alpha=0.4)
            ax_traj.add_patch(rect_start)
    else:
        for (x, y, r) in sp.obstacles:
            circle = plt.Circle((x, y), r, alpha=0.3, edgecolor='k', facecolor='gray')
            ax_traj.add_patch(circle)

    # 关键设置：因为区域现在是宽的，'equal' 比例不会再导致图像被压扁
    ax_traj.set_aspect('equal', adjustable='datalim')
    ax_traj.grid(True, linestyle=':', alpha=0.6)
    ax_traj.set_title("2D Trajectories (Top View)", fontsize=12, fontweight='bold')
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # --- 绘制下方数据曲线 ---

    # 1. 速度
    for i in range(v_mag.shape[0]):
        ax_spd.plot(t, v_mag[i, :], linewidth=1.2)
    ax_spd.axhline(sp.Vxy_min, linestyle="--", color='gray', linewidth=0.8)
    ax_spd.axhline(sp.Vxy_max, linestyle="--", color='gray', linewidth=0.8)
    ax_spd.set_title("Horiz. Speed (m/s)")
    ax_spd.set_xlabel("Time (s)")
    ax_spd.grid(True)

    # 2. 高度
    for i in range(Z.shape[0]):
        ax_alt.plot(t, Z[i, :], linewidth=1.2)
    ax_alt.axhline(sp.he, linestyle="--", color='gray', linewidth=0.8)
    ax_alt.set_title("Altitude (m)")
    ax_alt.set_xlabel("Time (s)")
    ax_alt.grid(True)

    # 3. 升降率
    if Z.shape[1] >= 2:
        dzdt = np.diff(Z, axis=1) / (t[1] - t[0])
        # 补齐长度以便绘图
        dzdt = np.column_stack([np.zeros(Z.shape[0]), dzdt])
        for i in range(Z.shape[0]):
            ax_dzdt.plot(t, dzdt[i, :], linewidth=1.0)
    ax_dzdt.set_title("Vertical Rate (m/s)")
    ax_dzdt.set_xlabel("Time (s)")
    ax_dzdt.grid(True)

    # 4. 航向
    for i in range(yaw.shape[0]):
        ax_yaw.plot(t, yaw[i, :], linewidth=1.0)
    ax_yaw.set_title("Yaw (rad)")
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.grid(True)

    plt.tight_layout()
    plt.show()
def _init_live_figure(num_drones: int, sp: "SimParams"):
    """创建简单 2D 轨迹窗口用于边跑边画（更稳：show + draw + flush）"""
    if not _HAS_MPL:
        return None

    plt.ion()  # 打开交互模式
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("UAV Swarm – 2D Trajectories (live)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    traj_lines = []
    start_scats = []
    end_scats = []

    for i in range(num_drones):
        (ln,) = ax.plot([], [], linewidth=1.5, label=f"UAV{i+1}")
        traj_lines.append(ln)
        st = ax.scatter([], [], s=30, marker="o")
        ed = ax.scatter([], [], s=50, marker="x")
        start_scats.append(st)
        end_scats.append(ed)

    # 画一下障碍物（2D 圆）
    obs_patches = []
    if hasattr(sp, "obstacles") and sp.obstacles is not None:
        for (ox, oy, r) in sp.obstacles:
            circ = plt.Circle((ox, oy), r, fill=False)
            ax.add_patch(circ)
            obs_patches.append(circ)

    ax.legend(loc="best", fontsize=9)

    # ✅ 关键：先 show，再 draw/flush，避免窗口白屏
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

    return {
        "fig": fig,
        "ax": ax,
        "traj_lines": traj_lines,
        "start_scats": start_scats,
        "end_scats": end_scats,
        "obs_patches": obs_patches,
    }


def _update_live_figure(state: dict, t: float, X, Y, obs_state=None):
    """用当前轨迹更新动画（更稳：draw_idle + flush_events + pause）"""
    if (not _HAS_MPL) or (state is None):
        return

    fig = state["fig"]
    ax = state["ax"]
    traj_lines = state["traj_lines"]
    start_scats = state["start_scats"]
    end_scats = state["end_scats"]

    num_drones, Nt = X.shape

    for i in range(num_drones):
        traj_lines[i].set_data(X[i, :], Y[i, :])
        start_scats[i].set_offsets([[X[i, 0], Y[i, 0]]])
        end_scats[i].set_offsets([[X[i, Nt - 1], Y[i, Nt - 1]]])

    # ✅ 更新动态障碍圆心
    if obs_state is not None and "obs_patches" in state:
        for j, circ in enumerate(state["obs_patches"]):
            if j < obs_state.shape[0]:
                circ.center = (obs_state[j, 0], obs_state[j, 1])

    ax.set_title(f"UAV Swarm – 2D Trajectories (t = {t:.1f} s)")

    # ✅ 关键：flush_events 能显著减少“白屏/未响应”
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


# ---------------------------- AirSim 适配提示 ----------------------------
AIRSIM_ADAPTATION_GUIDE = r"""
AirSim 适配：
1) 在每拍循环中，用本模块获得的 (Vxy_c, psi_c, h_c) 参考量，映射到 AirSim 指令：
   - vx = Vxy_c * cos(psi_c), vy = Vxy_c * sin(psi_c)
   - yaw（rad） = (pi/2 - psi_c)  （若你在 MATLAB→AirSim 做了 x/y 轴交换，此处需保持一致）
   - z（NED） = -h_c
   建议统一使用 moveByVelocityZAsync(duration=DT)，并且每拍 join()，避免指令覆盖。
2) 坐标系：文献+MATLAB 使用 ENU，AirSim 是 NED。常用变换：
   x_mat = -y_air, y_mat = x_air, z_mat = -z_air, psi_mat = pi/2 - yaw_air
3) 低速航向估计：速度极小时用姿态四元数估计 yaw，而不要用速度方向。
4) 允许误差吸附与饱和（Step 11, Eq.(4)）在 autopilot 里已实现；下发命令前不再重复裁剪。
"""


if __name__ == "__main__":
    params = SimParams()
    # 1) 新增 1 个动态障碍（位置+半径）
    dyn_obs = np.array([
        [200.0, 100.0, 5.0],
    ], dtype=float)

    # 2) 给这个动态障碍速度（m/s）
    dyn_vxy = np.array([
        [0.8, 0.2],
    ], dtype=float)

    # 3) 追加到原来的静态障碍后面
    params.obstacles = np.vstack([params.obstacles, dyn_obs])

    # 原来那 6 个默认是 0（静态），只给新增的 2 个非零速度
    params.obstacles_vxy = np.vstack([params.obstacles_vxy, dyn_vxy])

    # 方块横向动（替代原来第二个圆）
    params.square_enable = True
    params.square_center_xy = (320.0, 140.0)
    params.square_side = 20.0
    params.square_vxy = (0.6, 0.0)
    # 可选：动态障碍更稳（用一步预测位置做规划）
    params.obstacles_use_prediction = True
    params.verbose_uav1 = False
    params.csv_path = None  # 或者直接传 csv_path=None

    run_simulation(params, enable_plot=True, live_plot=True, csv_path=None)

