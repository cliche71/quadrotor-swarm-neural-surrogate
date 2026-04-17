import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np

from mpio_swarm_simv1 import (
    SimParams,
    run_episode,
    run_simulation,
    extract_features,
    neighbors_within,
    optimizePigeons_wrapper,
    _seg_point_dists,
)


SCENE_TYPES = {
    "b_chicane_chain": 0,
    "e_pillar_forest": 1,
    "g_forest_dynamic_spheres": 2,
}

MAP_X = (0.0, 450.0)
MAP_Y = (0.0, 200.0)
START_X = 20.0
GOAL_X = 430.0
START_Y_RANGE = (90.0, 110.0)

# Single source of truth for swarm size in dataset generation/eval/DAgger.
N_RANGE = (3, 9)  # inclusive
VE_MAG_RANGE = (6.0, 10.0)
LEVEL_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
FAMILY_ORDER = ["b_chicane_chain", "e_pillar_forest", "g_forest_dynamic_spheres"]
FAMILY_WEIGHTS = np.array([0.45, 0.10, 0.45], dtype=float)
PRECHECK_TIME = 2.0
PRECHECK_MIN_DX = 6.0
PRECHECK_MIN_V = 2.0
START_CLEAR_RADIUS = 50.0
GOAL_CLEAR_RADIUS = 50.0
MAX_SCENE_TRIES = 10
MIN_OBS = {
    "b_chicane_chain": 6,
    "e_pillar_forest": 10,
    "g_forest_dynamic_spheres": 10,
}


def _init_swarm(rng: np.random.Generator, n_uav: int, ve_mag: float,
                y_center: float = 100.0,
                R_desire: float = 10.0,
                he: float = 50.0,
                h_span: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    # 更紧凑的初始化：y 跨度 ≤25m，x 交错跨度 ≤25m
    x_lo, x_hi = START_X - 12.5, START_X + 12.5
    y_lo, y_hi = y_center - 12.5, y_center + 12.5
    h_lo, h_hi = he - h_span, he + h_span

    min_sep = 2.2
    max_tries = 20000

    pts = []
    tries = 0
    while len(pts) < n_uav and tries < max_tries:
        tries += 1
        x = float(rng.uniform(x_lo, x_hi))
        y = float(rng.uniform(y_lo, y_hi))
        ok = True
        for (px, py) in pts:
            if math.hypot(x - px, y - py) < min_sep:
                ok = False
                break
        if ok:
            pts.append((x, y))

    if len(pts) < n_uav:
        raise RuntimeError(
            f"init_swarm: failed to sample {n_uav} points with min_sep={min_sep} "
            f"in {max_tries} tries"
        )

    h = rng.uniform(h_lo, h_hi, size=n_uav)
    init_p = np.column_stack([
        np.array([p[0] for p in pts]),
        np.array([p[1] for p in pts]),
        h,
    ]).astype(float)
    init_vxy = np.tile(np.array([ve_mag, 0.0], dtype=float), (n_uav, 1))
    return init_p, init_vxy


def _clip(val: float, lo: float, hi: float) -> float:
    return lo if val < lo else hi if val > hi else val


def _add_wall(obs: list, x0: float, x1: float, y: float, r: float, spacing: float):
    x = x0
    while x <= x1:
        obs.append([x, y, r])
        x += spacing


def _rand_range(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _sample_n_uav(rng: np.random.Generator) -> int:
    n_min, n_max = N_RANGE
    return int(rng.integers(n_min, n_max + 1))


def _x_end_allow(r_max: float, margin: float = 8.0) -> float:
    """Obstacle center x must respect map bound and goal clear radius."""
    return min(
        MAP_X[1] - r_max - margin,
        GOAL_X - GOAL_CLEAR_RADIUS - r_max - margin,
    )


def _x_scene_span(r_max: float, x_start_hint: float = 110.0, margin: float = 8.0) -> Tuple[float, float]:
    """[x_start, x_end] where obstacles are allowed (also respects start clear)."""
    x_start = max(x_start_hint, START_X + START_CLEAR_RADIUS + r_max + margin)
    x_end = _x_end_allow(r_max=r_max, margin=margin)
    return x_start, x_end


def _pairwise_clear(obs: np.ndarray, min_sep: float) -> bool:
    if obs.size == 0:
        return True
    n = obs.shape[0]
    for i in range(n):
        xi, yi, ri = obs[i]
        for j in range(i + 1, n):
            xj, yj, rj = obs[j]
            if math.hypot(xi - xj, yi - yj) < (ri + rj + min_sep):
                return False
    return True


def _validate_obstacles(obs: np.ndarray,
                        start_xy: Tuple[float, float],
                        goal_xy: Tuple[float, float],
                        min_sep: float = 5.0) -> bool:
    if obs.size == 0:
        return True
    for x, y, r in obs:
        if x < MAP_X[0] + r or x > MAP_X[1] - r:
            return False
        if y < MAP_Y[0] + r or y > MAP_Y[1] - r:
            return False
        if math.hypot(x - start_xy[0], y - start_xy[1]) < (START_CLEAR_RADIUS + r):
            return False
        if math.hypot(x - goal_xy[0], y - goal_xy[1]) < (GOAL_CLEAR_RADIUS + r):
            return False
    if not _pairwise_clear(obs, min_sep):
        return False
    return True


def _make_gate(x: float, yc: float, r: float, d: float) -> List[List[float]]:
    return [
        [x, yc + d / 2.0, r],
        [x, yc - d / 2.0, r],
    ]


def _add_noise_pillars(rng: np.random.Generator,
                       obs: List[List[float]],
                       vxy: List[List[float]],
                       n_add: int,
                       x_range: Tuple[float, float],
                       y_center: float,
                       y_offset_range: Tuple[float, float],
                       r_range: Tuple[float, float],
                       min_sep: float = 5.0,
                       dynamic: bool = False,
                       gate_x: Optional[List[float]] = None,
                       min_gate_dx: float = 40.0) -> None:
    tries = 0
    max_tries = 2000
    while n_add > 0 and tries < max_tries:
        tries += 1
        x = _rand_range(rng, x_range[0], x_range[1])
        if gate_x:
            if min(abs(x - gx) for gx in gate_x) < min_gate_dx:
                continue
        side = 1.0 if rng.random() < 0.5 else -1.0
        y = y_center + side * _rand_range(rng, y_offset_range[0], y_offset_range[1])
        r = _rand_range(rng, r_range[0], r_range[1])
        if y < MAP_Y[0] + r or y > MAP_Y[1] - r:
            continue
        ok = True
        for ox, oy, orr in obs:
            if math.hypot(x - ox, y - oy) < (r + orr + min_sep):
                ok = False
                break
        if not ok:
            continue
        obs.append([x, y, r])
        if dynamic:
            speed = _rand_range(rng, 0.8, 1.5)
            ang = _rand_range(rng, 0.0, 2.0 * math.pi)
            vxy.append([speed * math.cos(ang), speed * math.sin(ang)])
        else:
            vxy.append([0.0, 0.0])
        n_add -= 1


def _sample_y0(rng: np.random.Generator, lo: float = START_Y_RANGE[0], hi: float = START_Y_RANGE[1]) -> float:
    return float(rng.uniform(lo, hi))


def _linspace_jitter(rng: np.random.Generator, x0: float, x1: float, n: int,
                     jitter: float, min_dx: float) -> Optional[List[float]]:
    if n <= 1:
        return [0.5 * (x0 + x1)]
    base = np.linspace(x0, x1, n, dtype=float)
    for _ in range(200):
        xs = [float(b + rng.uniform(-jitter, jitter)) for b in base]
        xs.sort()
        if all((xs[i + 1] - xs[i]) >= min_dx for i in range(len(xs) - 1)):
            return xs
    return None


def _family_b_chicane_chain(rng: np.random.Generator, level: int, r2_lim: float,
                            ve_mag: float, style: str = "tight", add_dyn: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float, float, List[float], Dict[str, object]]:
    """
    【物理回归版】蛇形通道：
    1. 几何：保持极其宽敞和平缓 (L=500, Gap=36)，这是低过载能过弯的前提。
    2. 物理：将过载限制在四旋翼的合理范围 (2.5g)。
    """
    y_center_base = 100.0
    
    # --- 1. 几何参数 (保持宽敞) ---
    r_obs = _rand_range(rng, 4.8, 5.2)
    hard_width = 2 * (r_obs + r2_lim)
    flyable_gap = _rand_range(rng, 34.0, 40.0)
    path_width = hard_width + flyable_gap
    
    # 振幅 A: 35~50m
    A = _rand_range(rng, 35.0, 50.0)
    
    # 波长 L: 450~600m (保持平缓，但允许曲率变化)
    # 只有弯道够缓，我们才能用 2.5g 的过载转过去
    L = _rand_range(rng, 450.0, 600.0)
    
    obs = []
    x_curr = 40.0
    step_x = 8.0 
    
    while x_curr <= 460.0:
        phase = (x_curr - 40.0) / L * 2 * math.pi
        y_c = y_center_base + A * math.sin(phase)
        
        k = A * (2 * math.pi / L) * math.cos(phase)
        angle = math.atan(k)
        dx = -math.sin(angle)
        dy = math.cos(angle)
        
        p_up = (x_curr + dx * (path_width/2), y_c + dy * (path_width/2))
        p_down = (x_curr - dx * (path_width/2), y_c - dy * (path_width/2))
        
        if 5.0 < p_up[1] < 195.0:
            obs.append([p_up[0], p_up[1], r_obs])
        if 5.0 < p_down[1] < 195.0:
            obs.append([p_down[0], p_down[1], r_obs])
            
        x_curr += step_x

    obstacles = np.array(obs, dtype=float)
    vxy = np.zeros((len(obs), 2), dtype=float)

    # --- 3. 物理与控制参数 (回归现实) ---
    extra = {
        # 【物理限制】
        # n_max = 2.5g (约25m/s^2)，这是高性能四旋翼的合理上限
        # 相比之前的 12g，这不仅真实，而且会让轨迹更符合动力学约束
        "n_max": 3.0, 
        
        # 【速度调整】
        # 训练速度：6~10 m/s
        "ve_mag": _clip(ve_mag, 6.0, 10.0),
        "ve_xy": (_clip(ve_mag, 6.0, 10.0), 0.0),
        
        # 【感知与编队】(保持之前的成功经验)
        "R2_comm": 40.0,        # 四旋翼感知范围
        "gap_lookahead": 50.0,
        
        "R_desire": 6.0,  # 紧缩
        "R1_comm": 45.0,  # 强连接
        
        "f1": 1.5, 
        "f2": 1.2, # 依然保持低速度耦合，允许自由机动
        
        "w_direction": 0.1,
    }

    return obstacles, vxy, y_center_base, GOAL_X, y_center_base, [], extra


def _family_e_pillar_forest(rng: np.random.Generator, level: int, r2_lim: float,
                            ve_mag: float, style: str = "tight") -> Tuple[np.ndarray, np.ndarray, float, float, float, List[float], Dict[str, object]]:
    """
    【柱林场景】模拟密林穿越（抖动网格 + 稀疏化）。
    """
    level = 3
    y_center = 100.0

    if level >= 3:
        r_obs = _rand_range(rng, 4.8, 5.2)
    else:
        r_obs = 5.0

    if level <= 1:
        grid_spacing_x = 75.0
        grid_spacing_y = 70.0
        jitter_amount = 15.0
        skip_prob = 0.30
        wave_amp = 10.0
        wave_len = 220.0
        rotate_deg = 6.0
        center_block_prob = 0.10
        corridor_half = 0.0
        corridor_place_prob = 0.0
        A_meander = 0.0
        L_meander = 1.0
        phase = 0.0
    elif level == 2:
        grid_spacing_x = 65.0
        grid_spacing_y = 60.0
        jitter_amount = 12.0
        skip_prob = 0.20
        wave_amp = 14.0
        wave_len = 200.0
        rotate_deg = 8.0
        center_block_prob = 0.15
        corridor_half = 0.0
        corridor_place_prob = 0.0
        A_meander = 0.0
        L_meander = 1.0
        phase = 0.0
    elif level == 3:
        grid_spacing_x = 36.0
        grid_spacing_y = 32.0
        jitter_amount = 4.0
        skip_prob = 0.01
        wave_amp = 20.0
        wave_len = 170.0
        rotate_deg = 30.0
        center_block_prob = 0.20
        corridor_half = 8.0
        corridor_place_prob = 0.10
        A_meander = 36.0
        L_meander = 1.0
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
    else:
        grid_spacing_x = 45.0
        grid_spacing_y = 40.0
        jitter_amount = 6.0
        skip_prob = 0.03
        wave_amp = 26.0
        wave_len = 150.0
        rotate_deg = 12.0
        center_block_prob = 0.25
        corridor_half = 10.0
        corridor_place_prob = 0.10
        A_meander = 32.0
        L_meander = 1.0
        phase = float(rng.uniform(0.0, 2.0 * math.pi))

    obs = []

    x_start, x_end = _x_scene_span(r_max=r_obs, x_start_hint=100.0)
    y_min = MAP_Y[0] + 20.0
    y_max = MAP_Y[1] - 20.0

    if level >= 3:
        L_meander = (x_end - x_start)
        phase_jump = phase + _rand_range(rng, 0.9 * math.pi, 1.2 * math.pi)
        x_mid = _rand_range(rng, x_start + 0.35 * L_meander, x_start + 0.65 * L_meander)
    if level >= 3:
        rot_mag = _rand_range(rng, 20.0, 50.0)
        rot_sign = -1.0 if rng.random() < 0.5 else 1.0
        theta = math.radians(rot_sign * rot_mag)
    else:
        theta = math.radians(_rand_range(rng, -rotate_deg, rotate_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot_cx = (x_start + x_end) * 0.5
    rot_cy = (y_min + y_max) * 0.5

    row_idx = 0
    curr_x = x_start
    while curr_x <= x_end:
        curr_y = y_min
        y_offset = 0.0 if (row_idx % 2 == 0) else (grid_spacing_y / 2.0)
        curr_y += y_offset
        while curr_y <= y_max:
            base_y = curr_y
            if level >= 3:
                phase_use = phase if curr_x <= x_mid else phase_jump
                corridor_y = y_center + A_meander * math.sin(
                    2.0 * math.pi * (curr_x - x_start) / L_meander + phase_use
                )
                in_corridor = abs(base_y - corridor_y) < corridor_half
            else:
                in_corridor = False

            if in_corridor:
                place_prob = corridor_place_prob
            else:
                place_prob = 1.0 - max(0.0, skip_prob - 0.05)

            if rng.random() < place_prob:
                wave = wave_amp * math.sin((curr_x - x_start) / wave_len * 2.0 * math.pi)
                jit_x = _rand_range(rng, -jitter_amount, jitter_amount)
                jit_y = _rand_range(rng, -jitter_amount, jitter_amount)
                base_x = curr_x + jit_x
                base_y = base_y + jit_y + wave

                dx = base_x - rot_cx
                dy = base_y - rot_cy
                final_x = rot_cx + dx * cos_t - dy * sin_t
                final_y = rot_cy + dx * sin_t + dy * cos_t
                if (MAP_Y[0] + 10.0) < final_y < (MAP_Y[1] - 10.0):
                    obs.append([final_x, final_y, r_obs])
            curr_y += grid_spacing_y
        curr_x += grid_spacing_x
        row_idx += 1

    # 额外在中心线附近打散“直线通道”
    if obs:
        x_mid = 0.5 * (x_start + x_end)
        for _ in range(int((x_end - x_start) / max(1.0, grid_spacing_x))):
            if rng.random() < center_block_prob:
                bx = _rand_range(rng, x_mid - 80.0, x_mid + 80.0)
                by = y_center + _rand_range(rng, -20.0, 20.0)
                obs.append([bx, by, r_obs])

    obstacles = np.array(obs, dtype=float)
    vxy = np.zeros((len(obs), 2), dtype=float)

    extra = {
        "R2_comm": 40.0,
        "gap_lookahead": 45.0,
        "R_desire": 6.5,
        "R1_comm": 40.0,
        "f1": 1.2,
        "f2": 0.2,
        "w_direction": 0.1,
        "n_max": 3.0,
        "collision_tol_obs": 1.0,
        "ve_mag": _clip(ve_mag, 6.0, 10.0),
        "ve_xy": (_clip(ve_mag, 6.0, 10.0), 0.0),
        "obstacles_bounds": (float(x_start), float(x_end), float(y_min), float(y_max)),
    }

    return obstacles, vxy, y_center, GOAL_X, y_center, [], extra


def _family_g_forest_dynamic_spheres(rng: np.random.Generator, r2_lim: float,
                                     ve_mag: float, he: float,
                                     style: str = "tight") -> Tuple[np.ndarray, np.ndarray, float, float, float, List[float], Dict[str, object]]:
    """
    【柱林 + 动态球】独立场景，不复用 e。
    """
    # --- g 场景参数区（松柱林 + 动态球） ---
    # 柱林：50/45/6/0.05（更宽松，便于动态球加入）
    # 动态球：2~6 个，2D 截面半径 3~6m，高度偏移 |dz|<=10m
    y_center = 100.0
    r_obs = _rand_range(rng, 4.8, 5.2)

    # 柱林（略松）：55/50/8/0.1
    grid_spacing_x = 45.0
    grid_spacing_y = 40.0
    jitter_amount = 6.0
    skip_prob = 0.03
    wave_amp = 12.0
    wave_len = 200.0

    obs = []

    x_start, x_end = _x_scene_span(r_max=r_obs, x_start_hint=100.0)
    y_min = MAP_Y[0] + 20.0
    y_max = MAP_Y[1] - 20.0

    rot_mag = _rand_range(rng, 20.0, 50.0)
    rot_sign = -1.0 if rng.random() < 0.5 else 1.0
    theta = math.radians(rot_sign * rot_mag)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot_cx = (x_start + x_end) * 0.5
    rot_cy = (y_min + y_max) * 0.5

    row_idx = 0
    curr_x = x_start
    while curr_x <= x_end:
        curr_y = y_min
        y_offset = 0.0 if (row_idx % 2 == 0) else (grid_spacing_y / 2.0)
        curr_y += y_offset
        while curr_y <= y_max:
            if rng.random() < (1.0 - skip_prob):
                wave = wave_amp * math.sin((curr_x - x_start) / wave_len * 2.0 * math.pi)
                jit_x = _rand_range(rng, -jitter_amount, jitter_amount)
                jit_y = _rand_range(rng, -jitter_amount, jitter_amount)
                base_x = curr_x + jit_x
                base_y = curr_y + jit_y + wave

                dx = base_x - rot_cx
                dy = base_y - rot_cy
                final_x = rot_cx + dx * cos_t - dy * sin_t
                final_y = rot_cy + dx * sin_t + dy * cos_t
                if (MAP_Y[0] + 10.0) < final_y < (MAP_Y[1] - 10.0):
                    obs.append([final_x, final_y, r_obs])
            curr_y += grid_spacing_y
        curr_x += grid_spacing_x
        row_idx += 1

    obstacles = np.array(obs, dtype=float)
    n_static = obstacles.shape[0]
    vxy = np.zeros((n_static, 2), dtype=float)

    # 动态球（速度随机 0.5~2.0 m/s，横/竖与之前一致）
    n_target = int(rng.integers(2, 7))
    spheres = []
    spheres_vxy = []
    spheres_3d = []
    max_trials = 2000
    trials = 0
    while len(spheres) < n_target and trials < max_trials:
        trials += 1
        cx = float(rng.uniform(x_start + 20.0, x_end - 20.0))
        cy = float(rng.uniform(y_center - 40.0, y_center + 40.0))
        r_eff = _rand_range(rng, 3.0, 6.0)
        dz = float(rng.uniform(-10.0, 10.0))
        r_sphere = math.sqrt(r_eff * r_eff + dz * dz)
        z_center = he + dz
        if obstacles.size > 0:
            d = np.linalg.norm(obstacles[:, :2] - np.array([cx, cy]), axis=1)
            min_clearance = obstacles[:, 2] + r_eff + 2.0
            if np.any(d < min_clearance):
                continue
        spheres.append([cx, cy, r_eff])
        spheres_3d.append([cx, cy, z_center, r_sphere, r_eff])
        speed = float(rng.uniform(0.5, 2.0))
        if rng.random() < 0.7:
            vx = 0.0
            vy = speed if rng.random() < 0.5 else -speed
        else:
            vy = 0.0
            vx = speed if rng.random() < 0.5 else -speed
        spheres_vxy.append([vx, vy])

    if spheres:
        obstacles = np.vstack([obstacles, np.asarray(spheres, dtype=float)])
        vxy = np.vstack([vxy, np.asarray(spheres_vxy, dtype=float)])

    extra = {
        "R2_comm": 40.0,
        "gap_lookahead": 45.0,
        "R_desire": 6.5,
        "R1_comm": 40.0,
        "f1": 1.2,
        "f2": 0.2,
        "w_direction": 0.1,
        "n_max": 3.0,
        "collision_tol_obs": 1.0,
        "ve_mag": _clip(ve_mag, 6.0, 10.0),
        "ve_xy": (_clip(ve_mag, 6.0, 10.0), 0.0),
        "obstacles_bounds": (float(x_start), float(x_end), float(y_min), float(y_max)),
        "obstacles_use_prediction": True,
        "dynamic_spheres_3d": spheres_3d,
    }

    return obstacles, vxy, y_center, GOAL_X, y_center, [], extra


def _family_a_gate_chain(*_args, **_kwargs):
    raise NotImplementedError("scene a_gate_chain is not available in this dataset_gen.py")


def _family_c_gap_lattice_dynamic(*_args, **_kwargs):
    raise NotImplementedError("scene c_gap_lattice_dynamic is not available in this dataset_gen.py")


def _family_d_gap_lattice(*_args, **_kwargs):
    raise NotImplementedError("scene d_gap_lattice is not available in this dataset_gen.py")


def _family_f_static_dynamic_mix(*_args, **_kwargs):
    raise NotImplementedError("scene f_static_dynamic_mix is not available in this dataset_gen.py")

def sample_scene(rng: np.random.Generator) -> Dict[str, object]:
    n_uav = _sample_n_uav(rng)
    ve_mag = float(rng.uniform(VE_MAG_RANGE[0], VE_MAG_RANGE[1]))
    he = float(rng.uniform(40.0, 70.0))

    if len(FAMILY_ORDER) != len(FAMILY_WEIGHTS) or len(FAMILY_ORDER) == 0:
        scene_type = "b_chicane_chain"
    else:
        weights = FAMILY_WEIGHTS / np.sum(FAMILY_WEIGHTS)
        scene_type = str(rng.choice(FAMILY_ORDER, p=weights))
    level = int(rng.choice([1, 2, 3, 4], p=LEVEL_WEIGHTS / np.sum(LEVEL_WEIGHTS)))

    r2_lim = float(SimParams().R2_lim)
    extra_cfg: Dict[str, object] = {}
    if scene_type == "b_chicane_chain":
        obstacles, obstacles_vxy, y0, x_goal, y_goal, gate_x, extra_cfg = _family_b_chicane_chain(
            rng, level, r2_lim, ve_mag
        )
    elif scene_type == "e_pillar_forest":
        obstacles, obstacles_vxy, y0, x_goal, y_goal, gate_x, extra_cfg = _family_e_pillar_forest(
            rng, level, r2_lim, ve_mag
        )
    elif scene_type == "g_forest_dynamic_spheres":
        obstacles, obstacles_vxy, y0, x_goal, y_goal, gate_x, extra_cfg = _family_g_forest_dynamic_spheres(
            rng, r2_lim, ve_mag, he
        )
        obstacles_use_prediction = True
    else:
        return {}

    min_obs = int(MIN_OBS.get(scene_type, 1))
    if obstacles.size == 0 or obstacles.shape[0] < min_obs:
        return {}

    obs_list = obstacles.tolist()
    vxy_list = obstacles_vxy.tolist()

    obstacles = np.array(obs_list, dtype=float)
    obstacles_vxy = np.array(vxy_list, dtype=float) if obs_list else np.zeros((0, 2), dtype=float)
    obstacles_use_prediction = bool(np.any(np.linalg.norm(obstacles_vxy, axis=1) > 1e-6))

    init_p, init_vxy = _init_swarm(rng, n_uav, ve_mag, y_center=y0, he=he)
    if scene_type != "b_chicane_chain":
        start_xy = (float(np.mean(init_p[:, 0])), float(np.mean(init_p[:, 1])))
        goal_xy = (x_goal, y_goal)
        if not _validate_obstacles(obstacles, start_xy, goal_xy):
            return {}

    scene_cfg = {
        "scene_type": scene_type,
        "scene_level": level,
        "N_uav": n_uav,
        "ve_mag": ve_mag,
        "he": he,
        "init_P": init_p,
        "init_Vxy": init_vxy,
        "ve_xy": (ve_mag, 0.0),
        "obstacles": obstacles,
        "obstacles_vxy": obstacles_vxy,
        "obstacles_use_prediction": obstacles_use_prediction,
        "x_goal": x_goal,
        "y_goal": y_goal,
        "square_enable": False,
        "formation_err_mean_thresh": 3.0,
        "formation_err_max_thresh": 10.0,
        "formation_over_limit": 0.20,
    }
    if extra_cfg:
        scene_cfg.update(extra_cfg)
    return scene_cfg


def write_npz(out_path: Path, X: np.ndarray, Y: np.ndarray, meta: Dict[str, np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X.astype(np.float32), Y=Y.astype(np.float32), **meta)


def _save_scene_json(scene_cfg: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serial = {}
    for k, v in scene_cfg.items():
        if isinstance(v, np.ndarray):
            serial[k] = v.tolist()
        else:
            serial[k] = v
    out_path.write_text(json.dumps(serial, indent=2), encoding="utf-8")


def _save_traj_npz(out_path: Path, traj: Dict[str, np.ndarray], meta: Dict[str, np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in traj.items():
        if isinstance(v, np.ndarray):
            payload[k] = v.astype(np.float32) if v.dtype.kind in "fc" else v
        else:
            payload[k] = v
    for k, v in meta.items():
        payload[k] = v
    np.savez_compressed(out_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_fail", action="store_true")
    parser.add_argument("--save_traj_train_every", type=int, default=2)
    parser.add_argument("--save_traj_fail_every", type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0, help="start index for ep_*.npz/json naming")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    fail_dir = out_dir / "fail"
    scene_dir = out_dir / "scenes"
    traj_dir = out_dir / "traj"
    traj_train_dir = traj_dir / "train"
    traj_fail_dir = traj_dir / "fail"
    train_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_train_dir.mkdir(parents=True, exist_ok=True)
    traj_fail_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "success": [],
        "fail": [],
        "scenes": [],
        "traj_train": [],
        "traj_fail": [],
        "num_episodes": args.num_episodes,
        "success_rate": 0.0,
        "mean_samples": 0.0,
        "feature_dim": None,
    }

    success_count = 0
    train_seen = 0
    fail_seen = 0
    total_samples = 0
    base_sp = SimParams()
    base_sp.verbose_uav1 = False
    base_sp.csv_path = None

    for ep in range(args.num_episodes):
        ep_id = int(args.start_index + ep)
        scene_cfg = {}
        for attempt in range(1, MAX_SCENE_TRIES + 1):
            scene_cfg = sample_scene(rng)
            if scene_cfg:
                break
        if not scene_cfg:
            print(f"[ep {ep:03d}] skipped (scene generation returned empty)", flush=True)
            continue
        scene_cfg["rng_seed"] = int(rng.integers(1, 1_000_000_000))

        X_rows: List[np.ndarray] = []
        Y_rows: List[np.ndarray] = []
        uav_ids: List[int] = []
        t_list: List[float] = []
        nbr_margin_list: List[float] = []
        obs_margin_list: List[float] = []
        episode_id = ep_id

        def policy_fn(i, P, V_xy, psi, lamb, obs_state, obstacles_plan, sp,
                      ff_raw, fa_raw, fc_raw, vf_z, vo_raw, step_k, t):
            w = optimizePigeons_wrapper(sp, P, i, V_xy, obstacles_plan,
                                        ff_raw, fa_raw, fc_raw, vf_z, vo_raw, lamb[i], psi[i])
            if step_k % args.stride != 0:
                return w
            min_nbr_margin = 1e9
            for j in range(P.shape[0]):
                if j == i:
                    continue
                d = np.linalg.norm(P[j, :2] - P[i, :2])
                min_nbr_margin = min(min_nbr_margin, d - sp.R1_lim)
            if not np.isfinite(min_nbr_margin):
                min_nbr_margin = 0.0
            min_obs_margin = 1e3
            if obstacles_plan is not None and np.size(obstacles_plan) > 0:
                obs_plan = np.asarray(obstacles_plan, dtype=float)
                centers = obs_plan[:, :2]
                radii = obs_plan[:, 2]
                dist = np.linalg.norm(centers - P[i, :2], axis=1)
                min_obs_margin = float(np.min(dist - (radii + sp.R2_lim)))
            nbr_idx = neighbors_within(P, i, sp.R1_comm)
            neighbors = np.array([
                [P[j, 0], P[j, 1], V_xy[j, 0], V_xy[j, 1]] for j in nbr_idx
            ], dtype=float)
            obstacles = np.hstack([obstacles_plan, obs_state[:, 3:5]])
            feat = extract_features(
                pos_i=P[i, :],
                vel_i=V_xy[i, :],
                psi_i=psi[i],
                lamb_i=lamb[i],
                neighbors=neighbors,
                obstacles=obstacles,
                ve_xy=sp.ve_xy,
            )
            X_rows.append(feat)
            Y_rows.append(np.array(w, dtype=float))
            uav_ids.append(int(i))
            t_list.append(float(t))
            nbr_margin_list.append(float(min_nbr_margin))
            obs_margin_list.append(float(min_obs_margin))
            return w

        results = run_episode(base_sp, scene_cfg, collect=False, stride=args.stride, policy_fn=policy_fn)
        scene_path = scene_dir / f"ep_{ep_id:06d}.json"
        _save_scene_json(scene_cfg, scene_path)
        index["scenes"].append(scene_path.as_posix())
        is_success = bool(results.get("safe_ok", False)) and bool(results.get("reach_ok", True)) and bool(results.get("formation_ok", True))
        if is_success:
            train_seen += 1
            save_traj = args.save_traj_train_every > 0 and ((train_seen - 1) % args.save_traj_train_every == 0)
        else:
            fail_seen += 1
            save_traj = args.save_traj_fail_every > 0 and ((fail_seen - 1) % args.save_traj_fail_every == 0)

        if save_traj:
            results_traj = run_episode(base_sp, scene_cfg, collect=True, stride=args.stride, policy_fn=None)
            traj = results_traj.get("traj", {})
            if traj:
                traj_meta = {
                    "episode_id": np.array([ep], dtype=np.int32),
                    "scene_type": np.array([SCENE_TYPES[scene_cfg["scene_type"]]], dtype=np.int16),
                    "scene_level": np.array([int(scene_cfg.get("scene_level", 1))], dtype=np.int16),
                    "N_uav": np.array([scene_cfg["N_uav"]], dtype=np.int16),
                    "ve_mag": np.array([scene_cfg["ve_mag"]], dtype=np.float32),
                    "safe_ok": np.array([int(results.get("safe_ok", False))], dtype=np.int16),
                    "reach_ok": np.array([int(results.get("reach_ok", True))], dtype=np.int16),
                    "formation_ok": np.array([int(results.get("formation_ok", True))], dtype=np.int16),
                }
                if is_success:
                    traj_path = traj_train_dir / f"ep_{ep_id:06d}.npz"
                    index["traj_train"].append(traj_path.as_posix())
                else:
                    traj_path = traj_fail_dir / f"ep_{ep_id:06d}.npz"
                    index["traj_fail"].append(traj_path.as_posix())
                _save_traj_npz(traj_path, traj, traj_meta)
        cost_flags = results.get("cost_flags", {})
        c3_any = cost_flags.get("c3_any", None)
        c4_any = cost_flags.get("c4_any", None)
        msg = (f"[ep {ep:03d}] safe={results['safe_ok']} reach={results['reach_ok']} "
               f"form={results['formation_ok']} "
               f"viol_rate={results.get('viol_rate')} "
               f"E_mean_avg={results.get('E_mean_avg')} "
               f"E_max_peak={results.get('E_max_peak')} "
               f"n_edges={results.get('n_edges')} "
               f"K_used={results.get('K_used')} "
               f"c3_any={c3_any} c4_any={c4_any}")
        print(msg, flush=True)
        if not results["safe_ok"]:
            print(f"  collision_obs={c3_any} collision_nbr={c4_any}", flush=True)

        if not X_rows:
            continue

        X = np.vstack(X_rows)
        Y = np.vstack(Y_rows)

        meta = {
            "episode_id": np.full((X.shape[0],), episode_id, dtype=np.int32),
            "uav_id": np.asarray(uav_ids, dtype=np.int16),
            "t": np.asarray(t_list, dtype=np.float32),
            "scene_type": np.full((X.shape[0],), SCENE_TYPES[scene_cfg["scene_type"]], dtype=np.int16),
            "scene_level": np.full((X.shape[0],), int(scene_cfg.get("scene_level", 1)), dtype=np.int16),
            "N_uav": np.full((X.shape[0],), scene_cfg["N_uav"], dtype=np.int16),
            "ve_mag": np.full((X.shape[0],), scene_cfg["ve_mag"], dtype=np.float32),
            "nbr_margin": np.asarray(nbr_margin_list, dtype=np.float32),
            "obs_margin": np.asarray(obs_margin_list, dtype=np.float32),
            "collision_nbr_ep": np.full((X.shape[0],), int(bool(c4_any)), dtype=np.int16),
            "collision_obs_ep": np.full((X.shape[0],), int(bool(c3_any)), dtype=np.int16),
        }

        if results["safe_ok"] and results["reach_ok"] and results["formation_ok"]:
            out_path = train_dir / f"ep_{ep_id:06d}.npz"
            write_npz(out_path, X, Y, meta)
            index["success"].append(out_path.as_posix())
            success_count += 1
            total_samples += int(X.shape[0])
        else:
            if args.save_fail:
                out_path = fail_dir / f"ep_{ep_id:06d}.npz"
                write_npz(out_path, X, Y, meta)
                index["fail"].append(out_path.as_posix())

        if index["feature_dim"] is None:
            index["feature_dim"] = int(X.shape[1])

    index["success_rate"] = float(success_count / max(args.num_episodes, 1))
    if success_count > 0:
        index["mean_samples"] = float(total_samples / success_count)

    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[dataset] success_rate={index['success_rate']:.2f}, mean_samples={index['mean_samples']:.1f}")
    print(f"[dataset] feature_dim={index['feature_dim']}")


if __name__ == "__main__":
    main()
