import argparse
import csv
import json
import os
import re
from pathlib import Path

import airsim
import numpy as np
import time
import math
import dataclasses
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# Public release: do not assume a user-specific AirSim settings.json path.
DEFAULT_AIRSIM_SETTINGS_JSON_CANDIDATES = []

# =====================================================
#          0. 数值工具 & 公共函数
# =====================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def norm2(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


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


def _wrap_angle(a: float) -> float:
    # wrap 到 (-pi, pi]
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _single_obstacle_turn_sign(P_i: np.ndarray, theta_e: float, obstacle: np.ndarray) -> float:
    drone = np.asarray(P_i[:2], dtype=float)
    cx, cy = float(obstacle[0]), float(obstacle[1])
    angle_ij = math.atan2(cy - drone[1], cx - drone[0])
    theta_rel = _wrap_angle(angle_ij - theta_e)
    return 1.0 if theta_rel >= 0.0 else -1.0


def _single_obstacle_yaw(P_i: np.ndarray,
                         theta_e: float,
                         obstacle: np.ndarray,
                         Rc: float,
                         turn_sign: Optional[float] = None) -> float:
    drone = np.asarray(P_i[:2], dtype=float)
    cx, cy = float(obstacle[0]), float(obstacle[1])
    s = _single_obstacle_turn_sign(P_i, theta_e, obstacle) if turn_sign is None else math.copysign(1.0, turn_sign)
    num = cy - s * Rc * math.cos(theta_e) - drone[1]
    den = cx + s * Rc * math.sin(theta_e) - drone[0]
    return _wrap_angle(math.atan2(num, den))


def load_mlp(model_dir: Path):
    scaler = np.load(model_dir / "scaler.npz")
    mean = scaler["mean"]
    std = scaler["std"]

    ckpt = torch.load(model_dir / "model.pt", map_location="cpu")
    input_dim = int(ckpt.get("input_dim", mean.shape[0]))
    if mean.shape[0] != input_dim:
        raise SystemExit("scaler dim != model dim")

    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
        torch.nn.Sigmoid(),
    )
    state = ckpt["model_state"]
    # 兼容 train_mlp.py 保存的 "net.xxx"
    if any(k.startswith("net.") for k in state.keys()):
        state = {k.replace("net.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, mean, std


def resolve_host_path(raw_path: str) -> Path:
    raw = str(raw_path).strip()
    if not raw:
        raise SystemExit("empty path")

    if re.match(r"^[A-Za-z]:[\\/]", raw):
        if os.name == "nt":
            return Path(raw)
        drive = raw[0].lower()
        rest = raw[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")

    if raw.startswith("/mnt/") and os.name == "nt":
        parts = raw.strip("/").split("/", 2)
        if len(parts) >= 2 and len(parts[1]) == 1 and parts[1].isalpha():
            drive = parts[1].upper()
            rest = parts[2] if len(parts) == 3 else ""
            rest = rest.replace("/", "\\")
            return Path(f"{drive}:\\{rest}" if rest else f"{drive}:\\")

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def load_scene_obstacles_and_overrides(scene_json: str) -> Tuple[np.ndarray, bool, Optional[Tuple[float, float, float, float]], Dict[str, float], np.ndarray]:
    scene_path = resolve_host_path(scene_json)
    if not scene_path.is_file():
        raise SystemExit(f"scene_json not found: {scene_path}")

    with scene_path.open("r", encoding="utf-8") as f:
        scene = json.load(f)

    if "obstacles" not in scene:
        raise SystemExit(f"scene_json missing 'obstacles': {scene_path}")

    obstacles = np.asarray(scene["obstacles"], dtype=float)
    if obstacles.size == 0:
        obstacles = obstacles.reshape(0, 3)
    else:
        if obstacles.ndim != 2 or obstacles.shape[1] != 3:
            raise SystemExit(
                f"scene_json 'obstacles' must have shape [N, 3], got {obstacles.shape} from {scene_path}"
            )
        if not np.isfinite(obstacles).all():
            raise SystemExit(f"scene_json 'obstacles' contains non-finite values: {scene_path}")

    obstacles_use_prediction = bool(scene.get("obstacles_use_prediction", False))

    dynamic_spheres_3d = np.asarray(scene.get("dynamic_spheres_3d", []), dtype=float)
    if dynamic_spheres_3d.size == 0:
        dynamic_spheres_3d = dynamic_spheres_3d.reshape(0, 5)
    else:
        if dynamic_spheres_3d.ndim != 2 or dynamic_spheres_3d.shape[1] != 5:
            raise SystemExit(
                f"scene_json 'dynamic_spheres_3d' must have shape [N, 5], got {dynamic_spheres_3d.shape} from {scene_path}"
            )
        if not np.isfinite(dynamic_spheres_3d).all():
            raise SystemExit(f"scene_json 'dynamic_spheres_3d' contains non-finite values: {scene_path}")
    dynamic_radii = dynamic_spheres_3d[:, 4].astype(float, copy=True)
    n_dynamic = int(dynamic_radii.shape[0])
    if n_dynamic > obstacles.shape[0]:
        raise SystemExit(
            f"scene_json dynamic sphere count {n_dynamic} exceeds obstacles count {obstacles.shape[0]}: {scene_path}"
        )
    static_obstacles = obstacles[:-n_dynamic].copy() if n_dynamic > 0 else obstacles.copy()

    obstacles_bounds = scene.get("obstacles_bounds", None)
    if obstacles_bounds is None:
        bounds_tuple = None
    else:
        bounds_arr = np.asarray(obstacles_bounds, dtype=float)
        if bounds_arr.shape != (4,) or not np.isfinite(bounds_arr).all():
            raise SystemExit(f"scene_json 'obstacles_bounds' must be 4 finite values: {scene_path}")
        bounds_tuple = tuple(float(x) for x in bounds_arr.tolist())

    allowed_keys = (
        "R2_comm",
        "gap_lookahead",
        "R_desire",
        "R1_comm",
        "f1",
        "f2",
        "w_direction",
        "n_max",
    )
    overrides: Dict[str, float] = {}
    for key in allowed_keys:
        if key in scene:
            val = float(scene[key])
            if not math.isfinite(val):
                raise SystemExit(f"scene_json '{key}' contains non-finite values: {scene_path}")
            overrides[key] = val
    return static_obstacles, obstacles_use_prediction, bounds_tuple, overrides, dynamic_radii


def query_dynamic_obstacles(client,
                            actor_specs: List[Dict[str, float]],
                            prev_cache: Dict[str, Tuple[float, float]],
                            dt: float,
                            he_plane: float) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
    rows: List[List[float]] = []
    new_cache: Dict[str, Tuple[float, float]] = {}

    for spec in actor_specs:
        name = str(spec["name"])
        radius = float(spec["radius"])
        pose = client.simGetObjectPose(name)
        p = pose.position
        xyz = np.array([p.x_val, p.y_val, p.z_val], dtype=float)
        if not np.isfinite(xyz).all():
            continue

        x_m = float(p.x_val)
        y_m = float(-p.y_val)
        z_m = float(-p.z_val)
        r_section = radius

        if name in prev_cache and dt > 1e-9:
            px, py = prev_cache[name]
            vx = (x_m - px) / dt
            vy = (y_m - py) / dt
        else:
            vx = 0.0
            vy = 0.0

        rows.append([x_m, y_m, r_section, vx, vy])
        new_cache[name] = (x_m, y_m)

    if rows:
        dyn = np.asarray(rows, dtype=float)
    else:
        dyn = np.zeros((0, 5), dtype=float)
    return dyn, new_cache


def _vehicle_name_sort_key(name: str):
    m = re.fullmatch(r"(.*?)(\d+)", name)
    if m:
        return (m.group(1), int(m.group(2)))
    return (name, float("inf"))


def load_airsim_vehicle_layout(settings_json: str) -> Tuple[List[str], np.ndarray]:
    settings_path = resolve_host_path(settings_json)
    if not settings_path.is_file():
        raise SystemExit(f"airsim_settings_json not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as f:
        settings = json.load(f)

    vehicles = settings.get("Vehicles")
    if not isinstance(vehicles, dict):
        raise SystemExit(f"airsim_settings_json missing 'Vehicles': {settings_path}")

    vehicle_names = sorted(
        [name for name, cfg in vehicles.items() if isinstance(cfg, dict)],
        key=_vehicle_name_sort_key,
    )
    if not vehicle_names:
        raise SystemExit(f"airsim_settings_json has no vehicle entries: {settings_path}")

    init_xy = np.zeros((len(vehicle_names), 2), dtype=float)
    for i, name in enumerate(vehicle_names):
        cfg = vehicles.get(name)
        if "X" not in cfg or "Y" not in cfg:
            raise SystemExit(f"vehicle '{name}' missing X/Y in {settings_path}")
        x = float(cfg["X"])
        y = float(cfg["Y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise SystemExit(f"vehicle '{name}' has non-finite X/Y in {settings_path}")
        init_xy[i, 0] = x
        init_xy[i, 1] = -y

    return vehicle_names, init_xy


def _parse_height_spec(raw_value, num_drones: int, field_name: str) -> np.ndarray:
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        if not math.isfinite(value):
            raise SystemExit(f"{field_name} must be finite")
        return np.full((num_drones,), value, dtype=float)

    arr = np.asarray(raw_value, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != num_drones:
        raise SystemExit(f"{field_name} must be a scalar or a length-{num_drones} list")
    if not np.isfinite(arr).all():
        raise SystemExit(f"{field_name} contains non-finite values")
    return arr


def load_swarm_profile(profile_json: str,
                       num_drones: int,
                       default_init_h: np.ndarray,
                       default_ve_xy: Tuple[float, float],
                       default_he: float) -> Tuple[np.ndarray, Tuple[float, float], float, Path]:
    profile_path = resolve_host_path(profile_json)
    if not profile_path.is_file():
        raise SystemExit(f"swarm_profile_json not found: {profile_path}")

    with profile_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)
    if not isinstance(profile, dict):
        raise SystemExit(f"swarm_profile_json root must be an object: {profile_path}")

    init_h_raw = profile.get("init_h", default_init_h.tolist())
    init_h = _parse_height_spec(init_h_raw, num_drones, "init_h")

    ve_xy_raw = profile.get("ve_xy", list(default_ve_xy))
    ve_xy = np.asarray(ve_xy_raw, dtype=float)
    if ve_xy.shape != (2,) or not np.isfinite(ve_xy).all():
        raise SystemExit(f"ve_xy must be a finite length-2 list in {profile_path}")

    if "he" in profile:
        he = float(profile["he"])
        if not math.isfinite(he):
            raise SystemExit(f"he must be finite in {profile_path}")
    elif np.allclose(init_h, init_h[0]):
        he = float(init_h[0])
    else:
        he = float(default_he)

    return init_h, (float(ve_xy[0]), float(ve_xy[1])), he, profile_path


@dataclass
class SimParams:
    T_v: float = 0.8
    T_psi: float = 0.6
    T_h: float = 0.8
    T_lambda: float = 0.25
    Vxy_min: float = 4.0
    Vxy_max: float = 12.0
    lambda_min: float = -5.0
    lambda_max: float = 5.0
    n_max: float = 3.0
    g: float = 9.8

    Kf: float = 0.25
    Kc: float = 100000.0
    Ka_vn: float = 0.1
    Ka_he: float = 3.0
    Kve: float = 1.0

    R1_comm: float = 20.0
    R_desire: float = 10.0
    R1_lim: float = 2.0
    R2_comm: float = 40.0
    R2_lim: float = 5.0
    cost1_buffer: float = 3.0
    cost1_sigma: float = 8.0
    collision_tol_obs: float = 1.0
    collision_tol_nbr: float = 0.0
    safe_soft_band: float = 1.0
    safe_clearance_m: float = 0.5

    gap_k_speed: float = 0.7
    gap_margin_deg: float = 8.0
    gap_samples: int = 7
    gap_blocked_samples: int = 31
    gap_w_clear: float = 0.4
    gap_w_width: float = 0.3
    gap_w_align: float = 3.0
    gap_w_turn: float = 0.1
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

    yaw_rate_max: float = math.radians(60.0)
    blocked_speed_scale: float = 0.6

    swarm_yaw_enable: bool = True
    swarm_yaw_k: int = 3

    he: float = 50.0
    ve3: float = 0.0
    ve_xy: Tuple[float, float] = (8.0, 0.0)

    theta_lim: float = math.pi / 2.0
    Rc: float = 10.0

    N: int = 58
    R: float = 0.3
    tr: float = 3.0
    Ncmax: int = 20
    Nd: int = 2
    p1: float = 0.9
    e_learn: float = 0.01
    sl: int = 2

    dt: float = 0.5
    sim_time: float = 59.5

    u_lim: float = 0.25
    f1: float = 1.0
    f2: float = 1.0
    w_speed: float = 1.0
    w_direction: float = 1.0
    Vxy_c_lim: float = 0.25
    psi_c_lim: float = 0.10

    obstacles: np.ndarray = field(default_factory=lambda: np.array([
        [120.0, 120.0, 5.0],
        [240.0, 75.0, 5.0],
        [350.0, 40.0, 5.0],
        [240.0, 155.0, 5.0],
        [360.0, 110.0, 5.0],
        [350.0, 180.0, 5.0],
    ], dtype=float))
    square_enable: bool = True
    square_center_xy: Tuple[float, float] = (320.0, 140.0)
    square_side: float = 20.0
    square_vxy: Tuple[float, float] = (0.6, 0.0)
    obstacles_vxy: np.ndarray = field(default_factory=lambda: np.zeros((6, 2), dtype=float))
    obstacles_bounds: Tuple[float, float, float, float] = (0.0, 400.0, 0.0, 200.0)
    obstacles_use_prediction: bool = False
    neighbors_use_prediction: bool = True

    init_P: np.ndarray = field(default_factory=lambda: np.array([
        [14.6929, 107.3676, 68.1682],
        [21.2809, 116.6406, 34.8423],
        [20.3911, 113.6529, 24.6351],
        [3.5699, 108.9509, 54.0000],
        [10.2116, 111.5580, 30.1431],
    ], dtype=float))
    init_Vxy: np.ndarray = field(default_factory=lambda: np.tile(np.array([8.0, 0.0], dtype=float), (5, 1)))

    rng_seed: Optional[int] = 42
    verbose_uav1: bool = True
    verbose_mpio: bool = False
    csv_path: Optional[str] = "uav_debug_log.csv"
    last_w: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float))
    last_w_valid: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))


def neighbors_within(P: np.ndarray, i: int, R: float) -> List[int]:
    idxs = []
    for j in range(P.shape[0]):
        if j == i:
            continue
        if np.linalg.norm(P[j, :2] - P[i, :2]) <= R:
            idxs.append(j)
    return idxs


def _ray_clearance(drone: np.ndarray,
                   obstacles: np.ndarray,
                   yaw: float,
                   r_eff: np.ndarray,
                   r_comm: float) -> float:
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
        dt_hit = math.sqrt(max(re * re - perp2, 0.0))
        hit = t - dt_hit
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


def extract_features(pos_i: np.ndarray,
                     vel_i: np.ndarray,
                     psi_i: float,
                     lamb_i: float,
                     neighbors: np.ndarray,
                     obstacles: np.ndarray,
                     ve_xy: Tuple[float, float],
                     k_n: int = 3,
                     k_o: int = 3) -> np.ndarray:
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


def predict_w_mlp(model,
                  mean: np.ndarray,
                  std: np.ndarray,
                  i: int,
                  P: np.ndarray,
                  V_xy: np.ndarray,
                  psi: np.ndarray,
                  lamb: np.ndarray,
                  obs_state: np.ndarray,
                  obstacles_eval: np.ndarray,
                  sp) -> np.ndarray:
    nbr_idx = neighbors_within(P, i, sp.R1_comm)
    neighbors = np.array([[P[j, 0], P[j, 1], V_xy[j, 0], V_xy[j, 1]] for j in nbr_idx], dtype=float)
    obstacles = np.hstack([obstacles_eval, obs_state[:, 3:5]])
    feat = extract_features(
        pos_i=P[i, :],
        vel_i=V_xy[i, :],
        psi_i=psi[i],
        lamb_i=lamb[i],
        neighbors=neighbors,
        obstacles=obstacles,
        ve_xy=sp.ve_xy,
    )
    x = (feat - mean) / std
    with torch.no_grad():
        w = model(torch.from_numpy(x.astype(np.float32))).numpy()
    return np.clip(w.astype(float), 0.0, 1.0)


# ---------------------------- 避障 ----------------------------

def obstacle_avoidance_uav(P_i: np.ndarray,
                           theta_e: float,
                           obstacles: np.ndarray,
                           sp: SimParams,
                           return_debug: bool = False,
                           v_mag: Optional[float] = None,
                           v_xy: Optional[np.ndarray] = None) -> Tuple[float, List[int], Optional[dict]]:
    drone = P_i[:2]
    R_comm = sp.R2_comm
    theta_v = sp.theta_lim
    R2_lim = sp.R2_lim

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
    alpha_e = _wrap_angle(theta_e - ref_yaw)

    blocked: List[Tuple[float, float]] = []
    sensed_idx: List[int] = []
    lookahead = float(getattr(sp, "gap_lookahead", 0.0))
    if lookahead <= 0.0:
        lookahead = min(0.7 * R_comm, 80.0)

    v_for_eff = float(v_norm) if v_norm > 0.0 else float(np.linalg.norm(sp.ve_xy))
    r_eff_all = obstacles[:, 2] + R2_lim + float(getattr(sp, "gap_k_speed", 0.7)) * v_for_eff * sp.dt
    for j in range(obstacles.shape[0]):
        cx, cy, _ = obstacles[j]
        rel = np.array([cx - drone[0], cy - drone[1]], dtype=float)
        d = float(np.linalg.norm(rel))
        if d < 1e-6:
            continue

        r_eff = float(r_eff_all[j])
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
        if d - r_eff > R_comm:
            continue

        alpha = math.atan2(rel[1], rel[0])
        delta = _wrap_angle(alpha - ref_yaw)

        if d <= r_eff:
            beta = theta_v
        else:
            beta = math.asin(min(1.0, r_eff / d))

        left = delta - beta
        right = delta + beta
        if right < -theta_v or left > theta_v:
            continue

        left = max(left, -theta_v)
        right = min(right, theta_v)
        if left < right:
            blocked.append((left, right))
            sensed_idx.append(j)

    if not blocked:
        if return_debug:
            debug = {
                "gap_type": "none",
                "blocked_cnt": 0,
                "gap_cnt": 0,
                "gap_left": math.nan,
                "gap_right": math.nan,
                "gap_width": math.nan,
                "local_angle": alpha_e,
                "blocked": False,
                "gap_score": math.nan,
                "gap_clear": math.nan,
                "ref_yaw": ref_yaw,
                "theta_e": theta_e,
                "alpha_e": alpha_e,
            }
            return theta_e, [], debug
        return theta_e, []

    blocked.sort(key=lambda seg: seg[0])
    merged = [blocked[0]]
    for a, b in blocked[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))

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

    k1 = float(getattr(sp, "gap_w_clear", 0.4))
    k2 = float(getattr(sp, "gap_w_width", 0.3))
    k3 = float(getattr(sp, "gap_w_align", 3.0))
    k4 = float(getattr(sp, "gap_w_turn", 0.1))

    def _candidate_fracs(n: int) -> List[float]:
        if n <= 2:
            return [0.5]
        if n == 3:
            return [0.25, 0.5, 0.75]
        if n == 4:
            return [0.2, 0.4, 0.6, 0.8]
        return [0.2, 0.35, 0.5, 0.65, 0.8]

    candidate_gaps: List[Tuple[str, float, float]] = []
    for a, b in interior_gaps:
        candidate_gaps.append(("interior", a, b))
    if bool(getattr(sp, "gap_boundary_enable", True)):
        for a, b in boundary_gaps:
            candidate_gaps.append(("boundary", a, b))

    best_gap = None
    best_angle = 0.0
    best_score = -1e9
    best_clear = 0.0
    gap_type = "none"
    gap_cnt = len(candidate_gaps)
    alpha_goal = _wrap_angle(theta_e - ref_yaw)

    # Phase 1: Collect all feasible candidates with safety metrics
    feasible_candidates = []
    for stage, a, b in candidate_gaps:
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
        if a <= alpha_goal <= b:
            candidates.append(alpha_goal)
        for alpha in sorted(set(candidates)):
            if abs(alpha) > (theta_v - edge_margin):
                continue
            yaw = _wrap_angle(ref_yaw + alpha)
            align = math.cos(_wrap_angle(yaw - theta_e))
            if align < prog_min:
                continue
            clear = _ray_clearance(drone, obs_use, yaw, r_eff_use, sp.R2_comm)
            clear_norm = clear / m_safe if m_safe > 1e-6 else 1.0
            clear_norm = clamp(clear_norm, 0.0, 1.0)
            if clear_norm < clear_min:
                continue
            feasible_candidates.append((stage, a, b, alpha, yaw, align, clear, clear_norm, width))

    # Phase 2: Score feasible candidates with alignment priority
    if feasible_candidates:
        for stage, a, b, alpha, yaw, align, clear, clear_norm, width in feasible_candidates:
            width_norm = width / (2.0 * theta_v)
            turn_term = -abs(alpha) / max(theta_v, 1e-6)
            score = (
                k3 * align +
                k1 * clear_norm +
                k2 * width_norm +
                k4 * turn_term
            )
            if stage == "boundary":
                score -= 0.5 * edge_pen * (abs(alpha) / max(theta_v, 1e-6))
            if score > best_score:
                best_score = score
                best_angle = alpha
                best_gap = (a, b)
                best_clear = clear
                gap_type = stage

    if best_gap is None:
        m = int(getattr(sp, "gap_blocked_samples", 31))
        angles = [(-theta_v + 2.0 * theta_v * i / (m - 1)) for i in range(m)]
        best_c = -1.0
        best_a = 0.0
        for ang in angles:
            if abs(ang) > (theta_v - edge_margin):
                continue
            yaw = _wrap_angle(ref_yaw + ang)
            clear = _ray_clearance(drone, obs_use, yaw, r_eff_use, sp.R2_comm)
            align = math.cos(_wrap_angle(yaw - theta_e))
            score = clear + 0.2 * align
            if score > best_c:
                best_c = score
                best_a = ang
        blocked_clear = _ray_clearance(drone, obs_use, _wrap_angle(ref_yaw + best_a), r_eff_use, sp.R2_comm)
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
                "gap_clear": blocked_clear,
                "ref_yaw": ref_yaw,
                "theta_e": theta_e,
                "alpha_e": alpha_e,
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
            "ref_yaw": ref_yaw,
            "theta_e": theta_e,
            "alpha_e": alpha_e,
        }
        return yaw_desired, sensed_idx, debug
    return yaw_desired, sensed_idx


# ---------------------------- MPIO 相关结构 & 函数 ----------------------------

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


def _resolve_rng(sp, rng: Optional[np.random.Generator]) -> np.random.Generator:
    """Return a generator without re-seeding on every call."""
    if rng is not None:
        return rng
    existing = getattr(sp, "_rng", None)
    if isinstance(existing, np.random.Generator):
        return existing
    new_rng = np.random.default_rng(getattr(sp, "rng_seed", None))
    setattr(sp, "_rng", new_rng)
    return new_rng


def nondominated_sort_safe(Costs: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    N = Costs.shape[0]
    dom_count = np.zeros(N, dtype=int)
    S: List[List[int]] = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
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

        for local_idx, pigeon_idx in enumerate(F):
            pop[pigeon_idx].CrowdingDistance = float(np.sum(d[local_idx, :]))
    return pop


def calculateCosts(P_i_next: np.ndarray, V_xy_i_next: np.ndarray, P_swarm: np.ndarray, i_uav: int,
                   V_xy_swarm: np.ndarray, obstacles: np.ndarray, sp) -> Tuple[float, float, int, int]:
    """
    成本函数：
      - cost3：与障碍物的硬约束
      - cost4：与邻居的硬约束
      - cost1：软约束（避障区：最大化投影；巡航区：速度&方向对齐）
      - cost2：软约束（编队质量与速度对齐）
    """
    pos_i_next = P_i_next[:2]
    vel_i_next = V_xy_i_next
    pos_i_cur = np.asarray(P_swarm[i_uav, :2], dtype=float)

    nbr_xy = P_swarm[:, :2]
    if bool(getattr(sp, "neighbors_use_prediction", False)):
        nbr_xy = nbr_xy + V_xy_swarm * sp.dt

    cost3 = 0
    if obstacles.size > 0:
        p0 = pos_i_cur
        p1 = pos_i_next
        centers = obstacles[:, :2]
        radii = obstacles[:, 2] + sp.R2_lim
        dists = _seg_point_dists(p0, p1, centers)
        if dists.size > 0 and np.any(dists < radii):
            cost3 = 1

    cost4 = 0
    for j in range(P_swarm.shape[0]):
        if i_uav != j and np.linalg.norm(nbr_xy[j] - pos_i_next) < sp.R1_lim:
            cost4 = 1
            break

    ve = np.array(sp.ve_xy, dtype=float)
    ve_mag = float(np.linalg.norm(ve))
    ve_unit = ve / (ve_mag if ve_mag > 1e-6 else 1.0)

    buffer = float(getattr(sp, "cost1_buffer", 3.0))
    sigma = float(getattr(sp, "cost1_sigma", 8.0))
    d_min = 1e9
    for j in range(obstacles.shape[0]):
        r_obs = obstacles[j, 2]
        d = np.linalg.norm(obstacles[j, :2] - pos_i_cur) - (r_obs + sp.R2_lim + buffer)
        if d < d_min:
            d_min = d

    alpha = float(np.clip(1.0 - d_min / sigma, 0.0, 1.0))
    forward = float(np.dot(vel_i_next, ve_unit))
    progress_cost = ve_mag - forward
    vel_match = abs(ve[0] - vel_i_next[0]) + abs(ve[1] - vel_i_next[1])
    cost1 = alpha * progress_cost + (1.0 - alpha) * vel_match

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


def update_drone_state(P: np.ndarray, V_xy: np.ndarray, psi: float, lamb: float,
                       u_prime: np.ndarray, sp) -> Tuple[np.ndarray, np.ndarray, float, float, Dict[str, int]]:
    """
    Eq.(25)(26)(3)：控制输入 + 自动驾驶仪动力学的一拍推进。
    这里仅在 MPIO 内部“预测下一拍”用，不直接作用于 AirSim。
    """
    flags = {"u_dead": 0, "clipL": 0, "clipH": 0, "rate_clamp": 0, "snap_v": 0, "snap_psi": 0}

    u_xy = u_prime[:2].copy()
    if norm2(u_xy) < sp.u_lim:
        u_xy[:] = 0.0
        flags["u_dead"] = 1

    V_xy_mag = float(np.linalg.norm(V_xy))

    V_xy_c = V_xy_mag + sp.T_v * (u_xy[0] * math.cos(psi) + u_xy[1] * math.sin(psi))
    if V_xy_mag < 0.1:
        psi_c = psi
    else:
        psi_c = psi + sp.T_psi / V_xy_mag * (-u_xy[0] * math.sin(psi) + u_xy[1] * math.cos(psi))

    h_c = P[2] + (sp.T_h / sp.T_lambda) * lamb + sp.T_h * u_prime[2]

    # 允许误差吸附
    ve_norm = float(np.linalg.norm(sp.ve_xy))
    if abs(V_xy_c - ve_norm) < sp.Vxy_c_lim:
        V_xy_c = ve_norm
        flags["snap_v"] = 1
    psi_m = math.atan2(sp.ve_xy[1], sp.ve_xy[0])
    if abs(psi_c - psi_m) < sp.psi_c_lim:
        psi_c = psi_m
        flags["snap_psi"] = 1

    # 动力学
    V_xy_dot = (V_xy_c - V_xy_mag) / sp.T_v
    psi_dot = (psi_c - psi) / sp.T_psi
    lambda_dot = ((h_c - P[2]) / sp.T_h - lamb / sp.T_lambda)

    # 航向角速率限
    v_for_limit = max(V_xy_mag, 0.1)
    psi_dot_lim = sp.n_max * sp.g / v_for_limit
    if abs(psi_dot) > psi_dot_lim:
        psi_dot = math.copysign(psi_dot_lim, psi_dot)
        flags["rate_clamp"] = 1

    V_xy_mag_next = V_xy_mag + V_xy_dot * sp.dt
    psi_next = psi + psi_dot * sp.dt
    lambda_next = lamb + lambda_dot * sp.dt
    lambda_next = clamp(lambda_next, sp.lambda_min, sp.lambda_max)

    h_next = P[2] + lambda_next * sp.dt
    x_next = P[0] + V_xy_mag_next * math.cos(psi_next) * sp.dt
    y_next = P[1] + V_xy_mag_next * math.sin(psi_next) * sp.dt

    V_xy_next = np.array(
        [V_xy_mag_next * math.cos(psi_next), V_xy_mag_next * math.sin(psi_next)],
        dtype=float,
    )

    if V_xy_mag_next < sp.Vxy_min:
        flags["clipL"] = 1
    if V_xy_mag_next > sp.Vxy_max:
        flags["clipH"] = 1

    P_next = np.array([x_next, y_next, h_next], dtype=float)
    return P_next, V_xy_next, psi_next, lambda_next, flags


def optimizePigeons_core(pop: List[Pigeon], sp, P_swarm: np.ndarray, i_uav: int,
                         V_xy_swarm: np.ndarray, obstacles: np.ndarray, ff_raw: np.ndarray,
                         fa_raw: np.ndarray, fc_raw: np.ndarray, vf_z: float, vo_raw: np.ndarray,
                         lamb_i: float, psi_i: float, rng: np.random.Generator) -> np.ndarray:
    """
    修改版 MPIO（带层级学习/拥挤度/精英保留/逐步减员）。
    最终从 Pareto 前沿中选 cost2 最小的个体。
    """
    rng = _resolve_rng(sp, rng)
    R_map = sp.R
    ft = sp.tr
    historical_A: List[Pigeon] = []
    archive_size = 50

    ff_raw = np.array(ff_raw, dtype=float)
    fa_raw = np.array(fa_raw, dtype=float)
    fc_raw = np.array(fc_raw, dtype=float)
    vo_raw = np.array(vo_raw, dtype=float)

    for Nc in range(1, max(1, sp.Ncmax) + 1):
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

        # 历史存档
        if not historical_A:
            combined_A = current_S1_pop
        else:
            combined_A = list(historical_A) + list(current_S1_pop)

        if combined_A:
            A_Costs = np.vstack([p.Cost12 for p in combined_A])
            _, A_fronts = nondominated_sort_safe(A_Costs)
            historical_A = [combined_A[idx] for idx in (A_fronts[0] if A_fronts and A_fronts[0] else [])]
            if len(historical_A) > archive_size:
                historical_A = CalcCrowdingDistance(historical_A, [list(range(len(historical_A)))])
                historical_A.sort(key=lambda pp: pp.CrowdingDistance)
                historical_A = historical_A[-archive_size:]
        else:
            historical_A = []

        # 选择 Xg / Xc
        if historical_A:
            Xg = historical_A[rng.integers(0, len(historical_A))].Position.copy()
        else:
            Xg = current_S1_pop[rng.integers(0, len(current_S1_pop))].Position.copy()
        Xc = np.mean(np.vstack([p.Position for p in current_S1_pop]), axis=0)

        # 领导者数量
        nPop = len(pop)
        nLeaders = max(1, min(nPop, int(math.ceil(sp.p1 * nPop))))

        # 按 Rank 排序保证“上层可学”
        order = np.argsort([p.Rank for p in pop])
        pop = [pop[idx] for idx in order]

        for i in range(nPop):
            old = dataclasses.replace(pop[i])  # 备份
            if pop[i].Rank <= nLeaders:
                rand1 = rng.random()
                rand2 = rng.random()
                term1 = math.exp(-R_map * Nc) * pop[i].V
                distXg = np.linalg.norm(Xg - pop[i].Position)
                distXc = np.linalg.norm(Xc - pop[i].Position)
                lgXg = math.log(distXg + 1.0)
                lgXc = math.log(distXc + 1.0)
                Vi_new = (
                    term1
                    + rand1 * ft * (1.0 - lgXg) * (Xg - pop[i].Position)
                    + rand2 * ft * lgXc * (Xc - pop[i].Position)
                )
                Vi_new = np.clip(Vi_new, -0.2, 0.2)
                pop[i].V = Vi_new
                pop[i].Position = np.clip(pop[i].Position + Vi_new, 0.0, 1.0)
            else:
                upper_indices = [idx for idx in range(nPop) if pop[idx].Rank < pop[i].Rank]
                if not upper_indices:
                    pop[i].Position = np.clip(
                        pop[i].Position + 0.01 * (2.0 * rng.random(2) - 1.0), 0.0, 1.0
                    )
                    pop[i].V = 0.01 * (2.0 * rng.random(2) - 1.0)
                else:
                    j_idx = upper_indices[rng.integers(0, len(upper_indices))]
                    for _ in range(sp.sl):
                        d_star = rng.integers(0, pop[i].Position.size)
                        pop[i].Position[d_star] = pop[j_idx].Position[d_star] + sp.e_learn * (
                            2.0 * rng.random() - 1.0
                        )
                        pop[i].Position[d_star] = clamp(pop[i].Position[d_star], 0.0, 1.0)
                    pop[i].V = np.clip(pop[i].Position - old.Position, -0.2, 0.2)

            # 重新评估
            w_p = pop[i].Position
            vf_prime_xy = w_p[0] * (ff_raw + fa_raw) + fc_raw
            vo_prime_xy = w_p[1] * vo_raw
            u_total_xy = vf_prime_xy + vo_prime_xy
            u_prime_p = np.array(
                [
                    u_total_xy[0] - V_xy_swarm[i_uav, 0],
                    u_total_xy[1] - V_xy_swarm[i_uav, 1],
                    vf_z,
                ],
                dtype=float,
            )
            P_next, V_xy_next, _, _, _ = update_drone_state(
                P_swarm[i_uav, :], V_xy_swarm[i_uav, :], psi_i, lamb_i, u_prime_p, sp
            )
            c1, c2, c3, c4 = calculateCosts(
                P_next, V_xy_next, P_swarm, i_uav, V_xy_swarm, obstacles, sp
            )
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

        # 减员
        if Nc <= sp.Ncmax and len(pop) > sp.Nd:
            Costs_after = np.vstack([p.Cost12 for p in pop])
            rank_after, fronts_after = nondominated_sort_safe(Costs_after)
            for ii, p in enumerate(pop):
                p.Rank = float(rank_after[ii])
            pop = CalcCrowdingDistance(pop, fronts_after)

            sort_keys = np.lexsort(
                (-np.array([p.CrowdingDistance for p in pop]), np.array([p.Rank for p in pop]))
            )
            current_N = len(pop)
            keep = max(2, current_N - sp.Nd)
            pop = [pop[idx] for idx in sort_keys[:keep]]

    if not pop:
        return np.array([0.2, 0.8], dtype=float)

    Costs = np.vstack([p.Cost12 for p in pop])
    _, frontsF = nondominated_sort_safe(Costs)
    if not frontsF or not frontsF[0]:
        return np.array([0.2, 0.8], dtype=float)

    s1_pop = [pop[idx] for idx in frontsF[0]]
    s1_costs = np.vstack([p.Cost12 for p in s1_pop])
    valid_rows = np.all(np.isfinite(s1_costs), axis=1)
    if not np.any(valid_rows):
        return np.array([0.2, 0.8], dtype=float)

    valid_costs = s1_costs[valid_rows, :]
    valid_pop = [p for (p, ok) in zip(s1_pop, valid_rows) if ok]
    best_idx = int(np.argmin(valid_costs[:, 1]))
    w = valid_pop[best_idx].Position.copy()
    return np.clip(w.astype(float), 0.0, 1.0)


def optimizePigeons_wrapper(sp, P_swarm: np.ndarray, i_uav: int, V_xy_swarm: np.ndarray,
                            obstacles: np.ndarray, ff_raw: np.ndarray, fa_raw: np.ndarray,
                            fc_raw: np.ndarray, vf_z: float, vo_raw: np.ndarray,
                            lamb_i: float, psi_i: float,
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
            u_prime_prev = np.array(
                [
                    u_total_prev[0] - V_xy_swarm[i_uav, 0],
                    u_total_prev[1] - V_xy_swarm[i_uav, 1],
                    vf_z,
                ],
                dtype=float,
            )

            P_prev, V_prev, _, _, _ = update_drone_state(
                P_swarm[i_uav, :], V_xy_swarm[i_uav, :], psi_i, lamb_i, u_prime_prev, sp
            )
            c1p, c2p, c3p, c4p = calculateCosts(
                P_prev, V_prev, P_swarm, i_uav, V_xy_swarm, obstacles, sp
            )
            infeasible_prev = (c3p == 1 or c4p == 1)
            if not infeasible_prev:
                pop.append(
                    Pigeon(
                        Position=w_prev,
                        V=np.zeros(2, dtype=float),
                        Cost12=np.array([c1p, c2p], dtype=float),
                        Rank=0.0,
                    )
                )

    for i in range(sp.N):
        w_p = X[i, :].copy()
        vf_prime_xy = w_p[0] * (ff_raw + fa_raw) + fc_raw
        vo_prime_xy = w_p[1] * vo_raw
        u_total_xy = vf_prime_xy + vo_prime_xy
        u_prime_p = np.array(
            [
                u_total_xy[0] - V_xy_swarm[i_uav, 0],
                u_total_xy[1] - V_xy_swarm[i_uav, 1],
                vf_z,
            ],
            dtype=float,
        )

        P_next, V_xy_next, _, _, _ = update_drone_state(
            P_swarm[i_uav, :], V_xy_swarm[i_uav, :], psi_i, lamb_i, u_prime_p, sp
        )
        c1, c2, c3, c4 = calculateCosts(
            P_next, V_xy_next, P_swarm, i_uav, V_xy_swarm, obstacles, sp
        )
        infeasible = (c3 == 1 or c4 == 1)
        init_cost = np.array([np.inf, np.inf], dtype=float) if infeasible else np.array([c1, c2], dtype=float)
        rank = float("inf") if infeasible else 0.0
        pop.append(
            Pigeon(
                Position=w_p,
                V=V[i, :].copy(),
                Cost12=init_cost,
                Rank=rank,
            )
        )

    if pop:
        valid0 = int(np.sum(np.all(np.isfinite(np.vstack([p.Cost12 for p in pop])), axis=1)))
    else:
        valid0 = 0
    if valid0 == 0:
        print(f"[MPIO] step infeasible: no valid candidates for uav={i_uav}", flush=True)

    return optimizePigeons_core(
        pop,
        sp,
        P_swarm,
        i_uav,
        V_xy_swarm,
        obstacles,
        ff_raw,
        fa_raw,
        fc_raw,
        vf_z,
        vo_raw,
        lamb_i,
        psi_i,
        rng=rng,
    )


# =====================================================
#          1. 仿真参数
# =====================================================

sp = SimParams()

dt = sp.dt
sim_time = sp.sim_time
steps = int(sim_time / dt) + 1  # 0, 0.5, ..., 59.5 共 120 步

vehicle_names = [f"UAV{i + 1}" for i in range(sp.init_P.shape[0])]
num_drones = len(vehicle_names)

# AirSim 指令级速度限制（与模型里的 Vxy_min/max 区分开）
CMD_VXY_MIN = 1.0
CMD_VXY_MAX = 12.0
CMD_VZ_MAX = 6.0

# Yaw ???????????? 10s??????????? True ? 0 ?????
YAW_CONTROL_ENABLED = False
YAW_DISABLE_FOR = 10.0


# 期望大方向（本地 SimParams）
global_ve = np.array(sp.ve_xy, dtype=float)

# 初始位置（MATLAB 坐标）
initial_positions_xy_matlab = {
    vehicle_names[i]: (float(sp.init_P[i, 0]), float(sp.init_P[i, 1]))
    for i in range(num_drones)
}

# 障碍物（MATLAB 坐标）
P_obstacles = np.asarray(sp.obstacles, dtype=float).copy()
if bool(getattr(sp, "square_enable", False)):
    square_r = (math.sqrt(2.0) / 2.0) * float(sp.square_side)
    square_obs = np.array(
        [[float(sp.square_center_xy[0]), float(sp.square_center_xy[1]), square_r]],
        dtype=float,
    )
    P_obstacles = np.vstack([P_obstacles, square_obs])

# =====================================================
#          2. MPIO / 模型参数（以 SimParams 为准）
# =====================================================

sp.dt = dt
sp.sim_time = sim_time



# =====================================================
#          3. 坐标变换工具：NED <-> MATLAB
# =====================================================
# =====================================================
#  坐标变换：X轴对应，Y轴取反 (镜像坐标系)
# =====================================================

def ned_to_matlab_pos(pos_ned):
    """
    AirSim NED -> MPIO 坐标
    X (前) -> x (前)
    Y (右) -> y (算法里的y是正的，AirSim里是负的，所以取反)
    Z (下) -> h (反号)
    """
    x_m = pos_ned.x_val        # X 轴保持一致！
    y_m = -pos_ned.y_val       # Y 轴取反！(-120 变 +120)
    z_m = -pos_ned.z_val
    return np.array([x_m, y_m, z_m], dtype=float)

def ned_to_matlab_vel(vel_ned):
    """
    速度同理：X不变，Y取反
    """
    vx_m = vel_ned.x_val
    vy_m = -vel_ned.y_val      # Y 轴取反
    return np.array([vx_m, vy_m], dtype=float)

def matlab_vel_to_ned(vx_m, vy_m, vz_m=0.0):
    """
    发指令时：X不变，Y取反转回去
    """
    vx_ned = vx_m
    vy_ned = -vy_m             # Y 轴取反
    vz_ned = -vz_m
    return vx_ned, vy_ned, vz_ned


def matlab_abs_pos_to_ned_plot(P_abs: np.ndarray) -> airsim.Vector3r:
    """
    算法绝对坐标 -> AirSim 世界 NED 轨迹绘制坐标。
    simPlotLineList 需要的是世界坐标，不是相对各自坑位的局部位移。
    """
    dx = float(P_abs[0])
    dy = float(P_abs[1])
    z_up = float(P_abs[2])
    return airsim.Vector3r(dx, -dy, -z_up)

def get_swarm_state(client, vehicle_names):
    num_drones = len(vehicle_names)
    P = np.zeros((num_drones, 3), dtype=float)
    V_xy = np.zeros((num_drones, 2), dtype=float)
    psi = np.zeros(num_drones, dtype=float)
    lam = np.zeros(num_drones, dtype=float)
    prev_psi = getattr(get_swarm_state, "_prev_psi", None)
    if prev_psi is None or len(prev_psi) != num_drones:
        prev_psi = np.zeros(num_drones, dtype=float)
        get_swarm_state._prev_psi = prev_psi

    for idx, name in enumerate(vehicle_names):
        state = client.getMultirotorState(vehicle_name=name)
        pos_ned = state.kinematics_estimated.position
        vel_ned = state.kinematics_estimated.linear_velocity

        # 1. 转换相对位移 (X对X, Y对Y, Z反号)
        displacement_m = ned_to_matlab_pos(pos_ned)

        # 2. 【必须加上】初始坐标
        # 理由：AirSim 返回的是相对(0,0)的位移，必须加上初始坑位，算法才知道它们没重叠
        P[idx, 0] = sp.init_P[idx][0] + displacement_m[0]
        P[idx, 1] = sp.init_P[idx][1] + displacement_m[1]
        P[idx, 2] = displacement_m[2]

        # 3. 速度转换
        V_xy[idx, :] = ned_to_matlab_vel(vel_ned)
        # 在你算完 V_xy[idx,:] 之后
        v = V_xy[idx, :]
        vnorm = np.linalg.norm(v)
        if vnorm > 0.3:
            psi[idx] = math.atan2(v[1], v[0])   # psi=0 指向 +x
        else:
            psi[idx] = prev_psi[idx]  # 速度太小保持上一拍 psi

        # 4. 垂直速度
        lam[idx] = -vel_ned.z_val
        prev_psi[idx] = psi[idx]

    return P, V_xy, psi, lam
# =====================================================
#          5. 邻居力计算（直接用 mpio_swarm_sim 里的形式）
# =====================================================

def compute_raw_forces(P: np.ndarray, V_xy: np.ndarray, i: int, sp) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 ff_raw, fa_raw, fc_raw （本地拷贝自 simv1）"""
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


# =====================================================
#          6. 主控制循环：在线 MPIO + AirSim 控制
# =====================================================

def run_on_airsim(backend: str = "mpio",
                  model_dir: Optional[str] = None,
                  log_path: Optional[str] = None,
                  scene_json: Optional[str] = None,
                  airsim_settings_json: Optional[str] = None,
                  swarm_profile_json: Optional[str] = None,
                  R2_comm: Optional[float] = None,
                  gap_lookahead: Optional[float] = None,
                  R_desire: Optional[float] = None,
                  R1_comm: Optional[float] = None,
                  f1: Optional[float] = None,
                  f2: Optional[float] = None,
                  w_direction: Optional[float] = None,
                  n_max: Optional[float] = None,
                  control_dt: Optional[float] = None,
                  hold_time: Optional[float] = None,
                  warmup_time: float = 2.0,
                  tau_v_smooth: float = 0.3,
                  a_max: float = 3.0,
                  uav1_debug_every: int = 0,
                  all_uav_debug_every: int = 0,
                  csv_flush_every: int = 100,
                  sense_debug_every: int = 0,
                  timing_log_every: int = 0,
                  status_every: int = 0,
                  plot_traj_every: int = 0,
                  plot_traj_thickness: float = 8.0,
                  save_logs: bool = True):
    backend = backend.lower()
    if backend not in {"mpio", "mlp"}:
        raise SystemExit(f"unknown backend: {backend}; valid=('mpio', 'mlp')")
    control_dt = float(control_dt if control_dt is not None else dt)
    if control_dt <= 0.0:
        raise SystemExit("control_dt must be > 0")
    hold = float(hold_time if hold_time is not None else max(control_dt * 1.1, control_dt + 0.01))
    if hold <= 0.0:
        raise SystemExit("hold_time must be > 0")
    if warmup_time < 0.0:
        raise SystemExit("warmup_time must be >= 0")
    if tau_v_smooth <= 0.0:
        raise SystemExit("tau_v_smooth must be > 0")
    if a_max <= 0.0:
        raise SystemExit("a_max must be > 0")
    if uav1_debug_every < 0:
        raise SystemExit("uav1_debug_every must be >= 0")
    if all_uav_debug_every < 0:
        raise SystemExit("all_uav_debug_every must be >= 0")
    if csv_flush_every <= 0:
        raise SystemExit("csv_flush_every must be >= 1")
    if sense_debug_every < 0:
        raise SystemExit("sense_debug_every must be >= 0")
    if timing_log_every < 0:
        raise SystemExit("timing_log_every must be >= 0")
    if status_every < 0:
        raise SystemExit("status_every must be >= 0")
    if plot_traj_every < 0:
        raise SystemExit("plot_traj_every must be >= 0")
    if plot_traj_thickness <= 0.0:
        raise SystemExit("plot_traj_thickness must be > 0")
    cli_scene_overrides = {
        "R2_comm": R2_comm,
        "gap_lookahead": gap_lookahead,
        "R_desire": R_desire,
        "R1_comm": R1_comm,
        "f1": f1,
        "f2": f2,
        "w_direction": w_direction,
        "n_max": n_max,
    }
    for key, val in cli_scene_overrides.items():
        if val is None:
            continue
        if not math.isfinite(float(val)):
            raise SystemExit(f"{key} must be finite when provided via command line")
    steps = int(sim_time / control_dt) + 1

    if backend == "mpio" and abs(control_dt - sp.dt) > 1e-9:
        print(
            f"[warn] control_dt={control_dt:.3f}s differs from solver sp.dt={sp.dt:.3f}s; "
            "this is mainly intended for the mlp backend.",
            flush=True,
        )

    mlp_model = None
    mean = None
    std = None
    if backend == "mlp":
        if not model_dir:
            raise SystemExit("model_dir is required when backend=mlp")
        mlp_model, mean, std = load_mlp(Path(model_dir))
        print(f"Loaded MLP backend from {model_dir}")

    vehicle_names = [f"UAV{i + 1}" for i in range(sp.init_P.shape[0])]
    num_drones = len(vehicle_names)
    runtime_init_xy = np.asarray(sp.init_P[:, :2], dtype=float).copy()

    settings_path = airsim_settings_json
    if not settings_path:
        for candidate in DEFAULT_AIRSIM_SETTINGS_JSON_CANDIDATES:
            if candidate.is_file():
                settings_path = str(candidate)
                break
    if settings_path:
        vehicle_names, runtime_init_xy = load_airsim_vehicle_layout(settings_path)
        num_drones = len(vehicle_names)
        print(
            f"Loaded AirSim spawn XY for {num_drones} UAVs from {Path(settings_path)}."
        )

    default_init_h = np.asarray(sp.init_P[:, 2], dtype=float)
    if swarm_profile_json:
        init_h, ve_xy_cfg, he_cfg, profile_path = load_swarm_profile(
            swarm_profile_json,
            num_drones,
            default_init_h if default_init_h.shape[0] == num_drones else np.full((num_drones,), float(sp.he), dtype=float),
            sp.ve_xy,
            sp.he,
        )
        print(
            f"Loaded swarm profile from {profile_path}; "
            f"init_h shape={init_h.shape[0]}, ve_xy=({ve_xy_cfg[0]:.2f},{ve_xy_cfg[1]:.2f}), he={he_cfg:.2f}."
        )
    else:
        if default_init_h.shape[0] != num_drones:
            raise SystemExit(
                f"settings.json defines {num_drones} UAVs but built-in init_P has {default_init_h.shape[0]}; "
                "provide --swarm_profile_json with matching init_h."
            )
        init_h = default_init_h.copy()
        ve_xy_cfg = (float(sp.ve_xy[0]), float(sp.ve_xy[1]))
        he_cfg = float(sp.he)

    sp.init_P = np.zeros((num_drones, 3), dtype=float)
    sp.init_P[:, :2] = runtime_init_xy
    sp.init_P[:, 2] = init_h
    sp.ve_xy = ve_xy_cfg
    sp.he = he_cfg

    obstacles_plan = np.asarray(P_obstacles, dtype=float)
    dynamic_radii = np.zeros((0,), dtype=float)
    if scene_json:
        obstacles_plan, obstacles_use_prediction, obstacles_bounds_scene, scene_overrides, dynamic_radii = load_scene_obstacles_and_overrides(scene_json)
        for key, val in scene_overrides.items():
            setattr(sp, key, val)
        sp.obstacles_use_prediction = obstacles_use_prediction
        if obstacles_bounds_scene is not None:
            sp.obstacles_bounds = obstacles_bounds_scene
        override_desc = ", ".join(f"{k}={scene_overrides[k]:.3f}" for k in sorted(scene_overrides))
        print(
            f"Loaded {obstacles_plan.shape[0]} obstacles from scene_json={Path(scene_json)}; "
            f"applied scene overrides: {override_desc if override_desc else 'none'}; "
            f"obstacles_use_prediction={sp.obstacles_use_prediction}; "
            f"obstacles_bounds=({sp.obstacles_bounds[0]:.2f},{sp.obstacles_bounds[1]:.2f},"
            f"{sp.obstacles_bounds[2]:.2f},{sp.obstacles_bounds[3]:.2f}); "
            f"dynamic_spheres={dynamic_radii.shape[0]}; "
            "keeping AirSim init_P / INIT_H / he / ve_xy unchanged."
        )
    applied_cli_overrides = {}
    for key, val in cli_scene_overrides.items():
        if val is None:
            continue
        setattr(sp, key, float(val))
        applied_cli_overrides[key] = float(val)
    if applied_cli_overrides:
        cli_override_desc = ", ".join(f"{k}={applied_cli_overrides[k]:.3f}" for k in sorted(applied_cli_overrides))
        print(f"Applied command-line scene param overrides: {cli_override_desc}")
    static_obs_state = np.zeros((obstacles_plan.shape[0], 5), dtype=float)
    static_obs_state[:, :3] = obstacles_plan

    # ---------- 连接 AirSim ----------
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    dynamic_actor_specs: List[Dict[str, float]] = []
    dyn_prev_cache: Dict[str, Tuple[float, float]] = {}
    if dynamic_radii.shape[0] > 0:
        actor_names = sorted(client.simListSceneObjects(".*BP_DynamicSphere.*"), key=_vehicle_name_sort_key)
        n_match = min(len(actor_names), int(dynamic_radii.shape[0]))
        if len(actor_names) != int(dynamic_radii.shape[0]):
            print(
                f"[warn] scene expects {int(dynamic_radii.shape[0])} dynamic spheres but AirSim found {len(actor_names)} "
                f"matching BP_DynamicSphere; using first {n_match}.",
                flush=True,
            )
        dynamic_actor_specs = [
            {"name": actor_names[k], "radius": float(dynamic_radii[k])}
            for k in range(n_match)
        ]
        if dynamic_actor_specs:
            dyn_info = ", ".join(
                f"{spec['name']}(R={spec['radius']:.1f})"
                for spec in dynamic_actor_specs
            )
            print(f"Tracking {len(dynamic_actor_specs)} dynamic obstacle actors from AirSim: {dyn_info}")
    obs_state = static_obs_state.copy()

    # ---------- 解锁并起飞 ----------
    for nm in vehicle_names:
        client.enableApiControl(True, vehicle_name=nm)
        client.armDisarm(True, vehicle_name=nm)

    takeoff_tasks = [client.takeoffAsync(vehicle_name=nm) for nm in vehicle_names]
    for t in takeoff_tasks:
        t.join()
    print("All UAVs takeoff.")

    # ---------- 垂直爬升到各自初始高度 ----------
    INIT_H = np.asarray(sp.init_P[:, 2], dtype=float).tolist()
    print("Climb to per-UAV initial altitudes...")
    for i, nm in enumerate(vehicle_names):
        client.moveToZAsync(-INIT_H[i], 5.0, vehicle_name=nm)

    tol = 1.0
    reached = False
    while not reached:
        reached = True
        time.sleep(0.2)
        for i, nm in enumerate(vehicle_names):
            state = client.getMultirotorState(vehicle_name=nm)
            alt = -state.kinematics_estimated.position.z_val
            if abs(alt - INIT_H[i]) > tol:
                reached = False
    print("All UAVs reached initial altitudes.")

    print(f">>> Initializing Velocity to match paper ve={sp.ve_xy} ...")

    # 计算 NED 坐标系下的初始速度向量
    vx_init_m = sp.ve_xy[0]  # 10.0
    vy_init_m = sp.ve_xy[1]  # 0.0
    # 这里的转换函数必须是你代码里那个正确的“旋转版”
    vx_init_ned, vy_init_ned, _ = matlab_vel_to_ned(vx_init_m, vy_init_m, 0)

    # 可选的速度预热：默认保留历史行为，但允许设为 0 直接进入主控制。
    if warmup_time > 0.0:
        warmup_steps = int(warmup_time / 0.1)
        for _ in range(warmup_steps):
            for nm in vehicle_names:
                client.moveByVelocityAsync(
                    vx_init_ned, vy_init_ned, 0,
                    duration=0.2,  # 稍微大于循环间隔，保持连贯
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
                    vehicle_name=nm
                )
            time.sleep(0.1)
        print(f">>> Velocity Initialized with warmup_time={warmup_time:.2f}s. Starting {backend.upper()} Control Loop.")
    else:
        print(f">>> Skipping velocity warmup. Starting {backend.upper()} Control Loop immediately after climb.")

    # ---------- 初始化 lambda/psi ----------
    P, V_xy, psi, lam = get_swarm_state(client,vehicle_names)

    print(f"Start online control for {sim_time:.1f}s, control_dt={control_dt:.3f}s, steps={steps}.")

    out_log_path = Path(log_path) if log_path else Path(f"airsim_{backend}_log.npz")
    debug_csv_file = None
    debug_writer = None
    debug_csv_path = None
    all_debug_csv_file = None
    all_debug_writer = None
    all_debug_csv_path = None
    sense_csv_file = None
    sense_writer = None
    sense_csv_path = None
    if save_logs:
        out_log_path.parent.mkdir(parents=True, exist_ok=True)
        if uav1_debug_every > 0:
            debug_csv_path = out_log_path.with_name(f"{out_log_path.stem}_uav1_debug.csv")
            debug_csv_path.parent.mkdir(parents=True, exist_ok=True)
            debug_csv_file = debug_csv_path.open("w", newline="", encoding="utf-8")
            debug_writer = csv.writer(debug_csv_file)
            debug_writer.writerow([
                "step", "t", "pos_x", "pos_y", "v",
                "psi_deg", "theta_e_deg", "ref_yaw_deg",
                "yaw_raw_deg", "yaw_cmd_deg",
                "yaw_err_raw_deg", "yaw_err_cmd_deg", "yaw_step_cap_deg",
                "vo_heading_deg", "vdes_heading_deg", "vcmd_heading_deg",
                "w0", "w1", "u_norm",
                "obs", "n_sensed", "sensed_idx",
                "nearest_obs_idx", "d_surface", "d_hard",
                "gap_type", "blocked", "gap_score", "gap_clear", "local_angle_deg",
                "ff_norm", "fa_norm", "fc_norm", "d_nbr_min",
                "vo_x", "vo_y", "u_x", "u_y", "vdes_x", "vdes_y", "vcmd_x", "vcmd_y",
            ])
        if all_uav_debug_every > 0:
            all_debug_csv_path = out_log_path.with_name(f"{out_log_path.stem}_all_uav_debug.csv")
            all_debug_csv_file = all_debug_csv_path.open("w", newline="", encoding="utf-8")
            all_debug_writer = csv.writer(all_debug_csv_file)
            all_debug_writer.writerow([
                "step", "t", "uav_idx", "uav_name",
                "pos_x", "pos_y", "v",
                "psi_deg", "theta_e_deg", "ref_yaw_deg",
                "yaw_raw_deg", "yaw_cmd_deg",
                "yaw_err_raw_deg", "yaw_err_cmd_deg", "yaw_step_cap_deg",
                "vo_heading_deg", "vdes_heading_deg", "vcmd_heading_deg",
                "w0", "w1", "u_norm",
                "obs", "n_sensed", "sensed_idx",
                "nearest_obs_idx", "d_surface", "d_hard",
                "gap_type", "blocked", "gap_score", "gap_clear", "local_angle_deg",
                "ff_norm", "fa_norm", "fc_norm", "d_nbr_min",
                "vo_x", "vo_y", "u_x", "u_y", "vdes_x", "vdes_y", "vcmd_x", "vcmd_y",
            ])
        if sense_debug_every > 0:
            sense_csv_path = out_log_path.with_name(f"{out_log_path.stem}_sense_debug.csv")
            sense_csv_file = sense_csv_path.open("w", newline="", encoding="utf-8")
            sense_writer = csv.writer(sense_csv_file)
            sense_writer.writerow([
                "step", "t", "uav_idx", "uav_name",
                "pos_x", "pos_y", "pos_z", "vxy", "psi_deg",
                "obs", "n_sensed", "sensed_idx",
                "nearest_obs_idx", "d_surface", "d_hard",
                "gap_type", "blocked", "gap_score", "gap_clear", "local_angle_deg", "ref_yaw_deg",
                "yaw_raw_deg", "yaw_cmd_deg", "yaw_err_raw_deg", "yaw_err_cmd_deg",
                "vo_heading_deg", "vdes_heading_deg", "vcmd_heading_deg",
                "w0", "w1", "u_norm",
                "ff_norm", "fa_norm", "fc_norm", "d_nbr_min",
                "vo_x", "vo_y", "u_x", "u_y", "vdes_x", "vdes_y", "vcmd_x", "vcmd_y",
            ])

    t_hist = np.zeros(steps, dtype=float)
    pos_hist = np.zeros((steps, num_drones, 3), dtype=float)
    vxy_hist = np.zeros((steps, num_drones, 2), dtype=float)
    vz_hist = np.zeros((steps, num_drones), dtype=float)
    w_hist = np.zeros((steps, num_drones, 2), dtype=float)
    u_hist = np.zeros((steps, num_drones), dtype=float)
    cmd_hist = np.zeros((steps, num_drones, 3), dtype=float)
    # 速度平滑状态
    alpha_v = math.exp(-control_dt / tau_v_smooth)
    v_lp_state = np.tile(sp.ve_xy, (num_drones, 1)).astype(float)
    v_cmd_state = np.tile(sp.ve_xy, (num_drones, 1)).astype(float)
    traj_plot_colors = [
        [1.0, 0.20, 0.20, 1.0],
        [0.10, 0.70, 1.0, 1.0],
        [0.15, 0.85, 0.25, 1.0],
        [1.0, 0.65, 0.10, 1.0],
        [1.0, 0.20, 0.85, 1.0],
    ]
    traj_prev_pts = [None] * num_drones
    if plot_traj_every > 0 and hasattr(client, "simFlushPersistentMarkers"):
        client.simFlushPersistentMarkers()
    if backend == "mpio":
        sp.last_w = np.full((num_drones, 2), [0.2, 0.8], dtype=float)
        sp.last_w_valid = np.zeros((num_drones,), dtype=bool)
        mpio_rng = _resolve_rng(sp, None)
    else:
        mpio_rng = None

    try:
        for step in range(steps):
            loop_start = time.time()
            t_sim = step * control_dt
            dyn_obs_state, dyn_prev_cache = query_dynamic_obstacles(
                client, dynamic_actor_specs, dyn_prev_cache, control_dt, sp.he
            )
            if static_obs_state.shape[0] > 0 and dyn_obs_state.shape[0] > 0:
                obs_state = np.vstack([static_obs_state, dyn_obs_state])
            elif static_obs_state.shape[0] > 0:
                obs_state = static_obs_state.copy()
            else:
                obs_state = dyn_obs_state.copy()

            obstacles_now = obs_state[:, :3].copy()
            if sp.obstacles_use_prediction and obs_state.shape[0] > 0:
                obstacles_eval = obstacles_now.copy()
                obstacles_eval[:, :2] += obs_state[:, 3:5] * control_dt
            else:
                obstacles_eval = obstacles_now.copy()
            t0_state = time.time()
            # 1) 从 AirSim 读当前真实状态
            P, V_xy, psi, lam = get_swarm_state(client,vehicle_names)
            t1_state = time.time()

            t_hist[step] = t_sim
            pos_hist[step, :, :] = P
            vxy_hist[step, :, :] = V_xy
            vz_hist[step, :] = lam

            theta_e = math.atan2(sp.ve_xy[1], sp.ve_xy[0])
            t0_control = time.time()

            # 2) 对每架无人机计算控制输入并发送命令（不 join）
            for i, nm in enumerate(vehicle_names):
                Pi = P[i, :]
                Vi = V_xy[i, :]
                P_for_sense = np.array([Pi[0], Pi[1], sp.he], dtype=float)
                yaw_raw_i, sensed_idx_local, avoid_dbg = obstacle_avoidance_uav(
                    P_for_sense,
                    theta_e,
                    obstacles_eval,
                    sp,
                    return_debug=True,
                    v_mag=float(np.linalg.norm(Vi)),
                    v_xy=Vi,
                )
                sensed_obstacles_local = (len(sensed_idx_local) > 0)
                yaw_cmd_i = _rate_limit_angle(yaw_raw_i, psi[i], sp.yaw_rate_max, control_dt)

                # --- (1) 邻居力 ---
                ff_raw, fa_raw, fc_raw = compute_raw_forces(P, V_xy, i, sp)

                # --- (2) 垂直控制量 vf_z ---
                vf_z = sp.Ka_he * (sp.he - Pi[2]) + sp.Kve * (sp.ve3 - lam[i])

                # --- (3) 避障方向 vo_raw ---
                ve_speed = float(np.linalg.norm(sp.ve_xy))
                # For the AirSim quadrotor velocity-control path, obstacle avoidance should
                # directly shape the commanded velocity direction. Keep yaw_cmd_i only as a
                # diagnostic/display heading reference instead of limiting vo_raw itself.
                vo_raw = ve_speed * np.array([math.cos(yaw_raw_i), math.sin(yaw_raw_i)], dtype=float)
                if avoid_dbg is not None and bool(avoid_dbg.get("blocked", False)):
                    vo_raw *= float(getattr(sp, "blocked_speed_scale", 0.6))

                # --- (4) 根据后端得到 w ---
                if backend == "mlp":
                    if mlp_model is None or mean is None or std is None:
                        raise RuntimeError("mlp backend requested but model is not loaded")
                    w = predict_w_mlp(
                        mlp_model, mean, std,
                        i, P, V_xy, psi, lam,
                        obs_state, obstacles_eval, sp
                    )
                else:
                    w = optimizePigeons_wrapper(
                        sp, P, i, V_xy, obstacles_eval,
                        ff_raw, fa_raw, fc_raw, vf_z, vo_raw,
                        lam[i], psi[i], rng=mpio_rng
                    )
                w = np.asarray(w, dtype=float).flatten()
                if w.size < 2:
                    w = np.array([0.5, 0.5], dtype=float)
                w = np.clip(w, 0.0, 1.0)
                if backend == "mpio":
                    sp.last_w[i] = w.copy()
                    sp.last_w_valid[i] = True

                # --- (5) 合成水平“目标速度向量” u_total_xy ---
                vf_prime_xy = w[0] * (ff_raw + fa_raw) + fc_raw
                vo_prime_xy = w[1] * vo_raw
                u_total_xy = vf_prime_xy + vo_prime_xy

                speed = np.linalg.norm(u_total_xy)

                if speed < 1e-3:
                    dir_xy = sp.ve_xy / np.linalg.norm(sp.ve_xy)
                    speed_cmd = CMD_VXY_MIN
                    vel_xy_des = speed_cmd * dir_xy
                else:
                    dir_xy = u_total_xy / speed
                    speed_cmd = float(np.clip(speed, CMD_VXY_MIN, CMD_VXY_MAX))
                    vel_xy_des = speed_cmd * dir_xy

                # --- (5') 速度平滑：低通 + 加速度限幅 ---
                v_lp = alpha_v * v_lp_state[i] + (1.0 - alpha_v) * vel_xy_des
                dv = v_lp - v_cmd_state[i]
                dv_norm = np.linalg.norm(dv)
                dv_max = a_max * control_dt
                if dv_norm > dv_max and dv_norm > 1e-6:
                    dv = dv * (dv_max / dv_norm)
                v_cmd = v_cmd_state[i] + dv
                v_lp_state[i] = v_lp
                v_cmd_state[i] = v_cmd

                vel_xy_m = v_cmd

                vx_m, vy_m = vel_xy_m[0], vel_xy_m[1]

                # --- (7) 垂直速度：锁定固定高度给 AirSim 内环 ---
                vz_m = 0.0
                z_ned_target = -sp.he

                # --- (8) MATLAB -> NED ---
                vx_ned, vy_ned, _ = matlab_vel_to_ned(vx_m, vy_m, 0.0)

                w_hist[step, i, :] = w
                u_hist[step, i] = speed
                cmd_hist[step, i, :] = np.array([vx_m, vy_m, vz_m], dtype=float)

                yaw_deg = math.degrees(math.atan2(vy_ned, vx_ned))
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

                client.moveByVelocityZAsync(
                    vx_ned, vy_ned, z_ned_target,
                    duration=hold,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=yaw_mode,
                    vehicle_name=nm
                )

                v_mag = np.linalg.norm(Vi)
                obs_flag = 1 if sensed_obstacles_local else 0
                num_sensed = len(sensed_idx_local)
                if obstacles_now.shape[0] > 0:
                    obs_center_dist = np.linalg.norm(obstacles_now[:, :2] - Pi[:2], axis=1)
                    nearest_obs_idx = int(np.argmin(obs_center_dist))
                    nearest_obs_surface_dist = float(
                        obs_center_dist[nearest_obs_idx] - obstacles_now[nearest_obs_idx, 2]
                    )
                    nearest_obs_hard_dist = float(
                        obs_center_dist[nearest_obs_idx] - (obstacles_now[nearest_obs_idx, 2] + sp.R2_lim)
                    )
                else:
                    nearest_obs_idx = -1
                    nearest_obs_surface_dist = float("nan")
                    nearest_obs_hard_dist = float("nan")

                nbr_dist = np.linalg.norm(P[:, :2] - Pi[:2], axis=1)
                nbr_dist[i] = np.inf
                d_nbr_min = float(np.min(nbr_dist)) if nbr_dist.size > 1 else float("nan")
                ff_norm = float(np.linalg.norm(ff_raw))
                fa_norm = float(np.linalg.norm(fa_raw))
                fc_norm = float(np.linalg.norm(fc_raw))
                gap_type = str(avoid_dbg.get("gap_type", "na")) if avoid_dbg else "na"
                blocked_flag = int(bool(avoid_dbg.get("blocked", False))) if avoid_dbg else 0
                gap_score = float(avoid_dbg.get("gap_score", math.nan)) if avoid_dbg else math.nan
                gap_clear = float(avoid_dbg.get("gap_clear", math.nan)) if avoid_dbg else math.nan
                local_angle = float(avoid_dbg.get("local_angle", math.nan)) if avoid_dbg else math.nan
                ref_yaw = float(avoid_dbg.get("ref_yaw", math.nan)) if avoid_dbg else math.nan
                sensed_idx_str = "|".join(str(int(j)) for j in sensed_idx_local)
                yaw_raw_deg = math.degrees(yaw_raw_i)
                yaw_cmd_deg = math.degrees(yaw_cmd_i)
                psi_deg = math.degrees(psi[i])
                theta_e_deg = math.degrees(theta_e)
                yaw_err_raw_deg = math.degrees(_wrap_angle(yaw_raw_i - psi[i]))
                yaw_err_cmd_deg = math.degrees(_wrap_angle(yaw_cmd_i - psi[i]))
                yaw_step_cap_deg = math.degrees(float(getattr(sp, "yaw_rate_max", 0.0)) * control_dt)
                vo_heading_deg = math.degrees(math.atan2(vo_raw[1], vo_raw[0]))
                vdes_heading_deg = math.degrees(math.atan2(vel_xy_des[1], vel_xy_des[0]))
                vcmd_heading_deg = math.degrees(math.atan2(v_cmd[1], v_cmd[0]))
                if sense_writer is not None and sense_debug_every > 0 and (step % sense_debug_every == 0):
                    sense_writer.writerow([
                        step, f"{t_sim:.6f}", i, nm,
                        f"{Pi[0]:.6f}", f"{Pi[1]:.6f}", f"{Pi[2]:.6f}", f"{v_mag:.6f}", f"{psi_deg:.6f}",
                        obs_flag, num_sensed, sensed_idx_str,
                        nearest_obs_idx,
                        f"{nearest_obs_surface_dist:.6f}" if math.isfinite(nearest_obs_surface_dist) else "",
                        f"{nearest_obs_hard_dist:.6f}" if math.isfinite(nearest_obs_hard_dist) else "",
                        gap_type, blocked_flag,
                        f"{gap_score:.6f}" if math.isfinite(gap_score) else "",
                        f"{gap_clear:.6f}" if math.isfinite(gap_clear) else "",
                        f"{math.degrees(local_angle):.6f}" if math.isfinite(local_angle) else "",
                        f"{math.degrees(ref_yaw):.6f}" if math.isfinite(ref_yaw) else "",
                        f"{yaw_raw_deg:.6f}", f"{yaw_cmd_deg:.6f}",
                        f"{yaw_err_raw_deg:.6f}", f"{yaw_err_cmd_deg:.6f}",
                        f"{vo_heading_deg:.6f}", f"{vdes_heading_deg:.6f}", f"{vcmd_heading_deg:.6f}",
                        f"{w[0]:.6f}", f"{w[1]:.6f}", f"{speed:.6f}",
                        f"{ff_norm:.6f}", f"{fa_norm:.6f}", f"{fc_norm:.6f}",
                        f"{d_nbr_min:.6f}" if math.isfinite(d_nbr_min) else "",
                        f"{vo_raw[0]:.6f}", f"{vo_raw[1]:.6f}",
                        f"{u_total_xy[0]:.6f}", f"{u_total_xy[1]:.6f}",
                        f"{vel_xy_des[0]:.6f}", f"{vel_xy_des[1]:.6f}",
                        f"{v_cmd[0]:.6f}", f"{v_cmd[1]:.6f}",
                    ])

                if uav1_debug_every > 0 and i == 0 and (step % uav1_debug_every == 0):

                    print(
                        f"[t={t_sim:5.1f}] UAV1 pos=({Pi[0]:6.2f},{Pi[1]:6.2f}) "
                        f"v={v_mag:5.2f} psi={psi_deg:6.1f} "
                        f"w=[{w[0]:.2f},{w[1]:.2f}] |u|={speed:5.2f} "
                        f"obs={obs_flag} n_sensed={num_sensed} obs_idx={nearest_obs_idx} "
                        f"d_surface={nearest_obs_surface_dist:6.2f} d_hard={nearest_obs_hard_dist:6.2f} "
                        f"yaw_raw={yaw_raw_deg:6.1f} yaw_cmd={yaw_cmd_deg:6.1f} "
                        f"dy_raw={yaw_err_raw_deg:6.1f} dy_cmd={yaw_err_cmd_deg:6.1f} "
                        f"cap={yaw_step_cap_deg:4.1f} gap={gap_type} "
                        f"|ff|={ff_norm:5.2f} |fa|={fa_norm:5.2f} |fc|={fc_norm:6.2f} d_nbr={d_nbr_min:5.2f} "
                        f"vo=({vo_raw[0]:5.2f},{vo_raw[1]:5.2f})@{vo_heading_deg:5.1f} "
                        f"vdes=({vel_xy_des[0]:5.2f},{vel_xy_des[1]:5.2f})@{vdes_heading_deg:5.1f} "
                        f"vcmd=({v_cmd[0]:5.2f},{v_cmd[1]:5.2f})@{vcmd_heading_deg:5.1f}"
                    )
                    if debug_writer is not None:
                        debug_writer.writerow([
                            step, f"{t_sim:.6f}", f"{Pi[0]:.6f}", f"{Pi[1]:.6f}", f"{v_mag:.6f}",
                            f"{psi_deg:.6f}", f"{theta_e_deg:.6f}",
                            f"{math.degrees(ref_yaw):.6f}" if math.isfinite(ref_yaw) else "",
                            f"{yaw_raw_deg:.6f}", f"{yaw_cmd_deg:.6f}",
                            f"{yaw_err_raw_deg:.6f}", f"{yaw_err_cmd_deg:.6f}", f"{yaw_step_cap_deg:.6f}",
                            f"{vo_heading_deg:.6f}", f"{vdes_heading_deg:.6f}", f"{vcmd_heading_deg:.6f}",
                            f"{w[0]:.6f}", f"{w[1]:.6f}", f"{speed:.6f}",
                            obs_flag, num_sensed, sensed_idx_str,
                            nearest_obs_idx,
                            f"{nearest_obs_surface_dist:.6f}" if math.isfinite(nearest_obs_surface_dist) else "",
                            f"{nearest_obs_hard_dist:.6f}" if math.isfinite(nearest_obs_hard_dist) else "",
                            gap_type, blocked_flag,
                            f"{gap_score:.6f}" if math.isfinite(gap_score) else "",
                            f"{gap_clear:.6f}" if math.isfinite(gap_clear) else "",
                            f"{math.degrees(local_angle):.6f}" if math.isfinite(local_angle) else "",
                            f"{ff_norm:.6f}", f"{fa_norm:.6f}", f"{fc_norm:.6f}",
                            f"{d_nbr_min:.6f}" if math.isfinite(d_nbr_min) else "",
                            f"{vo_raw[0]:.6f}", f"{vo_raw[1]:.6f}",
                            f"{u_total_xy[0]:.6f}", f"{u_total_xy[1]:.6f}",
                            f"{vel_xy_des[0]:.6f}", f"{vel_xy_des[1]:.6f}",
                            f"{v_cmd[0]:.6f}", f"{v_cmd[1]:.6f}",
                        ])
                    if debug_csv_file is not None and step % csv_flush_every == 0:
                        debug_csv_file.flush()
                if all_uav_debug_every > 0 and (step % all_uav_debug_every == 0):
                    print(
                        f"[t={t_sim:5.1f}] {nm} pos=({Pi[0]:6.2f},{Pi[1]:6.2f}) "
                        f"v={v_mag:5.2f} psi={psi_deg:6.1f} "
                        f"w=[{w[0]:.2f},{w[1]:.2f}] |u|={speed:5.2f} "
                        f"obs={obs_flag} n_sensed={num_sensed} obs_idx={nearest_obs_idx} "
                        f"d_surface={nearest_obs_surface_dist:6.2f} d_hard={nearest_obs_hard_dist:6.2f} "
                        f"yaw_raw={yaw_raw_deg:6.1f} yaw_cmd={yaw_cmd_deg:6.1f} "
                        f"dy_raw={yaw_err_raw_deg:6.1f} dy_cmd={yaw_err_cmd_deg:6.1f} "
                        f"cap={yaw_step_cap_deg:4.1f} gap={gap_type} "
                        f"|ff|={ff_norm:5.2f} |fa|={fa_norm:5.2f} |fc|={fc_norm:6.2f} d_nbr={d_nbr_min:5.2f} "
                        f"vo=({vo_raw[0]:5.2f},{vo_raw[1]:5.2f})@{vo_heading_deg:5.1f} "
                        f"vdes=({vel_xy_des[0]:5.2f},{vel_xy_des[1]:5.2f})@{vdes_heading_deg:5.1f} "
                        f"vcmd=({v_cmd[0]:5.2f},{v_cmd[1]:5.2f})@{vcmd_heading_deg:5.1f}"
                    )
                    if all_debug_writer is not None:
                        all_debug_writer.writerow([
                            step, f"{t_sim:.6f}", i, nm,
                            f"{Pi[0]:.6f}", f"{Pi[1]:.6f}", f"{v_mag:.6f}",
                            f"{psi_deg:.6f}", f"{theta_e_deg:.6f}",
                            f"{math.degrees(ref_yaw):.6f}" if math.isfinite(ref_yaw) else "",
                            f"{yaw_raw_deg:.6f}", f"{yaw_cmd_deg:.6f}",
                            f"{yaw_err_raw_deg:.6f}", f"{yaw_err_cmd_deg:.6f}", f"{yaw_step_cap_deg:.6f}",
                            f"{vo_heading_deg:.6f}", f"{vdes_heading_deg:.6f}", f"{vcmd_heading_deg:.6f}",
                            f"{w[0]:.6f}", f"{w[1]:.6f}", f"{speed:.6f}",
                            obs_flag, num_sensed, sensed_idx_str,
                            nearest_obs_idx,
                            f"{nearest_obs_surface_dist:.6f}" if math.isfinite(nearest_obs_surface_dist) else "",
                            f"{nearest_obs_hard_dist:.6f}" if math.isfinite(nearest_obs_hard_dist) else "",
                            gap_type, blocked_flag,
                            f"{gap_score:.6f}" if math.isfinite(gap_score) else "",
                            f"{gap_clear:.6f}" if math.isfinite(gap_clear) else "",
                            f"{math.degrees(local_angle):.6f}" if math.isfinite(local_angle) else "",
                            f"{ff_norm:.6f}", f"{fa_norm:.6f}", f"{fc_norm:.6f}",
                            f"{d_nbr_min:.6f}" if math.isfinite(d_nbr_min) else "",
                            f"{vo_raw[0]:.6f}", f"{vo_raw[1]:.6f}",
                            f"{u_total_xy[0]:.6f}", f"{u_total_xy[1]:.6f}",
                            f"{vel_xy_des[0]:.6f}", f"{vel_xy_des[1]:.6f}",
                            f"{v_cmd[0]:.6f}", f"{v_cmd[1]:.6f}",
                        ])
                    if all_debug_csv_file is not None and step % csv_flush_every == 0:
                        all_debug_csv_file.flush()
                if sense_csv_file is not None and step % csv_flush_every == 0:
                    sense_csv_file.flush()
            t1_control = time.time()

            if plot_traj_every > 0 and step % plot_traj_every == 0:
                for i in range(num_drones):
                    curr_pt = matlab_abs_pos_to_ned_plot(P[i, :])
                    prev_pt = traj_prev_pts[i]
                    traj_prev_pts[i] = curr_pt
                    if prev_pt is None:
                        continue
                    client.simPlotLineList(
                        [prev_pt, curr_pt],
                        color_rgba=traj_plot_colors[i % len(traj_plot_colors)],
                        thickness=plot_traj_thickness,
                        duration=-1.0,
                        is_persistent=True,
                    )
            t2_plot = time.time()

            # 3) 周期性打印总状态
            if status_every > 0 and step % status_every == 0:
                parts = []
                for i, nm in enumerate(vehicle_names):
                    st = client.getMultirotorState(vehicle_name=nm)
                    p = st.kinematics_estimated.position
                    v = st.kinematics_estimated.linear_velocity
                    v_xy_mag = math.hypot(v.x_val, v.y_val)
                    parts.append(
                        f"UAV{i+1} p=({p.x_val:6.1f},{p.y_val:6.1f},{-p.z_val:5.1f}) vxy={v_xy_mag:4.1f}"
                    )
                print(f"[t={t_sim:5.1f}] | " + " | ".join(parts), flush=True)
            t3_status = time.time()

            if timing_log_every > 0 and step % timing_log_every == 0:
                dt_state_ms = (t1_state - t0_state) * 1000.0
                dt_control_ms = (t1_control - t0_control) * 1000.0
                dt_plot_ms = (t2_plot - t1_control) * 1000.0
                dt_status_ms = (t3_status - t2_plot) * 1000.0
                dt_total_ms = (t3_status - loop_start) * 1000.0
                print(
                    f"[timing] step={step} state={dt_state_ms:5.2f}ms "
                    f"control={dt_control_ms:5.2f}ms plot={dt_plot_ms:5.2f}ms "
                    f"status={dt_status_ms:5.2f}ms total={dt_total_ms:6.2f}ms",
                    flush=True,
                )

            # 4) 控制频率：按 control_dt 调度（不阻塞在指令上）
            elapsed = time.time() - loop_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
            else:
                print(f"[warn] step={step} elapsed={elapsed:.3f}s > control_dt={control_dt:.3f}s", flush=True)
    finally:
        if debug_csv_file is not None:
            debug_csv_file.close()
        if all_debug_csv_file is not None:
            all_debug_csv_file.close()
        if sense_csv_file is not None:
            sense_csv_file.close()

    if save_logs:
        np.savez(
            out_log_path,
            t=t_hist,
            pos=pos_hist,
            vxy=vxy_hist,
            vz=vz_hist,
            w=w_hist,
            u=u_hist,
            cmd=cmd_hist,
        )
        print(f"Saved log to {out_log_path}")
        print(f"Saved UAV1 debug csv to {debug_csv_path}")
        if all_debug_csv_path is not None:
            print(f"Saved all-UAV debug csv to {all_debug_csv_path}")
        if sense_csv_path is not None:
            print(f"Saved sense debug csv to {sense_csv_path}")

    # ---------- 结束：降落并释放控制 ----------
    print("Simulation finished. Landing all UAVs...")
    land_tasks = [client.landAsync(vehicle_name=nm) for nm in vehicle_names]
    for t in land_tasks:
        t.join()

    for nm in vehicle_names:
        client.armDisarm(False, vehicle_name=nm)
        client.enableApiControl(False, vehicle_name=nm)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="mpio", choices=("mpio", "mlp"))
    parser.add_argument("--model_dir", type=str, default="", help="required when backend=mlp")
    parser.add_argument("--log_path", type=str, default="", help="optional output npz path")
    parser.add_argument("--scene_json", type=str, default="", help="optional scene json; overrides obstacles plus selected scene params")
    parser.add_argument("--airsim_settings_json", type=str, default="", help="optional AirSim settings.json; overrides sp.init_P XY from Vehicles")
    parser.add_argument("--swarm_profile_json", type=str, default="", help="optional swarm profile json for init_h / ve_xy / he")
    parser.add_argument("--R2_comm", type=float, default=None, help="optional override for scene/solver R2_comm")
    parser.add_argument("--gap_lookahead", type=float, default=None, help="optional override for scene/solver gap_lookahead")
    parser.add_argument("--R_desire", type=float, default=None, help="optional override for scene/solver R_desire")
    parser.add_argument("--R1_comm", type=float, default=None, help="optional override for scene/solver R1_comm")
    parser.add_argument("--f1", type=float, default=None, help="optional override for scene/solver f1")
    parser.add_argument("--f2", type=float, default=None, help="optional override for scene/solver f2")
    parser.add_argument("--w_direction", type=float, default=None, help="optional override for scene/solver w_direction")
    parser.add_argument("--n_max", type=float, default=None, help="optional override for scene/solver n_max")
    parser.add_argument("--control_dt", type=float, default=dt, help="outer-loop control period in seconds")
    parser.add_argument("--hold_time", type=float, default=0.0, help="AirSim command duration; default tracks control_dt")
    parser.add_argument("--warmup_time", type=float, default=2.0, help="pre-control fixed forward-velocity warmup in seconds; 0 disables")
    parser.add_argument("--tau_v_smooth", type=float, default=1.0, help="velocity low-pass time constant")
    parser.add_argument("--a_max", type=float, default=3.0, help="command acceleration limit in m/s^2")
    parser.add_argument("--uav1_debug_every", type=int, default=0, help="print/log UAV1 debug every N steps; 0 disables")
    parser.add_argument("--all_uav_debug_every", type=int, default=0, help="print/log all-UAV detailed debug every N steps; 0 disables")
    parser.add_argument("--csv_flush_every", type=int, default=100, help="flush debug csv files every N steps")
    parser.add_argument("--sense_debug_every", type=int, default=0, help="write all-UAV obstacle sensing debug csv every N steps; 0 disables")
    parser.add_argument("--timing_log_every", type=int, default=0, help="print loop timing breakdown every N steps; 0 disables")
    parser.add_argument("--status_every", type=int, default=0, help="print swarm status every N steps; 0 disables")
    parser.add_argument("--plot_traj_every", type=int, default=0, help="plot AirSim trajectory segments every N steps; 0 disables")
    parser.add_argument("--plot_traj_thickness", type=float, default=8.0, help="AirSim plotted trajectory line thickness")
    parser.add_argument("--no_save_logs", action="store_true", help="disable npz/csv log files")
    args = parser.parse_args()
    run_on_airsim(
        backend=args.backend,
        model_dir=args.model_dir or None,
        log_path=args.log_path or None,
        scene_json=args.scene_json or None,
        airsim_settings_json=args.airsim_settings_json or None,
        swarm_profile_json=args.swarm_profile_json or None,
        R2_comm=args.R2_comm,
        gap_lookahead=args.gap_lookahead,
        R_desire=args.R_desire,
        R1_comm=args.R1_comm,
        f1=args.f1,
        f2=args.f2,
        w_direction=args.w_direction,
        n_max=args.n_max,
        control_dt=args.control_dt,
        hold_time=(args.hold_time if args.hold_time > 0.0 else None),
        warmup_time=args.warmup_time,
        tau_v_smooth=args.tau_v_smooth,
        a_max=args.a_max,
        uav1_debug_every=args.uav1_debug_every,
        all_uav_debug_every=args.all_uav_debug_every,
        csv_flush_every=args.csv_flush_every,
        sense_debug_every=args.sense_debug_every,
        timing_log_every=args.timing_log_every,
        status_every=args.status_every,
        plot_traj_every=args.plot_traj_every,
        plot_traj_thickness=args.plot_traj_thickness,
        save_logs=not args.no_save_logs,
    )
