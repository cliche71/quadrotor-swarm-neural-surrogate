from __future__ import annotations

import copy
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from mpio_swarm_sim import (
    CalcCrowdingDistance,
    Pigeon,
    SimParams,
    calculateCosts,
    compute_raw_forces,
    initializePigeons,
    neighbors_within,
    nondominated_sort_safe,
    norm2,
    obstacle_avoidance_uav,
    update_drone_state,
)

_VEL_LIM = 0.05
_DEFAULT_W = np.array([0.2, 0.8], dtype=float)


def _filter_finite_pigeons(pop: List[Pigeon]) -> List[Pigeon]:
    return [p for p in pop if np.all(np.isfinite(p.Cost12))]


def _seg_point_dists(p0: np.ndarray, p1: np.ndarray, points: np.ndarray) -> np.ndarray:
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


def _append_square_as_circle(sp: SimParams) -> Optional[int]:
    if not bool(getattr(sp, "square_enable", False)):
        setattr(sp, "square_idx", None)
        return None
    if bool(getattr(sp, "_square_appended", False)):
        return getattr(sp, "square_idx", None)

    a = float(getattr(sp, "square_side", 0.0))
    cx, cy = getattr(sp, "square_center_xy", (0.0, 0.0))
    r = (math.sqrt(2.0) / 2.0) * a
    sp.obstacles = np.vstack([np.asarray(sp.obstacles, dtype=float), np.array([[cx, cy, r]], dtype=float)])

    vxy = np.asarray(getattr(sp, "obstacles_vxy", np.zeros((0, 2), dtype=float)), dtype=float)
    if vxy.ndim == 1 and vxy.size == 2:
        vxy = np.tile(vxy, (sp.obstacles.shape[0] - 1, 1))
    if vxy.shape[0] < sp.obstacles.shape[0] - 1:
        pad = np.zeros((sp.obstacles.shape[0] - 1 - vxy.shape[0], 2), dtype=float)
        vxy = np.vstack([vxy, pad]) if vxy.size else pad
    elif vxy.shape[0] > sp.obstacles.shape[0] - 1:
        vxy = vxy[: sp.obstacles.shape[0] - 1, :]

    square_vxy = np.array([list(getattr(sp, "square_vxy", (0.0, 0.0)))], dtype=float)
    sp.obstacles_vxy = np.vstack([vxy, square_vxy])
    setattr(sp, "square_idx", int(sp.obstacles.shape[0] - 1))
    setattr(sp, "_square_appended", True)
    return getattr(sp, "square_idx")


def _init_obstacle_state(sp: SimParams) -> np.ndarray:
    obs = np.asarray(sp.obstacles, dtype=float)
    obs_state = np.zeros((obs.shape[0], 5), dtype=float)
    obs_state[:, :3] = obs

    vxy = np.asarray(getattr(sp, "obstacles_vxy", np.zeros((obs.shape[0], 2), dtype=float)), dtype=float)
    if vxy.ndim == 1 and vxy.size == 2:
        vxy = np.tile(vxy, (obs.shape[0], 1))
    if vxy.shape != (obs.shape[0], 2):
        vxy = np.zeros((obs.shape[0], 2), dtype=float)
    obs_state[:, 3:5] = vxy
    return obs_state


def _update_obstacle_state(obs_state: np.ndarray, sp: SimParams) -> None:
    xmin, xmax, ymin, ymax = getattr(sp, "obstacles_bounds", (0.0, 400.0, 0.0, 200.0))
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


def optimizePigeons_core(
    pop: List[Pigeon],
    sp: SimParams,
    P_swarm: np.ndarray,
    i_uav: int,
    V_xy_swarm: np.ndarray,
    obstacles: np.ndarray,
    ff_raw: np.ndarray,
    fa_raw: np.ndarray,
    fc_raw: np.ndarray,
    vf_z: float,
    vo_raw: np.ndarray,
    lamb_i: float,
    psi_i: float,
    rng: np.random.Generator,
) -> np.ndarray:
    historical_A: List[Pigeon] = []
    ff_raw = np.asarray(ff_raw, dtype=float)
    fa_raw = np.asarray(fa_raw, dtype=float)
    fc_raw = np.asarray(fc_raw, dtype=float)
    vo_raw = np.asarray(vo_raw, dtype=float)

    if not pop:
        return _DEFAULT_W.copy()

    current_n = len(pop)
    max_iter = max(1, int(sp.Ncmax))
    for Nc in range(1, max_iter + 1):
        pop = _filter_finite_pigeons(pop)
        if not pop:
            break

        costs = np.vstack([p.Cost12 for p in pop])
        rank, fronts = nondominated_sort_safe(costs)
        if not fronts or not fronts[0]:
            break

        for idx, pigeon in enumerate(pop):
            pigeon.Rank = float(rank[idx])
        pop = CalcCrowdingDistance(pop, fronts)
        s1_pop = [pop[idx] for idx in fronts[0]]
        x_center = np.mean(np.vstack([p.Position for p in s1_pop]), axis=0)

        combined_archive = list(historical_A) + list(s1_pop)
        archive_costs = np.vstack([p.Cost12 for p in combined_archive])
        _, archive_fronts = nondominated_sort_safe(archive_costs)
        historical_A = [combined_archive[idx] for idx in archive_fronts[0]]
        if not historical_A:
            break
        x_gbest = historical_A[int(rng.integers(0, len(historical_A)))].Position.copy()

        x_center = np.asarray(x_center, dtype=float)
        transition = math.log10(float(Nc)) / float(max_iter) if max_iter > 0 else 0.0
        transition = float(np.clip(transition, 0.0, 1.0))
        next_pop: List[Pigeon] = []
        for pigeon in pop:
            old = copy.deepcopy(pigeon)
            rand1 = float(rng.random())
            rand2 = float(rng.random())
            v_new = (
                math.exp(-sp.R * Nc) * pigeon.V
                + rand1 * sp.tr * (1.0 - transition) * (x_gbest - pigeon.Position)
                + rand2 * sp.tr * transition * (x_center - pigeon.Position)
            )
            v_new = np.clip(v_new, -_VEL_LIM, _VEL_LIM)
            x_new = np.clip(pigeon.Position + v_new, 0.0, 1.0)

            vf_prime_xy = x_new[0] * (ff_raw + fa_raw) + fc_raw
            vo_prime_xy = x_new[1] * vo_raw
            u_total_xy = vf_prime_xy + vo_prime_xy
            u_prime = np.array(
                [
                    u_total_xy[0] - V_xy_swarm[i_uav, 0],
                    u_total_xy[1] - V_xy_swarm[i_uav, 1],
                    vf_z,
                ],
                dtype=float,
            )
            p_next, v_next, _, _, _ = update_drone_state(
                P_swarm[i_uav, :],
                V_xy_swarm[i_uav, :],
                psi_i,
                lamb_i,
                u_prime,
                sp,
            )
            c1, c2, c3, c4 = calculateCosts(p_next, v_next, P_swarm, i_uav, V_xy_swarm, obstacles, sp)
            if c3 == 1 or c4 == 1:
                new_cost = np.array([np.inf, np.inf], dtype=float)
            else:
                new_cost = np.array([c1, c2], dtype=float)

            if np.all(old.Cost12 <= new_cost) and np.any(old.Cost12 < new_cost):
                next_pop.append(old)
                continue

            pigeon.Position = x_new
            pigeon.V = v_new
            pigeon.Cost12 = new_cost
            next_pop.append(pigeon)

        pop = next_pop
        if not pop:
            break

        current_n = max(2, current_n - int(sp.Nd))
        if Nc < max_iter and len(pop) > current_n:
            pop = _filter_finite_pigeons(pop)
            if not pop:
                break

            costs = np.vstack([p.Cost12 for p in pop])
            rank, fronts = nondominated_sort_safe(costs)
            for idx, pigeon in enumerate(pop):
                pigeon.Rank = float(rank[idx])
            pop = CalcCrowdingDistance(pop, fronts)
            order = np.lexsort(
                (
                    -np.array([p.CrowdingDistance for p in pop], dtype=float),
                    np.array([p.Rank for p in pop], dtype=float),
                )
            )
            pop = [pop[idx] for idx in order[:current_n]]

    if not pop:
        return _DEFAULT_W.copy()

    pop = _filter_finite_pigeons(pop)
    if not pop:
        return _DEFAULT_W.copy()

    costs = np.vstack([p.Cost12 for p in pop])
    _, fronts = nondominated_sort_safe(costs)
    if not fronts or not fronts[0]:
        return _DEFAULT_W.copy()

    s1_pop = [pop[idx] for idx in fronts[0]]
    s1_costs = np.vstack([p.Cost12 for p in s1_pop])
    valid = np.all(np.isfinite(s1_costs), axis=1)
    if not np.any(valid):
        return _DEFAULT_W.copy()

    valid_pop = [p for p, ok in zip(s1_pop, valid) if ok]
    valid_costs = s1_costs[valid, :]
    best_idx = int(np.argmin(valid_costs[:, 1]))
    return np.clip(valid_pop[best_idx].Position.astype(float), 0.0, 1.0)


def optimizePigeons_wrapper(
    sp: SimParams,
    P_swarm: np.ndarray,
    i_uav: int,
    V_xy_swarm: np.ndarray,
    obstacles: np.ndarray,
    ff_raw: np.ndarray,
    fa_raw: np.ndarray,
    fc_raw: np.ndarray,
    vf_z: float,
    vo_raw: np.ndarray,
    lamb_i: float,
    psi_i: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(sp.rng_seed)
    X, V = initializePigeons(sp.N, rng)
    pop: List[Pigeon] = []
    for idx in range(sp.N):
        w_p = X[idx, :].copy()
        vf_prime_xy = w_p[0] * (ff_raw + fa_raw) + fc_raw
        vo_prime_xy = w_p[1] * vo_raw
        u_total_xy = vf_prime_xy + vo_prime_xy
        u_prime = np.array(
            [
                u_total_xy[0] - V_xy_swarm[i_uav, 0],
                u_total_xy[1] - V_xy_swarm[i_uav, 1],
                vf_z,
            ],
            dtype=float,
        )
        p_next, v_next, _, _, _ = update_drone_state(
            P_swarm[i_uav, :],
            V_xy_swarm[i_uav, :],
            psi_i,
            lamb_i,
            u_prime,
            sp,
        )
        c1, c2, c3, c4 = calculateCosts(p_next, v_next, P_swarm, i_uav, V_xy_swarm, obstacles, sp)
        if c3 == 1 or c4 == 1:
            cost12 = np.array([np.inf, np.inf], dtype=float)
            rank = float("inf")
        else:
            cost12 = np.array([c1, c2], dtype=float)
            rank = 0.0
        pop.append(Pigeon(Position=w_p, V=V[idx, :].copy(), Cost12=cost12, Rank=rank))

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
        rng,
    )


def run_simulation(sp: Optional[SimParams] = None) -> Dict[str, np.ndarray]:
    if sp is None:
        sp = SimParams()
    rng = np.random.default_rng(sp.rng_seed)

    P = np.asarray(sp.init_P, dtype=float).copy()
    V_xy = np.asarray(sp.init_Vxy, dtype=float).copy()
    psi = np.zeros(P.shape[0], dtype=float)
    lamb = np.zeros(P.shape[0], dtype=float)

    _append_square_as_circle(sp)
    obs_state = _init_obstacle_state(sp)

    total_steps = int(round(sp.sim_time / sp.dt)) + 1
    time_vector = np.arange(total_steps, dtype=float) * sp.dt
    num_drones = P.shape[0]
    n_obs = obs_state.shape[0]

    X_uav = np.zeros((num_drones, total_steps), dtype=float)
    Y_uav = np.zeros((num_drones, total_steps), dtype=float)
    Z_uav = np.zeros((num_drones, total_steps), dtype=float)
    Vx_uav = np.zeros((num_drones, total_steps), dtype=float)
    Vy_uav = np.zeros((num_drones, total_steps), dtype=float)
    obs_hist = np.zeros((n_obs, total_steps, 3), dtype=float)
    decision_latency_ms: List[float] = []
    step_decision_latency_ms: List[float] = []

    X_uav[:, 0] = P[:, 0]
    Y_uav[:, 0] = P[:, 1]
    Z_uav[:, 0] = P[:, 2]
    Vx_uav[:, 0] = V_xy[:, 0]
    Vy_uav[:, 0] = V_xy[:, 1]
    if n_obs > 0:
        obs_hist[:, 0, :] = obs_state[:, :3]

    for k in range(total_steps - 1):
        if (k % 20) == 0:
            print(f"[base-mpio] k={k + 1}/{total_steps - 1}", flush=True)

        step_decision_ms = 0.0
        P_k = P.copy()
        V_k = V_xy.copy()
        psi_k = psi.copy()
        lamb_k = lamb.copy()

        if bool(getattr(sp, "obstacles_use_prediction", False)) and obs_state.size > 0:
            obs_pred = obs_state.copy()
            obs_pred[:, 0] += obs_pred[:, 3] * sp.dt
            obs_pred[:, 1] += obs_pred[:, 4] * sp.dt
            obstacles_plan = obs_pred[:, :3]
        else:
            obstacles_plan = obs_state[:, :3]

        P_next = np.zeros_like(P_k)
        V_next = np.zeros_like(V_k)
        psi_next = np.zeros_like(psi_k)
        lamb_next = np.zeros_like(lamb_k)

        for i in range(num_drones):
            ff_raw, fa_raw, fc_raw = compute_raw_forces(P_k, V_k, i, sp)
            vf_z = sp.Ka_he * (sp.he - P_k[i, 2]) + sp.Kve * (sp.ve3 - lamb_k[i])

            theta_e = math.atan2(sp.ve_xy[1], sp.ve_xy[0])
            yaw_desired, _ = obstacle_avoidance_uav(P_k[i, :], theta_e, obstacles_plan, sp)
            vo_raw = np.array(
                [
                    np.linalg.norm(sp.ve_xy) * math.cos(yaw_desired),
                    np.linalg.norm(sp.ve_xy) * math.sin(yaw_desired),
                ],
                dtype=float,
            )

            t0 = time.perf_counter()
            w = optimizePigeons_wrapper(
                sp,
                P_k,
                i,
                V_k,
                obstacles_plan,
                ff_raw,
                fa_raw,
                fc_raw,
                vf_z,
                vo_raw,
                lamb_k[i],
                psi_k[i],
                rng=rng,
            )
            decision_ms = (time.perf_counter() - t0) * 1000.0
            decision_latency_ms.append(decision_ms)
            step_decision_ms += decision_ms

            vf_prime_xy = w[0] * (ff_raw + fa_raw) + fc_raw
            vo_prime_xy = w[1] * vo_raw
            u_total_xy = vf_prime_xy + vo_prime_xy

            max_delta = sp.n_max * sp.g * sp.dt
            cur_norm = norm2(V_k[i, :])
            des_norm = norm2(u_total_xy)
            if des_norm > cur_norm + max_delta:
                u_total_xy = u_total_xy / (des_norm + 1e-12) * (cur_norm + max_delta)
            elif des_norm < cur_norm - max_delta:
                u_total_xy = u_total_xy / (des_norm + 1e-12) * (cur_norm - max_delta)

            u_prime = np.array(
                [
                    u_total_xy[0] - V_k[i, 0],
                    u_total_xy[1] - V_k[i, 1],
                    vf_z,
                ],
                dtype=float,
            )
            p_next, v_next, psi_i_next, lamb_i_next, _ = update_drone_state(
                P_k[i, :],
                V_k[i, :],
                psi_k[i],
                lamb_k[i],
                u_prime,
                sp,
            )
            P_next[i, :] = p_next
            V_next[i, :] = v_next
            psi_next[i] = psi_i_next
            lamb_next[i] = lamb_i_next

        P = P_next
        V_xy = V_next
        psi = psi_next
        lamb = lamb_next
        step_decision_latency_ms.append(step_decision_ms)

        X_uav[:, k + 1] = P[:, 0]
        Y_uav[:, k + 1] = P[:, 1]
        Z_uav[:, k + 1] = P[:, 2]
        Vx_uav[:, k + 1] = V_xy[:, 0]
        Vy_uav[:, k + 1] = V_xy[:, 1]

        if obs_state.size > 0:
            _update_obstacle_state(obs_state, sp)
            obs_hist[:, k + 1, :] = obs_state[:, :3]

    return {
        "t": time_vector,
        "X": X_uav,
        "Y": Y_uav,
        "Z": Z_uav,
        "Vx": Vx_uav,
        "Vy": Vy_uav,
        "obs_hist": obs_hist,
        "decision_latency_ms": np.asarray(decision_latency_ms, dtype=np.float32),
        "step_decision_latency_ms": np.asarray(step_decision_latency_ms, dtype=np.float32),
    }


def run_episode(
    sp: Optional[SimParams],
    scene_cfg: Optional[Dict[str, object]],
    collect: bool = True,
    stride: int = 2,
) -> Dict[str, object]:
    if sp is None:
        sp = SimParams()
    sp_local = copy.deepcopy(sp)
    if scene_cfg:
        for key, value in scene_cfg.items():
            setattr(sp_local, key, value)

    results = run_simulation(sp_local)
    X = results["X"]
    Y = results["Y"]
    Z = results["Z"]
    Vx = results["Vx"]
    Vy = results["Vy"]
    obs_hist = results["obs_hist"]
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
        traj = {
            "X": X,
            "Y": Y,
            "Z": Z,
            "Vx": Vx,
            "Vy": Vy,
            "obs_hist": obs_hist,
            "t": results["t"],
        }
    else:
        X_list, Y_list, traj = [], [], {}

    collision_obs = False
    collision_obs_hard = False
    collision_nbr = False
    obs_count_total = 0
    obs_count_steps = 0
    soft_band = float(getattr(sp_local, "safe_soft_band", 1.0))
    safe_clearance = float(getattr(sp_local, "safe_clearance_m", 0.0))
    for k in range(max(0, total_steps - 1)):
        pos0 = np.stack([X[:, k], Y[:, k]], axis=1)
        pos1 = np.stack([X[:, k + 1], Y[:, k + 1]], axis=1)
        obs_k = obs_hist[:, k, :] if obs_hist.size > 0 else np.asarray(sp_local.obstacles, dtype=float)
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

    k_n = int(getattr(sp_local, "formation_k_neighbors", 3))
    err_mean_list: List[float] = []
    err_max_list: List[float] = []
    for k in range(total_steps):
        Pk = np.stack([X[:, k], Y[:, k]], axis=1)
        e_list: List[float] = []
        for i in range(num_drones):
            dists = np.linalg.norm(Pk - Pk[i, :], axis=1)
            dists[i] = np.inf
            order = np.argsort(dists)
            for j in order[:k_n]:
                if np.isfinite(dists[j]):
                    e_list.append(abs(dists[j] - sp_local.R_desire))
        if e_list:
            err_mean_list.append(float(np.mean(e_list)))
            err_max_list.append(float(np.max(e_list)))

    if err_mean_list:
        err_mean = float(np.mean(err_mean_list))
        err_max = float(np.max(err_max_list))
        mean_thresh = float(getattr(sp_local, "formation_err_mean_thresh", 2.0))
        max_thresh = float(getattr(sp_local, "formation_err_max_thresh", 4.0))
        over_limit_ratio = float(
            np.mean((np.asarray(err_mean_list) > mean_thresh) | (np.asarray(err_max_list) > max_thresh))
        )
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
