import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from mpio_swarm_simv1 import (
    SimParams,
    run_episode,
    extract_features,
    neighbors_within,
    optimizePigeons_wrapper,
)

# 复用主数据集场景分布，避免 DAgger 与训练/评估口径不一致
from dataset_gen import sample_scene, SCENE_TYPES, N_RANGE


def sample_scene_focus_n(rng: np.random.Generator, n_min: int, n_max: int) -> Dict[str, object]:
    max_tries = 200
    for _ in range(max_tries):
        scene_cfg = sample_scene(rng)
        if not scene_cfg:
            continue
        n_uav = int(scene_cfg.get("N_uav", 0))
        if n_min <= n_uav <= n_max:
            return scene_cfg
    raise RuntimeError(f"failed to sample scene with N in [{n_min}, {n_max}] after {max_tries} tries")


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


def write_npz(out_path: Path, X: np.ndarray, Y: np.ndarray, meta: Dict[str, np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X.astype(np.float32), Y=Y.astype(np.float32), **meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="dataset_dagger_v1")
    ap.add_argument("--num_episodes", type=int, default=30)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_min", type=int, default=N_RANGE[0])
    ap.add_argument("--n_max", type=int, default=N_RANGE[1])
    ap.add_argument("--start_index", type=int, default=0, help="start index for ep_*.npz naming")
    ap.add_argument("--nbr_margin_trigger", type=float, default=0.5)
    ap.add_argument("--obs_margin_trigger", type=float, default=1.0)
    ap.add_argument("--max_trigger_per_step", type=int, default=1)

    # 可选：加速 teacher（会改变 teacher 分布；想严格对齐 baseline 就别开）
    ap.add_argument("--teacher_fast", action="store_true")
    args = ap.parse_args()
    if args.n_min > args.n_max:
        raise SystemExit("n_min must be <= n_max")
    if args.n_min < N_RANGE[0] or args.n_max > N_RANGE[1]:
        raise SystemExit(f"n_min/n_max must stay within dataset_gen.N_RANGE={N_RANGE}")

    model_dir = Path(args.model_dir)
    model, mean, std = load_mlp(model_dir)

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    index_path = out_dir / "index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}
    else:
        index = {}
    index.setdefault("episodes", [])
    index.setdefault("num_episodes", 0)
    index.setdefault("feature_dim", None)

    base_sp = SimParams()
    base_sp.verbose_uav1 = False
    base_sp.csv_path = None
    if args.teacher_fast:
        base_sp.N = 20
        base_sp.Ncmax = 8

    for ep in range(args.num_episodes):
        ep_id = int(args.start_index + ep)
        scene_cfg = sample_scene_focus_n(rng, args.n_min, args.n_max)
        scene_cfg["rng_seed"] = int(rng.integers(1, 1_000_000_000))

        X_rows: List[np.ndarray] = []
        Y_rows: List[np.ndarray] = []
        uav_ids: List[int] = []
        t_list: List[float] = []
        nbr_margin_list: List[float] = []
        obs_margin_list: List[float] = []
        trigger_state = {"step": None, "count": 0}

        def policy_fn(i, P, V_xy, psi, lamb, obs_state, obstacles_plan, sp,
                      ff_raw, fa_raw, fc_raw, vf_z, vo_raw, step_k, t):

            # ----- student action (MLP) -----
            nbr_idx = neighbors_within(P, i, sp.R1_comm)
            neighbors = np.array([[P[j, 0], P[j, 1], V_xy[j, 0], V_xy[j, 1]] for j in nbr_idx], dtype=float)
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
            x = (feat - mean) / std
            with torch.no_grad():
                w_student = model(torch.from_numpy(x.astype(np.float32))).numpy()
            w_student = np.clip(w_student.astype(float), 0.0, 1.0)

            # ----- collect & label with teacher (MPIO) -----
            min_nbr_margin = 1e9
            for j in range(P.shape[0]):
                if j == i:
                    continue
                d = np.linalg.norm(P[j, :2] - P[i, :2])
                min_nbr_margin = min(min_nbr_margin, d - sp.R1_lim)
            min_obs_margin = 1e3
            if obstacles_plan is not None and np.size(obstacles_plan) > 0:
                obs_plan = np.asarray(obstacles_plan, dtype=float)
                centers = obs_plan[:, :2]
                radii = obs_plan[:, 2]
                dist = np.linalg.norm(centers - P[i, :2], axis=1)
                min_obs_margin = float(np.min(dist - (radii + sp.R2_lim)))

            if trigger_state["step"] != step_k:
                trigger_state["step"] = step_k
                trigger_state["count"] = 0

            trigger = (
                (min_nbr_margin < args.nbr_margin_trigger) or
                (min_obs_margin < args.obs_margin_trigger)
            )
            should_collect = (step_k % args.stride == 0) or (
                trigger and trigger_state["count"] < args.max_trigger_per_step
            )
            if should_collect:
                if trigger and step_k % args.stride != 0:
                    trigger_state["count"] += 1
                # teacher label on the SAME state
                w_teacher = optimizePigeons_wrapper(
                    sp, P, i, V_xy, obstacles_plan,
                    ff_raw, fa_raw, fc_raw, vf_z, vo_raw, lamb[i], psi[i]
                )
                X_rows.append(feat)
                Y_rows.append(np.array(w_teacher, dtype=float))
                uav_ids.append(int(i))
                t_list.append(float(t))
                nbr_margin_list.append(float(min_nbr_margin))
                obs_margin_list.append(float(min_obs_margin))

            return w_student

        res = run_episode(base_sp, scene_cfg, collect=False, stride=args.stride, policy_fn=policy_fn)

        if not X_rows:
            print(f"[dagger ep {ep:03d}] no samples")
            continue

        X = np.vstack(X_rows)
        Y = np.vstack(Y_rows)

        meta = {
            "episode_id": np.full((X.shape[0],), ep_id, dtype=np.int32),
            "uav_id": np.asarray(uav_ids, dtype=np.int16),
            "t": np.asarray(t_list, dtype=np.float32),
            "scene_type": np.full((X.shape[0],), SCENE_TYPES[scene_cfg["scene_type"]], dtype=np.int16),
            "N_uav": np.full((X.shape[0],), scene_cfg["N_uav"], dtype=np.int16),
            "ve_mag": np.full((X.shape[0],), scene_cfg["ve_mag"], dtype=np.float32),
            # 记录 episode 结果，方便你后面筛选
            "safe_ok": np.full((X.shape[0],), int(res["safe_ok"]), dtype=np.int8),
            "formation_ok": np.full((X.shape[0],), int(res["formation_ok"]), dtype=np.int8),
            "nbr_margin": np.asarray(nbr_margin_list, dtype=np.float32),
            "obs_margin": np.asarray(obs_margin_list, dtype=np.float32),
            "collision_nbr_ep": np.full((X.shape[0],), int(res["cost_flags"]["c4_any"]), dtype=np.int16),
            "collision_obs_ep": np.full((X.shape[0],), int(res["cost_flags"]["c3_any"]), dtype=np.int16),
        }

        out_path = train_dir / f"ep_{ep_id:06d}.npz"
        write_npz(out_path, X, Y, meta)
        index["episodes"].append(out_path.as_posix())
        if index["feature_dim"] is None:
            index["feature_dim"] = int(X.shape[1])

        print(f"[dagger ep {ep:03d}] N={scene_cfg['N_uav']} type={scene_cfg['scene_type']} "
              f"samples={X.shape[0]} safe={res['safe_ok']} form={res['formation_ok']}")

    index["num_episodes"] = int(len(index["episodes"]))
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[dagger] wrote {len(index['episodes'])} episodes to {out_dir}")


if __name__ == "__main__":
    main()
