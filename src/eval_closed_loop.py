import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from base_mpio_solver import SimParams as BaseSimParams, run_episode as run_base_episode
from mpio_swarm_simv1 import (
    SimParams as ModifiedSimParams,
    extract_features,
    neighbors_within,
    run_episode as run_modified_episode,
)
from dataset_gen import sample_scene, _init_swarm, N_RANGE

TARGET_SCENE_TYPES = (
    "b_chicane_chain",
    "e_pillar_forest",
    "g_forest_dynamic_spheres",
)
METHODS = ("mlp", "base_mpio", "mpio")


def _n_bucket(n_uav: int) -> str:
    if n_uav <= 5:
        return "3-5"
    if n_uav <= 8:
        return "6-8"
    return f"9-{N_RANGE[1]}"


def _parse_csv_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _progress_path(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.name}.progress.json"


def _save_scene_json(scene_cfg: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_jsonable(scene_cfg), indent=2), encoding="utf-8")


def sample_scene_legacy_easy(rng: np.random.Generator) -> Dict[str, object]:
    n_uav = int(rng.integers(5, 7))
    ve_mag = float(rng.uniform(6.0, 12.0))
    try:
        from dataset_gen import _scene_legacy_easy
    except ImportError as exc:
        raise SystemExit(
            "legacy_easy scene is not available in dataset_gen.py; "
            "run without --legacy_easy_only or add _scene_legacy_easy."
        ) from exc
    obstacles, obstacles_vxy = _scene_legacy_easy(rng)
    init_p, init_vxy = _init_swarm(rng, n_uav, ve_mag)
    return {
        "scene_type": "legacy_easy",
        "N_uav": n_uav,
        "ve_mag": ve_mag,
        "init_P": init_p,
        "init_Vxy": init_vxy,
        "ve_xy": (ve_mag, 0.0),
        "obstacles": obstacles,
        "obstacles_vxy": obstacles_vxy,
        "obstacles_use_prediction": True,
        "square_enable": False,
        "formation_err_mean_thresh": 3.0,
        "formation_err_max_thresh": 10.0,
        "formation_over_limit": 0.20,
    }


def _load_scene_cfg(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    array_keys = {"init_P", "init_Vxy", "obstacles", "obstacles_vxy"}
    tuple_keys = {"ve_xy", "square_center_xy", "square_vxy", "obstacles_bounds"}

    scene_cfg: Dict[str, object] = {}
    for key, value in data.items():
        if value is None:
            scene_cfg[key] = None
        elif key in array_keys:
            scene_cfg[key] = np.asarray(value, dtype=float)
        elif key in tuple_keys:
            scene_cfg[key] = tuple(value)
        else:
            scene_cfg[key] = value

    scene_cfg.setdefault("scene_type", "custom")
    scene_cfg.setdefault("scene_level", 1)
    scene_cfg.setdefault("N_uav", 0)
    scene_cfg.setdefault("ve_mag", 0.0)
    scene_cfg.setdefault("he", 50.0)
    scene_cfg.setdefault("ve_xy", (0.0, 0.0))
    scene_cfg.setdefault("obstacles_use_prediction", False)
    scene_cfg.setdefault("square_enable", False)
    scene_cfg.setdefault("square_center_xy", (0.0, 0.0))
    scene_cfg.setdefault("square_side", 0.0)
    scene_cfg.setdefault("square_vxy", (0.0, 0.0))
    scene_cfg.setdefault("x_goal", 0.0)
    scene_cfg.setdefault("y_goal", 0.0)
    scene_cfg.setdefault("rng_seed", None)
    return scene_cfg


def _new_method_stats(selected_methods: Iterable[str]) -> Dict[str, Dict[str, float]]:
    return {method: {} for method in selected_methods}


def _accumulate(stats: Dict[str, float], results: Dict[str, object], formation_metrics: Dict[str, float]) -> None:
    stats["count"] = stats.get("count", 0) + 1
    stats["safe_ok"] = stats.get("safe_ok", 0) + int(results["safe_ok"])
    stats["reach_ok"] = stats.get("reach_ok", 0) + int(results["reach_ok"])
    stats["formation_ok"] = stats.get("formation_ok", 0) + int(results["formation_ok"])
    stats["formation_mean"] = stats.get("formation_mean", 0.0) + formation_metrics["mean_err"]
    stats["formation_max"] = stats.get("formation_max", 0.0) + formation_metrics["max_err"]
    cost_flags = results.get("cost_flags", {})
    stats["collision_obs"] = stats.get("collision_obs", 0) + int(cost_flags.get("c3_any", False))
    stats["collision_obs_hard"] = stats.get("collision_obs_hard", 0) + int(cost_flags.get("c3_hard_any", False))
    stats["collision_nbr"] = stats.get("collision_nbr", 0) + int(cost_flags.get("c4_any", False))
    hard_collision = bool(cost_flags.get("c3_hard_any", False)) or bool(cost_flags.get("c4_any", False))
    stats["collision_free"] = stats.get("collision_free", 0) + int(not hard_collision)
    latency = results.get("latency", {})
    stats["lat_step_mean_sum"] = stats.get("lat_step_mean_sum", 0.0) + float(latency.get("step_mean_ms", 0.0))
    stats["lat_step_std_sum"] = stats.get("lat_step_std_sum", 0.0) + float(latency.get("step_std_ms", 0.0))
    stats["lat_step_p95_sum"] = stats.get("lat_step_p95_sum", 0.0) + float(latency.get("step_p95_ms", 0.0))
    stats["lat_decision_mean_sum"] = stats.get("lat_decision_mean_sum", 0.0) + float(latency.get("decision_mean_ms", 0.0))
    stats["lat_decision_std_sum"] = stats.get("lat_decision_std_sum", 0.0) + float(latency.get("decision_std_ms", 0.0))
    stats["lat_decision_p95_sum"] = stats.get("lat_decision_p95_sum", 0.0) + float(latency.get("decision_p95_ms", 0.0))
    stats["lat_step_overrun_count"] = stats.get("lat_step_overrun_count", 0) + int(latency.get("step_overrun_count", 0))
    stats["lat_step_count"] = stats.get("lat_step_count", 0) + int(latency.get("step_count", 0))
    stats["lat_budget_ms"] = float(latency.get("budget_ms", stats.get("lat_budget_ms", 0.0)))


def _episode_diag(results: Dict[str, object]) -> Dict[str, object]:
    collision = results.get("collision")
    latency = results.get("latency", {})
    return {
        "safe_ok": bool(results.get("safe_ok")),
        "reach_ok": bool(results.get("reach_ok")),
        "formation_ok": bool(results.get("formation_ok")),
        "collision_obs": bool(results.get("cost_flags", {}).get("c3_any", False)),
        "collision_obs_hard": bool(results.get("cost_flags", {}).get("c3_hard_any", False)),
        "collision_nbr": bool(results.get("cost_flags", {}).get("c4_any", False)),
        "latency_step_mean_ms": float(latency.get("step_mean_ms", 0.0)),
        "latency_step_p95_ms": float(latency.get("step_p95_ms", 0.0)),
        "latency_step_overrun_ratio": float(latency.get("step_overrun_ratio", 0.0)),
        "failure_step": collision.get("step") if isinstance(collision, dict) else None,
        "failure_type": collision.get("type") if isinstance(collision, dict) else None,
        "min_clearance": results.get("min_clearance"),
    }


def _finalize(stats: Dict[str, float]) -> Dict[str, float]:
    count = max(int(stats.get("count", 0)), 1)
    lat_step_count = max(int(stats.get("lat_step_count", 0)), 1)
    return {
        "count": int(stats.get("count", 0)),
        "collision_free_rate": float(stats.get("collision_free", 0) / count),
        "safe_ok_rate": float(stats.get("safe_ok", 0) / count),
        "reach_ok_rate": float(stats.get("reach_ok", 0) / count),
        "formation_ok_rate": float(stats.get("formation_ok", 0) / count),
        "formation_mean": float(stats.get("formation_mean", 0.0) / count),
        "formation_max": float(stats.get("formation_max", 0.0) / count),
        "collision_obs_rate": float(stats.get("collision_obs", 0) / count),
        "obstacle_margin_violation_rate": float(stats.get("collision_obs", 0) / count),
        "obstacle_hard_collision_rate": float(stats.get("collision_obs_hard", 0) / count),
        "collision_nbr_rate": float(stats.get("collision_nbr", 0) / count),
        "latency_budget_ms": float(stats.get("lat_budget_ms", 0.0)),
        "latency_step_mean_ms": float(stats.get("lat_step_mean_sum", 0.0) / count),
        "latency_step_std_ms": float(stats.get("lat_step_std_sum", 0.0) / count),
        "latency_step_p95_ms": float(stats.get("lat_step_p95_sum", 0.0) / count),
        "latency_decision_mean_ms": float(stats.get("lat_decision_mean_sum", 0.0) / count),
        "latency_decision_std_ms": float(stats.get("lat_decision_std_sum", 0.0) / count),
        "latency_decision_p95_ms": float(stats.get("lat_decision_p95_sum", 0.0) / count),
        "latency_step_overrun_ratio": float(stats.get("lat_step_overrun_count", 0) / lat_step_count),
    }


def _build_final_summary(summary_raw: Dict[str, object], selected_methods: Tuple[str, ...]) -> Dict[str, object]:
    summary = copy.deepcopy(summary_raw)
    for scene_key, stats in summary["scene_stats"].items():
        for method in selected_methods:
            summary["scene_stats"][scene_key][method] = _finalize(stats[method])
    for bucket_key, stats in summary["bucket_stats"].items():
        for method in selected_methods:
            summary["bucket_stats"][bucket_key][method] = _finalize(stats[method])
    for method in selected_methods:
        summary["overall"][method] = _finalize(summary["overall"][method])
    return summary


def _write_progress(
    progress_path: Path,
    summary_raw: Dict[str, object],
    rng: np.random.Generator,
    valid_eps: int,
    target_scene_counts: Dict[str, int],
    scene_ptr: int,
) -> None:
    payload = {
        "summary_raw": summary_raw,
        "rng_state": _to_jsonable(rng.bit_generator.state),
        "valid_eps": int(valid_eps),
        "target_scene_counts": {k: int(v) for k, v in target_scene_counts.items()},
        "scene_ptr": int(scene_ptr),
    }
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_base_params() -> BaseSimParams:
    sp = BaseSimParams()
    sp.dt = 0.5
    sp.sim_time = 59.5
    setattr(sp, "safe_clearance_m", 0.5)
    setattr(sp, "safe_soft_band", 1.0)
    return sp


def _make_modified_params() -> ModifiedSimParams:
    sp = ModifiedSimParams()
    setattr(sp, "safe_clearance_m", 0.5)
    return sp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="mlp_out")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--episodes_per_scene", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--out_path", type=str, default="results.json")
    parser.add_argument("--legacy_easy_only", action="store_true")
    parser.add_argument("--scene_dir", type=str, default="", help="use scene jsons from this dir instead of sample_scene")
    parser.add_argument("--methods", type=str, default="mlp,base_mpio,mpio", help="comma-separated subset of: mlp,base_mpio,mpio")
    parser.add_argument("--scene_types", type=str, default="", help="comma-separated scene types to run; default uses protocol scene set")
    parser.add_argument("--resume", action="store_true", help="resume from out_path.progress.json")
    parser.add_argument("--save_scene_dir", type=str, default="", help="save accepted eval scenes as ep_XXXXXX.json for later replay")
    args = parser.parse_args()

    selected_methods = tuple(_parse_csv_list(args.methods))
    if not selected_methods:
        raise SystemExit("no methods selected")
    invalid_methods = [m for m in selected_methods if m not in METHODS]
    if invalid_methods:
        raise SystemExit(f"unknown methods: {invalid_methods}; valid={METHODS}")

    requested_scene_types = tuple(_parse_csv_list(args.scene_types))
    invalid_scene_types = [s for s in requested_scene_types if s not in TARGET_SCENE_TYPES]
    if invalid_scene_types:
        raise SystemExit(f"unknown scene_types: {invalid_scene_types}; valid={TARGET_SCENE_TYPES}")

    model = None
    mean = std = None
    if "mlp" in selected_methods:
        model_dir = Path(args.model_dir)
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
        if any(k.startswith("net.") for k in state.keys()):
            state = {k.replace("net.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()

    rng = np.random.default_rng(args.seed)

    def mlp_policy(i, P, V_xy, psi, lamb, obs_state, obstacles_plan, sp,
                   ff_raw, fa_raw, fc_raw, vf_z, vo_raw, step_k, t):
        if model is None or mean is None or std is None:
            raise RuntimeError("mlp policy requested but model is not loaded")
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
        x = (feat - mean) / std
        with torch.no_grad():
            w = model(torch.from_numpy(x.astype(np.float32))).numpy()
        return np.clip(w.astype(float), 0.0, 1.0)

    balanced_mode = not args.scene_dir and not args.legacy_easy_only
    target_scene_types = list(requested_scene_types) if requested_scene_types else (
        list(TARGET_SCENE_TYPES) if balanced_mode else []
    )
    target_total_episodes = (
        len(target_scene_types) * args.episodes_per_scene
        if balanced_mode else args.num_episodes
    )

    summary = {
        "episodes": target_total_episodes,
        "valid_episodes": 0,
        "scene_sample_attempts": 0,
        "invalid_scene_skips": 0,
        "balanced_scene_eval": bool(balanced_mode),
        "episodes_per_scene": int(args.episodes_per_scene) if balanced_mode else None,
        "target_scene_types": target_scene_types,
        "selected_methods": list(selected_methods),
        "obstacle_metric_definition": "collision_obs_rate is evaluated on inflated boundary: dist_to_center < r_obs + safe_clearance_m",
        "latency_protocol": "step-level decision latency (sum over UAVs per sim step); overrun if latency > dt",
        "scene_stats": {},
        "bucket_stats": {},
        "overall": {method: {} for method in selected_methods},
        "episodes_detail": [],
    }

    scene_paths: List[Path] = []
    if args.scene_dir:
        scene_dir = Path(args.scene_dir)
        if not scene_dir.exists():
            raise SystemExit(f"scene_dir not found: {scene_dir}")
        scene_paths = sorted(scene_dir.glob("ep_*.json"))
        if requested_scene_types:
            scene_paths = [p for p in scene_paths if _load_scene_cfg(p).get("scene_type") in requested_scene_types]
        if not scene_paths:
            raise SystemExit(f"no scene jsons in {scene_dir} after scene_type filter")
        target_total_episodes = len(scene_paths) if args.num_episodes <= 0 else min(args.num_episodes, len(scene_paths))
        summary["episodes"] = target_total_episodes
        summary["balanced_scene_eval"] = False
        summary["episodes_per_scene"] = None
        summary["target_scene_types"] = list(requested_scene_types)

    sample_scene_fn = sample_scene_legacy_easy if args.legacy_easy_only else sample_scene
    target_scene_counts = {scene_type: 0 for scene_type in target_scene_types}
    valid_eps = 0
    scene_ptr = 0
    progress_path = _progress_path(Path(args.out_path))
    save_scene_dir = Path(args.save_scene_dir) if args.save_scene_dir else None
    if save_scene_dir is not None:
        save_scene_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not progress_path.exists():
            raise SystemExit(f"resume requested but progress file not found: {progress_path}")
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        summary = payload["summary_raw"]
        valid_eps = int(payload.get("valid_eps", summary.get("valid_episodes", 0)))
        summary["valid_episodes"] = valid_eps
        target_scene_counts = {k: int(v) for k, v in payload.get("target_scene_counts", target_scene_counts).items()}
        scene_ptr = int(payload.get("scene_ptr", 0))
        rng.bit_generator.state = payload["rng_state"]
        print(f"[eval] resumed progress: {valid_eps}/{summary['episodes']} from {progress_path}", flush=True)

    max_attempts = max(target_total_episodes * 50, 1000)
    while valid_eps < target_total_episodes:
        summary["scene_sample_attempts"] += 1
        if summary["scene_sample_attempts"] > max_attempts:
            raise SystemExit(
                f"failed to collect {target_total_episodes} valid episodes after {summary['scene_sample_attempts']} attempts; "
                f"invalid_scene_skips={summary['invalid_scene_skips']}"
            )

        if scene_paths:
            if scene_ptr >= len(scene_paths):
                break
            scene_cfg = _load_scene_cfg(scene_paths[scene_ptr])
            scene_ptr += 1
            if scene_cfg.get("rng_seed") is None:
                scene_cfg["rng_seed"] = int(rng.integers(1, 1_000_000_000))
        elif balanced_mode:
            remaining_types = [
                scene_type for scene_type, count in target_scene_counts.items()
                if count < args.episodes_per_scene
            ]
            if not remaining_types:
                break
            scene_type = remaining_types[0]
            max_scene_tries = 500
            scene_cfg = {}
            for _ in range(max_scene_tries):
                scene_cfg = sample_scene(rng)
                if not scene_cfg:
                    continue
                if scene_cfg.get("scene_type") != scene_type:
                    continue
                break
            if not scene_cfg:
                raise SystemExit(f"failed to sample enough scenes for type={scene_type} after {max_scene_tries} tries")
            scene_cfg["rng_seed"] = int(rng.integers(1, 1_000_000_000))
        else:
            scene_cfg = sample_scene_fn(rng)
            if not scene_cfg or "scene_type" not in scene_cfg:
                summary["invalid_scene_skips"] += 1
                continue
            if requested_scene_types and scene_cfg["scene_type"] not in requested_scene_types:
                continue
            scene_cfg["rng_seed"] = int(rng.integers(1, 1_000_000_000))

        n_uav = int(scene_cfg.get("N_uav", 0))
        if not (N_RANGE[0] <= n_uav <= N_RANGE[1]):
            summary["invalid_scene_skips"] += 1
            continue
        scene_cfg["verbose_uav1"] = False
        scene_cfg["csv_path"] = ""

        results_by_method = {}
        if "mlp" in selected_methods:
            results_by_method["mlp"] = run_modified_episode(_make_modified_params(), scene_cfg, collect=False, policy_fn=mlp_policy)
        if "mpio" in selected_methods:
            results_by_method["mpio"] = run_modified_episode(_make_modified_params(), scene_cfg, collect=False, policy_fn=None)
        if "base_mpio" in selected_methods:
            results_by_method["base_mpio"] = run_base_episode(_make_base_params(), scene_cfg, collect=False)

        scene_key = scene_cfg["scene_type"]
        if balanced_mode and scene_key in target_scene_counts:
            if target_scene_counts[scene_key] >= args.episodes_per_scene:
                continue
            target_scene_counts[scene_key] += 1
        bucket_key = _n_bucket(scene_cfg["N_uav"])
        if save_scene_dir is not None:
            scene_path = save_scene_dir / f"ep_{valid_eps:06d}.json"
            _save_scene_json(scene_cfg, scene_path)
        summary["scene_stats"].setdefault(scene_key, _new_method_stats(selected_methods))
        summary["bucket_stats"].setdefault(bucket_key, _new_method_stats(selected_methods))

        ep_diag = {
            "scene_type": scene_key,
            "N_uav": int(scene_cfg["N_uav"]),
            "rng_seed": int(scene_cfg["rng_seed"]),
        }
        for method, result in results_by_method.items():
            ep_diag[method] = _episode_diag(result)
            _accumulate(summary["scene_stats"][scene_key][method], result, result["formation_metrics"])
            _accumulate(summary["bucket_stats"][bucket_key][method], result, result["formation_metrics"])
            _accumulate(summary["overall"][method], result, result["formation_metrics"])
        summary["episodes_detail"].append(ep_diag)

        valid_eps += 1
        summary["valid_episodes"] = valid_eps
        if balanced_mode:
            summary["scene_target_counts"] = {k: int(args.episodes_per_scene) for k in target_scene_types}
            summary["scene_actual_counts"] = {k: int(v) for k, v in target_scene_counts.items()}

        progress_scene = ""
        if balanced_mode and scene_key in target_scene_counts:
            progress_scene = f" [{scene_key} {target_scene_counts[scene_key]}/{args.episodes_per_scene}]"
        print(f"[eval] progress {valid_eps}/{target_total_episodes}{progress_scene}", flush=True)
        _write_progress(progress_path, summary, rng, valid_eps, target_scene_counts, scene_ptr)

    final_summary = _build_final_summary(summary, selected_methods)
    Path(args.out_path).write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"[eval] wrote {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
