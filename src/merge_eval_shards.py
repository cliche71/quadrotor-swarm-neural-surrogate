import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FILE_RE = re.compile(r"^(?P<prefix>[a-z_]+)_s(?P<seed>\d{4})_(?P<scene>.+)\.json$")


def _weighted_merge_metric_dict(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(items)
    total_count = sum(int(item.get("count", 0)) for item in items)
    if total_count <= 0:
        return {}

    merged: Dict[str, float] = {"count": int(total_count)}
    keys = set()
    for item in items:
        keys.update(item.keys())
    keys.discard("count")

    for key in sorted(keys):
        values: List[Tuple[float, int]] = []
        for item in items:
            if key not in item:
                continue
            values.append((float(item[key]), int(item.get("count", 0))))
        if not values:
            continue
        denom = sum(weight for _, weight in values)
        if denom <= 0:
            continue
        merged[key] = sum(value * weight for value, weight in values) / denom
    return merged


def _merge_nested_stats(dicts: Iterable[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    dicts = list(dicts)
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())

    merged: Dict[str, Dict[str, float]] = {}
    for key in sorted(all_keys):
        parts = [d[key] for d in dicts if key in d]
        merged[key] = _weighted_merge_metric_dict(parts)
    return merged


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_seed(path: Path) -> str:
    m = FILE_RE.match(path.name)
    if not m:
        raise SystemExit(f"cannot infer seed from filename: {path.name}")
    return m.group("seed")


def _infer_method(payload: Dict[str, object], path: Path) -> str:
    selected = payload.get("selected_methods")
    if isinstance(selected, list) and len(selected) == 1:
        return str(selected[0])
    overall = payload.get("overall", {})
    if isinstance(overall, dict) and len(overall) == 1:
        return str(next(iter(overall.keys())))
    raise SystemExit(f"cannot infer single method from {path}")


def _merge_shard_payloads(payloads: List[Dict[str, object]], method: str) -> Dict[str, object]:
    if not payloads:
        raise SystemExit("no payloads to merge")

    total_valid = sum(int(p.get("valid_episodes", 0)) for p in payloads)
    total_target = sum(int(p.get("episodes", 0)) for p in payloads)

    merged = {
        "episodes": int(total_target),
        "valid_episodes": int(total_valid),
        "scene_sample_attempts": sum(int(p.get("scene_sample_attempts", 0)) for p in payloads),
        "invalid_scene_skips": sum(int(p.get("invalid_scene_skips", 0)) for p in payloads),
        "balanced_scene_eval": False,
        "episodes_per_scene": None,
        "target_scene_types": sorted({
            scene_type
            for p in payloads
            for scene_type in p.get("target_scene_types", [])
        }),
        "selected_methods": [method],
        "obstacle_metric_definition": payloads[0].get("obstacle_metric_definition"),
        "latency_protocol": payloads[0].get("latency_protocol"),
        "scene_stats": _merge_nested_stats([p.get("scene_stats", {}) for p in payloads]),
        "bucket_stats": _merge_nested_stats([p.get("bucket_stats", {}) for p in payloads]),
        "overall": {
            method: _weighted_merge_metric_dict(
                [p.get("overall", {}).get(method, {}) for p in payloads if method in p.get("overall", {})]
            )
        },
        "episodes_detail": [],
    }

    episodes_detail: List[Dict[str, object]] = []
    for p in payloads:
        episodes_detail.extend(p.get("episodes_detail", []))
    merged["episodes_detail"] = episodes_detail
    return merged


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output")
    parser.add_argument("--prefixes", type=str, default="base,mpio", help="comma-separated filename prefixes to merge")
    parser.add_argument("--out_dir", type=str, default="output/merged_eval")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    prefixes = [item.strip() for item in args.prefixes.split(",") if item.strip()]
    if not prefixes:
        raise SystemExit("no prefixes selected")

    grouped: Dict[str, Dict[str, List[Tuple[Path, Dict[str, object]]]]] = {}
    for prefix in prefixes:
        for path in sorted(input_dir.glob(f"{prefix}_s*.json")):
            if path.name.endswith(".progress.json"):
                continue
            payload = _load_json(path)
            method = _infer_method(payload, path)
            seed = _infer_seed(path)
            grouped.setdefault(method, {}).setdefault(seed, []).append((path, payload))

    if not grouped:
        raise SystemExit(f"no shard jsons found in {input_dir} for prefixes={prefixes}")

    for method, seed_map in sorted(grouped.items()):
        seed_payloads: List[Dict[str, object]] = []
        for seed, shard_items in sorted(seed_map.items()):
            paths = [path for path, _ in shard_items]
            payloads = [payload for _, payload in shard_items]
            merged_seed = _merge_shard_payloads(payloads, method)
            merged_seed["source_files"] = [path.name for path in paths]
            seed_out = out_dir / f"{method}_seed_{seed}.json"
            _write_json(seed_out, merged_seed)
            seed_payloads.append(merged_seed)
            print(f"[merge] wrote {seed_out}")

        merged_overall = _merge_shard_payloads(seed_payloads, method)
        merged_overall["source_files"] = [f"{method}_seed_{seed}.json" for seed in sorted(seed_map.keys())]
        overall_out = out_dir / f"{method}_overall.json"
        _write_json(overall_out, merged_overall)
        print(f"[merge] wrote {overall_out}")


if __name__ == "__main__":
    main()
