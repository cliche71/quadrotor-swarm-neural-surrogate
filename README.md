# Neural Surrogate for Real-Time UAV Swarm Flocking in Cluttered Environments

This repository accompanies our paper on replacing online modified-MPIO weight search with a lightweight neural surrogate for distributed UAV swarm control in cluttered environments.

## Release Scope

This repository includes:
- source code for simulation, training, evaluation, and AirSim deployment,
- the trained surrogate checkpoint used in the reported results,
- merged evaluation summaries used for the main quantitative tables.

This repository does **not** include:
- the exact frozen training dataset used for the final reported model.

## Main Released Artifacts

- Trained model:
  - `output/mlp_out_v7_mix50/model.pt`
  - `output/mlp_out_v7_mix50/scaler.npz`
  - `output/mlp_out_v7_mix50/train_log.json`
- Reported evaluation summaries:
  - `output/merged_eval/mlp_overall.json`
  - `output/merged_eval/mpio_overall.json`
  - `output/merged_eval/base_mpio_overall.json`

## Repository Structure

- `src/`: simulation, training, evaluation, and AirSim deployment code
- `output/mlp_out_v7_mix50/`: released trained surrogate checkpoint
- `output/merged_eval/`: merged evaluation outputs used for the reported tables
- `data/README.md`: data-availability statement for the frozen training artifact

## Installation

Install the core offline dependencies:

```bash
pip install -r requirements.txt
```

Install the additional AirSim runtime dependency only if you want to run the online deployment entry:

```bash
pip install -r requirements-airsim.txt
```

## Quick Inspection of the Reported Results

The paper-facing aggregate numbers are released directly under `output/merged_eval/`. The three main summary files are:

- `output/merged_eval/mlp_overall.json`
- `output/merged_eval/mpio_overall.json`
- `output/merged_eval/base_mpio_overall.json`

## AirSim Deployment

Minimal command:

```bash
python src/run_on_airsim.py \
  --backend mlp \
  --model_dir output/mlp_out_v7_mix50 \
  --swarm_profile_json src/swarm_profile.json \
  --airsim_settings_json /path/to/AirSim/settings.json
```

Recommended command for a smoother release demo:

```bash
python src/run_on_airsim.py \
  --backend mlp \
  --model_dir output/mlp_out_v7_mix50 \
  --swarm_profile_json src/swarm_profile.json \
  --airsim_settings_json /path/to/AirSim/settings.json \
  --control_dt 0.025 \
  --a_max 6.0 \
  --no_save_logs
```

This public release does not ship the frozen scene JSON files used in our internal dataset artifacts. If you want to use `--scene_json`, provide your own compatible scene file.

## Reproducibility Note

This release is intended to support paper-level inspection and deployment-oriented use:

- the full source code for simulation, training, evaluation, and AirSim deployment is included;
- the final reported surrogate checkpoint is included;
- the merged evaluation summaries used for the reported tables are included.

The exact frozen training dataset used for the final reported model is not publicly distributed in this repository.
