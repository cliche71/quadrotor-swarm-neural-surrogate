import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _load_npz_files(files: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_list = []
    Y_list = []
    nbr_margin_list = []
    collision_nbr_list = []
    obs_margin_list = []
    collision_obs_list = []
    for fp in files:
        data = np.load(fp)
        X = data["X"].astype(np.float32)
        Y = data["Y"].astype(np.float32)
        X_list.append(X)
        Y_list.append(Y)
        if "nbr_margin" in data:
            nbr_margin = data["nbr_margin"].astype(np.float32)
        else:
            nbr_margin = np.zeros((X.shape[0],), dtype=np.float32)
        if "collision_nbr_ep" in data:
            collision_nbr = data["collision_nbr_ep"].astype(np.float32)
        else:
            collision_nbr = np.zeros((X.shape[0],), dtype=np.float32)
        if "obs_margin" in data:
            obs_margin = data["obs_margin"].astype(np.float32)
        else:
            obs_margin = np.full((X.shape[0],), 1e3, dtype=np.float32)
        if "collision_obs_ep" in data:
            collision_obs = data["collision_obs_ep"].astype(np.float32)
        else:
            collision_obs = np.zeros((X.shape[0],), dtype=np.float32)
        nbr_margin_list.append(nbr_margin)
        collision_nbr_list.append(collision_nbr)
        obs_margin_list.append(obs_margin)
        collision_obs_list.append(collision_obs)
    if not X_list:
        empty_x = np.zeros((0, 0), dtype=np.float32)
        empty_y = np.zeros((0, 2), dtype=np.float32)
        empty_meta = np.zeros((0,), dtype=np.float32)
        return empty_x, empty_y, empty_meta, empty_meta, empty_meta, empty_meta
    return (
        np.vstack(X_list),
        np.vstack(Y_list),
        np.concatenate(nbr_margin_list),
        np.concatenate(collision_nbr_list),
        np.concatenate(obs_margin_list),
        np.concatenate(collision_obs_list),
    )


def _split_episodes(files: List[Path], seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n = len(files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return [files[i] for i in train_idx], [files[i] for i in val_idx], [files[i] for i in test_idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sample_weights(
    nbr_margin: torch.Tensor,
    collision_nbr_ep: torch.Tensor,
    obs_margin: torch.Tensor,
    collision_obs_ep: torch.Tensor,
    nbr_margin_ref: float,
    obs_margin_ref: float,
    nbr_margin_weight: float,
    obs_margin_weight: float,
    nbr_collision_mult: float,
    obs_collision_mult: float,
) -> torch.Tensor:
    w = torch.ones_like(nbr_margin)
    w = w + nbr_margin_weight * torch.clamp((nbr_margin_ref - nbr_margin) / max(nbr_margin_ref, 1e-6), 0.0, 1.0)
    w = w + obs_margin_weight * torch.clamp((obs_margin_ref - obs_margin) / max(obs_margin_ref, 1e-6), 0.0, 1.0)
    w = w * torch.where(collision_nbr_ep > 0.5, nbr_collision_mult, 1.0)
    w = w * torch.where(collision_obs_ep > 0.5, obs_collision_mult, 1.0)
    return w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/train")
    parser.add_argument("--out_dir", type=str, default="mlp_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--nbr_margin_ref", type=float, default=0.5)
    parser.add_argument("--obs_margin_ref", type=float, default=1.0)
    parser.add_argument("--nbr_margin_weight", type=float, default=5.0)
    parser.add_argument("--obs_margin_weight", type=float, default=4.0)
    parser.add_argument("--nbr_collision_mult", type=float, default=3.0)
    parser.add_argument("--obs_collision_mult", type=float, default=2.5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("ep_*.npz"))
    if not files:
        raise SystemExit(f"no data in {data_dir}")

    train_files, val_files, test_files = _split_episodes(files, args.seed)

    X_train, Y_train, nbr_train, col_train, obs_train, col_obs_train = _load_npz_files(train_files)
    X_val, Y_val, nbr_val, col_val, obs_val, col_obs_val = _load_npz_files(val_files)
    X_test, Y_test, nbr_test, col_test, obs_test, col_obs_test = _load_npz_files(test_files)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    np.savez_compressed(out_dir / "scaler.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))

    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    X_test_n = (X_test - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss(reduction="none")

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(Y_train),
        torch.from_numpy(nbr_train),
        torch.from_numpy(col_train),
        torch.from_numpy(obs_train),
        torch.from_numpy(col_obs_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_n),
        torch.from_numpy(Y_val),
        torch.from_numpy(nbr_val),
        torch.from_numpy(col_val),
        torch.from_numpy(obs_val),
        torch.from_numpy(col_obs_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_n),
        torch.from_numpy(Y_test),
        torch.from_numpy(nbr_test),
        torch.from_numpy(col_test),
        torch.from_numpy(obs_test),
        torch.from_numpy(col_obs_test),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    best_val = float("inf")
    best_state = None
    no_improve = 0
    log = {"train_loss": [], "val_loss": [], "val_loss_weighted": [], "test_mae": None}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, nbr_margin, collision_nbr_ep, obs_margin, collision_obs_ep in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            nbr_margin = nbr_margin.to(device=device, dtype=torch.float32)
            collision_nbr_ep = collision_nbr_ep.to(device=device, dtype=torch.float32)
            obs_margin = obs_margin.to(device=device, dtype=torch.float32)
            collision_obs_ep = collision_obs_ep.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()
            pred = model(xb)
            loss_raw = criterion(pred, yb).mean(dim=1)
            w = _sample_weights(
                nbr_margin, collision_nbr_ep, obs_margin, collision_obs_ep,
                args.nbr_margin_ref, args.obs_margin_ref,
                args.nbr_margin_weight, args.obs_margin_weight,
                args.nbr_collision_mult, args.obs_collision_mult,
            )
            loss = (loss_raw * w).mean()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * xb.size(0)
        train_loss /= max(len(train_ds), 1)

        model.eval()
        val_loss = 0.0
        val_loss_weighted = 0.0
        with torch.no_grad():
            for xb, yb, nbr_margin, collision_nbr_ep, obs_margin, collision_obs_ep in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)
                nbr_margin = nbr_margin.to(device=device, dtype=torch.float32)
                collision_nbr_ep = collision_nbr_ep.to(device=device, dtype=torch.float32)
                obs_margin = obs_margin.to(device=device, dtype=torch.float32)
                collision_obs_ep = collision_obs_ep.to(device=device, dtype=torch.float32)
                pred = model(xb)
                loss_raw = criterion(pred, yb).mean(dim=1)
                loss = loss_raw.mean()
                w = _sample_weights(
                    nbr_margin, collision_nbr_ep, obs_margin, collision_obs_ep,
                    args.nbr_margin_ref, args.obs_margin_ref,
                    args.nbr_margin_weight, args.obs_margin_weight,
                    args.nbr_collision_mult, args.obs_collision_mult,
                )
                loss_weighted = (loss_raw * w).mean()
                val_loss += float(loss.item()) * xb.size(0)
                val_loss_weighted += float(loss_weighted.item()) * xb.size(0)
        val_loss /= max(len(val_ds), 1)
        val_loss_weighted /= max(len(val_ds), 1)

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        log["val_loss_weighted"].append(val_loss_weighted)
        print(
            f"[epoch {epoch:03d}] train={train_loss:.6f} "
            f"val={val_loss:.6f} val_w={val_loss_weighted:.6f}"
        )

        if val_loss_weighted < best_val - 1e-6:
            best_val = val_loss_weighted
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("[early-stop] no improvement")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        mae_sum = 0.0
        for xb, yb, nbr_margin, collision_nbr_ep, obs_margin, collision_obs_ep in test_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            nbr_margin = nbr_margin.to(device=device, dtype=torch.float32)
            collision_nbr_ep = collision_nbr_ep.to(device=device, dtype=torch.float32)
            obs_margin = obs_margin.to(device=device, dtype=torch.float32)
            collision_obs_ep = collision_obs_ep.to(device=device, dtype=torch.float32)
            pred = model(xb)
            mae_sum += float(torch.abs(pred - yb).mean().item()) * xb.size(0)
        test_mae = mae_sum / max(len(test_ds), 1)
    log["test_mae"] = test_mae

    torch.save({"model_state": model.state_dict(), "input_dim": X_train.shape[1]}, out_dir / "model.pt")
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[test] mae={test_mae:.6f}")


if __name__ == "__main__":
    main()
