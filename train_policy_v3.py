#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


DIRS_8: List[Tuple[int, int]] = [
    (-1, 0),   # 0: up
    (-1, 1),   # 1: up-right
    (0, 1),    # 2: right
    (1, 1),    # 3: down-right
    (1, 0),    # 4: down
    (1, -1),   # 5: down-left
    (0, -1),   # 6: left
    (-1, -1),  # 7: up-left
]
DIR_TO_ACTION: Dict[Tuple[int, int], int] = {d: i for i, d in enumerate(DIRS_8)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ưu tiên reproducible khi debug rollout
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def rotate_dir_ccw_90(dr: int, dc: int) -> Tuple[int, int]:
    return -dc, dr


def hflip_dir(dr: int, dc: int) -> Tuple[int, int]:
    return dr, -dc


def vflip_dir(dr: int, dc: int) -> Tuple[int, int]:
    return -dr, dc


def rotate_rc_ccw_90(r: int, c: int, h: int, w: int) -> Tuple[int, int]:
    # (r, c) -> (w - 1 - c, r)
    return w - 1 - c, r


def hflip_rc(r: int, c: int, h: int, w: int) -> Tuple[int, int]:
    return r, w - 1 - c


def vflip_rc(r: int, c: int, h: int, w: int) -> Tuple[int, int]:
    return h - 1 - r, c


class PolicyDataset(Dataset):
    """
    Expected keys in each .npz sample:
      - x            : [5, H, W] float32
      - action       : scalar int
      - current_rc   : [2] int
      - goal_rc      : [2] int
      - obstacle     : [H, W] float32
      - gt_path_mask : [H, W] float32
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        enable_augmentation: bool = False,
        aug_rot90: bool = True,
        aug_hflip: bool = True,
        aug_vflip: bool = False,
        aug_prob_rot90: float = 0.75,
        aug_prob_hflip: float = 0.50,
        aug_prob_vflip: float = 0.20,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.files: List[Path] = sorted((self.root_dir / split).glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz found in: {self.root_dir / split}")

        self.enable_augmentation = enable_augmentation and split == "train"
        self.aug_rot90 = aug_rot90
        self.aug_hflip = aug_hflip
        self.aug_vflip = aug_vflip
        self.aug_prob_rot90 = aug_prob_rot90
        self.aug_prob_hflip = aug_prob_hflip
        self.aug_prob_vflip = aug_prob_vflip

        self.actions: List[int] = []
        for f in self.files:
            with np.load(f, allow_pickle=False) as data:
                self.actions.append(int(data["action"]))

    def __len__(self) -> int:
        return len(self.files)

    def _augment_sample(
        self,
        x: np.ndarray,
        action: int,
        current_rc: np.ndarray,
        goal_rc: np.ndarray,
        obstacle: np.ndarray,
        gt_path_mask: np.ndarray,
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h, w = obstacle.shape
        cur_r, cur_c = int(current_rc[0]), int(current_rc[1])
        goal_r, goal_c = int(goal_rc[0]), int(goal_rc[1])
        dr, dc = DIRS_8[action]

        # Random rot90 CCW, k in {0,1,2,3}
        if self.aug_rot90 and random.random() < self.aug_prob_rot90:
            k = random.randint(0, 3)
            if k > 0:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                obstacle = np.rot90(obstacle, k=k, axes=(0, 1)).copy()
                gt_path_mask = np.rot90(gt_path_mask, k=k, axes=(0, 1)).copy()

                for _ in range(k):
                    cur_r, cur_c = rotate_rc_ccw_90(cur_r, cur_c, h, w)
                    goal_r, goal_c = rotate_rc_ccw_90(goal_r, goal_c, h, w)
                    dr, dc = rotate_dir_ccw_90(dr, dc)
                    h, w = w, h

        # Horizontal flip
        if self.aug_hflip and random.random() < self.aug_prob_hflip:
            x = np.flip(x, axis=2).copy()
            obstacle = np.flip(obstacle, axis=1).copy()
            gt_path_mask = np.flip(gt_path_mask, axis=1).copy()

            cur_r, cur_c = hflip_rc(cur_r, cur_c, h, w)
            goal_r, goal_c = hflip_rc(goal_r, goal_c, h, w)
            dr, dc = hflip_dir(dr, dc)

        # Vertical flip
        if self.aug_vflip and random.random() < self.aug_prob_vflip:
            x = np.flip(x, axis=1).copy()
            obstacle = np.flip(obstacle, axis=0).copy()
            gt_path_mask = np.flip(gt_path_mask, axis=0).copy()

            cur_r, cur_c = vflip_rc(cur_r, cur_c, h, w)
            goal_r, goal_c = vflip_rc(goal_r, goal_c, h, w)
            dr, dc = vflip_dir(dr, dc)

        new_action = DIR_TO_ACTION[(dr, dc)]
        new_current_rc = np.asarray([cur_r, cur_c], dtype=np.int64)
        new_goal_rc = np.asarray([goal_r, goal_c], dtype=np.int64)

        return x, new_action, new_current_rc, new_goal_rc, obstacle, gt_path_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self.files[idx]
        with np.load(f, allow_pickle=False) as data:
            x = data["x"].astype(np.float32)
            action = int(data["action"])
            current_rc = data["current_rc"].astype(np.int64)
            goal_rc = data["goal_rc"].astype(np.int64)
            obstacle = data["obstacle"].astype(np.float32)
            gt_path_mask = data["gt_path_mask"].astype(np.float32)

        if self.enable_augmentation:
            x, action, current_rc, goal_rc, obstacle, gt_path_mask = self._augment_sample(
                x=x,
                action=action,
                current_rc=current_rc,
                goal_rc=goal_rc,
                obstacle=obstacle,
                gt_path_mask=gt_path_mask,
            )

        return {
            "x": torch.from_numpy(x.copy()),
            "action": torch.tensor(action, dtype=torch.long),
            "current_rc": torch.from_numpy(current_rc.copy()),
            "goal_rc": torch.from_numpy(goal_rc.copy()),
            "obstacle": torch.from_numpy(obstacle.copy()).unsqueeze(0),
            "gt_path_mask": torch.from_numpy(gt_path_mask.copy()).unsqueeze(0),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PolicyPlannerNet(nn.Module):
    """
    model(x) -> action_logits
    model(x, return_aux=True) -> (action_logits, path_logits)
    """

    def __init__(self, in_ch: int = 5, base_ch: int = 32, num_actions: int = 8, dropout_p: float = 0.10) -> None:
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bot = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

        self.fuse = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(base_ch * 4, base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(base_ch * 4, num_actions),
        )

        self.path_head = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))
        feat = self.fuse(b)
        return feat

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        feat = self.encode(x)

        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        action_logits = self.head(pooled)

        if not return_aux:
            return action_logits

        path_logits = self.path_head(feat)
        path_logits = F.interpolate(
            path_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return action_logits, path_logits


@dataclass
class TrainConfig:
    dataset_root: str
    save_dir: str

    batch_size: int = 64
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    base_ch: int = 32
    dropout_p: float = 0.10

    label_smoothing: float = 0.05
    path_loss_weight: float = 0.20

    sampler_power: float = 0.50
    max_sample_weight: float = 3.0
    disable_sampler: bool = False

    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    grad_clip_norm: float = 1.0

    enable_augmentation: bool = True
    aug_rot90: bool = True
    aug_hflip: bool = True
    aug_vflip: bool = False
    aug_prob_rot90: float = 0.75
    aug_prob_hflip: float = 0.50
    aug_prob_vflip: float = 0.20

    eval_rollout_max_factor: int = 6
    loop_fail_visit_count: int = 2
    eval_rollout_samples_per_batch: int = 4

    # Chỉ bật nếu runtime thật cũng dùng cùng logic này
    enable_action_cost_rerank: bool = False
    action_cost_penalty_gain: float = 0.75

    early_stop_patience: int = 6
    min_epochs_before_stop: int = 10


def action_loss_fn(logits: torch.Tensor, target: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    return F.cross_entropy(logits, target, label_smoothing=label_smoothing)


def path_mask_loss_fn(path_logits: torch.Tensor, gt_path_mask: torch.Tensor) -> torch.Tensor:
    target = gt_path_mask.float()
    pos = target.sum()
    neg = torch.tensor(target.numel(), device=target.device, dtype=target.dtype) - pos
    pos_weight_value = (neg / pos.clamp_min(1.0)).clamp(min=1.0, max=20.0)
    pos_weight = torch.as_tensor(pos_weight_value, dtype=path_logits.dtype, device=path_logits.device)
    return F.binary_cross_entropy_with_logits(path_logits, target, pos_weight=pos_weight)


def compute_acc(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item())


def build_weighted_sampler(
    dataset: PolicyDataset,
    power: float = 0.5,
    max_weight: float = 3.0,
) -> Tuple[WeightedRandomSampler, Counter]:
    counts = Counter(dataset.actions)
    if not counts:
        raise RuntimeError("Dataset actions are empty")

    max_count = max(counts.values())
    class_weights: Dict[int, float] = {}
    for a in range(8):
        c = counts.get(a, 0)
        if c <= 0:
            class_weights[a] = float(max_weight)
        else:
            w = (max_count / float(c)) ** float(power)
            class_weights[a] = min(float(w), float(max_weight))

    sample_weights = [class_weights[a] for a in dataset.actions]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, counts


def choose_action(
    logits: torch.Tensor,
    x_base: torch.Tensor,
    obstacle: torch.Tensor,
    current: Tuple[int, int],
    enable_action_cost_rerank: bool,
    action_cost_penalty_gain: float,
) -> Optional[int]:
    """
    logits shape: [8]
    Return:
      - int action nếu có action hợp lệ
      - None nếu không còn action hợp lệ nào
    """
    h, w = obstacle.shape[-2:]

    valid_actions: List[Tuple[int, int, int]] = []
    for a, (dr, dc) in enumerate(DIRS_8):
        nr = current[0] + dr
        nc = current[1] + dc

        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            continue
        if obstacle[0, 0, nr, nc] >= 0.5:
            continue

        valid_actions.append((a, nr, nc))

    if not valid_actions:
        return None

    if not enable_action_cost_rerank:
        best_action = max(valid_actions, key=lambda t: float(logits[t[0]].item()))[0]
        return int(best_action)

    best_score = -1e18
    best_action = None

    for a, nr, nc in valid_actions:
        score = float(logits[a].item()) - float(action_cost_penalty_gain) * float(x_base[0, 1, nr, nc].item())
        if score > best_score:
            best_score = score
            best_action = a

    return int(best_action) if best_action is not None else None


def rollout_policy(
    model: nn.Module,
    x_base: torch.Tensor,
    obstacle: torch.Tensor,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    max_steps: int = 512,
    loop_fail_visit_count: int = 2,
    enable_action_cost_rerank: bool = False,
    action_cost_penalty_gain: float = 0.75,
) -> Tuple[bool, int]:
    device = x_base.device
    h, w = obstacle.shape[-2:]
    current = (int(start_rc[0]), int(start_rc[1]))
    goal = (int(goal_rc[0]), int(goal_rc[1]))

    visit_counts: Dict[Tuple[int, int], int] = {current: 1}
    steps = 0

    while steps < max_steps:
        if current == goal:
            return True, steps

        cur_map = torch.zeros((1, 1, h, w), dtype=x_base.dtype, device=device)
        goal_map = torch.zeros((1, 1, h, w), dtype=x_base.dtype, device=device)
        cur_map[0, 0, current[0], current[1]] = 1.0
        goal_map[0, 0, goal[0], goal[1]] = 1.0

        x = x_base.clone()
        x[:, 3:4] = cur_map
        x[:, 4:5] = goal_map

        with torch.no_grad():
            logits = model(x).squeeze(0)

        action = choose_action(
            logits=logits,
            x_base=x_base,
            obstacle=obstacle,
            current=current,
            enable_action_cost_rerank=enable_action_cost_rerank,
            action_cost_penalty_gain=action_cost_penalty_gain,
        )

        if action is None:
            return False, steps

        dr, dc = DIRS_8[action]
        nr = current[0] + dr
        nc = current[1] + dc
        nxt = (nr, nc)
        steps += 1

        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            return False, steps
        if obstacle[0, 0, nr, nc] >= 0.5:
            return False, steps

        current = nxt
        visit_counts[current] = visit_counts.get(current, 0) + 1
        if visit_counts[current] >= loop_fail_visit_count:
            return False, steps

    return False, steps


def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_action_loss = 0.0
    total_path_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    rollout_success = 0
    rollout_total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(cfg.device, non_blocking=True)
            action = batch["action"].to(cfg.device, non_blocking=True)
            obstacle = batch["obstacle"].to(cfg.device, non_blocking=True)
            gt_path_mask = batch["gt_path_mask"].to(cfg.device, non_blocking=True)

            logits, path_logits = model(x, return_aux=True)
            action_loss = action_loss_fn(logits, action, cfg.label_smoothing)
            path_loss = path_mask_loss_fn(path_logits, gt_path_mask)
            loss = action_loss + cfg.path_loss_weight * path_loss
            acc = compute_acc(logits, action)

            total_loss += float(loss.item())
            total_action_loss += float(action_loss.item())
            total_path_loss += float(path_loss.item())
            total_acc += acc
            total_batches += 1

            rollout_n = min(cfg.eval_rollout_samples_per_batch, x.shape[0])
            for i in range(rollout_n):
                x0 = x[i:i + 1].clone()
                obs0 = obstacle[i:i + 1]
                start_rc0 = tuple(int(v) for v in batch["current_rc"][i].tolist())
                goal_rc0 = tuple(int(v) for v in batch["goal_rc"][i].tolist())

                max_steps = int(cfg.eval_rollout_max_factor * (obs0.shape[-2] + obs0.shape[-1]))
                ok, _ = rollout_policy(
                    model=model,
                    x_base=x0,
                    obstacle=obs0,
                    start_rc=start_rc0,
                    goal_rc=goal_rc0,
                    max_steps=max_steps,
                    loop_fail_visit_count=cfg.loop_fail_visit_count,
                    enable_action_cost_rerank=cfg.enable_action_cost_rerank,
                    action_cost_penalty_gain=cfg.action_cost_penalty_gain,
                )
                rollout_success += int(ok)
                rollout_total += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "action_loss": total_action_loss / max(total_batches, 1),
        "path_loss": total_path_loss / max(total_batches, 1),
        "acc": total_acc / max(total_batches, 1),
        "rollout": rollout_success / max(rollout_total, 1),
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    val_metrics: Dict[str, float],
) -> None:
    ckpt: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg.__dict__,
        "val_loss": val_metrics["loss"],
        "val_action_loss": val_metrics["action_loss"],
        "val_path_loss": val_metrics["path_loss"],
        "val_acc": val_metrics["acc"],
        "val_rollout": val_metrics["rollout"],
    }
    torch.save(ckpt, path)


def is_improved(
    val_rollout: float,
    val_acc: float,
    val_loss: float,
    best_rollout: float,
    best_acc: float,
    best_val_loss: float,
) -> bool:
    if val_rollout > best_rollout:
        return True
    if val_rollout == best_rollout and val_acc > best_acc:
        return True
    if val_rollout == best_rollout and val_acc == best_acc and val_loss < best_val_loss:
        return True
    return False


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    sampler,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PolicyDataset(
        cfg.dataset_root,
        "train",
        enable_augmentation=cfg.enable_augmentation,
        aug_rot90=cfg.aug_rot90,
        aug_hflip=cfg.aug_hflip,
        aug_vflip=cfg.aug_vflip,
        aug_prob_rot90=cfg.aug_prob_rot90,
        aug_prob_hflip=cfg.aug_prob_hflip,
        aug_prob_vflip=cfg.aug_prob_vflip,
    )
    val_ds = PolicyDataset(cfg.dataset_root, "val", enable_augmentation=False)

    test_ds = None
    test_dir = Path(cfg.dataset_root) / "test"
    if test_dir.exists() and any(test_dir.glob("*.npz")):
        test_ds = PolicyDataset(cfg.dataset_root, "test", enable_augmentation=False)

    train_sampler = None
    action_counts = Counter(train_ds.actions)
    if not cfg.disable_sampler:
        train_sampler, action_counts = build_weighted_sampler(
            train_ds,
            power=cfg.sampler_power,
            max_weight=cfg.max_sample_weight,
        )

    train_loader = make_loader(
        dataset=train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=("cuda" in cfg.device),
    )
    val_loader = make_loader(
        dataset=val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=cfg.num_workers,
        pin_memory=("cuda" in cfg.device),
    )
    test_loader = None
    if test_ds is not None:
        test_loader = make_loader(
            dataset=test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=cfg.num_workers,
            pin_memory=("cuda" in cfg.device),
        )

    model = PolicyPlannerNet(
        in_ch=5,
        base_ch=cfg.base_ch,
        num_actions=8,
        dropout_p=cfg.dropout_p,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in cfg.device and cfg.amp))

    best_rollout = -1.0
    best_acc = -1.0
    best_val_loss = float("inf")
    epochs_since_improve = 0

    print("=" * 100, flush=True)
    print("TRAIN CONFIG", flush=True)
    for k, v in cfg.__dict__.items():
        print(f"{k:28s}: {v}", flush=True)
    print(f"train samples               : {len(train_ds)}", flush=True)
    print(f"val samples                 : {len(val_ds)}", flush=True)
    if test_ds is not None:
        print(f"test samples                : {len(test_ds)}", flush=True)
    print(f"train action counts         : {dict(sorted(action_counts.items()))}", flush=True)
    print("=" * 100, flush=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_action_loss = 0.0
        running_path_loss = 0.0
        running_acc = 0.0
        num_steps = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch["x"].to(cfg.device, non_blocking=True)
            action = batch["action"].to(cfg.device, non_blocking=True)
            gt_path_mask = batch["gt_path_mask"].to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda" if "cuda" in cfg.device else "cpu",
                enabled=(cfg.amp and "cuda" in cfg.device),
            ):
                logits, path_logits = model(x, return_aux=True)
                action_loss = action_loss_fn(logits, action, cfg.label_smoothing)
                path_loss = path_mask_loss_fn(path_logits, gt_path_mask)
                loss = action_loss + cfg.path_loss_weight * path_loss

            if "cuda" in cfg.device and cfg.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            acc = compute_acc(logits.detach(), action)
            running_loss += float(loss.item())
            running_action_loss += float(action_loss.item())
            running_path_loss += float(path_loss.item())
            running_acc += acc
            num_steps += 1

            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(
                    f"Epoch {epoch:03d}/{cfg.epochs} | "
                    f"batch {batch_idx:04d}/{len(train_loader)} | "
                    f"loss={running_loss / num_steps:.4f} | "
                    f"action_loss={running_action_loss / num_steps:.4f} | "
                    f"path_loss={running_path_loss / num_steps:.4f} | "
                    f"acc={running_acc / num_steps:.4f}",
                    flush=True,
                )

        train_metrics = {
            "loss": running_loss / max(num_steps, 1),
            "action_loss": running_action_loss / max(num_steps, 1),
            "path_loss": running_path_loss / max(num_steps, 1),
            "acc": running_acc / max(num_steps, 1),
        }

        val_metrics = evaluate(model, val_loader, cfg)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_action_loss={train_metrics['action_loss']:.4f} "
            f"train_path_loss={train_metrics['path_loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} || "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_action_loss={val_metrics['action_loss']:.4f} "
            f"val_path_loss={val_metrics['path_loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_rollout={val_metrics['rollout']:.4f}",
            flush=True,
        )

        save_checkpoint(
            path=save_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            val_metrics=val_metrics,
        )

        improved = is_improved(
            val_rollout=val_metrics["rollout"],
            val_acc=val_metrics["acc"],
            val_loss=val_metrics["loss"],
            best_rollout=best_rollout,
            best_acc=best_acc,
            best_val_loss=best_val_loss,
        )

        if improved:
            best_rollout = val_metrics["rollout"]
            best_acc = val_metrics["acc"]
            best_val_loss = val_metrics["loss"]
            epochs_since_improve = 0

            save_checkpoint(
                path=save_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                val_metrics=val_metrics,
            )
            print(f"Saved best checkpoint -> {save_dir / 'best.pt'}", flush=True)
        else:
            epochs_since_improve += 1
            print(
                f"No improvement for {epochs_since_improve} epoch(s). "
                f"Best val_rollout={best_rollout:.4f}, best val_acc={best_acc:.4f}, best val_loss={best_val_loss:.4f}",
                flush=True,
            )

        if epoch >= cfg.min_epochs_before_stop and epochs_since_improve >= cfg.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(patience={cfg.early_stop_patience}, "
                f"best_val_rollout={best_rollout:.4f}, best_val_acc={best_acc:.4f}, best_val_loss={best_val_loss:.4f})",
                flush=True,
            )
            break

    # Final test evaluation on best checkpoint
    best_ckpt_path = save_dir / "best.pt"
    if best_ckpt_path.exists():
        print("=" * 100, flush=True)
        print(f"Loading best checkpoint for final evaluation: {best_ckpt_path}", flush=True)
        ckpt = torch.load(best_ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        val_metrics_best = evaluate(model, val_loader, cfg)
        print(
            f"[BEST-VAL] "
            f"val_loss={val_metrics_best['loss']:.4f} "
            f"val_action_loss={val_metrics_best['action_loss']:.4f} "
            f"val_path_loss={val_metrics_best['path_loss']:.4f} "
            f"val_acc={val_metrics_best['acc']:.4f} "
            f"val_rollout={val_metrics_best['rollout']:.4f}",
            flush=True,
        )

        if test_loader is not None:
            test_metrics = evaluate(model, test_loader, cfg)
            print(
                f"[BEST-TEST] "
                f"test_loss={test_metrics['loss']:.4f} "
                f"test_action_loss={test_metrics['action_loss']:.4f} "
                f"test_path_loss={test_metrics['path_loss']:.4f} "
                f"test_acc={test_metrics['acc']:.4f} "
                f"test_rollout={test_metrics['rollout']:.4f}",
                flush=True,
            )
        print("=" * 100, flush=True)
    else:
        print("WARNING: best.pt not found; skipped final best-checkpoint evaluation.", flush=True)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_policy")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--dropout_p", type=float, default=0.10)

    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--path_loss_weight", type=float, default=0.20)

    parser.add_argument("--sampler_power", type=float, default=0.50)
    parser.add_argument("--max_sample_weight", type=float, default=3.0)
    parser.add_argument("--disable_sampler", action="store_true")

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument(
        "--enable_augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--aug_rot90",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--aug_hflip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--aug_vflip",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--aug_prob_rot90", type=float, default=0.75)
    parser.add_argument("--aug_prob_hflip", type=float, default=0.50)
    parser.add_argument("--aug_prob_vflip", type=float, default=0.20)

    parser.add_argument("--eval_rollout_max_factor", type=int, default=6)
    parser.add_argument("--loop_fail_visit_count", type=int, default=2)
    parser.add_argument("--eval_rollout_samples_per_batch", type=int, default=4)

    parser.add_argument(
        "--enable_action_cost_rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--action_cost_penalty_gain", type=float, default=0.75)

    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--min_epochs_before_stop", type=int, default=10)

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)