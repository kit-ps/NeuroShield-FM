#!/usr/bin/env python3
"""
eeg_auth_train_optuna.py

End-to-end single-file script for:
  1) Packed EEG layouts -> canonical reorder + channel mask + canonical 3D positions
  2) Optuna tuning (SQLite resume-safe) in a single unified study

Architecture:
  - architecture: patch tokenization + ALiBi temporal encoder (fixed)
  - embedder: {"linear", "mlp_patch", "cnn_simple", "cnn_multiscale"}
  - channel positional encoding: coord-MLP (fixed)

Tuning (single stage):
  - coord_scale, embedder, embed_dim, num_heads, channel_depth, temporal_depth,
    patch_len, lr, weight_decay, clip_value,
    chan_norm, emb_norm, mask_ratio
  - per-batch temporal-length augmentation is enabled by default (edit TRAIN_BATCH_LENGTHS in code)
- Defaults when not tuned (fixed params fallback):
    emb_norm="l2", mask_ratio=0.10

Assumes you have:
  utils_evaluation.py with:
    compute_similarity_scores, get_enrollment_verification_indices
"""

import os
import argparse
import math
import sys
import gc
from functools import partial
from typing import Optional, Dict, Any

VALID_EVAL_CROP_MODES = ("center", "left")

# ---------------------------------------------------------------------
# CLI (parse + set CUDA_VISIBLE_DEVICES before importing torch)
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--cuda_devices", type=str, default="0",
                    help="Value for CUDA_VISIBLE_DEVICES (e.g., '0' or '0,1')")

parser.add_argument("--train_dir", type=str, default="./data/train_packed")
parser.add_argument("--val_dir",   type=str, default="./data/val_packed")
parser.add_argument("--test_dir",  type=str, default="./data/test_packed")

parser.add_argument("--save_dir", type=str, default="./ckpts_bw")

# tuning
parser.add_argument("--study_db", type=str, default="sqlite:///bw_optuna.db",
                    help="Optuna RDBStorage URL (SQLite).")
parser.add_argument("--study_name", type=str, default="BrainWave",
                    help="Optuna base study name (unique names per experiment are recommended).")
parser.add_argument("--n_trials", type=int, default=40)
parser.add_argument("--tuning_epochs", type=int, default=8)

# data/loader
parser.add_argument("--crop_len", type=int, default=500)
parser.add_argument(
    "--eval_crop_mode",
    type=str,
    default="center",
    choices=VALID_EVAL_CROP_MODES,
    help="Deterministic crop policy for val/test when random_crop=False.",
)

# reproducibility
parser.add_argument("--seed", type=int, default=42)

# In notebooks, ipykernel injects extra argv entries (e.g. "-f <kernel.json>").
# Use parse_known_args there so importing/running this file inside Jupyter works.
if "ipykernel" in sys.modules:
    args, _unknown = parser.parse_known_args()
else:
    args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.5",
)

# ---------------------------------------------------------------------
# Static config constants
# ---------------------------------------------------------------------
DEFAULT_MAX_3D_DIST = 0.015
LOSS_LR_SCALE = 0.5
CLIP_GRAD_MAX_NORM = 1.0

TUNING_LENGTH_MULTIPLIER = 1000

EPISODE_NUM_SUBJECTS = 30
EPISODE_MAX_SESSIONS = 5
EPISODE_PREFER_MULTISESSION = True
EPISODE_NO_DUPLICATES = True
EPISODE_MAX_TRIES = 50

MAX_NATIVE_CHANNELS = 64

# Essential defaults kept in-code to keep CLI small/clean.
CHOSEN_LOSS = "SupConLoss"
NUM_WORKERS_TRAIN = 15
NUM_WORKERS_EVAL = 2
CACHE_EVAL_BATCH_SIZE = 64
TOP_K = 40
TRAIN_BATCH_SIZE = 128

DEFAULT_MASK_RATIO = 0.10
DEFAULT_EMB_NORM = "l2"

USE_TORCH_COMPILE = False
LOG_CUDA_MEM = False
FAIL_ON_NAN = False

# Main training behavior: always sample temporal target length per batch.
TRAIN_BATCH_LENGTHS = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
TRAIN_LENGTH_CROP_MODE = "random"  # {"random", "center"}

# Optional second EER variant:
# if enabled, cap each subject EER at random level (0.5) before aggregation.
USE_CAPPED_SUBJECT_EER = False
SUBJECT_EER_CAP = 0.5

if not (0.0 <= DEFAULT_MASK_RATIO <= 1.0):
    raise ValueError(f"DEFAULT_MASK_RATIO must be in [0, 1], got {DEFAULT_MASK_RATIO}")
if TRAIN_LENGTH_CROP_MODE not in ("random", "center"):
    raise ValueError(f"TRAIN_LENGTH_CROP_MODE must be 'random' or 'center', got {TRAIN_LENGTH_CROP_MODE}")
if any(int(x) <= 0 for x in TRAIN_BATCH_LENGTHS):
    raise ValueError(f"TRAIN_BATCH_LENGTHS must contain positive integers, got {TRAIN_BATCH_LENGTHS}")


def normalize_eval_crop_mode(mode: Any) -> str:
    value = "center" if mode is None else str(mode).strip().lower()
    if value == "":
        value = "center"
    if value not in VALID_EVAL_CROP_MODES:
        raise ValueError(
            f"eval_crop_mode must be one of {VALID_EVAL_CROP_MODES}, got {mode!r}"
        )
    return value

# ---------------------------------------------------------------------
# Imports (after CUDA env set)
# ---------------------------------------------------------------------
import glob
import h5py
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import optim, amp
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import mne
from pytorch_metric_learning import losses, miners

# Local evaluation helpers
from utils_evaluation import (
    compute_similarity_scores,
    get_enrollment_verification_indices,
)

import optuna
from optuna.storages import RDBStorage

# Favor reproducibility over cudnn autotuning.
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ---------------------------------------------------------------------
# Search-space tracking (auto-collected from suggest_categorical calls)
# ---------------------------------------------------------------------
def suggest_cat(trial, name: str, choices, registry: Optional[Dict[str, tuple]] = None):
    frozen = tuple(choices)
    if registry is not None:
        prev = registry.get(name)
        if prev is None:
            registry[name] = frozen
        elif prev != frozen:
            raise ValueError(f"Inconsistent choices for '{name}': {prev} vs {frozen}")
    return trial.suggest_categorical(name, list(frozen))


def unique_combo_count(registry: Dict[str, tuple]) -> int:
    if not registry:
        return 0
    return math.prod(len(v) for v in registry.values())


def is_disallowed_embed_batch_combo(embed_dim: int, batch_size: int) -> bool:
    # Memory/speed guardrail.
    return int(embed_dim) == 256 and int(batch_size) == 512


def _is_unformable_batch_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        "No layout can form a duplicate-free same-layout batch" in msg
        or "Could not form a full batch after max_tries" in msg
    )


def _is_cuda_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda out of memory" in msg)


# ---------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int):
    # Keep numpy/python RNG deterministic per worker given the DataLoader seed.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


seed_everything(args.seed)


def format_cuda_mem(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cuda_mem=NA(cpu)"
    dev = device.index if device.index is not None else torch.cuda.current_device()
    alloc = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
    max_alloc = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(dev) / (1024 ** 3)
    free_b, total_b = torch.cuda.mem_get_info(dev)
    free = free_b / (1024 ** 3)
    total = total_b / (1024 ** 3)
    return (
        f"alloc={alloc:.2f}G reserved={reserved:.2f}G "
        f"max_alloc={max_alloc:.2f}G max_reserved={max_reserved:.2f}G "
        f"free={free:.2f}G total={total:.2f}G"
    )


def _loader_perf_kwargs(num_workers: int, pin_memory: bool = True) -> Dict[str, Any]:
    """
    Shared DataLoader performance options.
    """
    kw: Dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kw["prefetch_factor"] = 4
        kw["worker_init_fn"] = _seed_worker
    return kw


def _shutdown_dataloader(loader: Optional[DataLoader]) -> None:
    """
    Force worker teardown instead of waiting for Python GC.

    With persistent workers, relying on object destruction across many Optuna
    trials can leak file descriptors long enough to hit the OS open-file limit.
    """
    if loader is None:
        return
    try:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator = None
    except Exception:
        pass


def _sample_train_batch_length() -> Optional[int]:
    lengths = [int(x) for x in TRAIN_BATCH_LENGTHS if int(x) > 0]
    if len(lengths) == 0:
        return None
    return int(np.random.choice(lengths))


def _fit_batch_to_target_length(
    eeg: torch.Tensor,
    t_lengths: torch.Tensor,
    target_len: int,
    crop_mode: str,
) -> torch.Tensor:
    """
    eeg: [B, C, T] -> [B, C, target_len]
    """
    bsz, csz, t_cur = eeg.shape
    tgt = int(target_len)
    if tgt <= 0:
        raise ValueError(f"target_len must be > 0, got {tgt}")
    out = eeg.new_zeros((bsz, csz, tgt))
    for bi in range(bsz):
        t_i = int(t_lengths[bi].item()) if t_lengths is not None else int(t_cur)
        t_i = max(0, min(t_i, int(t_cur)))
        x_i = eeg[bi, :, :t_i]
        if t_i >= tgt:
            if crop_mode == "center":
                start = int((t_i - tgt) // 2)
            else:
                start = int(np.random.randint(0, t_i - tgt + 1)) if t_i > tgt else 0
            out[bi] = x_i[:, start:start + tgt]
        else:
            out[bi, :, :t_i] = x_i
    return out

from pyeer.eer_info import get_eer_stats
def compute_eer(result_array):
    result_array = np.array(result_array)
    genuine = result_array[result_array[:, 1] == 1][:, 0]
    impostor = result_array[result_array[:, 1] == 0][:, 0]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"It is possible that you had set the wrong score type\..*",
            category=UserWarning,
        )
        stats = get_eer_stats(genuine, impostor)
    return stats.eer, stats.fmr100


def evaluate_eer_per_class_capped(similarity_results, cap: float = 0.5):
    cap = float(cap)
    if not (0.0 <= cap <= 1.0):
        raise ValueError(f"cap must be in [0, 1], got {cap}")

    eer_list = []
    for subject_id, results in similarity_results.items():
        eer, _ = compute_eer(np.array(results))
        eer = float(eer)
        if np.isfinite(eer) and eer > cap:
            print(f"[WARN] subject={subject_id} raw_eer={eer:.4f} capped_to={cap:.4f}")
            eer = cap
        eer_list.append(eer)

    avg_eer = np.mean(eer_list) * 100
    std_eer = np.std(eer_list) * 100
    return avg_eer, std_eer

def evaluate_eer_per_class(similarity_results):
    if USE_CAPPED_SUBJECT_EER:
        return evaluate_eer_per_class_capped(similarity_results, cap=SUBJECT_EER_CAP)

    eer_list = []
    for results in similarity_results.values():
        eer, _ = compute_eer(np.array(results))
        eer_list.append(eer)

    avg_eer = np.mean(eer_list) * 100
    std_eer = np.std(eer_list) * 100
    return avg_eer, std_eer


# ---------------------------------------------------------------------
# Normalization + canonical list
# ---------------------------------------------------------------------
def normalize_label(raw):
    if raw is None:
        return None
    lab = raw.strip().upper()
    lab = lab.replace("EEG", "")
    lab = lab.replace("-REF", "")
    lab = lab.replace("-LE", "")
    lab = lab.replace("-RE", "")
    lab = lab.replace("REF", "")
    lab = lab.replace(" ", "")
    lab = lab.replace("[", "").replace("]", "")

    # TUH-like fixes
    if lab == "T1":
        return "FT9"
    if lab == "T2":
        return "FT10"
    return lab


# canonical channels (as you used)
eeg_clist = [
    'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FT10',
    'FT9', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'
]
REF_CLIST_NORM = [normalize_label(ch) for ch in eeg_clist]

def build_ref_pos_dict_from_mne(montage_names=("standard_1005",)):
    pos_dict = {}

    for mname in montage_names:
        montage = mne.channels.make_standard_montage(mname)

        ch_pos = montage.get_positions().get("ch_pos") or {}
        print(f"[montage OK] {mname}: ch_pos={len(ch_pos)}")

        kept = 0
        for raw_name, xyz in ch_pos.items():
            nname = normalize_label(raw_name)
            if not nname:
                continue
            if nname not in pos_dict:
                pos_dict[nname] = np.asarray(xyz, dtype=np.float32)
                kept += 1

        print(f"[normalize] kept={kept}, total_dict={len(pos_dict)}")

    return pos_dict


REF_POS_DICT_2 = build_ref_pos_dict_from_mne(("standard_1005",))
REF_POS_biosemi256 = build_ref_pos_dict_from_mne(("biosemi256",))
_MISSING_POS_WARNED = set()

EXTRA_REF_POS = {
    "O9": np.array([-0.029818400740623474, -0.1145699992775917, -0.02921600081026554], dtype=np.float32),
    "O10": np.array([0.029741600155830383, -0.1142600029706955, -0.029255999252200127], dtype=np.float32),
}

for ch, xyz in EXTRA_REF_POS.items():
    REF_POS_DICT_2.setdefault(ch, xyz)


def parse_layout_str(layout_str: str):
    return [normalize_label(ch) for ch in layout_str.split("_")]


def _choose_ref_pos_dict(names_norm):
    valid = [n for n in names_norm if n]
    if len(valid) == 0:
        return REF_POS_DICT_2

    uniq = set(valid)
    hit_std = len(uniq.intersection(REF_POS_DICT_2.keys()))
    hit_bio = len(uniq.intersection(REF_POS_biosemi256.keys()))
    cov_std = hit_std / max(len(uniq), 1)
    cov_bio = hit_bio / max(len(uniq), 1)

    # Tie-breaker: canonical BioSemi A* anchors.
    if cov_bio == cov_std and ("A1" in uniq or "A2" in uniq):
        return REF_POS_biosemi256

    return REF_POS_biosemi256 if cov_bio > cov_std else REF_POS_DICT_2


def names_norm_to_ref_pos3d(names_norm, ref_dict=None):
    if ref_dict is None:
        ref_dict = _choose_ref_pos_dict(names_norm)

    pos = np.zeros((len(names_norm), 3), dtype=np.float32)
    missing_now = []
    for i, n in enumerate(names_norm):
        if n in ref_dict:
            pos[i] = np.asarray(ref_dict[n], dtype=np.float32)
        elif n and n not in _MISSING_POS_WARNED:
            _MISSING_POS_WARNED.add(n)
            missing_now.append(n)
    if missing_now:
        print(
            "[WARN] Missing 3D position for channels "
            f"(using [0,0,0]): {sorted(missing_now)}"
        )
    return pos


def build_channel_index_map(
    sample_names,
    ref_names,
    sample_pos_arr=None,
    ref_pos=None,
    max_3d_dist=0.05,
):
    """
    Map canonical ref_names -> sample indices.
      1) exact name match (first unused sample)
      2) optional 3D greedy fallback for still-unmatched refs
    """
    name_to_indices = defaultdict(list)
    for si, sname in enumerate(sample_names):
        name_to_indices[sname].append(si)

    out = [-1] * len(ref_names)
    used_sample = set()

    # 1) exact name match
    for ri, rname in enumerate(ref_names):
        for si in name_to_indices.get(rname, []):
            if si not in used_sample:
                out[ri] = si
                used_sample.add(si)
                break

    # 2) optional 3D fallback
    if sample_pos_arr is None or ref_pos is None:
        return out

    candidates = []
    for ri, rname in enumerate(ref_names):
        if out[ri] != -1 or rname not in ref_pos:
            continue
        rxyz = np.asarray(ref_pos[rname], dtype=np.float32)
        for si in range(len(sample_names)):
            if si in used_sample or si >= len(sample_pos_arr):
                continue
            sxyz = np.asarray(sample_pos_arr[si], dtype=np.float32)
            d = float(np.linalg.norm(sxyz - rxyz))
            if d <= max_3d_dist:
                candidates.append((d, ri, si))

    candidates.sort(key=lambda t: t[0])

    for d, ri, si in candidates:
        if out[ri] == -1 and si not in used_sample:
            out[ri] = si
            used_sample.add(si)

    return out


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class PackedLayoutDataset2(Dataset):
    """
    Each item:
      - eeg: (C, T)
      - y: label (int, 0-based)
      - meta including session_id/layout_id
    """
    def __init__(
        self,
        packed_dir,
        crop_len=500,
        random_crop=True,
        only_basename=None,
        deterministic_crop_mode="center",
    ):
        self.packed_dir = os.path.abspath(packed_dir)

        if only_basename is not None:
            single_path = os.path.join(self.packed_dir, only_basename)
            if not os.path.exists(single_path):
                raise RuntimeError(f"Requested file {single_path} not found")
            self.files = [single_path]
        else:
            self.files = glob.glob(os.path.join(self.packed_dir, "packed_*.h5"))

        if len(self.files) == 0:
            raise RuntimeError(f"No packed_*.h5 in {self.packed_dir}")

        self.crop_len = int(crop_len)
        self.random_crop = bool(random_crop)
        self.deterministic_crop_mode = normalize_eval_crop_mode(deterministic_crop_mode)

        self.index = []
        self.subject_ids = []
        self.session_ids = []
        self.file_layouts = []
        self.layout_ids = []

        for fi, fpath in enumerate(self.files):
            with h5py.File(fpath, "r") as hf:
                n_seg = hf["data"].shape[0]
                layout_id = hf.attrs["layout_id"]
                if isinstance(layout_id, bytes):
                    layout_id = layout_id.decode("utf-8")
                chan_list = [normalize_label(ch) for ch in layout_id.split("_")]
                self.file_layouts.append(chan_list)

                subs = hf["subject_id"][...]
                sess = hf["session_id"][...]

            for si in range(n_seg):
                subj = subs[si]
                if isinstance(subj, bytes):
                    subj = subj.decode("utf-8")
                self.index.append((fi, si))
                self.subject_ids.append(subj)
                self.session_ids.append(int(sess[si]))
                self.layout_ids.append(layout_id)

        uniq_subj = sorted(set(self.subject_ids))
        self.subj2lab = {s: i for i, s in enumerate(uniq_subj)}  # 0-based for class-indexed losses
        self.targets = torch.tensor([self.subj2lab[s] for s in self.subject_ids], dtype=torch.long)

        print(f"[INFO] dataset {self.packed_dir}")
        print(f"       files: {len(self.files)}")
        print(f"       segments: {len(self.index)}")
        print(f"       unique subjects: {len(uniq_subj)}")

    def __len__(self):
        return len(self.index)

    def _crop(self, x):
        C, T = x.shape
        L = self.crop_len
        if T == L:
            return x
        if T > L:
            if self.random_crop:
                start = random.randint(0, T - L)
            else:
                if self.deterministic_crop_mode == "left":
                    start = 0
                else:
                    start = (T - L) // 2
            return x[:, start:start + L]
        out = np.zeros((C, L), dtype=x.dtype)
        out[:, :T] = x
        return out

    def __getitem__(self, idx):
        fi, si = self.index[idx]
        with h5py.File(self.files[fi], "r") as hf:
            x = hf["data"][si]
            subj_b = hf["subject_id"][si]
            sess_i = int(hf["session_id"][si])
            file_b = hf["file_id"][si]
            layout_b = hf.attrs["layout_id"]
            sfreq = float(hf.attrs["sfreq"])

        subj = subj_b.decode("utf-8") if isinstance(subj_b, bytes) else subj_b
        y = self.subj2lab[subj]

        x = self._crop(x)

        layout_id = layout_b.decode("utf-8") if isinstance(layout_b, bytes) else layout_b
        file_id = file_b.decode("utf-8") if isinstance(file_b, bytes) else file_b

        return {
            "eeg": x,
            "y": y,
            "subject_id": subj,
            "session_id": sess_i,
            "file_id": file_id,
            "layout_id": layout_id,
            "sfreq": sfreq,
        }


class PackedLayoutDatasetNoCrop(PackedLayoutDataset2):
    """
    Train-time dataset variant: keep native temporal length (no crop/pad here).
    """
    def _crop(self, x):
        return x


# ---------------------------------------------------------------------
# Collate (canonical reorder + mask + canonical 3D pos)
# Returns: eeg, mask, labels, pos3d, meta
# ---------------------------------------------------------------------
def build_layout_cache(dataset: PackedLayoutDataset2) -> Dict[str, Dict[str, Any]]:
    """
    Cache normalized channel names and canonical 3D positions per layout_id.
    """
    cache: Dict[str, Dict[str, Any]] = {}
    for layout_id in sorted(set(dataset.layout_ids)):
        names_norm = tuple(parse_layout_str(layout_id))
        pos3d = names_norm_to_ref_pos3d(names_norm).astype(np.float32, copy=False)
        cache[str(layout_id)] = {"names_norm": names_norm, "pos3d": pos3d}
    return cache


def _get_layout_cached(layout_id: str, n_chan: int, layout_cache: Optional[Dict[str, Dict[str, Any]]]):
    lid = str(layout_id)
    if layout_cache is not None:
        entry = layout_cache.get(lid)
        if entry is not None:
            return entry["names_norm"][:n_chan], entry["pos3d"][:n_chan]
    names_norm = tuple(parse_layout_str(lid)[:n_chan])
    pos3d = names_norm_to_ref_pos3d(names_norm).astype(np.float32, copy=False)
    return names_norm, pos3d


def _collate_native_item(
    item,
    Cmax,
    L,
    deterministic_when_capped=False,
    shared_keep_idx=None,
    layout_cache: Optional[Dict[str, Dict[str, Any]]] = None,
):
    x = item["eeg"]
    C = int(x.shape[0])
    T = int(x.shape[1])
    _sample_norm, sample_pos = _get_layout_cached(item["layout_id"], C, layout_cache)

    eeg_pad = np.zeros((Cmax, L), dtype=np.float32)
    mask = np.zeros((Cmax,), dtype=bool)
    pos_pad = np.zeros((Cmax, 3), dtype=np.float32)

    if shared_keep_idx is not None:
        keep_idx = np.asarray(shared_keep_idx, dtype=np.int64)
    elif C > Cmax:
        if deterministic_when_capped:
            # Deterministic cap: keep earliest channels in native order.
            keep_idx = np.arange(Cmax, dtype=np.int64)
        else:
            keep_idx = np.random.choice(C, size=Cmax, replace=False)
            keep_idx.sort()  # keep original channel order among sampled channels
    else:
        keep_idx = np.arange(C)

    C_sel = len(keep_idx)
    if C_sel > 0:
        c_use = min(C_sel, Cmax)
        sel = keep_idx[:c_use]
        eeg_pad[:c_use, :T] = x[sel, :]
        pos_pad[:c_use, :] = sample_pos[sel]
        mask[:c_use] = True

    return eeg_pad, mask, pos_pad


def _collate_keep_item(
    item,
    keep_norm,
    keep_pos_template,
    ref_pos,
    Cmax,
    L,
    max_3d_dist,
    layout_cache: Optional[Dict[str, Dict[str, Any]]] = None,
):
    x = item["eeg"]
    C = int(x.shape[0])
    T = int(x.shape[1])
    sample_norm, _sample_pos = _get_layout_cached(item["layout_id"], C, layout_cache)
    sample_template_pos = names_norm_to_ref_pos3d(sample_norm, ref_dict=ref_pos)

    eeg_pad = np.zeros((Cmax, L), dtype=np.float32)
    mask = np.zeros((Cmax,), dtype=bool)
    pos_pad = np.zeros((Cmax, 3), dtype=np.float32)

    idx_map = build_channel_index_map(
        sample_names=sample_norm,
        ref_names=keep_norm,
        sample_pos_arr=sample_template_pos,
        ref_pos=ref_pos,
        max_3d_dist=max_3d_dist,
    )

    c_use = min(len(idx_map), Cmax)
    for ci in range(c_use):
        si = idx_map[ci]
        if si < 0 or si >= x.shape[0]:
            continue
        eeg_pad[ci, :T] = x[si, :]
        mask[ci] = True
        pos_pad[ci, :] = keep_pos_template[ci]

    return eeg_pad, mask, pos_pad


def _collate_native_batch_homogeneous(
    batch,
    Cmax: int,
    L: int,
    deterministic_when_capped: bool,
    shared_keep_idx,
    layout_cache: Optional[Dict[str, Dict[str, Any]]],
):
    """
    Fast path: same layout_id and same channel count for all samples in batch.
    """
    B = len(batch)
    C = int(batch[0]["eeg"].shape[0])
    eeg_src = np.stack([item["eeg"] for item in batch], axis=0).astype(np.float32, copy=False)  # [B,C,L]

    if shared_keep_idx is not None:
        keep_idx = np.asarray(shared_keep_idx, dtype=np.int64)
    elif C > Cmax:
        if deterministic_when_capped:
            keep_idx = np.arange(Cmax, dtype=np.int64)
        else:
            keep_idx = np.random.choice(C, size=Cmax, replace=False)
            keep_idx.sort()
    else:
        keep_idx = np.arange(C, dtype=np.int64)

    c_use = min(len(keep_idx), Cmax)
    sel = keep_idx[:c_use]

    eeg_batch = np.zeros((B, Cmax, L), dtype=np.float32)
    if c_use > 0:
        eeg_batch[:, :c_use, :] = eeg_src[:, sel, :]

    mask_batch = np.zeros((B, Cmax), dtype=bool)
    if c_use > 0:
        mask_batch[:, :c_use] = True

    _sample_norm, sample_pos = _get_layout_cached(batch[0]["layout_id"], C, layout_cache)
    pos_template = np.zeros((Cmax, 3), dtype=np.float32)
    if c_use > 0:
        pos_template[:c_use, :] = sample_pos[sel]
    pos3d_batch = np.broadcast_to(pos_template, (B, Cmax, 3)).copy()

    labels = np.asarray([int(item["y"]) for item in batch], dtype=np.int64)
    meta = {
        "subject_id": [item["subject_id"] for item in batch],
        "session_id": [item["session_id"] for item in batch],
        "layout_id": [item["layout_id"] for item in batch],
        "file_id": [item["file_id"] for item in batch],
        "sfreq": [item["sfreq"] for item in batch],
        "t_len": [int(item["eeg"].shape[1]) for item in batch],
    }
    return (
        torch.from_numpy(eeg_batch).float(),
        torch.from_numpy(mask_batch).bool(),
        torch.from_numpy(labels).long(),
        torch.from_numpy(pos3d_batch).float(),
        meta,
    )


def collate_mixed2(
    batch,
    keep_channels=None,
    fixed_Cmax=None,
    max_3d_dist=DEFAULT_MAX_3D_DIST,
    layout_cache: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    If keep_channels is not None:
      - output channels follow keep_channels canonical order
      - exact normalized-name match first
      - unmatched canonical slots are optionally filled by nearest 3D channel match
        (one-to-one, with distance threshold max_3d_dist)
      - pad ALWAYS to Cmax = len(keep_channels)  (constant across all batches)
    Otherwise:
      - keep native channel order
      - pad to per-batch max channels (variable across batches)
    """
    L = max(int(item["eeg"].shape[1]) for item in batch) if batch else 0
    ref_pos = None

    deterministic_when_capped = False
    if keep_channels is not None:
        keep_norm = [normalize_label(c) for c in keep_channels]
        keep_norm = [c for c in keep_norm if c]
        keep_set = set(keep_norm)
        ref_pos = _choose_ref_pos_dict(keep_norm)
        keep_pos_template = names_norm_to_ref_pos3d(keep_norm, ref_dict=ref_pos)
        Cmax = len(keep_norm)
    else:
        keep_norm, keep_set, keep_pos_template = None, None, None
        kept_counts = [item["eeg"].shape[0] for item in batch]
        Cmax = min(max(kept_counts), MAX_NATIVE_CHANNELS) if kept_counts else 0

    if fixed_Cmax is not None:
        Cmax = int(fixed_Cmax)
        # Only requested behavior: deterministic cap when fixed_Cmax is explicitly set.
        if keep_set is None:
            deterministic_when_capped = True

    shared_keep_idx = None
    if keep_set is None and (not deterministic_when_capped):
        # If this native batch is homogeneous (same layout + same channel count),
        # share one random capped subset across all samples in the batch.
        layout_ids = [item["layout_id"] for item in batch]
        ch_counts = [int(item["eeg"].shape[0]) for item in batch]
        if (
            len(set(layout_ids)) == 1
            and len(set(ch_counts)) == 1
            and len(ch_counts) > 0
            and ch_counts[0] > Cmax
            and Cmax > 0
        ):
            C0 = ch_counts[0]
            shared_keep_idx = np.random.choice(C0, size=Cmax, replace=False)
            shared_keep_idx.sort()  # keep original channel order among selected channels

    # Fast path: homogeneous native batch (common for same-layout episode sampler).
    if keep_set is None and len(batch) > 0:
        lid0 = batch[0]["layout_id"]
        c0 = int(batch[0]["eeg"].shape[0])
        t0 = int(batch[0]["eeg"].shape[1])
        homogeneous = all(
            (
                item["layout_id"] == lid0
                and int(item["eeg"].shape[0]) == c0
                and int(item["eeg"].shape[1]) == t0
            )
            for item in batch
        )
        if homogeneous:
            return _collate_native_batch_homogeneous(
                batch=batch,
                Cmax=Cmax,
                L=L,
                deterministic_when_capped=deterministic_when_capped,
                shared_keep_idx=shared_keep_idx,
                layout_cache=layout_cache,
            )

    eegs, masks, labels, pos3ds = [], [], [], []
    meta = {k: [] for k in ["subject_id", "session_id", "layout_id", "file_id", "sfreq", "t_len"]}

    for item in batch:
        if keep_set is None:
            eeg_pad, mask, pos_pad = _collate_native_item(
                item,
                Cmax=Cmax,
                L=L,
                deterministic_when_capped=deterministic_when_capped,
                shared_keep_idx=shared_keep_idx,
                layout_cache=layout_cache,
            )
        else:
            eeg_pad, mask, pos_pad = _collate_keep_item(
                item,
                keep_norm=keep_norm,
                keep_pos_template=keep_pos_template,
                ref_pos=ref_pos,
                Cmax=Cmax,
                L=L,
                max_3d_dist=max_3d_dist,
                layout_cache=layout_cache,
            )

        eegs.append(torch.from_numpy(eeg_pad))
        masks.append(torch.from_numpy(mask))
        pos3ds.append(torch.from_numpy(pos_pad))
        labels.append(item["y"])

        for k in meta:
            if k == "t_len":
                meta[k].append(int(item["eeg"].shape[1]))
            else:
                meta[k].append(item[k])

    eeg_batch = torch.stack(eegs, 0).float()
    mask_batch = torch.stack(masks, 0).bool()
    pos3d_batch = torch.stack(pos3ds, 0).float()
    labels = torch.tensor(labels, dtype=torch.long)

    return eeg_batch, mask_batch, labels, pos3d_batch, meta


# ---------------------------------------------------------------------
# Episode batch sampler
# ---------------------------------------------------------------------
def _safe_choice(lst, size=1):
    arr = list(lst)
    replace = len(arr) < size
    return np.random.choice(arr, size=size, replace=replace)

class LayoutAwareEpisodeMultiSessionBatchSampler(Sampler):
    """
    Layout-aware episode sampler.

    Per batch:
      1) choose one layout_id (weighted by layout sample count)
      2) episode part:
         - pick K subjects within that layout
         - for each subject: pick up to S distinct sessions, sample 1 index per session
         - pack round-robin across subjects (max K*S episode samples)
      3) fill remaining slots from that same layout only

    no_duplicates=True: no repeated sample indices within a batch.
    """
    def __init__(
        self,
        labels,
        sessions,
        layouts,
        batch_size: int,
        num_subjects_per_batch: int = 30,
        max_sessions_per_subject: int = 5,
        length_before_new_iter: int = 100000,
        prefer_multisession_subjects: bool = True,
        no_duplicates: bool = True,
        max_tries: int = 50,
    ):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        self.labels = np.asarray(labels)
        self.sessions = np.asarray(sessions)
        self.layouts = np.asarray(layouts)

        self.N = int(len(self.labels))
        self.batch_size = int(batch_size)
        self.K = int(num_subjects_per_batch)
        self.S = int(max_sessions_per_subject)

        self.prefer_multisession_subjects = bool(prefer_multisession_subjects)
        self.no_duplicates = bool(no_duplicates)
        self.max_tries = int(max_tries)

        if self.no_duplicates and self.batch_size > self.N:
            raise RuntimeError(
                f"no_duplicates=True but batch_size={self.batch_size} > dataset size N={self.N}."
            )

        length_before_new_iter = int(length_before_new_iter)
        self.length = length_before_new_iter - (length_before_new_iter % self.batch_size)

        # --- layout -> indices ---
        self.layout_to_indices = defaultdict(list)
        for i, lay in enumerate(self.layouts):
            self.layout_to_indices[lay].append(i)

        self.unique_layouts = list(self.layout_to_indices.keys())
        if len(self.unique_layouts) == 0:
            raise RuntimeError("No layouts found.")

        # --- build per-layout maps: subject->session->indices (restricted to that layout) ---
        self.maps = {}
        layout_counts = []
        for lay in self.unique_layouts:
            idxs = self.layout_to_indices[lay]
            layout_counts.append(len(idxs))

            lab_sess_to_idx = defaultdict(lambda: defaultdict(list))
            lab_to_sessions = defaultdict(set)

            for i in idxs:
                y = self.labels[i]
                s = self.sessions[i]
                lab_sess_to_idx[y][s].append(i)
                lab_to_sessions[y].add(s)

            unique_labels = list(lab_sess_to_idx.keys())

            if self.prefer_multisession_subjects:
                multi_labels = [y for y in unique_labels if len(lab_to_sessions[y]) >= 2]
                single_labels = [y for y in unique_labels if len(lab_to_sessions[y]) < 2]
            else:
                multi_labels = unique_labels
                single_labels = []

            self.maps[lay] = {
                "idxs": np.asarray(idxs, dtype=np.int64),
                "lab_sess_to_idx": lab_sess_to_idx,
                "lab_to_sessions": lab_to_sessions,
                "unique_labels": unique_labels,
                "multi_labels": multi_labels,
                "single_labels": single_labels,
            }

        # layout sampling probs ~ number of samples in layout
        layout_counts = np.asarray(layout_counts, dtype=np.float64)
        self.layout_probs = layout_counts / layout_counts.sum()

        # Same-layout duplicate-free batches require at least batch_size samples
        # in the chosen layout. Exclude smaller layouts up front so training uses
        # only layouts that can actually form a full batch.
        if self.no_duplicates:
            self.eligible_layouts = [
                lay for lay in self.unique_layouts
                if len(self.layout_to_indices[lay]) >= self.batch_size
            ]
            self.ineligible_layouts = [
                lay for lay in self.unique_layouts
                if len(self.layout_to_indices[lay]) < self.batch_size
            ]
            if len(self.eligible_layouts) == 0:
                raise RuntimeError(
                    "No layout can form a duplicate-free same-layout batch: "
                    f"batch_size={self.batch_size}, max_layout_size={max(len(v) for v in self.layout_to_indices.values())}."
                )
            if len(self.ineligible_layouts) > 0:
                kept_samples = int(sum(len(self.layout_to_indices[l]) for l in self.eligible_layouts))
                dropped_samples = int(sum(len(self.layout_to_indices[l]) for l in self.ineligible_layouts))
                print(
                    "[INFO] Sampler excluded ineligible layouts: "
                    f"kept_layouts={len(self.eligible_layouts)} "
                    f"dropped_layouts={len(self.ineligible_layouts)} "
                    f"kept_samples={kept_samples} dropped_samples={dropped_samples} "
                    f"(batch_size={self.batch_size})."
                )
            counts = np.array([len(self.layout_to_indices[l]) for l in self.eligible_layouts], dtype=np.float64)
            self.eligible_probs = counts / counts.sum()
        else:
            self.eligible_layouts = self.unique_layouts
            self.eligible_probs = self.layout_probs
        self._warned_unformable_batch = False

    def __len__(self):
        return self.length // self.batch_size

    def _sample_episode_from_layout(self, lay, used_idx, B):
        m = self.maps[lay]
        lab_sess_to_idx = m["lab_sess_to_idx"]
        unique_labels = m["unique_labels"]
        multi_labels = m["multi_labels"]

        # pick subjects without repetition (within layout)
        K_eff = min(self.K, B, len(unique_labels))
        if K_eff == 0:
            return []

        subjects = []
        used_subj = set()

        if self.prefer_multisession_subjects and len(multi_labels) > 0:
            pool = multi_labels.copy()
            np.random.shuffle(pool)
            for y in pool:
                if len(subjects) >= K_eff:
                    break
                subjects.append(y)
                used_subj.add(y)

        if len(subjects) < K_eff:
            pool = [y for y in unique_labels if y not in used_subj]
            np.random.shuffle(pool)
            for y in pool:
                if len(subjects) >= K_eff:
                    break
                subjects.append(y)
                used_subj.add(y)

        # build per-subject up-to-S session samples
        per_subj_items = []
        for y in subjects:
            sess_keys = list(lab_sess_to_idx[y].keys())
            np.random.shuffle(sess_keys)
            sess_keys = sess_keys[: min(self.S, len(sess_keys))]

            items = []
            for s in sess_keys:
                # pick 1 index for that session, but avoid duplicates if needed
                candidates = lab_sess_to_idx[y][s]
                if self.no_duplicates:
                    candidates = [i for i in candidates if i not in used_idx]
                if len(candidates) == 0:
                    continue
                idx = _safe_choice(candidates, size=1)[0]
                items.append(int(idx))

            per_subj_items.append(items)

        # round-robin pack
        episode = []
        for r in range(self.S):
            for items in per_subj_items:
                if len(episode) >= B:
                    break
                if r < len(items):
                    idx = items[r]
                    if (not self.no_duplicates) or (idx not in used_idx):
                        episode.append(idx)
                        used_idx.add(idx)
            if len(episode) >= B:
                break

        return episode

    def __iter__(self):
        for _ in range(len(self)):
            B = self.batch_size

            # Try multiple layouts if needed (mainly useful for same_layout mode with tight constraints)
            for _try in range(self.max_tries):
                lay = np.random.choice(self.eligible_layouts, p=self.eligible_probs)

                batch = []
                used_idx = set()

                # 1) episode from chosen layout
                episode = self._sample_episode_from_layout(lay, used_idx, B)
                batch.extend(episode)

                # 2) fill remaining
                need = B - len(batch)
                if need <= 0:
                    yield batch[:B]
                    break

                pool = self.maps[lay]["idxs"]

                if self.no_duplicates:
                    pool = np.setdiff1d(pool, np.fromiter(used_idx, dtype=np.int64), assume_unique=False)
                    if len(pool) < need:
                        # cannot fill under current constraints/layout choice
                        continue
                    fill = np.random.choice(pool, size=need, replace=False)
                else:
                    fill = np.random.choice(pool, size=need, replace=True)

                batch.extend([int(x) for x in fill])
                yield batch
                break
            else:
                raise RuntimeError(
                    "Could not form a full batch after max_tries even though eligible layouts "
                    "were prefiltered. This indicates a sampler logic issue."
                )



# ---------------------------------------------------------------------
# Embedding modules
# ---------------------------------------------------------------------
class PatchEmbedLinear(nn.Module):
    """x_patches: [B*C, N, L] -> [B*C, N, D]"""
    def __init__(self, patch_len: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(patch_len, embed_dim)

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        return self.proj(x_patches)


class PatchEmbedMLP(nn.Module):
    """Patch-wise MLP without extra normalization."""
    def __init__(self, patch_len: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(patch_len, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        return self.net(x_patches)


class PatchEmbedCNNSimple(nn.Module):
    """
    Simple CNN over each patch independently without aggressive downsampling:
      input  : [B*C, N, L]
      reshape: [B*C*N, 1, L]
      output : [B*C, N, out_dim]
    """
    def __init__(self, patch_len: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(64, out_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        Bc, N, L = x_patches.shape
        z = x_patches.reshape(Bc * N, 1, L)
        z = self.conv(z)              # [Bc*N, out_dim, L']
        z = self.pool(z).squeeze(-1)  # [Bc*N, out_dim]
        return z.view(Bc, N, -1)


class PatchEmbedCNNMultiScale(nn.Module):
    """
    Parallel temporal kernels to capture complementary short- and medium-range
    temporal structure inside each patch.
    """
    def __init__(self, patch_len: int, out_dim: int):
        super().__init__()
        del patch_len  # kept for interface symmetry with other embedders

        branch_dim = max(16, out_dim // 4)
        kernels = (3, 5, 9)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(1, branch_dim, kernel_size=k, stride=1, padding=k // 2),
                nn.GELU(),
                nn.Conv1d(branch_dim, branch_dim, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
            )
            for k in kernels
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_dim * len(kernels), out_dim, kernel_size=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        Bc, N, L = x_patches.shape
        z = x_patches.reshape(Bc * N, 1, L)
        z = torch.cat([branch(z) for branch in self.branches], dim=1)
        z = self.fuse(z)
        z = self.pool(z).squeeze(-1)
        return z.view(Bc, N, -1)


class CoordMLP(nn.Module):
    """pos3d: [B,C,3] -> [B,C,D]"""
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, pos3d: torch.Tensor) -> torch.Tensor:
        return self.net(pos3d)


def _parse_coord_scale(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return float(value)


def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    ALiBi slopes from the reference schedule.
    """
    if n_heads <= 0:
        raise ValueError(f"n_heads must be > 0, got {n_heads}")

    def _get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        vals = _get_slopes_power_of_2(n_heads)
    else:
        closest_pow2 = 2 ** math.floor(math.log2(n_heads))
        vals = _get_slopes_power_of_2(closest_pow2)
        vals_extra = _get_alibi_slopes(2 * closest_pow2).tolist()[0::2]
        vals.extend(vals_extra[: n_heads - closest_pow2])
    return torch.tensor(vals, dtype=torch.float32)


class TemporalALiBiEncoderLayer(nn.Module):
    """
    Pre-norm transformer encoder layer with bidirectional ALiBi-style bias.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = self.d_model // self.nhead
        self.dropout_p = float(dropout)

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.drop_attn = nn.Dropout(self.dropout_p)
        self.drop_ff = nn.Dropout(self.dropout_p)

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, int(dim_feedforward)),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(int(dim_feedforward), self.d_model),
        )

        self.register_buffer("alibi_slopes", _get_alibi_slopes(self.nhead), persistent=False)

    def _alibi_bias(self, n_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(n_tokens, device=device, dtype=torch.float32)
        # Bidirectional setting: penalize attention by absolute distance.
        dist = (pos[None, :] - pos[:, None]).abs()  # [L, L]
        slopes = self.alibi_slopes.to(device=device, dtype=torch.float32).view(1, self.nhead, 1, 1)
        bias = -slopes * dist.view(1, 1, n_tokens, n_tokens)  # [1, H, L, L]
        return bias.to(dtype=dtype)

    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, _ = x.shape
        qkv = self.qkv(x).view(bsz, n_tokens, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, Dh]

        attn_bias = self._alibi_bias(n_tokens, device=x.device, dtype=q.dtype)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=(self.dropout_p if self.training else 0.0),
            is_causal=False,
        )
        out = attn.transpose(1, 2).contiguous().view(bsz, n_tokens, self.d_model)
        out = self.out_proj(out)
        return self.drop_attn(out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_ff(self.ffn(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x


class TemporalALiBiEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalALiBiEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(int(num_layers))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------
# Encoder (patch-only, fixed coord-MLP channel position encoding)
# ---------------------------------------------------------------------
class BrainWaveTokEmbEncoder(nn.Module):
    def __init__(
        self,
        input_len: int,
        embed_dim: int,
        num_heads: int,
        channel_depth: int,
        num_channels: int,
        mask_ratio: float,
        embedder: str,         # {"linear","mlp_patch","cnn_simple","cnn_multiscale"}
        emb_norm: str = "l2",
        patch_len: int = 50,
        temporal_depth: int = 4,
        cnn_out_dim: Optional[int] = None,
        coord_scale_init: Optional[float] = 100.0,
        clip_value: float = 300.0,
        chan_norm: str = "none",
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.channel_depth = int(channel_depth)
        self.num_channels = int(num_channels)
        self.mask_ratio = float(mask_ratio)
        self.embedder = str(embedder)
        self.emb_norm = str(emb_norm)
        self.clip_value = float(clip_value)
        self.chan_norm = str(chan_norm)

        if self.embedder not in ("linear", "mlp_patch", "cnn_simple", "cnn_multiscale"):
            raise ValueError(f"Unknown embedder: {self.embedder}")
        if self.emb_norm not in ("none", "l2"):
            raise ValueError(f"Unknown emb_norm: {self.emb_norm}")
        if self.chan_norm not in ("none", "zscore", "robust", "rms"):
            raise ValueError(f"Unknown chan_norm: {self.chan_norm}")

        self._init_posenc(coord_scale_init)
        self._init_patch_branch(patch_len, temporal_depth, cnn_out_dim)
        self._init_channel_encoder()

    def _init_posenc(self, coord_scale_init: Optional[float]):
        self.pos_coord = CoordMLP(self.embed_dim)
        if coord_scale_init is None:
            self.coord_scale = None
        else:
            # Static (non-trainable) coordinate scale.
            self.coord_scale = float(coord_scale_init)

    def _init_patch_branch(self, patch_len: int, temporal_depth: int, cnn_out_dim: Optional[int]):
        # patch_len is fixed at model init. Runtime variable T is handled by
        # padding each sample to the nearest multiple of patch_len.
        self.patch_len = int(patch_len)
        if self.patch_len <= 0:
            raise ValueError(f"patch_len must be >= 1, got {self.patch_len}")
        self.temporal_depth = int(temporal_depth)

        out_dim = int(cnn_out_dim) if cnn_out_dim is not None else self.embed_dim
        self.fuse_proj = nn.Linear(out_dim, self.embed_dim) if out_dim != self.embed_dim else None

        if self.embedder == "linear":
            self.patch_embed = PatchEmbedLinear(self.patch_len, self.embed_dim)
        elif self.embedder == "mlp_patch":
            self.patch_embed = PatchEmbedMLP(self.patch_len, self.embed_dim)
        elif self.embedder == "cnn_simple":
            self.patch_embed = PatchEmbedCNNSimple(self.patch_len, out_dim)
        elif self.embedder == "cnn_multiscale":
            self.patch_embed = PatchEmbedCNNMultiScale(self.patch_len, out_dim)
        else:
            raise ValueError(f"Unsupported embedder: {self.embedder}")

        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.temporal_encoder = TemporalALiBiEncoder(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            num_layers=self.temporal_depth,
            dropout=0.1,
        )

    def _init_channel_encoder(self):
        channel_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            batch_first=True,
            dropout=0.1,
            norm_first=True,
        )
        self.channel_encoder = TransformerEncoder(channel_layer, num_layers=self.channel_depth)
        self.channel_cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

    def _preprocess_x(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip_value > 0:
            x = x.clamp(-self.clip_value, self.clip_value)
        if self.chan_norm == "none":
            return x

        eps = 1e-6
        if self.chan_norm == "zscore":
            mu = x.mean(dim=-1, keepdim=True)
            sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
            return (x - mu) / sd
        if self.chan_norm == "robust":
            med = x.median(dim=-1, keepdim=True).values
            mad = (x - med).abs().median(dim=-1, keepdim=True).values
            mad = (1.4826 * mad).clamp_min(eps)
            return (x - med) / mad

        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        return x / rms

    def _apply_random_channel_mask(self, chan_mask: torch.Tensor, device: torch.device):
        if (not self.training) or self.mask_ratio <= 0 or chan_mask is None:
            return chan_mask
        chan_mask = chan_mask.clone()

        # Batch-wise masking: drop the same channel indices for all samples.
        # To keep this valid for every sample, choose only channels that are valid
        # across the whole batch (intersection over sample masks).
        common_valid = chan_mask.all(dim=0).nonzero(as_tuple=False).squeeze(1)
        if common_valid.numel() == 0:
            return chan_mask

        k = max(1, int(common_valid.numel() * self.mask_ratio))
        k = min(k, int(common_valid.numel()))
        idx = common_valid[torch.randperm(common_valid.numel(), device=device)[:k]]
        chan_mask[:, idx] = False
        return chan_mask

    def _embed_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        if T <= 0:
            raise ValueError(f"Invalid temporal length T={T}")

        # Dynamic patch count at runtime (patch_len fixed from model setup).
        rem = T % self.patch_len
        if rem != 0:
            pad_t = self.patch_len - rem
            x = torch.nn.functional.pad(x, (0, pad_t))
            T = T + pad_t
        n_patches = T // self.patch_len
        x_patches = x.view(B * C, n_patches, self.patch_len)
        patch_tok = self.patch_embed(x_patches)
        if self.fuse_proj is not None:
            patch_tok = self.fuse_proj(patch_tok)

        cls = self.temporal_cls_token.expand(B * C, 1, self.embed_dim)
        seq = torch.cat([cls, patch_tok], dim=1)
        enc = self.temporal_encoder(seq)
        return enc[:, 0].view(B, C, self.embed_dim)

    def forward(self, x: torch.Tensor, chan_mask: torch.Tensor = None, pos3d: torch.Tensor = None) -> torch.Tensor:
        B, _C, _T = x.shape
        if pos3d is None:
            raise ValueError("pos3d is required")

        x = self._preprocess_x(x)
        chan_mask = self._apply_random_channel_mask(chan_mask, x.device)
        z = self._embed_patch_tokens(x)

        cls_tok = self.channel_cls_token.expand(B, 1, self.embed_dim)
        tok = torch.cat([cls_tok, z], dim=1)
        if self.coord_scale is None:
            pos3d_scaled = pos3d
        else:
            pos3d_scaled = pos3d * self.coord_scale
        pe = self.pos_coord(pos3d_scaled)
        pe = torch.cat([torch.zeros(B, 1, self.embed_dim, device=tok.device), pe], dim=1)
        tok = tok + pe

        if chan_mask is not None:
            pad_mask = torch.cat(
                [torch.zeros(B, 1, dtype=torch.bool, device=tok.device), ~chan_mask],
                dim=1,
            )
        else:
            pad_mask = None

        out = self.channel_encoder(tok, src_key_padding_mask=pad_mask)
        emb = out[:, 0]
        if self.emb_norm == "l2":
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1, eps=1e-12)
        return emb


# ---------------------------------------------------------------------
# Builders (model + loss/miner + optimizer)
# ---------------------------------------------------------------------
def build_model(
    trial: Optional[Any],
    num_channels: int,
    device: torch.device,
    fixed_params: Optional[Dict[str, Any]] = None,
    space_registry: Optional[Dict[str, tuple]] = None,
):
    """
    Patch+ALiBi architecture is fixed in this code version.
    """
    # defaults (overridable via args/fixed_params)
    emb_norm = str(DEFAULT_EMB_NORM)
    mask_ratio = float(DEFAULT_MASK_RATIO)

    # Search parameters
    if fixed_params is None:
        if trial is None:
            raise ValueError("trial must be provided when fixed_params is None")
        embedder = suggest_cat(
            trial,
            "embedder",
            ["linear", "mlp_patch", "cnn_simple", "cnn_multiscale"],
            registry=space_registry,
        )
        embed_dim = int(suggest_cat(trial, "embed_dim", [64, 128], registry=space_registry))
        num_heads = int(suggest_cat(trial, "num_heads", [4, 8], registry=space_registry))
        channel_depth = int(suggest_cat(trial, "channel_depth", [2, 4, 6], registry=space_registry))
        temporal_depth = int(suggest_cat(trial, "temporal_depth", [2, 4], registry=space_registry))

        coord_scale_choice = suggest_cat(
            trial,
            "coord_scale",
            ["none", 100.0],
            registry=space_registry,
        )
        coord_scale_init = _parse_coord_scale(coord_scale_choice)

        clip_value = float(suggest_cat(trial, "clip_value", [0.0, 200.0, 400.0], registry=space_registry))
        chan_norm = suggest_cat(trial, "chan_norm", ["none", "zscore", "rms"], registry=space_registry)
        emb_norm = str(suggest_cat(trial, "emb_norm", ["none", "l2"], registry=space_registry))
        mask_ratio = float(suggest_cat(trial, "mask_ratio", [0.0, 0.1, 0.2], registry=space_registry))
        patch_len = int(suggest_cat(trial, "patch_len", [50, 100], registry=space_registry))
    else:
        if "posenc" in fixed_params and str(fixed_params["posenc"]) != "coord_mlp":
            raise ValueError(
                f"Unsupported posenc in fixed params: {fixed_params['posenc']}. "
                "This code version requires coord_mlp channel positional encoding."
            )
        if "temporal_posenc" in fixed_params and str(fixed_params["temporal_posenc"]) != "alibi":
            raise ValueError(
                f"Unsupported temporal_posenc in fixed params: {fixed_params['temporal_posenc']}. "
                "This code version requires temporal_posenc='alibi'."
            )
        if "tokenization" in fixed_params and str(fixed_params["tokenization"]) != "patch":
            raise ValueError(
                f"Unsupported tokenization in fixed params: {fixed_params['tokenization']}. "
                "This code version requires tokenization='patch'."
            )
        embedder = fixed_params["embedder"]
        embed_dim = int(fixed_params["embed_dim"])
        num_heads = int(fixed_params.get("num_heads", 8))
        channel_depth = int(fixed_params["channel_depth"])
        if "temporal_depth" not in fixed_params:
            raise ValueError(
                "Missing required key 'temporal_depth' in fixed params. "
                "Use a study generated by this code version."
            )
        temporal_depth = int(fixed_params["temporal_depth"])

        coord_scale_init = _parse_coord_scale(fixed_params.get("coord_scale", 100.0))

        clip_value = float(fixed_params.get("clip_value", 300.0))
        chan_norm = str(fixed_params.get("chan_norm", "none"))
        emb_norm = str(fixed_params.get("emb_norm", emb_norm))
        mask_ratio = float(fixed_params.get("mask_ratio", mask_ratio))
        patch_len = int(fixed_params["patch_len"])

    if emb_norm not in ("none", "l2"):
        raise ValueError(f"Unsupported emb_norm: {emb_norm}")
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

    # CNN output width is kept fixed to embed_dim.
    cnn_out_dim = embed_dim

    model = BrainWaveTokEmbEncoder(
        input_len=args.crop_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        channel_depth=channel_depth,
        num_channels=num_channels,
        mask_ratio=mask_ratio,
        embedder=embedder,
        patch_len=patch_len,
        temporal_depth=temporal_depth,
        cnn_out_dim=cnn_out_dim,
        emb_norm=emb_norm,
        coord_scale_init=coord_scale_init,
        clip_value=clip_value,
        chan_norm=chan_norm,
    ).to(device)

    if USE_TORCH_COMPILE:
        model = torch.compile(model)

    return model, int(embed_dim)


def build_loss_and_miner(chosen_loss: str, num_classes: int, embed_dim: int, device: torch.device):
    """
    No miner_epsilon / temperature / triplet_margin hyperparameters: use library defaults.
    """
    miner_fn = None

    if chosen_loss == "SupConLoss":
        loss_fn = losses.SupConLoss().to(device)
        miner_fn = miners.MultiSimilarityMiner()

    elif chosen_loss == "LiftedStructureLoss":
        loss_fn = losses.LiftedStructureLoss().to(device)
        miner_fn = miners.MultiSimilarityMiner()

    elif chosen_loss == "TripletMarginLoss":
        loss_fn = losses.TripletMarginLoss().to(device)
        miner_fn = miners.TripletMarginMiner(margin=0.05, type_of_triplets="semihard")

    elif chosen_loss == "ArcFaceLoss":
        loss_fn = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embed_dim).to(device)

    elif chosen_loss == "SoftTripleLoss":
        loss_fn = losses.SoftTripleLoss(num_classes=num_classes, embedding_size=embed_dim).to(device)

    else:
        raise ValueError(f"Unsupported loss: {chosen_loss}")

    return loss_fn, miner_fn


def build_optimizer(
    trial: Optional[Any],
    model: nn.Module,
    loss_fn: nn.Module,
    fixed_params: Optional[Dict[str, Any]] = None,
    space_registry: Optional[Dict[str, tuple]] = None,
):
    if fixed_params is None:
        if trial is None:
            raise ValueError("trial must be provided when fixed_params is None")
        lr = float(suggest_cat(trial, "lr", [1e-5, 1e-4, 1e-3], registry=space_registry))
        opt_name = "AdamW"
        wd = float(suggest_cat(trial, "weight_decay", [0.0, 1e-5], registry=space_registry))
    else:
        lr = float(fixed_params["lr"])
        opt_name = str(fixed_params.get("optimizer", "AdamW"))
        wd = float(fixed_params["weight_decay"])

    model_params = [p for p in model.parameters() if p.requires_grad]
    loss_params = [p for p in loss_fn.parameters() if p.requires_grad] if hasattr(loss_fn, "parameters") else []

    param_groups = [{"params": model_params, "lr": lr}]
    if len(loss_params) > 0:
        param_groups.append({"params": loss_params, "lr": lr * LOSS_LR_SCALE})

    if opt_name == "AdamW":
        optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer for this code version: {opt_name}. Expected 'AdamW'.")

    return optimizer


# ---------------------------------------------------------------------
# Train + cached evaluation
# ---------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, loss_fn, miner_fn, device):
    model.train()
    total, n = 0.0, 0
    skipped_nonfinite = 0
    skipped_bad_grad = 0
    use_autocast = (device.type == "cuda")
    clip_params = [p for p in model.parameters() if p.requires_grad]
    if hasattr(loss_fn, "parameters"):
        clip_params += [p for p in loss_fn.parameters() if p.requires_grad]

    for step, (eeg, mask, labels, pos3d, _meta) in enumerate(loader, start=1):
        t_lengths = torch.as_tensor(_meta.get("t_len", [int(eeg.shape[-1])] * int(eeg.shape[0])), dtype=torch.long)
        target_len = _sample_train_batch_length()
        if target_len is not None:
            eeg = _fit_batch_to_target_length(
                eeg,
                t_lengths=t_lengths,
                target_len=target_len,
                crop_mode=str(TRAIN_LENGTH_CROP_MODE),
            )

        eeg = eeg.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pos3d = pos3d.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Keep forward under autocast, but compute metric losses in fp32 for stability.
        ctx = amp.autocast(device_type="cuda") if use_autocast else nullcontext()
        with ctx:
            emb = model(eeg, mask, pos3d=pos3d)
        emb = emb.float()
        if not torch.isfinite(emb).all():
            skipped_nonfinite += 1
            if FAIL_ON_NAN:
                raise FloatingPointError(f"Non-finite embedding at step={step}")
            continue

        if miner_fn is not None:
            mined = miner_fn(emb, labels)
            loss = loss_fn(emb, labels, mined)
        else:
            loss = loss_fn(emb, labels)

        if loss.ndim != 0:
            loss = loss.mean()
        if not torch.isfinite(loss).item():
            skipped_nonfinite += 1
            if FAIL_ON_NAN:
                raise FloatingPointError(f"Non-finite loss at step={step}: {loss}")
            continue

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        if len(clip_params) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=CLIP_GRAD_MAX_NORM)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                skipped_bad_grad += 1
                if FAIL_ON_NAN:
                    raise FloatingPointError(f"Non-finite grad norm at step={step}: {grad_norm}")
                # Keep GradScaler state consistent for next iteration.
                scaler.step(optimizer)
                scaler.update()
                continue

        scaler.step(optimizer)
        scaler.update()

        total += float(loss.item())
        n += 1

    if skipped_nonfinite > 0 or skipped_bad_grad > 0:
        print(
            f"[WARN] skipped_batches nonfinite_emb_or_loss={skipped_nonfinite} "
            f"nonfinite_grad={skipped_bad_grad} used_batches={n}"
        )
    if n == 0:
        raise FloatingPointError("All batches were skipped due to non-finite values.")
    return total / n


def collect_arrays(loader, T_expected=500):
    xs, ms, ys, ss, ps = [], [], [], [], []
    for eeg, mask, labels, pos3d, meta in tqdm(loader, desc="collect", leave=False):
        B, Cb, T = eeg.shape
        if T != T_expected:
            raise ValueError(f"segment length {T} != {T_expected}")
        xs.append(eeg.numpy())
        ms.append(mask.numpy())
        ys.append(labels.numpy())
        ss.append(np.array(meta["session_id"], dtype=np.int64))
        ps.append(pos3d.numpy())
    X_all = np.concatenate(xs, axis=0)
    M_all = np.concatenate(ms, axis=0)
    Y_all = np.concatenate(ys, axis=0)
    S_all = np.concatenate(ss, axis=0)
    P_all = np.concatenate(ps, axis=0)
    return X_all, M_all, Y_all, S_all, P_all


def generate_embeddings_eeg_masked(X, M, P, model, device, batch_size=256):
    model.eval()
    embs = []
    N = X.shape[0]
    use_autocast = (device.type == "cuda")
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_t = torch.from_numpy(X[start:end]).to(device)
            m_t = torch.from_numpy(M[start:end]).to(device)
            p_t = torch.from_numpy(P[start:end]).to(device)
            ctx = amp.autocast(device_type="cuda") if use_autocast else nullcontext()
            with ctx:
                out = model(x_t, m_t, pos3d=p_t)
            embs.append(out.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def evaluate_avg_eer_from_cached(model, device, X_all, M_all, Y_all, S_all, P_all, top_k=40):
    enroll_idxs, verify_idxs = get_enrollment_verification_indices(Y_all, S_all)

    X_enroll = X_all[enroll_idxs]
    M_enroll = M_all[enroll_idxs]
    P_enroll = P_all[enroll_idxs]
    y_enroll = Y_all[enroll_idxs]

    X_verify = X_all[verify_idxs]
    M_verify = M_all[verify_idxs]
    P_verify = P_all[verify_idxs]
    y_verify = Y_all[verify_idxs]

    en_emb = generate_embeddings_eeg_masked(
        X_enroll, M_enroll, P_enroll, model, device=device, batch_size=CACHE_EVAL_BATCH_SIZE
    )
    ve_emb = generate_embeddings_eeg_masked(
        X_verify, M_verify, P_verify, model, device=device, batch_size=CACHE_EVAL_BATCH_SIZE
    )

    sim_dict = compute_similarity_scores(en_emb, y_enroll, ve_emb, y_verify, distance="cd", top_k=top_k)
    avg_eer, std_eer = evaluate_eer_per_class(sim_dict)
    return float(avg_eer), float(std_eer)


# ---------------------------------------------------------------------
# Optuna objective (single-stage; supports fixed_params for final rebuild)
# ---------------------------------------------------------------------
def objective(trial, train_ds, Xv, Mv, Yv, Sv, Pv, device,
              fixed_params: Optional[Dict[str, Any]] = None,
              train_layout_cache: Optional[Dict[str, Dict[str, Any]]] = None,
              space_registry: Optional[Dict[str, tuple]] = None,
              print_space_once: Optional[Dict[str, bool]] = None,
              space_label: str = "Search"):
    if LOG_CUDA_MEM and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        print(f"[MEM][T{trial.number:04d}] start {format_cuda_mem(device)}")

    bs = int(TRAIN_BATCH_SIZE)
    if fixed_params is not None and ("batch_size" in fixed_params):
        bs_fixed = int(fixed_params["batch_size"])
        if bs_fixed != bs:
            raise ValueError(
                f"fixed_params['batch_size']={bs_fixed} does not match TRAIN_BATCH_SIZE={bs}. "
                "This code version keeps batch size fixed."
            )

    model = None
    optimizer = None
    scaler = None
    train_sampler = None
    train_loader = None
    best_eer = float("inf")
    try:
        model, embed_dim = build_model(
            trial,
            num_channels=len(REF_CLIST_NORM),
            device=device,
            fixed_params=fixed_params,
            space_registry=space_registry,
        )
        if is_disallowed_embed_batch_combo(embed_dim, bs):
            raise optuna.TrialPruned(
                f"Disallowed combo: embed_dim={embed_dim}, batch_size={bs}"
            )

        num_classes = int(train_ds.targets.unique().numel())
        loss_fn, miner_fn = build_loss_and_miner(CHOSEN_LOSS, num_classes, embed_dim, device)

        optimizer = build_optimizer(trial, model, loss_fn, fixed_params=fixed_params, space_registry=space_registry)

        # Skip duplicate parameter sets already running/completed in this study.
        cur_params = dict(trial.params)
        if cur_params:
            prior = trial.study.get_trials(
                deepcopy=False,
                states=(optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.COMPLETE),
            )
            for t in prior:
                if t.number == trial.number:
                    continue
                if t.params == cur_params:
                    raise optuna.TrialPruned(
                        f"Duplicate params of trial {t.number} ({t.state.name}): {cur_params}"
                    )

        if print_space_once is not None and not print_space_once.get("done", False):
            print(f"[INFO] {space_label} unique combinations: {unique_combo_count(space_registry):,}")
            print_space_once["done"] = True

        trial_params_for_print = dict(fixed_params) if fixed_params is not None else dict(trial.params)
        trial_params_for_print["batch_size"] = int(bs)
        print(f"[TRIAL {trial.number:04d}] start params={trial_params_for_print}")

        scaler = amp.GradScaler(enabled=(device.type == "cuda"))
        new_iter = bs * TUNING_LENGTH_MULTIPLIER

        try:
            train_sampler = LayoutAwareEpisodeMultiSessionBatchSampler(
                labels=train_ds.targets,
                sessions=train_ds.session_ids,
                layouts=train_ds.layout_ids,
                batch_size=bs,
                num_subjects_per_batch=EPISODE_NUM_SUBJECTS,
                max_sessions_per_subject=EPISODE_MAX_SESSIONS,
                length_before_new_iter=new_iter,
                prefer_multisession_subjects=EPISODE_PREFER_MULTISESSION,
                no_duplicates=EPISODE_NO_DUPLICATES,
                max_tries=EPISODE_MAX_TRIES,
            )
        except RuntimeError as exc:
            if _is_unformable_batch_error(exc):
                raise optuna.TrialPruned(str(exc))
            if _is_cuda_oom_error(exc):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"OOM while building sampler/loader: {exc}")
            raise
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=partial(collate_mixed2, layout_cache=train_layout_cache),
            **_loader_perf_kwargs(NUM_WORKERS_TRAIN, pin_memory=True),
        )
        iters_per_epoch = len(train_loader)

        for ep in range(1, args.tuning_epochs + 1):
            try:
                tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, miner_fn, device)
                avg_eer, std_eer = evaluate_avg_eer_from_cached(
                    model, device, Xv, Mv, Yv, Sv, Pv, top_k=TOP_K
                )
            except RuntimeError as exc:
                if _is_unformable_batch_error(exc):
                    raise optuna.TrialPruned(str(exc))
                if _is_cuda_oom_error(exc):
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned(f"OOM in trial {trial.number} epoch {ep}: {exc}")
                raise

            print(
                f"[E{ep:03d}] iters={iters_per_epoch} bs={bs}  "
                f"train_loss={tr_loss:.4f}  val_eer={avg_eer:.4f}+-{std_eer:.4f}"
            )
            if LOG_CUDA_MEM and device.type == "cuda":
                print(f"[MEM][T{trial.number:04d} E{ep:03d}] {format_cuda_mem(device)}")

            trial.report(avg_eer, step=ep)
            best_eer = min(best_eer, avg_eer)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_eer
    finally:
        if LOG_CUDA_MEM and device.type == "cuda":
            print(f"[MEM][T{trial.number:04d}] before_cleanup {format_cuda_mem(device)}")
        _shutdown_dataloader(train_loader)
        model = None
        optimizer = None
        scaler = None
        train_loader = None
        train_sampler = None
        gc.collect()
        torch.cuda.empty_cache()
        if LOG_CUDA_MEM and device.type == "cuda":
            print(f"[MEM][T{trial.number:04d}] after_cleanup {format_cuda_mem(device)}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    print(
        f"[INFO] train-length augmentation: lengths={list(TRAIN_BATCH_LENGTHS)} "
        f"mode={TRAIN_LENGTH_CROP_MODE}"
    )
    print(
        f"[INFO] train dataset mode: no-crop (native lengths). "
        f"crop_len={args.crop_len} is used for val/test only; patch_len controls tokenization."
    )
    if LOG_CUDA_MEM and device.type == "cuda":
        print(f"[MEM] startup {format_cuda_mem(device)}")

    train_ds = PackedLayoutDatasetNoCrop(args.train_dir, crop_len=args.crop_len, random_crop=True)
    val_ds = PackedLayoutDataset2(
        args.val_dir,
        crop_len=args.crop_len,
        random_crop=False,
        deterministic_crop_mode=args.eval_crop_mode,
    )
    test_ds = PackedLayoutDataset2(
        args.test_dir,
        crop_len=args.crop_len,
        random_crop=False,
        deterministic_crop_mode=args.eval_crop_mode,
    )
    train_layout_cache = build_layout_cache(train_ds)
    val_layout_cache = build_layout_cache(val_ds)
    test_layout_cache = build_layout_cache(test_ds)

    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=partial(collate_mixed2, fixed_Cmax=32, layout_cache=val_layout_cache),
        **_loader_perf_kwargs(NUM_WORKERS_EVAL, pin_memory=True),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        collate_fn=partial(collate_mixed2, fixed_Cmax=32, layout_cache=test_layout_cache),
        **_loader_perf_kwargs(NUM_WORKERS_EVAL, pin_memory=True),
    )

    # cache arrays once (critical for tuning speed)
    Xv, Mv, Yv, Sv, Pv = collect_arrays(val_loader, T_expected=args.crop_len)
    Xt, Mt, Yt, St, Pt = collect_arrays(test_loader, T_expected=args.crop_len)
    _shutdown_dataloader(val_loader)
    _shutdown_dataloader(test_loader)
    del val_loader, test_loader, val_ds, test_ds
    gc.collect()
    
    if LOG_CUDA_MEM and device.type == "cuda":
        print(f"[MEM] after_cache_collect {format_cuda_mem(device)}")

    # Optuna storage (resume-safe)
    storage = RDBStorage(
        url=args.study_db,
        engine_kwargs={"connect_args": {"timeout": 600, "check_same_thread": False}}
    )

    search_registry: Dict[str, tuple] = {}
    search_printed = {"done": False}

    def run_study(objective_fn):
        study = optuna.create_study(
            study_name=f"{args.study_name}_{CHOSEN_LOSS}",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=optuna.samplers.RandomSampler(seed=None),
            pruner=optuna.pruners.NopPruner(),
        )
        if args.n_trials > 0:
            max_complete_cb = optuna.study.MaxTrialsCallback(
                args.n_trials,
                states=(optuna.trial.TrialState.COMPLETE,),
            )
            study.optimize(
                objective_fn,
                n_trials=None,
                callbacks=[max_complete_cb],
                catch=(optuna.TrialPruned, ValueError, FloatingPointError, torch.OutOfMemoryError),
            )
        else:
            print("[INFO] n_trials=0, skipping search.")
        complete_trials = study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.COMPLETE,),
        )
        if len(complete_trials) == 0:
            raise RuntimeError("Search has no completed trials. Increase n_trials or reuse an existing study.")
        print("[RESULT] SEARCH BEST EER:", study.best_value)
        print("[RESULT] SEARCH BEST PARAMS:", study.best_params)
        return study

    study = run_study(
        lambda t: objective(
            t, train_ds, Xv, Mv, Yv, Sv, Pv, device,
            fixed_params=None,
            train_layout_cache=train_layout_cache,
            space_registry=search_registry,
            print_space_once=search_printed,
            space_label="Search",
        ),
    )
    best_params = dict(study.best_params)

    if "tokenization" in best_params and str(best_params["tokenization"]) != "patch":
        raise RuntimeError(
            "Loaded study contains tokenization != 'patch' (older study schema). "
            "Use a fresh --study_name/--study_db for this code version."
        )
    if "temporal_posenc" in best_params and str(best_params["temporal_posenc"]) != "alibi":
        raise RuntimeError(
            "Loaded study contains temporal_posenc != 'alibi' (older study schema). "
            "Use a fresh --study_name/--study_db for this code version."
        )
    if is_disallowed_embed_batch_combo(int(best_params["embed_dim"]), int(TRAIN_BATCH_SIZE)):
        raise RuntimeError(
            "Best params use disallowed combo embed_dim=256 with batch_size=512. "
            "Use a fresh --study_name/--study_db with this constraint enabled."
        )
    if best_params.get("embedder") not in ("linear", "mlp_patch", "cnn_simple", "cnn_multiscale"):
        raise RuntimeError(
            "Best params contain unsupported architecture options from an older study. "
            "Use a fresh --study_name/--study_db for this code version."
        )
    if "temporal_depth" not in best_params:
        raise RuntimeError(
            "Loaded study is missing temporal_depth (older schema). "
            "Use a fresh --study_name/--study_db for this code version."
        )

    print("[RESULT] SELECTED BEST PARAMS:", best_params)
    print("[INFO] Search completed. Use eeg_auth_replicate_best_varlen.py or another dedicated runner to retrain the selected best params.")


if __name__ == "__main__":
    main()
