#!/usr/bin/env python3
"""
Run shared channel-level inference experiments for a NeuroShield checkpoint
and save all results to CSV.

This script supports both:
  - downstream fine-tuned checkpoints
  - zero-shot / pretrained foundation checkpoints

For the zero-shot case, point CONFIG["meta_json"] at the foundation-model
metadata, optionally set CONFIG["hard_ckpt_path"], and choose a separate
CONFIG["save_output_dir"] so the outputs do not overwrite fine-tuned CSVs.

Current experiment families:
  - full baseline
  - leave-one-channel-out
  - single-channel-only
  - random three-channel subsets
  - region ablation
  - region pair only
  - zero-pos3d full-channel control

Edit CONFIG and run:
  python evaluate_channel_level_experiments.py
"""

import csv
import gc
import importlib
import json
import math
import os
import sys
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

import numpy as np


CONFIG: Dict[str, Any] = {
    "cuda_devices": "0",
    "finetune_dir": "./ckpts_bw_replicate_varlen",
    "meta_json": "./ckpts_bw_replicate_varlen/replicate_best_trial_varlen_meta.json",
    "hard_ckpt_path": "./ckpts_bw_replicate_varlen/replicate_best_trial_varlen_final.pth",
    "ckpt_kind": "final",  # {"best", "final"}
    "eval_dir": "./data/downstream/test_packed",
    "crop_len": 500,  # 0 -> use checkpoint/native default
    "eval_crop_mode": "auto",  # {"auto", "center", "left"}
    "top_k": 0,  # 0 -> use finetune metadata/default
    "use_capped_subject_eer": None,  # None -> keep finetune/current default
    "exact_match_only": True,
    "channel_names": [],  # empty -> use full canonical REF_CLIST_NORM
    "experiments": [
        "leave_one_out",
        "single_channel_only",
        "random_three_channels",
        "region_pair_only",
        "zero_pos3d",
    ],
    "random_three_channel_trials": 20,
    "random_seed": 101,
    "region_schemes": [
        "anatomical_6",
        "hemisphere_3",
    ],
    "save_csv_prefix": "channel_level",
    "save_output_dir": "",  # empty -> use finetune_dir
}


def _import_eval_module(cuda_devices: str):
    if "torch" in sys.modules:
        torch_mod = sys.modules["torch"]
        try:
            if torch_mod.cuda.is_available() and torch_mod.cuda.is_initialized():
                print(
                    "[WARN] torch.cuda already initialized before CUDA_VISIBLE_DEVICES set. "
                    "Restart kernel for reliable GPU switching."
                )
        except Exception:
            pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], "--cuda_devices", str(cuda_devices)]
    try:
        if "eeg_auth_train_optuna" in sys.modules:
            mod = importlib.reload(sys.modules["eeg_auth_train_optuna"])
        else:
            mod = importlib.import_module("eeg_auth_train_optuna")
    finally:
        sys.argv = old_argv
    return mod


def _resolve_device(mod):
    if not mod.torch.cuda.is_available():
        return mod.torch.device("cpu"), "cpu"

    cfg = str(CONFIG["cuda_devices"]).strip()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    first = cfg.split(",")[0].strip() if cfg else "0"
    logical_idx = 0 if (vis and vis == cfg) else int(first)
    n_visible = int(mod.torch.cuda.device_count())

    if logical_idx < 0 or logical_idx >= n_visible:
        print(
            f"[WARN] Requested logical cuda:{logical_idx} is out of visible range [0,{max(n_visible - 1, 0)}]. "
            "Falling back to cuda:0."
        )
        logical_idx = 0

    mod.torch.cuda.set_device(logical_idx)
    dev_name = mod.torch.cuda.get_device_name(logical_idx)
    info = (
        f"logical=cuda:{logical_idx} name='{dev_name}' visible_count={n_visible} "
        f"CUDA_VISIBLE_DEVICES='{vis}' requested='{cfg}'"
    )
    if vis and "," not in vis:
        info += " (single visible physical GPU is remapped to logical cuda:0)"
    return mod.torch.device(f"cuda:{logical_idx}"), info


def _choose_ckpt(meta: Dict[str, Any], ckpt_kind: str) -> str:
    if ckpt_kind == "best":
        return str(meta["best_ckpt"])
    if ckpt_kind == "final":
        return str(meta["final_ckpt"])
    raise ValueError("CONFIG['ckpt_kind'] must be 'best' or 'final'.")


def _resolve_meta_json() -> str:
    explicit = str(CONFIG.get("meta_json", "")).strip()
    if explicit:
        meta_path = os.path.abspath(explicit)
        if not os.path.exists(meta_path):
            raise RuntimeError(f"meta_json not found: {meta_path}")
        return meta_path

    finetune_dir = os.path.abspath(str(CONFIG.get("finetune_dir", "")).strip())
    if finetune_dir == "":
        raise RuntimeError("Either CONFIG['meta_json'] or CONFIG['finetune_dir'] must be provided.")

    candidates = [
        os.path.join(finetune_dir, "finetune_meta.json"),
        os.path.join(finetune_dir, "replicate_best_trial_varlen_meta.json"),
        os.path.join(finetune_dir, "replicate_best_trial_hardware_filtered_meta.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise RuntimeError(f"Could not find metadata JSON in {finetune_dir}")


def _resolve_ckpt_path(meta: Dict[str, Any], ckpt_kind: str) -> str:
    explicit = str(CONFIG.get("hard_ckpt_path", "")).strip()
    if explicit:
        ckpt_path = os.path.abspath(explicit)
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"hard_ckpt_path not found: {ckpt_path}")
        return ckpt_path

    meta_key = "best_ckpt" if ckpt_kind == "best" else "final_ckpt"
    meta_path = str(meta.get(meta_key, "")).strip()
    if meta_path:
        ckpt_path = os.path.abspath(meta_path)
        if os.path.exists(ckpt_path):
            return ckpt_path
        print(f"[WARN] metadata {meta_key} not found on disk: {ckpt_path}")

    run_dir = os.path.abspath(str(CONFIG.get("finetune_dir", "")).strip()) if str(CONFIG.get("finetune_dir", "")).strip() else ""
    if run_dir != "":
        fallback_names = {
            "best": [
                "replicate_best_trial_varlen_best.pth",
                "replicate_best_trial_best.pth",
                "replicate_best_trial_hardware_filtered_best.pth",
                "finetune_best.pth",
                "best.pth",
            ],
            "final": [
                "replicate_best_trial_varlen_final.pth",
                "replicate_best_trial_final.pth",
                "replicate_best_trial_hardware_filtered_final.pth",
                "finetune_final.pth",
                "final.pth",
            ],
        }
        for name in fallback_names[str(ckpt_kind)]:
            candidate = os.path.join(run_dir, name)
            if os.path.exists(candidate):
                print(f"[INFO] using fallback checkpoint: {candidate}")
                return candidate

    raise RuntimeError(
        f"Could not resolve checkpoint for ckpt_kind={ckpt_kind!r}. "
        f"Checked CONFIG['hard_ckpt_path'], metadata key {meta_key!r}, and standard run-directory fallbacks."
    )


def _resolve_trial_params(meta: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [
        meta.get("trial_params"),
        meta.get("trial_params_effective"),
        meta.get("best_params_effective"),
        meta.get("best_params"),
    ]
    for item in candidates:
        if isinstance(item, dict) and len(item) > 0:
            return dict(item)
    raise RuntimeError("Could not find model trial parameters in metadata.")


def _resolve_save_output_dir() -> str:
    explicit = str(CONFIG.get("save_output_dir", "")).strip()
    if explicit:
        return os.path.abspath(explicit)
    fallback = str(CONFIG.get("finetune_dir", "")).strip()
    if fallback == "":
        raise RuntimeError(
            "Could not resolve save output directory. Set CONFIG['save_output_dir'] "
            "explicitly when running without CONFIG['finetune_dir']."
        )
    return os.path.abspath(fallback)


def _infer_native_len(eval_dir: str) -> int:
    import h5py
    import glob

    files = sorted(glob.glob(os.path.join(eval_dir, "packed_*.h5")))
    if len(files) == 0:
        raise RuntimeError(f"No packed_*.h5 files found in {eval_dir}")

    with h5py.File(files[0], "r") as hf:
        native_len = int(hf["data"].shape[-1])
    return native_len


def _resolve_crop_len(meta: Dict[str, Any], eval_dir: str) -> int:
    crop_len = int(CONFIG.get("crop_len", 0) or 0)
    if crop_len > 0:
        return crop_len

    cfg = meta.get("config", {})
    maybe = int(cfg.get("crop_len", 0) or 0)
    if maybe > 0:
        return maybe

    return _infer_native_len(eval_dir)


def _resolve_eval_crop_mode(mod, meta: Dict[str, Any]) -> str:
    requested = str(CONFIG.get("eval_crop_mode", "auto")).strip().lower()
    if requested == "auto":
        fallback = meta.get("config", {}).get("eval_crop_mode", meta.get("eval_crop_mode", "center"))
        resolved = mod.normalize_eval_crop_mode(fallback)
    else:
        resolved = mod.normalize_eval_crop_mode(requested)
    print(f"[INFO] eval_crop_mode: requested={requested} resolved={resolved}")
    return resolved


def _resolve_top_k(meta: Dict[str, Any]) -> int:
    top_k = int(CONFIG.get("top_k", 0) or 0)
    if top_k > 0:
        return top_k
    return int(meta.get("config", {}).get("top_k", 40))


def _resolve_use_capped_subject_eer(meta: Dict[str, Any], mod) -> bool:
    override = CONFIG.get("use_capped_subject_eer", None)
    if override is None:
        return bool(meta.get("config", {}).get("use_capped_subject_eer", getattr(mod, "USE_CAPPED_SUBJECT_EER", False)))
    return bool(override)


def _infer_dataset_channel_names(mod, eval_dir: str, crop_len: int, eval_crop_mode: str) -> List[str]:
    dataset = mod.PackedLayoutDataset2(
        eval_dir,
        crop_len=crop_len,
        random_crop=False,
        deterministic_crop_mode=eval_crop_mode,
    )
    channel_names: List[str] = []
    seen = set()
    for layout_id in dataset.layout_ids:
        for ch in mod.parse_layout_str(str(layout_id)):
            if not ch or ch in seen:
                continue
            seen.add(ch)
            channel_names.append(ch)

    if len(channel_names) == 0:
        raise RuntimeError(f"Could not infer any valid channels from eval_dir: {eval_dir}")
    return channel_names


def _resolve_channel_names(mod, eval_dir: str, crop_len: int, eval_crop_mode: str) -> List[str]:
    cfg_names = CONFIG.get("channel_names", [])
    if not cfg_names:
        return _infer_dataset_channel_names(mod, eval_dir=eval_dir, crop_len=crop_len, eval_crop_mode=eval_crop_mode)

    resolved: List[str] = []
    seen = set()
    for raw_name in cfg_names:
        norm = mod.normalize_label(str(raw_name))
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        resolved.append(norm)

    if not resolved:
        raise RuntimeError("CONFIG['channel_names'] did not resolve to any valid canonical channels.")
    return resolved


def _channel_region_anatomical_6(channel_name: str) -> str:
    ch = str(channel_name).upper().strip()
    if ch.startswith(("FP", "AF")):
        return "frontopolar_prefrontal"
    if ch.startswith(("F", "FC")):
        return "frontal"
    if ch.startswith(("C", "CZ")):
        return "central"
    if ch.startswith(("T", "FT", "TP")):
        return "temporal"
    if ch.startswith(("CP", "P", "PZ")):
        return "parietal"
    if ch.startswith(("PO", "O", "OZ", "I", "IZ")):
        return "occipital"
    return "other"


def _channel_region_hemisphere_3(channel_name: str) -> str:
    ch = str(channel_name).upper().strip()
    digits = "".join(c for c in ch if c.isdigit())
    if digits:
        try:
            last_digit = int(digits[-1])
            return "left" if (last_digit % 2 == 1) else "right"
        except ValueError:
            pass
    return "midline"


def _build_region_groups(channel_names: Sequence[str], scheme: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for ch in channel_names:
        if scheme == "anatomical_6":
            region = _channel_region_anatomical_6(ch)
        elif scheme == "hemisphere_3":
            region = _channel_region_hemisphere_3(ch)
        else:
            raise ValueError(f"Unsupported region scheme: {scheme}")
        if region == "other":
            print(f"[WARN] channel '{ch}' did not match a named anatomical_6 region.")
            continue
        groups.setdefault(region, []).append(str(ch))
    return groups


def _resolve_region_schemes() -> List[str]:
    raw = CONFIG.get("region_schemes", [])
    if raw is None:
        return ["anatomical_6", "hemisphere_3"]
    if not isinstance(raw, list) or len(raw) == 0:
        raise RuntimeError("CONFIG['region_schemes'] must be a non-empty list when provided.")

    valid = {"anatomical_6", "hemisphere_3"}
    resolved: List[str] = []
    seen = set()
    for item in raw:
        name = str(item).strip().lower()
        if name == "":
            continue
        if name not in valid:
            raise ValueError(f"Unsupported region scheme '{item}'. Valid values: {sorted(valid)}")
        if name in seen:
            continue
        seen.add(name)
        resolved.append(name)
    if len(resolved) == 0:
        raise RuntimeError("CONFIG['region_schemes'] did not contain any valid scheme names.")
    return resolved


def _parse_bilateral_key(channel_name: str) -> Optional[Tuple[str, int]]:
    ch = str(channel_name).upper().strip()
    m = re.fullmatch(r"([A-Z]+)(\d+)", ch)
    if m is None:
        return None
    stem = str(m.group(1))
    number = int(m.group(2))
    return stem, number


def _build_region_pair_rows(channel_names: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Build deterministic left/right symmetric pairs such as F3/F4, F7/F8, FC5/FC6.
    Pairs are grouped by the anatomical_6 region of their channel family.
    """
    by_key: Dict[Tuple[str, int], str] = {}
    for ch in channel_names:
        parsed = _parse_bilateral_key(ch)
        if parsed is None:
            continue
        by_key[parsed] = str(ch)

    rows: List[Dict[str, Any]] = []
    seen_pairs = set()
    for (stem, number), left_name in by_key.items():
        if number % 2 == 0:
            continue
        right_key = (stem, number + 1)
        right_name = by_key.get(right_key)
        if right_name is None:
            continue

        pair_key = (left_name, right_name)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        region = _channel_region_anatomical_6(left_name)
        if region == "other":
            print(f"[WARN] skipping unmatched bilateral pair {left_name}/{right_name}")
            continue
        rows.append(
            {
                "scheme": "anatomical_6",
                "region": region,
                "pair_name": f"{left_name}_{right_name}",
                "channels": [left_name, right_name],
            }
        )

    rows.sort(key=lambda row: (str(row["region"]), str(row["pair_name"])))
    return rows


def _resolve_random_three_channel_trials() -> int:
    trials = int(CONFIG.get("random_three_channel_trials", 20) or 0)
    if trials <= 0:
        raise RuntimeError("CONFIG['random_three_channel_trials'] must be a positive integer.")
    return trials


def _resolve_random_seed() -> int:
    return int(CONFIG.get("random_seed", 101))


def _build_random_three_channel_rows(
    channel_names: Sequence[str],
    trials: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if len(channel_names) < 3:
        raise RuntimeError("random_three_channels requires at least 3 available channels.")

    max_unique = math.comb(len(channel_names), 3)
    if trials > max_unique:
        print(
            f"[WARN] random_three_channel_trials={trials} exceeds unique 3-channel combinations={max_unique}. "
            f"Clamping to {max_unique}."
        )
        trials = max_unique

    rng = np.random.default_rng(seed)
    chosen = set()
    rows: List[Dict[str, Any]] = []

    while len(rows) < trials:
        sample = tuple(sorted(rng.choice(channel_names, size=3, replace=False).tolist()))
        if sample in chosen:
            continue
        chosen.add(sample)
        rows.append(
            {
                "trial_index": len(rows),
                "subset_name": "__".join(sample),
                "target_channel": f"random3_{len(rows):03d}",
                "channels": list(sample),
            }
        )

    return rows


def _resolve_experiments() -> List[str]:
    raw = CONFIG.get("experiments", [])
    if not isinstance(raw, list) or len(raw) == 0:
        raise RuntimeError("CONFIG['experiments'] must be a non-empty list.")

    valid = {"leave_one_out", "single_channel_only", "random_three_channels", "region_ablation", "region_pair_only", "zero_pos3d"}
    resolved: List[str] = []
    seen = set()
    for item in raw:
        name = str(item).strip().lower()
        if name == "":
            continue
        if name not in valid:
            raise ValueError(f"Unsupported experiment '{item}'. Valid values: {sorted(valid)}")
        if name in seen:
            continue
        seen.add(name)
        resolved.append(name)

    if len(resolved) == 0:
        raise RuntimeError("CONFIG['experiments'] did not contain any valid experiment names.")
    return resolved


def _build_eval_loader(
    mod,
    eval_dir: str,
    crop_len: int,
    eval_crop_mode: str,
    keep_channels: Sequence[str],
    max_3d_dist: float,
):
    test_ds = mod.PackedLayoutDataset2(
        eval_dir,
        crop_len=crop_len,
        random_crop=False,
        deterministic_crop_mode=eval_crop_mode,
    )
    layout_cache = mod.build_layout_cache(test_ds)
    collate_fn = partial(
        mod.collate_mixed2,
        keep_channels=list(keep_channels),
        max_3d_dist=float(max_3d_dist),
        layout_cache=layout_cache,
    )
    loader = mod.DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
        **mod._loader_perf_kwargs(mod.NUM_WORKERS_EVAL, pin_memory=True),
    )
    return test_ds, loader


def collect_embedding_arrays(mod, loader, model, device, embed_batch_size: int, zero_pos3d: bool = False):
    """
    Run the model batch-by-batch and keep only embeddings plus the metadata
    needed for enrollment/verification and channel-presence summaries.
    """
    embs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ss: List[np.ndarray] = []

    active_counts: Optional[np.ndarray] = None
    total_samples = 0

    model.eval()
    use_autocast = (device.type == "cuda")
    batch_size = max(1, int(embed_batch_size))

    with mod.torch.no_grad():
        for eeg, mask, labels, pos3d, meta in mod.tqdm(loader, desc="embed", leave=False):
            B = int(eeg.shape[0])
            sess = np.asarray(meta["session_id"], dtype=np.int64)
            mask_np = mask.numpy().astype(np.int64, copy=False)

            if active_counts is None:
                active_counts = np.zeros(mask_np.shape[1], dtype=np.int64)
            active_counts += mask_np.sum(axis=0)
            total_samples += B

            for start in range(0, B, batch_size):
                end = min(start + batch_size, B)
                x_t = eeg[start:end].to(device, non_blocking=True)
                m_t = mask[start:end].to(device, non_blocking=True)
                p_t = pos3d[start:end].to(device, non_blocking=True)
                if zero_pos3d:
                    p_t = mod.torch.zeros_like(p_t)

                ctx = mod.amp.autocast(device_type="cuda") if use_autocast else mod.nullcontext()
                with ctx:
                    out = model(x_t, m_t, pos3d=p_t)

                embs.append(out.float().cpu().numpy())

            ys.append(labels.numpy())
            ss.append(sess)

    if len(embs) == 0 or active_counts is None or total_samples <= 0:
        raise RuntimeError("No embeddings were collected from the evaluation loader.")

    active_fraction = active_counts.astype(np.float64) / float(total_samples)
    return (
        np.concatenate(embs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(ss, axis=0),
        active_fraction,
    )


def evaluate_avg_eer_from_embedding_arrays(mod, E_all, Y_all, S_all, top_k: int):
    enroll_idxs, verify_idxs = mod.get_enrollment_verification_indices(Y_all, S_all)
    en_emb = E_all[enroll_idxs]
    ve_emb = E_all[verify_idxs]
    y_enroll = Y_all[enroll_idxs]
    y_verify = Y_all[verify_idxs]

    sim_dict = mod.compute_similarity_scores(
        en_emb,
        y_enroll,
        ve_emb,
        y_verify,
        distance="cd",
        top_k=top_k,
    )
    avg_eer, std_eer = mod.evaluate_eer_per_class(sim_dict)
    return float(avg_eer), float(std_eer)


def _evaluate_subset(
    mod,
    model,
    device,
    eval_dir: str,
    crop_len: int,
    eval_crop_mode: str,
    keep_channels: Sequence[str],
    max_3d_dist: float,
    top_k: int,
    zero_pos3d: bool = False,
) -> Dict[str, Any]:
    test_ds, test_loader = _build_eval_loader(
        mod,
        eval_dir=eval_dir,
        crop_len=crop_len,
        eval_crop_mode=eval_crop_mode,
        keep_channels=keep_channels,
        max_3d_dist=max_3d_dist,
    )

    E, Y, S, active_fraction = collect_embedding_arrays(
        mod,
        test_loader,
        model,
        device,
        embed_batch_size=mod.CACHE_EVAL_BATCH_SIZE,
        zero_pos3d=zero_pos3d,
    )

    if float(active_fraction.sum()) <= 0.0:
        eer = float("nan")
        std = float("nan")
        status = "no_active_channels"
    else:
        eer, std = evaluate_avg_eer_from_embedding_arrays(
            mod,
            E,
            Y,
            S,
            top_k=top_k,
        )
        status = "ok"

    shutdown_fn = getattr(mod, "_shutdown_dataloader", None)
    if shutdown_fn is not None:
        shutdown_fn(test_loader)

    test_loader = None
    test_ds = None
    E = Y = S = None
    gc.collect()
    if device.type == "cuda":
        mod.torch.cuda.empty_cache()

    return {
        "eer": float(eer),
        "std": float(std),
        "status": status,
        "active_fraction_mean": float(np.mean(active_fraction)),
        "active_fraction_min": float(np.min(active_fraction)),
        "active_fraction_max": float(np.max(active_fraction)),
    }


def _make_row(
    experiment: str,
    subset_name: str,
    kept_channels: Sequence[str],
    dropped_channels: Sequence[str],
    target_channel: str,
    result: Dict[str, Any],
    baseline_eer: float,
    baseline_std: float,
) -> Dict[str, Any]:
    eer = float(result["eer"])
    row = {
        "experiment": str(experiment),
        "subset_name": str(subset_name),
        "target_channel": str(target_channel),
        "n_channels": int(len(kept_channels)),
        "eer": eer,
        "std": float(result["std"]),
        "delta_vs_baseline": float(eer - baseline_eer) if np.isfinite(eer) and np.isfinite(baseline_eer) else float("nan"),
        "baseline_eer": float(baseline_eer),
        "baseline_std": float(baseline_std),
        "active_fraction_mean": float(result["active_fraction_mean"]),
        "active_fraction_min": float(result["active_fraction_min"]),
        "active_fraction_max": float(result["active_fraction_max"]),
        "status": str(result["status"]),
        "kept_channels_json": json.dumps(list(kept_channels)),
        "dropped_channels_json": json.dumps(list(dropped_channels)),
    }
    return row


def _save_experiment_csvs(
    out_dir: str,
    save_csv_prefix: str,
    baseline_row: Dict[str, Any],
    experiment_rows: Dict[str, List[Dict[str, Any]]],
) -> None:
    fieldnames = [
        "experiment",
        "subset_name",
        "target_channel",
        "n_channels",
        "eer",
        "std",
        "delta_vs_baseline",
        "baseline_eer",
        "baseline_std",
        "active_fraction_mean",
        "active_fraction_min",
        "active_fraction_max",
        "status",
        "kept_channels_json",
        "dropped_channels_json",
    ]

    for experiment, rows in experiment_rows.items():
        csv_path = os.path.join(out_dir, f"{save_csv_prefix}_{experiment}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(baseline_row)
            writer.writerows(rows)
        print("[INFO] saved csv:", csv_path)
        print("[INFO] rows:", 1 + len(rows))


def run() -> None:
    finetune_dir = os.path.abspath(str(CONFIG.get("finetune_dir", "")).strip()) if str(CONFIG.get("finetune_dir", "")).strip() else ""
    meta_json = _resolve_meta_json()
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    trial_params = _resolve_trial_params(meta)
    ckpt_path = _resolve_ckpt_path(meta, str(CONFIG["ckpt_kind"]).strip().lower())
    eval_dir_cfg = str(CONFIG.get("eval_dir", "")).strip()
    eval_dir = os.path.abspath(eval_dir_cfg) if eval_dir_cfg else os.path.abspath(str(meta["config"]["test_dir"]))
    save_output_dir = _resolve_save_output_dir()
    os.makedirs(save_output_dir, exist_ok=True)

    mod = _import_eval_module(str(CONFIG["cuda_devices"]))
    device, device_info = _resolve_device(mod)
    crop_len = _resolve_crop_len(meta, eval_dir)
    eval_crop_mode = _resolve_eval_crop_mode(mod, meta)
    top_k = _resolve_top_k(meta)

    mod.NUM_WORKERS_EVAL = int(meta.get("config", {}).get("num_workers_eval", 2))
    mod.CACHE_EVAL_BATCH_SIZE = int(meta.get("config", {}).get("cache_eval_batch_size", 64))
    mod.TOP_K = top_k
    mod.USE_TORCH_COMPILE = False
    mod.USE_CAPPED_SUBJECT_EER = _resolve_use_capped_subject_eer(meta, mod)

    channel_names = _resolve_channel_names(
        mod,
        eval_dir=eval_dir,
        crop_len=crop_len,
        eval_crop_mode=eval_crop_mode,
    )
    region_schemes = _resolve_region_schemes()
    region_groups_by_scheme = {scheme: _build_region_groups(channel_names, scheme=scheme) for scheme in region_schemes}
    region_pair_rows = _build_region_pair_rows(channel_names)
    experiments = _resolve_experiments()
    random_three_channel_trials = _resolve_random_three_channel_trials() if "random_three_channels" in experiments else 0
    random_seed = _resolve_random_seed() if "random_three_channels" in experiments else 0
    random_three_channel_rows = (
        _build_random_three_channel_rows(
            channel_names,
            trials=random_three_channel_trials,
            seed=random_seed,
        )
        if "random_three_channels" in experiments
        else []
    )
    save_csv_prefix = str(CONFIG.get("save_csv_prefix", "channel_level")).strip() or "channel_level"
    max_3d_dist = 0.0 if bool(CONFIG.get("exact_match_only", True)) else float(mod.DEFAULT_MAX_3D_DIST)

    if finetune_dir != "":
        print("[INFO] finetune_dir:", finetune_dir)
    print("[INFO] meta_json:", meta_json)
    print("[INFO] checkpoint:", ckpt_path)
    print("[INFO] eval_dir:", eval_dir)
    print("[INFO] save_output_dir:", save_output_dir)
    print("[INFO] crop_len:", crop_len)
    print("[INFO] top_k:", top_k)
    print("[INFO] exact_match_only:", bool(CONFIG.get("exact_match_only", True)))
    print("[INFO] max_3d_dist:", max_3d_dist)
    print("[INFO] channel_count:", len(channel_names))
    print("[INFO] experiments:", experiments)
    print("[INFO] zero_pos3d meaning: explicit pos3d is replaced with zeros; channel slot order stays unchanged")
    print("[INFO] region_schemes:", region_schemes)
    print("[INFO] region_groups:", {scheme: {k: len(v) for k, v in groups.items()} for scheme, groups in region_groups_by_scheme.items()})
    print("[INFO] region_pair_count:", len(region_pair_rows))
    if "random_three_channels" in experiments:
        print("[INFO] random_three_channel_trials:", random_three_channel_trials)
        print("[INFO] random_seed:", random_seed)
    print("[INFO] device:", device)
    print("[INFO] device_info:", device_info)
    print("[INFO] trial_params:", trial_params)

    model, embed_dim = mod.build_model(
        trial=None,
        num_channels=len(mod.REF_CLIST_NORM),
        device=device,
        fixed_params=trial_params,
    )
    state = mod.torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("[INFO] loaded checkpoint with embed_dim:", embed_dim)

    experiment_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in experiments}

    print("[EVAL] baseline full canonical subset")
    baseline_result = _evaluate_subset(
        mod,
        model,
        device,
        eval_dir=eval_dir,
        crop_len=crop_len,
        eval_crop_mode=eval_crop_mode,
        keep_channels=channel_names,
        max_3d_dist=max_3d_dist,
        top_k=top_k,
    )
    baseline_eer = float(baseline_result["eer"])
    baseline_std = float(baseline_result["std"])
    baseline_row = _make_row(
        experiment="baseline",
        subset_name="full_canonical",
        kept_channels=channel_names,
        dropped_channels=[],
        target_channel="",
        result=baseline_result,
        baseline_eer=baseline_eer,
        baseline_std=baseline_std,
    )
    print(f"[RESULT] baseline -> EER={baseline_eer:.4f} +- {baseline_std:.4f}")

    if "leave_one_out" in experiments:
        print("[EVAL] leave-one-channel-out")
        for ch in channel_names:
            keep = [c for c in channel_names if c != ch]
            result = _evaluate_subset(
                mod,
                model,
                device,
                eval_dir=eval_dir,
                crop_len=crop_len,
                eval_crop_mode=eval_crop_mode,
                keep_channels=keep,
                max_3d_dist=max_3d_dist,
                top_k=top_k,
            )
            experiment_rows["leave_one_out"].append(
                _make_row(
                    experiment="leave_one_out",
                    subset_name=f"full_minus_{ch}",
                    kept_channels=keep,
                    dropped_channels=[ch],
                    target_channel=ch,
                    result=result,
                    baseline_eer=baseline_eer,
                    baseline_std=baseline_std,
                )
            )
            print(
                f"[RESULT] leave_one_out {ch:>6s} -> "
                f"EER={result['eer']:.4f} +- {result['std']:.4f} "
                f"(delta={float(result['eer']) - baseline_eer:+.4f})"
            )

    if "zero_pos3d" in experiments:
        print("[EVAL] zero-pos3d full-channel control")
        result = _evaluate_subset(
            mod,
            model,
            device,
            eval_dir=eval_dir,
            crop_len=crop_len,
            eval_crop_mode=eval_crop_mode,
            keep_channels=channel_names,
            max_3d_dist=max_3d_dist,
            top_k=top_k,
            zero_pos3d=True,
        )
        experiment_rows["zero_pos3d"].append(
            _make_row(
                experiment="zero_pos3d",
                subset_name="full_channels_zero_pos3d",
                kept_channels=channel_names,
                dropped_channels=[],
                target_channel="all_channels_zero_pos3d",
                result=result,
                baseline_eer=baseline_eer,
                baseline_std=baseline_std,
            )
        )
        print(
            f"[RESULT] zero_pos3d -> "
            f"EER={result['eer']:.4f} +- {result['std']:.4f} "
            f"(delta={float(result['eer']) - baseline_eer:+.4f})"
        )

    if "single_channel_only" in experiments:
        print("[EVAL] single-channel-only")
        for ch in channel_names:
            keep = [ch]
            result = _evaluate_subset(
                mod,
                model,
                device,
                eval_dir=eval_dir,
                crop_len=crop_len,
                eval_crop_mode=eval_crop_mode,
                keep_channels=keep,
                max_3d_dist=max_3d_dist,
                top_k=top_k,
            )
            experiment_rows["single_channel_only"].append(
                _make_row(
                    experiment="single_channel_only",
                    subset_name=ch,
                    kept_channels=keep,
                    dropped_channels=[c for c in channel_names if c != ch],
                    target_channel=ch,
                    result=result,
                    baseline_eer=baseline_eer,
                    baseline_std=baseline_std,
                )
            )
            print(
                f"[RESULT] single_channel_only {ch:>6s} -> "
                f"EER={result['eer']:.4f} +- {result['std']:.4f}"
            )

    if "random_three_channels" in experiments:
        print("[EVAL] random three-channel subsets")
        for triplet_row in random_three_channel_rows:
            keep = list(triplet_row["channels"])
            result = _evaluate_subset(
                mod,
                model,
                device,
                eval_dir=eval_dir,
                crop_len=crop_len,
                eval_crop_mode=eval_crop_mode,
                keep_channels=keep,
                max_3d_dist=max_3d_dist,
                top_k=top_k,
            )
            dropped = [c for c in channel_names if c not in set(keep)]
            experiment_rows["random_three_channels"].append(
                _make_row(
                    experiment="random_three_channels",
                    subset_name=triplet_row["subset_name"],
                    kept_channels=keep,
                    dropped_channels=dropped,
                    target_channel=str(triplet_row["target_channel"]),
                    result=result,
                    baseline_eer=baseline_eer,
                    baseline_std=baseline_std,
                )
            )
            print(
                f"[RESULT] random_three_channels {triplet_row['target_channel']} -> "
                f"channels={keep} "
                f"EER={result['eer']:.4f} +- {result['std']:.4f}"
            )

    if "region_ablation" in experiments:
        print("[EVAL] region ablation")
        for scheme_name, region_groups in region_groups_by_scheme.items():
            print(f"[EVAL] region scheme={scheme_name}")
            for region_name, drop_channels in region_groups.items():
                if len(drop_channels) == 0:
                    continue

                drop_set = set(drop_channels)
                keep = [c for c in channel_names if c not in drop_set]
                if len(keep) == 0:
                    print(f"[WARN] region_ablation {scheme_name}:{region_name} would remove all channels. Skipping.")
                    continue

                result = _evaluate_subset(
                    mod,
                    model,
                    device,
                    eval_dir=eval_dir,
                    crop_len=crop_len,
                    eval_crop_mode=eval_crop_mode,
                    keep_channels=keep,
                    max_3d_dist=max_3d_dist,
                    top_k=top_k,
                )
                experiment_rows["region_ablation"].append(
                    _make_row(
                        experiment="region_ablation",
                        subset_name=f"{scheme_name}__drop_{region_name}",
                        kept_channels=keep,
                        dropped_channels=drop_channels,
                        target_channel=f"{scheme_name}__{region_name}",
                        result=result,
                        baseline_eer=baseline_eer,
                        baseline_std=baseline_std,
                    )
                )
                print(
                    f"[RESULT] region_ablation {scheme_name}:{region_name} -> "
                    f"EER={result['eer']:.4f} +- {result['std']:.4f} "
                    f"(delta={float(result['eer']) - baseline_eer:+.4f}; dropped={drop_channels})"
                )

    if "region_pair_only" in experiments:
        print("[EVAL] region pair only")
        for pair_row in region_pair_rows:
            keep = list(pair_row["channels"])
            result = _evaluate_subset(
                mod,
                model,
                device,
                eval_dir=eval_dir,
                crop_len=crop_len,
                eval_crop_mode=eval_crop_mode,
                keep_channels=keep,
                max_3d_dist=max_3d_dist,
                top_k=top_k,
            )
            dropped = [c for c in channel_names if c not in set(keep)]
            experiment_rows["region_pair_only"].append(
                _make_row(
                    experiment="region_pair_only",
                    subset_name=f"{pair_row['scheme']}__{pair_row['region']}__{pair_row['pair_name']}",
                    kept_channels=keep,
                    dropped_channels=dropped,
                    target_channel=f"{pair_row['scheme']}__{pair_row['region']}__{pair_row['pair_name']}",
                    result=result,
                    baseline_eer=baseline_eer,
                    baseline_std=baseline_std,
                )
            )
            print(
                f"[RESULT] region_pair_only {pair_row['scheme']}:{pair_row['region']}:{pair_row['pair_name']} -> "
                f"EER={result['eer']:.4f} +- {result['std']:.4f}"
            )

    _save_experiment_csvs(
        out_dir=save_output_dir,
        save_csv_prefix=save_csv_prefix,
        baseline_row=baseline_row,
        experiment_rows=experiment_rows,
    )


if __name__ == "__main__":
    run()
