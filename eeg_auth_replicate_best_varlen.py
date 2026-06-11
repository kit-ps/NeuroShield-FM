#!/usr/bin/env python3
"""
Replicate the best Optuna trial using the exact training pipeline in
`eeg_auth_train_optuna.py`.

This script is config-driven (no CLI). Edit CONFIG and run:
  python eeg_auth_replicate_best_varlen.py
"""

import copy
import csv
import importlib
import json
import os
import sys
import gc
from functools import partial
from typing import Any, Dict

import optuna

TRAIN_MODULE_NAME = "eeg_auth_train_optuna"


CONFIG: Dict[str, Any] = {
    "cuda_devices": "0",
    "study_db": "bw_optuna.db",  # path or sqlite:/// URL
    "study_name": "BrainAuth_SupConLoss",  # should match training study name
    "train_dir": "./data/train_packed",
    "val_dir": "./data/val_packed",
    "test_dir": "./data/test_packed",
    "save_dir": "./ckpts_bw_replicate_varlen",
    "best_ckpt_name": "replicate_best_trial_varlen_best.pth",
    "final_ckpt_name": "replicate_best_trial_varlen_final.pth",
    "log_csv_name": "replicate_best_trial_varlen_log.csv",
    "meta_json_name": "replicate_best_trial_varlen_meta.json",
    "crop_len": 500,
    "eval_crop_mode": "center",
    "num_workers_train": 15,
    "num_workers_eval": 2,
    "cache_eval_batch_size": 64,
    "train_batch_size": 128,
    "top_k": 40,
    "seed": 42,
    "tuning_epochs": 99,
    "compile": False,
    "cuda_alloc_conf": "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.5",
}


def _normalize_db_url(study_db: str) -> str:
    return study_db if "://" in study_db else f"sqlite:///{study_db}"


def _load_study(study_db: str, study_name: str) -> optuna.Study:
    storage = optuna.storages.RDBStorage(
        url=_normalize_db_url(study_db),
        engine_kwargs={"connect_args": {"timeout": 600, "check_same_thread": False}},
    )
    return optuna.load_study(study_name=study_name, storage=storage)


def _import_train_module(cuda_devices: str):
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
    alloc_conf = str(CONFIG.get("cuda_alloc_conf", "")).strip()
    if alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)

    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], "--cuda_devices", str(cuda_devices)]
    try:
        if TRAIN_MODULE_NAME in sys.modules:
            mod = importlib.reload(sys.modules[TRAIN_MODULE_NAME])
        else:
            mod = importlib.import_module(TRAIN_MODULE_NAME)
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


def _sync_train_module_config(mod) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG["cuda_devices"])
    mod.args.cuda_devices = str(CONFIG["cuda_devices"])
    mod.args.train_dir = str(CONFIG["train_dir"])
    mod.args.val_dir = str(CONFIG["val_dir"])
    if str(CONFIG.get("test_dir", "")).strip():
        mod.args.test_dir = str(CONFIG["test_dir"])
    mod.args.crop_len = int(CONFIG["crop_len"])
    mod.args.eval_crop_mode = mod.normalize_eval_crop_mode(CONFIG.get("eval_crop_mode", "center"))
    mod.args.seed = int(CONFIG["seed"])
    mod.args.tuning_epochs = int(CONFIG["tuning_epochs"])

    # Overwrite in-module knobs so replication honors CONFIG.
    mod.NUM_WORKERS_TRAIN = int(CONFIG["num_workers_train"])
    mod.NUM_WORKERS_EVAL = int(CONFIG["num_workers_eval"])
    mod.CACHE_EVAL_BATCH_SIZE = int(CONFIG["cache_eval_batch_size"])
    if hasattr(mod, "TRAIN_BATCH_SIZE"):
        mod.TRAIN_BATCH_SIZE = int(CONFIG.get("train_batch_size", mod.TRAIN_BATCH_SIZE))
    mod.TOP_K = int(CONFIG["top_k"])
    mod.USE_TORCH_COMPILE = bool(CONFIG["compile"])


def _required_trial_keys() -> set:
    return {
        "embedder",
        "embed_dim",
        "channel_depth",
        "temporal_depth",
        "lr",
        "weight_decay",
        "patch_len",
    }


def _shutdown_loader(mod, loader) -> None:
    fn = getattr(mod, "_shutdown_dataloader", None)
    if fn is not None:
        fn(loader)


def run() -> None:
    mod = _import_train_module(CONFIG["cuda_devices"])
    _sync_train_module_config(mod)

    device, device_info = _resolve_device(mod)
    print("[INFO] device:", device)
    print("[INFO] device_info:", device_info)

    study_db = str(CONFIG["study_db"])
    study_name = str(CONFIG["study_name"])
    study = _load_study(study_db, study_name)
    best_trial = study.best_trial
    trial_params = dict(best_trial.params)


    

    

    missing = sorted(_required_trial_keys() - set(trial_params.keys()))
    if missing:
        raise ValueError(f"Best trial params are missing required keys: {missing}.")

    bs_fixed = int(getattr(mod, "TRAIN_BATCH_SIZE", int(CONFIG.get("train_batch_size", 128))))
    if mod.is_disallowed_embed_batch_combo(int(trial_params["embed_dim"]), bs_fixed):
        raise RuntimeError("Best trial uses disallowed combo embed_dim=256 and batch_size=512.")

    run_seed = int(mod.args.seed) + int(best_trial.number)
    mod.seed_everything(run_seed)
    print("[INFO] study:", study_name)
    print("[INFO] best_trial_number:", best_trial.number)
    print("[INFO] best_trial_value:", float(best_trial.value))
    print("[INFO] run_seed:", run_seed)
    print("[INFO] fixed_train_batch_size:", bs_fixed)
    print("[INFO] trial_params:", trial_params)

    train_ds = mod.PackedLayoutDatasetNoCrop(mod.args.train_dir, crop_len=mod.args.crop_len, random_crop=True)
    val_ds = mod.PackedLayoutDataset2(
        mod.args.val_dir,
        crop_len=mod.args.crop_len,
        random_crop=False,
        deterministic_crop_mode=mod.args.eval_crop_mode,
    )
    test_enabled = bool(str(CONFIG.get("test_dir", "")).strip())
    test_ds = (
        mod.PackedLayoutDataset2(
            str(CONFIG["test_dir"]),
            crop_len=mod.args.crop_len,
            random_crop=False,
            deterministic_crop_mode=mod.args.eval_crop_mode,
        )
        if test_enabled
        else None
    )

    train_layout_cache = mod.build_layout_cache(train_ds)
    val_layout_cache = mod.build_layout_cache(val_ds)
    test_layout_cache = mod.build_layout_cache(test_ds) if test_ds is not None else None

    val_loader = mod.DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=partial(mod.collate_mixed2, fixed_Cmax=32, layout_cache=val_layout_cache),
        **mod._loader_perf_kwargs(mod.NUM_WORKERS_EVAL, pin_memory=True),
    )
    test_loader = None
    if test_ds is not None:
        test_loader = mod.DataLoader(
            test_ds,
            batch_size=256,
            shuffle=False,
            collate_fn=partial(mod.collate_mixed2, fixed_Cmax=32, layout_cache=test_layout_cache),
            **mod._loader_perf_kwargs(mod.NUM_WORKERS_EVAL, pin_memory=True),
        )

    # Cache fixed-length eval arrays once.
    Xv, Mv, Yv, Sv, Pv = mod.collect_arrays(val_loader, T_expected=mod.args.crop_len)
    Xt = Mt = Yt = St = Pt = None
    if test_loader is not None:
        Xt, Mt, Yt, St, Pt = mod.collect_arrays(test_loader, T_expected=mod.args.crop_len)
    _shutdown_loader(mod, val_loader)
    _shutdown_loader(mod, test_loader)
    val_loader = None
    test_loader = None
    val_ds = None
    test_ds = None
    gc.collect()

    model, embed_dim = mod.build_model(
        None,
        num_channels=len(mod.REF_CLIST_NORM),
        device=device,
        fixed_params=trial_params,
    )
    num_classes = int(train_ds.targets.unique().numel())
    loss_fn, miner_fn = mod.build_loss_and_miner(mod.CHOSEN_LOSS, num_classes, embed_dim, device)
    optimizer = mod.build_optimizer(None, model, loss_fn, fixed_params=trial_params)
    scaler = mod.amp.GradScaler(enabled=(device.type == "cuda"))

    bs = int(getattr(mod, "TRAIN_BATCH_SIZE", int(CONFIG.get("train_batch_size", 128))))
    new_iter = int(bs * mod.TUNING_LENGTH_MULTIPLIER)
    train_sampler = mod.LayoutAwareEpisodeMultiSessionBatchSampler(
        labels=train_ds.targets,
        sessions=train_ds.session_ids,
        layouts=train_ds.layout_ids,
        batch_size=bs,
        num_subjects_per_batch=mod.EPISODE_NUM_SUBJECTS,
        max_sessions_per_subject=mod.EPISODE_MAX_SESSIONS,
        length_before_new_iter=new_iter,
        prefer_multisession_subjects=mod.EPISODE_PREFER_MULTISESSION,
        no_duplicates=mod.EPISODE_NO_DUPLICATES,
        max_tries=mod.EPISODE_MAX_TRIES,
    )
    train_loader = mod.DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=partial(mod.collate_mixed2, layout_cache=train_layout_cache),
        **mod._loader_perf_kwargs(mod.NUM_WORKERS_TRAIN, pin_memory=True),
    )
    iters_per_epoch = len(train_loader)

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    best_ckpt = os.path.join(CONFIG["save_dir"], CONFIG["best_ckpt_name"])
    final_ckpt = os.path.join(CONFIG["save_dir"], CONFIG["final_ckpt_name"])
    log_csv = os.path.join(CONFIG["save_dir"], CONFIG["log_csv_name"])
    meta_json = os.path.join(CONFIG["save_dir"], CONFIG["meta_json_name"])

    logs = []
    best_val = float("inf")
    best_ep = -1
    best_test = float("nan")
    best_test_std = float("nan")

    print(
        f"[INFO] Replication start: epochs={mod.args.tuning_epochs}, iters={iters_per_epoch}, "
        f"bs={bs}, loss={mod.CHOSEN_LOSS} (val/test evaluated every epoch)"
    )
    for ep in range(1, int(mod.args.tuning_epochs) + 1):
        tr_loss = mod.train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, miner_fn, device)
        val_eer, val_std = mod.evaluate_avg_eer_from_cached(
            model, device, Xv, Mv, Yv, Sv, Pv, top_k=mod.TOP_K
        )
        test_eer = float("nan")
        test_std = float("nan")
        if Xt is not None:
            test_eer, test_std = mod.evaluate_avg_eer_from_cached(
                model, device, Xt, Mt, Yt, St, Pt, top_k=mod.TOP_K
            )

        improved = val_eer < best_val
        if improved:
            best_val = float(val_eer)
            best_ep = int(ep)
            best_test = float(test_eer)
            best_test_std = float(test_std)
            mod.torch.save(model.state_dict(), best_ckpt)

        logs.append((ep, tr_loss, val_eer, val_std, test_eer, test_std, int(improved)))
        print(
            f"[E{ep:03d}] iters={iters_per_epoch} bs={bs} "
            f"train_loss={tr_loss:.4f} val_eer={val_eer:.4f}+-{val_std:.4f} "
            f"test_eer={test_eer:.4f}+-{test_std:.4f} improved={improved}"
        )

    _shutdown_loader(mod, train_loader)
    train_loader = None
    train_sampler = None
    train_ds = None
    gc.collect()

    mod.torch.save(model.state_dict(), final_ckpt)

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_eer", "val_std", "test_eer", "test_std", "is_improved"])
        writer.writerows(logs)

    meta = {
        "study_db": study_db,
        "study_name": study_name,
        "best_trial_number": int(best_trial.number),
        "best_trial_value": float(best_trial.value),
        "run_seed": run_seed,
        "trial_params_effective": trial_params,
        "eval_crop_mode": str(mod.args.eval_crop_mode),
        "best_val_eer_replicated": best_val,
        "best_epoch_replicated": best_ep,
        "best_test_eer_replicated": best_test,
        "best_test_std_replicated": best_test_std,
        "best_ckpt": best_ckpt,
        "final_ckpt": final_ckpt,
        "log_csv": log_csv,
        "module": TRAIN_MODULE_NAME,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] Saved best checkpoint:", best_ckpt)
    print("[INFO] Saved final checkpoint:", final_ckpt)
    print("[INFO] Saved log:", log_csv)
    print("[INFO] Saved meta:", meta_json)
    print(
        f"[RESULT] replicated_best_val_eer={best_val:.6f} "
        f"(study_best={float(best_trial.value):.6f}) best_epoch={best_ep}"
    )

    global RUN_OUTPUT
    RUN_OUTPUT = copy.deepcopy(meta)
    RUN_OUTPUT["logs"] = logs


if __name__ == "__main__":
    run()
