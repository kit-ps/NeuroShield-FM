#!/usr/bin/env python3
"""
Finetune a pretrained checkpoint using the current `eeg_auth_train_optuna.py`
pipeline (patch-only + ALiBi temporal encoder + coord-MLP channel PE).

Edit CONFIG / FINETUNE / EPISODE and run:
  python eeg_auth_finetune.py

Notebook behavior:
  - if executed inside Jupyter (for example, via %run), the script auto-builds
    the session and exposes objects (model/optimizer/loaders) without forcing
    full finetuning.
"""

import copy
import csv
import gc
import importlib
import json
import os
import sys
from functools import partial
from typing import Any, Dict, Optional, Tuple


TRAIN_MODULE_NAME = "eeg_auth_train_optuna"
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.5",
)


CONFIG: Dict[str, Any] = {
    "cuda_devices": "0",
    "pretrained_ckpt": "./ckpts_bw_replicate_varlen/replicate_best_trial_varlen_final.pth",
    "replica_meta_json": "./ckpts_bw_replicate_varlen/replicate_best_trial_varlen_meta.json",
    "train_dir": "./data/downstream/train_packed",
    "val_dir": "./data/downstream/val_packed",
    "test_dir": "./data/downstream/test_packed",  # optional: set "" to disable
    "save_dir": "./ckpts_bw_finetune",
    "best_ckpt_name": "finetune_best.pth",
    "final_ckpt_name": "finetune_final.pth",
    "log_csv_name": "finetune_log.csv",
    "meta_json_name": "finetune_meta.json",
    "crop_len": 500,
    "eval_crop_mode": "center",
    "num_workers_train": 15,
    "num_workers_eval": 2,
    "cache_eval_batch_size": 64,
    "top_k": 40,
    "max_native_channels": 64,
    "seed": 42,
    "compile": False,
    "allow_partial_ckpt_load": True,
    "notebook_auto_setup": True,
    "notebook_auto_train": False,
    # Applied after loading trial_params_effective from replica_meta_json.
    # Use this to override finetuning-time optimization knobs such as lr/batch_size
    # while keeping the replicated architecture aligned with the checkpoint.
    "trial_param_overrides": {
        "lr": 1e-5,
        "mask_ratio": 0.0,
    },
    # Fallback only. If replica_meta_json exists, trial_params_effective will be
    # loaded from there and override this block.
    "trial_params": {
        "batch_size": 128,
        "embedder": "cnn_simple",
        "embed_dim": 128,
        "num_heads": 8,
        "channel_depth": 2,
        "temporal_depth": 2,
        "coord_scale": "none",
        "clip_value": 0.0,
        "chan_norm": "zscore",
        "emb_norm": "l2",
        "mask_ratio": 0.0,
        "patch_len": 100,
        "lr": 1e-5,
        "weight_decay": 1e-5,
    },
}

FINETUNE: Dict[str, Any] = {
    "num_epochs": 99,
    "steps_per_epoch": 200,
    "save_best_after_epoch": 1,
    "patience": 10,
    "min_delta": 0.0,
    "early_stop_start_epoch": 1,
    "eval_fixed_cmax": 32,   # <= 0 => variable Cmax in eval collate
    "eval_test_every_epoch": False,
    "finetune_mode": "full",  # {"full", "last_k", "tokens_pos", "loss_only"}
    "unfreeze_last_k_channel": 1,
    "unfreeze_last_k_temporal": 1,
}

EPISODE: Dict[str, Any] = {
    "num_subjects": 30,
    "max_sessions": 5,
    "prefer_multisession": True,
    "no_duplicates": True,
    "max_tries": 50,
}


SESSION: Dict[str, Any] = {}
RUN_OUTPUT: Dict[str, Any] = {}
mod = None
device = None
model = None
optimizer = None
scaler = None
loss_fn = None
miner_fn = None
train_loader = None
train_sampler = None


def _is_notebook_runtime() -> bool:
    return "ipykernel" in sys.modules


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
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], "--cuda_devices", str(cuda_devices)]
    try:
        if TRAIN_MODULE_NAME in sys.modules:
            mod_local = importlib.reload(sys.modules[TRAIN_MODULE_NAME])
        else:
            mod_local = importlib.import_module(TRAIN_MODULE_NAME)
    finally:
        sys.argv = old_argv
    return mod_local


def _resolve_device(mod_local):
    if not mod_local.torch.cuda.is_available():
        return mod_local.torch.device("cpu"), "cpu"

    cfg = str(CONFIG["cuda_devices"]).strip()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    first = cfg.split(",")[0].strip() if cfg else "0"
    logical_idx = 0 if (vis and vis == cfg) else int(first)
    n_visible = int(mod_local.torch.cuda.device_count())

    if logical_idx < 0 or logical_idx >= n_visible:
        print(
            f"[WARN] Requested logical cuda:{logical_idx} is out of visible range [0,{max(n_visible - 1, 0)}]. "
            "Falling back to cuda:0."
        )
        logical_idx = 0

    mod_local.torch.cuda.set_device(logical_idx)
    dev_name = mod_local.torch.cuda.get_device_name(logical_idx)
    info = (
        f"logical=cuda:{logical_idx} name='{dev_name}' visible_count={n_visible} "
        f"CUDA_VISIBLE_DEVICES='{vis}' requested='{cfg}'"
    )
    if vis and "," not in vis:
        info += " (single visible physical GPU is remapped to logical cuda:0)"
    return mod_local.torch.device(f"cuda:{logical_idx}"), info


def _sync_train_module_config(mod_local) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG["cuda_devices"])
    mod_local.args.cuda_devices = str(CONFIG["cuda_devices"])
    mod_local.args.train_dir = str(CONFIG["train_dir"])
    mod_local.args.val_dir = str(CONFIG["val_dir"])
    if str(CONFIG.get("test_dir", "")).strip():
        mod_local.args.test_dir = str(CONFIG["test_dir"])
    mod_local.args.crop_len = int(CONFIG["crop_len"])
    mod_local.args.eval_crop_mode = mod_local.normalize_eval_crop_mode(CONFIG.get("eval_crop_mode", "center"))
    mod_local.args.seed = int(CONFIG["seed"])

    mod_local.NUM_WORKERS_TRAIN = int(CONFIG["num_workers_train"])
    mod_local.NUM_WORKERS_EVAL = int(CONFIG["num_workers_eval"])
    mod_local.CACHE_EVAL_BATCH_SIZE = int(CONFIG["cache_eval_batch_size"])
    mod_local.TOP_K = int(CONFIG["top_k"])
    mod_local.MAX_NATIVE_CHANNELS = int(CONFIG["max_native_channels"])
    mod_local.USE_TORCH_COMPILE = bool(CONFIG["compile"])
    if "train_batch_lengths" in CONFIG and CONFIG["train_batch_lengths"] is not None:
        mod_local.TRAIN_BATCH_LENGTHS = [int(x) for x in CONFIG["train_batch_lengths"]]


def _required_param_keys() -> set:
    return {
        "batch_size",
        "embedder",
        "embed_dim",
        "channel_depth",
        "temporal_depth",
        "patch_len",
        "lr",
        "weight_decay",
    }


def _validate_config() -> None:
    if not str(CONFIG.get("pretrained_ckpt", "")).strip():
        raise ValueError("CONFIG['pretrained_ckpt'] is required.")
    meta_json = str(CONFIG.get("replica_meta_json", "")).strip()
    if meta_json and (not os.path.exists(meta_json)):
        raise ValueError(f"CONFIG['replica_meta_json'] does not exist: {meta_json}")
    missing = sorted(_required_param_keys() - set(CONFIG["trial_params"].keys()))
    if missing:
        raise ValueError(f"CONFIG['trial_params'] missing required keys: {missing}")

    if int(FINETUNE["steps_per_epoch"]) <= 0:
        raise ValueError("FINETUNE['steps_per_epoch'] must be > 0.")
    if int(FINETUNE["num_epochs"]) <= 0:
        raise ValueError("FINETUNE['num_epochs'] must be > 0.")
    if int(FINETUNE["save_best_after_epoch"]) <= 0:
        raise ValueError("FINETUNE['save_best_after_epoch'] must be > 0.")
    if int(FINETUNE["early_stop_start_epoch"]) <= 0:
        raise ValueError("FINETUNE['early_stop_start_epoch'] must be > 0.")
    if float(FINETUNE["min_delta"]) < 0:
        raise ValueError("FINETUNE['min_delta'] must be >= 0.")

    mode = str(FINETUNE["finetune_mode"])
    if mode not in ("full", "last_k", "tokens_pos", "loss_only"):
        raise ValueError(f"Unsupported FINETUNE['finetune_mode']: {mode}")


def extract_state_dict(raw_obj: Any) -> Dict[str, Any]:
    if not isinstance(raw_obj, dict):
        raise ValueError("Checkpoint must be a dict or contain a dict state_dict.")
    for key in ("state_dict", "model_state_dict", "model"):
        maybe = raw_obj.get(key)
        if isinstance(maybe, dict):
            return maybe
    return raw_obj


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if len(state_dict) == 0:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        plen = len(prefix)
        return {k[plen:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint_flexible(torch_mod, model_local, ckpt_path: str) -> Tuple[str, int, int]:
    try:
        raw = torch_mod.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch_mod.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(raw)

    variants = []
    variants.append(("raw", state_dict))
    variants.append(("strip_module", _strip_prefix_if_present(state_dict, "module.")))
    variants.append(("strip_orig_mod", _strip_prefix_if_present(state_dict, "_orig_mod.")))
    strip_both = _strip_prefix_if_present(_strip_prefix_if_present(state_dict, "module."), "_orig_mod.")
    variants.append(("strip_module_and_orig_mod", strip_both))
    variants.append(("add_orig_mod", {f"_orig_mod.{k}": v for k, v in state_dict.items()}))

    seen_signatures = set()
    last_err = None
    for tag, cand in variants:
        signature = tuple(sorted(cand.keys()))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        try:
            model_local.load_state_dict(cand, strict=True)
            return tag, 0, 0
        except RuntimeError as err:
            last_err = err

    missing, unexpected = model_local.load_state_dict(strip_both, strict=False)
    if last_err is not None:
        print(f"[WARN] strict checkpoint load failed ({last_err}). Used strict=False fallback.")
    return "strict_false_fallback", len(missing), len(unexpected)


def set_requires_grad(module, flag: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = bool(flag)


def configure_finetune_mode(model_local, mode: str, last_k_channel: int, last_k_temporal: int) -> str:
    for p in model_local.parameters():
        p.requires_grad = False

    if mode == "full":
        for p in model_local.parameters():
            p.requires_grad = True
        return "all model parameters trainable"

    if mode == "tokens_pos":
        set_requires_grad(getattr(model_local, "pos_coord", None), True)
        if getattr(model_local, "channel_cls_token", None) is not None:
            model_local.channel_cls_token.requires_grad = True
        if getattr(model_local, "temporal_cls_token", None) is not None:
            model_local.temporal_cls_token.requires_grad = True
        return "CoordMLP + CLS tokens trainable (ALiBi is fixed)"

    if mode == "last_k":
        set_requires_grad(getattr(model_local, "pos_coord", None), True)
        if getattr(model_local, "channel_cls_token", None) is not None:
            model_local.channel_cls_token.requires_grad = True
        if getattr(model_local, "temporal_cls_token", None) is not None:
            model_local.temporal_cls_token.requires_grad = True
        if getattr(model_local, "fuse_proj", None) is not None:
            set_requires_grad(model_local.fuse_proj, True)

        if hasattr(model_local.channel_encoder, "layers") and len(model_local.channel_encoder.layers) > 0:
            k = max(1, int(last_k_channel))
            k = min(k, len(model_local.channel_encoder.layers))
            for layer in model_local.channel_encoder.layers[-k:]:
                set_requires_grad(layer, True)

        if hasattr(model_local.temporal_encoder, "layers") and len(model_local.temporal_encoder.layers) > 0:
            kt = max(1, int(last_k_temporal))
            kt = min(kt, len(model_local.temporal_encoder.layers))
            for layer in model_local.temporal_encoder.layers[-kt:]:
                set_requires_grad(layer, True)

        return "last temporal/channel blocks + CoordMLP + CLS tokens + fuse_proj trainable"

    if mode == "loss_only":
        return "model frozen; only parametric loss parameters (if any) are trainable"

    raise ValueError(f"Unknown finetune mode: {mode}")


def count_trainable_params(module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def _publish_session_globals(state: Dict[str, Any]) -> None:
    global SESSION, mod, device, model, optimizer, scaler, loss_fn, miner_fn, train_loader, train_sampler
    SESSION = state
    mod = state["mod"]
    device = state["device"]
    model = state["model"]
    optimizer = state["optimizer"]
    scaler = state["scaler"]
    loss_fn = state["loss_fn"]
    miner_fn = state["miner_fn"]
    train_loader = state["train_loader"]
    train_sampler = state["train_sampler"]


def _build_eval_loader(mod_local, ds, layout_cache, fixed_cmax: int):
    if fixed_cmax > 0:
        collate_fn = partial(mod_local.collate_mixed2, fixed_Cmax=fixed_cmax, layout_cache=layout_cache)
    else:
        collate_fn = partial(mod_local.collate_mixed2, layout_cache=layout_cache)
    return mod_local.DataLoader(
        ds,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
        **mod_local._loader_perf_kwargs(mod_local.NUM_WORKERS_EVAL, pin_memory=True),
    )


def setup_session() -> Dict[str, Any]:
    _validate_config()
    trial_params = dict(CONFIG["trial_params"])
    meta_json = str(CONFIG.get("replica_meta_json", "")).strip()
    if meta_json:
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        loaded = meta.get("trial_params_effective")
        if not isinstance(loaded, dict):
            raise ValueError(
                "replica_meta_json must contain a dict under 'trial_params_effective'."
            )
        trial_params = dict(loaded)
    trial_params.update(dict(CONFIG.get("trial_param_overrides", {})))

    mod_local = _import_train_module(CONFIG["cuda_devices"])
    _sync_train_module_config(mod_local)
    device_local, device_info = _resolve_device(mod_local)

    run_seed = int(CONFIG["seed"])
    mod_local.seed_everything(run_seed)

    print("[INFO] device:", device_local)
    print("[INFO] device_info:", device_info)
    print("[INFO] finetune_mode:", FINETUNE["finetune_mode"])
    print("[INFO] steps_per_epoch:", FINETUNE["steps_per_epoch"])
    print("[INFO] trial_params:", trial_params)
    print(
        "[INFO] train-length augmentation:",
        {
            "lengths": [int(x) for x in getattr(mod_local, "TRAIN_BATCH_LENGTHS", [])],
            "mode": str(getattr(mod_local, "TRAIN_LENGTH_CROP_MODE", "random")),
        },
    )

    if mod_local.is_disallowed_embed_batch_combo(int(trial_params["embed_dim"]), int(trial_params["batch_size"])):
        raise RuntimeError("Disallowed combo: embed_dim=256 and batch_size=512.")

    train_ds = mod_local.PackedLayoutDatasetNoCrop(mod_local.args.train_dir, crop_len=mod_local.args.crop_len, random_crop=True)
    val_ds = mod_local.PackedLayoutDataset2(
        mod_local.args.val_dir,
        crop_len=mod_local.args.crop_len,
        random_crop=False,
        deterministic_crop_mode=mod_local.args.eval_crop_mode,
    )
    has_test = bool(str(CONFIG.get("test_dir", "")).strip())
    test_ds = (
        mod_local.PackedLayoutDataset2(
            str(CONFIG["test_dir"]),
            crop_len=mod_local.args.crop_len,
            random_crop=False,
            deterministic_crop_mode=mod_local.args.eval_crop_mode,
        )
        if has_test
        else None
    )

    train_layout_cache = mod_local.build_layout_cache(train_ds)
    val_layout_cache = mod_local.build_layout_cache(val_ds)
    test_layout_cache = mod_local.build_layout_cache(test_ds) if test_ds is not None else None

    eval_fixed_cmax = int(FINETUNE["eval_fixed_cmax"])
    val_loader = _build_eval_loader(mod_local, val_ds, val_layout_cache, fixed_cmax=eval_fixed_cmax)
    Xv, Mv, Yv, Sv, Pv = mod_local.collect_arrays(val_loader, T_expected=mod_local.args.crop_len)

    Xt = Mt = Yt = St = Pt = None
    if test_ds is not None:
        test_loader = _build_eval_loader(mod_local, test_ds, test_layout_cache, fixed_cmax=eval_fixed_cmax)
        Xt, Mt, Yt, St, Pt = mod_local.collect_arrays(test_loader, T_expected=mod_local.args.crop_len)
        test_loader = None

    val_loader = None
    val_ds = None
    test_ds = None
    gc.collect()

    model_local, embed_dim = mod_local.build_model(
        None,
        num_channels=len(mod_local.REF_CLIST_NORM),
        device=device_local,
        fixed_params=trial_params,
    )

    load_tag, n_missing, n_unexpected = load_checkpoint_flexible(mod_local.torch, model_local, CONFIG["pretrained_ckpt"])
    print(
        f"[INFO] Loaded checkpoint '{CONFIG['pretrained_ckpt']}' with mode={load_tag}, "
        f"missing={n_missing}, unexpected={n_unexpected}"
    )
    if n_missing > 0 or n_unexpected > 0:
        mismatch_msg = (
            "[WARN] Partial checkpoint load detected. This usually means CONFIG['trial_params'] "
            "does not exactly match the source architecture."
        )
        if bool(CONFIG.get("allow_partial_ckpt_load", True)):
            print(mismatch_msg, "Continuing because CONFIG['allow_partial_ckpt_load']=True.")
        else:
            raise RuntimeError(mismatch_msg + " Set allow_partial_ckpt_load=True to force-continue.")

    mode_desc = configure_finetune_mode(
        model_local,
        mode=str(FINETUNE["finetune_mode"]),
        last_k_channel=int(FINETUNE["unfreeze_last_k_channel"]),
        last_k_temporal=int(FINETUNE["unfreeze_last_k_temporal"]),
    )
    print("[INFO] Finetune mode detail:", mode_desc)

    num_classes = int(train_ds.targets.unique().numel())
    loss_fn_local, miner_fn_local = mod_local.build_loss_and_miner(mod_local.CHOSEN_LOSS, num_classes, embed_dim, device_local)
    optimizer_local = mod_local.build_optimizer(None, model_local, loss_fn_local, fixed_params=trial_params)
    scaler_local = mod_local.amp.GradScaler(enabled=(device_local.type == "cuda"))

    trainable_model = count_trainable_params(model_local)
    trainable_loss = count_trainable_params(loss_fn_local) if hasattr(loss_fn_local, "parameters") else 0
    total_trainable = trainable_model + trainable_loss
    print(
        f"[INFO] trainable params: model={trainable_model:,}, "
        f"loss={trainable_loss:,}, total={total_trainable:,}"
    )
    if total_trainable == 0:
        raise RuntimeError(
            "No trainable parameters. Use finetune_mode=full/last_k/tokens_pos, "
            "or use a parametric loss with finetune_mode=loss_only."
        )

    bs = int(trial_params["batch_size"])
    steps_per_epoch = int(FINETUNE["steps_per_epoch"])
    length_before_new_iter = int(bs * steps_per_epoch)
    train_sampler_local = mod_local.LayoutAwareEpisodeMultiSessionBatchSampler(
        labels=train_ds.targets,
        sessions=train_ds.session_ids,
        layouts=train_ds.layout_ids,
        batch_size=bs,
        num_subjects_per_batch=int(EPISODE["num_subjects"]),
        max_sessions_per_subject=int(EPISODE["max_sessions"]),
        length_before_new_iter=length_before_new_iter,
        prefer_multisession_subjects=bool(EPISODE["prefer_multisession"]),
        no_duplicates=bool(EPISODE["no_duplicates"]),
        max_tries=int(EPISODE["max_tries"]),
    )
    train_loader_local = mod_local.DataLoader(
        train_ds,
        batch_sampler=train_sampler_local,
        collate_fn=partial(mod_local.collate_mixed2, layout_cache=train_layout_cache),
        **mod_local._loader_perf_kwargs(mod_local.NUM_WORKERS_TRAIN, pin_memory=True),
    )
    print(f"[INFO] Train loader ready: batch_size={bs}, steps_per_epoch={len(train_loader_local)}")

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    best_ckpt = os.path.join(CONFIG["save_dir"], CONFIG["best_ckpt_name"])
    final_ckpt = os.path.join(CONFIG["save_dir"], CONFIG["final_ckpt_name"])
    log_csv = os.path.join(CONFIG["save_dir"], CONFIG["log_csv_name"])
    meta_json = os.path.join(CONFIG["save_dir"], CONFIG["meta_json_name"])

    print("[INFO] Baseline evaluation before finetuning...")
    base_val_eer, base_val_std = mod_local.evaluate_avg_eer_from_cached(
        model_local, device_local, Xv, Mv, Yv, Sv, Pv, top_k=mod_local.TOP_K
    )
    print(f"[BASE] val_eer={base_val_eer:.4f}+-{base_val_std:.4f}")

    base_test_eer = float("nan")
    base_test_std = float("nan")
    if Xt is not None:
        base_test_eer, base_test_std = mod_local.evaluate_avg_eer_from_cached(
            model_local, device_local, Xt, Mt, Yt, St, Pt, top_k=mod_local.TOP_K
        )
        print(f"[BASE] test_eer={base_test_eer:.4f}+-{base_test_std:.4f}")

    state = {
        "mod": mod_local,
        "device": device_local,
        "device_info": device_info,
        "run_seed": run_seed,
        "trial_params": trial_params,
        "mode_desc": mode_desc,
        "train_ds": train_ds,
        "train_sampler": train_sampler_local,
        "train_loader": train_loader_local,
        "iters_per_epoch": len(train_loader_local),
        "model": model_local,
        "embed_dim": embed_dim,
        "loss_fn": loss_fn_local,
        "miner_fn": miner_fn_local,
        "optimizer": optimizer_local,
        "scaler": scaler_local,
        "Xv": Xv,
        "Mv": Mv,
        "Yv": Yv,
        "Sv": Sv,
        "Pv": Pv,
        "Xt": Xt,
        "Mt": Mt,
        "Yt": Yt,
        "St": St,
        "Pt": Pt,
        "best_ckpt": best_ckpt,
        "final_ckpt": final_ckpt,
        "log_csv": log_csv,
        "meta_json": meta_json,
        "logs": [],
        "best_val": float("inf"),
        "best_ep": -1,
        "best_test": float("nan"),
        "best_test_std": float("nan"),
        "baseline_val_eer": float(base_val_eer),
        "baseline_val_std": float(base_val_std),
        "baseline_test_eer": float(base_test_eer),
        "baseline_test_std": float(base_test_std),
        "last_epoch": 0,
        "early_stop_wait": 0,
    }
    _publish_session_globals(state)
    return state


def train_epochs(state: Dict[str, Any], num_epochs: int, write_files: bool = True) -> Dict[str, Any]:
    if int(num_epochs) <= 0:
        return state

    mod_local = state["mod"]
    device_local = state["device"]
    model_local = state["model"]
    optimizer_local = state["optimizer"]
    scaler_local = state["scaler"]
    loss_fn_local = state["loss_fn"]
    miner_fn_local = state["miner_fn"]
    train_loader_local = state["train_loader"]

    print(
        f"[INFO] Finetuning start: epochs={int(num_epochs)}, iters={int(state['iters_per_epoch'])}, "
        f"bs={int(state['trial_params']['batch_size'])}, patience={int(FINETUNE['patience'])}, "
        f"min_delta={float(FINETUNE['min_delta'])}"
    )

    for _ in range(int(num_epochs)):
        ep = int(state["last_epoch"]) + 1
        tr_loss = mod_local.train_one_epoch(model_local, train_loader_local, optimizer_local, scaler_local, loss_fn_local, miner_fn_local, device_local)
        val_eer, val_std = mod_local.evaluate_avg_eer_from_cached(
            model_local, device_local, state["Xv"], state["Mv"], state["Yv"], state["Sv"], state["Pv"], top_k=mod_local.TOP_K
        )

        test_eer = float("nan")
        test_std = float("nan")
        if state["Xt"] is not None and bool(FINETUNE["eval_test_every_epoch"]):
            test_eer, test_std = mod_local.evaluate_avg_eer_from_cached(
                model_local, device_local, state["Xt"], state["Mt"], state["Yt"], state["St"], state["Pt"], top_k=mod_local.TOP_K
            )

        improved = val_eer < (float(state["best_val"]) - float(FINETUNE["min_delta"]))
        if improved:
            state["best_val"] = float(val_eer)
            state["best_ep"] = int(ep)
            state["early_stop_wait"] = 0
            if not mod_local.math.isnan(test_eer):
                state["best_test"] = float(test_eer)
                state["best_test_std"] = float(test_std)
            if ep >= int(FINETUNE["save_best_after_epoch"]):
                mod_local.torch.save(model_local.state_dict(), state["best_ckpt"])
        elif ep >= int(FINETUNE["early_stop_start_epoch"]):
            state["early_stop_wait"] = int(state["early_stop_wait"]) + 1

        state["logs"].append((ep, tr_loss, val_eer, val_std, test_eer, test_std, int(improved), int(state["early_stop_wait"])))
        state["last_epoch"] = int(ep)

        print(
            f"[FT E{ep:03d}] loss={tr_loss:.4f} val_eer={val_eer:.4f}+-{val_std:.4f} "
            f"test_eer={test_eer:.4f}+-{test_std:.4f} improved={improved} wait={int(state['early_stop_wait'])}"
        )

        patience = int(FINETUNE["patience"])
        if patience > 0 and ep >= int(FINETUNE["early_stop_start_epoch"]) and int(state["early_stop_wait"]) >= patience:
            print(
                f"[EARLY STOP] epoch={ep}, best_epoch={int(state['best_ep'])}, "
                f"best_val_eer={float(state['best_val']):.4f}"
            )
            break

    if write_files:
        write_artifacts(state)
    _publish_session_globals(state)
    return state


def write_artifacts(state: Dict[str, Any]) -> Dict[str, Any]:
    mod_local = state["mod"]
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    mod_local.torch.save(state["model"].state_dict(), state["final_ckpt"])

    if not os.path.exists(state["best_ckpt"]):
        mod_local.torch.save(state["model"].state_dict(), state["best_ckpt"])

    final_test_eer = float("nan")
    final_test_std = float("nan")
    best_test_eer = float(state["best_test"])
    best_test_std = float(state["best_test_std"])

    if state["Xt"] is not None:
        final_test_eer, final_test_std = mod_local.evaluate_avg_eer_from_cached(
            state["model"], state["device"], state["Xt"], state["Mt"], state["Yt"], state["St"], state["Pt"], top_k=mod_local.TOP_K
        )

        best_model, _ = mod_local.build_model(
            None,
            num_channels=len(mod_local.REF_CLIST_NORM),
            device=state["device"],
            fixed_params=state["trial_params"],
        )
        load_checkpoint_flexible(mod_local.torch, best_model, state["best_ckpt"])
        best_test_eer, best_test_std = mod_local.evaluate_avg_eer_from_cached(
            best_model, state["device"], state["Xt"], state["Mt"], state["Yt"], state["St"], state["Pt"], top_k=mod_local.TOP_K
        )

    with open(state["log_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_eer", "val_std", "test_eer", "test_std", "is_improved", "early_stop_wait"])
        writer.writerows(state["logs"])

    meta = {
        "config": CONFIG,
        "finetune": FINETUNE,
        "episode": EPISODE,
        "run_seed": state["run_seed"],
        "trial_params": state["trial_params"],
        "mode_desc": state["mode_desc"],
        "effective_train_batch_lengths": [
            int(x) for x in getattr(mod_local, "TRAIN_BATCH_LENGTHS", [])
        ],
        "effective_train_length_crop_mode": str(
            getattr(mod_local, "TRAIN_LENGTH_CROP_MODE", "random")
        ),
        "baseline_val_eer": state["baseline_val_eer"],
        "baseline_val_std": state["baseline_val_std"],
        "baseline_test_eer": state["baseline_test_eer"],
        "baseline_test_std": state["baseline_test_std"],
        "best_val_eer": float(state["best_val"]),
        "best_epoch": int(state["best_ep"]),
        "final_test_eer": float(final_test_eer),
        "final_test_std": float(final_test_std),
        "best_test_eer": float(best_test_eer),
        "best_test_std": float(best_test_std),
        "best_ckpt": state["best_ckpt"],
        "final_ckpt": state["final_ckpt"],
        "log_csv": state["log_csv"],
        "trained_epochs": int(state["last_epoch"]),
        "module": TRAIN_MODULE_NAME,
    }
    with open(state["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] Saved best checkpoint:", state["best_ckpt"])
    print("[INFO] Saved final checkpoint:", state["final_ckpt"])
    print("[INFO] Saved log:", state["log_csv"])
    print("[INFO] Saved meta:", state["meta_json"])
    print(f"[RESULT] best_val_eer={float(state['best_val']):.6f} best_epoch={int(state['best_ep'])}")
    if state["Xt"] is not None:
        print(
            f"[RESULT] final_test_eer={float(final_test_eer):.6f} "
            f"best_test_eer={float(best_test_eer):.6f}"
        )

    global RUN_OUTPUT
    RUN_OUTPUT = copy.deepcopy(meta)
    RUN_OUTPUT["logs"] = list(state["logs"])
    return meta


def cleanup_session(state: Dict[str, Any]) -> None:
    state["train_loader"] = None
    state["train_sampler"] = None
    state["train_ds"] = None
    gc.collect()


def run() -> None:
    state = setup_session()
    train_epochs(state, num_epochs=int(FINETUNE["num_epochs"]), write_files=True)


def run_with_overrides(
    config_overrides: Optional[Dict[str, Any]] = None,
    finetune_overrides: Optional[Dict[str, Any]] = None,
    trial_param_overrides: Optional[Dict[str, Any]] = None,
    episode_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    config_backup = copy.deepcopy(CONFIG)
    finetune_backup = copy.deepcopy(FINETUNE)
    episode_backup = copy.deepcopy(EPISODE)
    try:
        if config_overrides:
            CONFIG.update(config_overrides)
        if finetune_overrides:
            FINETUNE.update(finetune_overrides)
        if episode_overrides:
            EPISODE.update(episode_overrides)
        if trial_param_overrides:
            CONFIG["trial_param_overrides"].update(trial_param_overrides)
        run()
    finally:
        CONFIG.clear()
        CONFIG.update(config_backup)
        FINETUNE.clear()
        FINETUNE.update(finetune_backup)
        EPISODE.clear()
        EPISODE.update(episode_backup)


def run_finetune_low_lr(lr: float = 1e-5) -> None:
    run_with_overrides(
        finetune_overrides={"finetune_mode": "full"},
        trial_param_overrides={"lr": float(lr)},
    )


def run_finetune_last_k(
    lr: float = 1e-5,
    unfreeze_last_k_channel: int = 1,
    unfreeze_last_k_temporal: int = 1,
) -> None:
    run_with_overrides(
        finetune_overrides={
            "finetune_mode": "last_k",
            "unfreeze_last_k_channel": int(unfreeze_last_k_channel),
            "unfreeze_last_k_temporal": int(unfreeze_last_k_temporal),
        },
        trial_param_overrides={"lr": float(lr)},
    )


def run_finetune_tokens_pos(lr: float = 1e-5) -> None:
    run_with_overrides(
        finetune_overrides={"finetune_mode": "tokens_pos"},
        trial_param_overrides={"lr": float(lr)},
    )


if __name__ == "__main__":
    if _is_notebook_runtime() and bool(CONFIG.get("notebook_auto_setup", True)):
        state = setup_session()
        if bool(CONFIG.get("notebook_auto_train", False)):
            train_epochs(state, num_epochs=int(FINETUNE["num_epochs"]), write_files=True)
        else:
            print(
                "[INFO] Notebook mode: session initialized. "
                "Objects are available as globals: model, optimizer, scaler, loss_fn, miner_fn, "
                "train_loader, train_sampler, SESSION."
            )
    else:
        run()
