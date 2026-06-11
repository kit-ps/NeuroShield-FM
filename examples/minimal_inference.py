#!/usr/bin/env python3
"""
Minimal NeuroShield inference example.

Expected input:
  - a .npz file containing:
      X: float32 array of shape [N, C, T]
      M: bool array of shape [N, C]
      P: float32 array of shape [N, C, 3]

Example:
  python examples/minimal_inference.py --input-npz sample_batch.npz
"""

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-json",
        type=str,
        default=os.path.join(REPO_ROOT, "ckpts_bw_replicate_varlen", "replicate_best_trial_varlen_meta.json"),
        help="Path to the NeuroShield metadata JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(REPO_ROOT, "ckpts_bw_replicate_varlen", "replicate_best_trial_varlen_final.pth"),
        help="Path to the pretrained NeuroShield checkpoint.",
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help="Input .npz file containing X, M, and P arrays.",
    )
    parser.add_argument(
        "--output-npy",
        type=str,
        default="",
        help="Optional output path for the embeddings .npy file.",
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES value. Use an empty string or unavailable GPU to fall back to CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def _import_train_module(cuda_devices: str):
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


def _resolve_device(mod, requested_cuda: str):
    if not mod.torch.cuda.is_available():
        return mod.torch.device("cpu")

    cfg = str(requested_cuda).strip()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    first = cfg.split(",")[0].strip() if cfg else "0"
    logical_idx = 0 if (vis and vis == cfg) else int(first)
    n_visible = int(mod.torch.cuda.device_count())
    if logical_idx < 0 or logical_idx >= n_visible:
        logical_idx = 0
    mod.torch.cuda.set_device(logical_idx)
    return mod.torch.device(f"cuda:{logical_idx}")


def _load_trial_params(meta: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("trial_params", "trial_params_effective", "best_params_effective", "best_params"):
        value = meta.get(key)
        if isinstance(value, dict) and len(value) > 0:
            return dict(value)
    raise RuntimeError("Could not find trial parameters in metadata JSON.")


def _validate_inputs(X: np.ndarray, M: np.ndarray, P: np.ndarray) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must have shape [N, C, T], got {X.shape}")
    if M.ndim != 2:
        raise ValueError(f"M must have shape [N, C], got {M.shape}")
    if P.ndim != 3 or P.shape[-1] != 3:
        raise ValueError(f"P must have shape [N, C, 3], got {P.shape}")
    if X.shape[0] != M.shape[0] or X.shape[0] != P.shape[0]:
        raise ValueError("X, M, and P must have the same N dimension.")
    if X.shape[1] != M.shape[1] or X.shape[1] != P.shape[1]:
        raise ValueError("X, M, and P must have the same C dimension.")


def main() -> None:
    args = _parse_args()
    meta_json = os.path.abspath(str(args.meta_json))
    checkpoint = os.path.abspath(str(args.checkpoint))
    input_npz = os.path.abspath(str(args.input_npz))
    output_npy = (
        os.path.abspath(str(args.output_npy))
        if str(args.output_npy).strip()
        else os.path.splitext(input_npz)[0] + "_embeddings.npy"
    )

    if not os.path.exists(meta_json):
        raise RuntimeError(f"Metadata JSON not found: {meta_json}")
    if not os.path.exists(checkpoint):
        raise RuntimeError(f"Checkpoint not found: {checkpoint}")
    if not os.path.exists(input_npz):
        raise RuntimeError(f"Input NPZ not found: {input_npz}")

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    trial_params = _load_trial_params(meta)

    batch = np.load(input_npz)
    X = np.asarray(batch["X"], dtype=np.float32)
    M = np.asarray(batch["M"], dtype=bool)
    P = np.asarray(batch["P"], dtype=np.float32)
    _validate_inputs(X, M, P)

    mod = _import_train_module(str(args.cuda_devices))
    device = _resolve_device(mod, str(args.cuda_devices))
    model, embed_dim = mod.build_model(
        trial=None,
        num_channels=len(mod.REF_CLIST_NORM),
        device=device,
        fixed_params=trial_params,
    )
    state = mod.torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    embeddings = mod.generate_embeddings_eeg_masked(
        X,
        M,
        P,
        model,
        device=device,
        batch_size=int(args.batch_size),
    )
    np.save(output_npy, embeddings)

    print("[INFO] meta_json:", meta_json)
    print("[INFO] checkpoint:", checkpoint)
    print("[INFO] input_npz:", input_npz)
    print("[INFO] output_npy:", output_npy)
    print("[INFO] device:", device)
    print("[INFO] model_embed_dim:", embed_dim)
    print("[INFO] X.shape:", tuple(X.shape))
    print("[INFO] M.shape:", tuple(M.shape))
    print("[INFO] P.shape:", tuple(P.shape))
    print("[INFO] embeddings.shape:", tuple(embeddings.shape))


if __name__ == "__main__":
    main()
