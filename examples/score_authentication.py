#!/usr/bin/env python3
"""
Compute simple template-based authentication scores from saved embeddings.

Typical workflow:
  1) Generate enrollment embeddings with examples/minimal_inference.py
  2) Generate query embeddings with examples/minimal_inference.py
  3) Score the query embeddings against the mean enrollment template

Example:
  python examples/score_authentication.py ^
    --enroll-npy enroll_embeddings.npy ^
    --query-npy query_embeddings.npy
"""

import argparse
import os
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils_evaluation import build_mean_template, score_mean_template


def _parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value: {text!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enroll-npy",
        type=str,
        required=True,
        help="Path to enrollment embeddings saved as a .npy array with shape [N, D].",
    )
    parser.add_argument(
        "--query-npy",
        type=str,
        required=True,
        help="Path to query embeddings saved as a .npy array with shape [D] or [M, D].",
    )
    parser.add_argument(
        "--normalize-template",
        type=str,
        default="true",
        help="Whether to L2-normalize the mean template after averaging (true/false).",
    )
    parser.add_argument(
        "--save-scores-npy",
        type=str,
        default="",
        help="Optional output path for the score array.",
    )
    return parser.parse_args()


def _validate_enroll(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"Enrollment embeddings must have shape [N, D], got {out.shape}")
    if out.shape[0] == 0:
        raise ValueError("Enrollment embeddings array is empty.")
    return out


def _validate_query(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim not in {1, 2}:
        raise ValueError(f"Query embeddings must have shape [D] or [M, D], got {out.shape}")
    if out.ndim == 2 and out.shape[0] == 0:
        raise ValueError("Query embeddings array is empty.")
    return out


def main() -> None:
    args = _parse_args()
    enroll_path = os.path.abspath(str(args.enroll_npy))
    query_path = os.path.abspath(str(args.query_npy))
    save_scores_path = os.path.abspath(str(args.save_scores_npy)) if str(args.save_scores_npy).strip() else ""
    normalize_template = _parse_bool(str(args.normalize_template))

    if not os.path.exists(enroll_path):
        raise RuntimeError(f"Enrollment embeddings file not found: {enroll_path}")
    if not os.path.exists(query_path):
        raise RuntimeError(f"Query embeddings file not found: {query_path}")

    enroll = _validate_enroll(np.load(enroll_path))
    query = _validate_query(np.load(query_path))

    template = build_mean_template(
        enroll,
        normalize_template=normalize_template,
    )
    scores = score_mean_template(
        enroll,
        query,
        normalize_template=normalize_template,
    )

    if save_scores_path:
        np.save(save_scores_path, scores)

    print("[INFO] enroll_npy:", enroll_path)
    print("[INFO] query_npy:", query_path)
    print("[INFO] normalize_template:", normalize_template)
    print("[INFO] enroll.shape:", tuple(enroll.shape))
    print("[INFO] query.shape:", tuple(query.shape))
    print("[INFO] template.shape:", tuple(template.shape))
    print("[INFO] scores.shape:", tuple(scores.shape))
    print("[INFO] score_mean:", float(np.mean(scores)))
    print("[INFO] score_min:", float(np.min(scores)))
    print("[INFO] score_max:", float(np.max(scores)))
    if scores.size == 1:
        print("[RESULT] score:", float(scores[0]))
    if save_scores_path:
        print("[INFO] saved_scores_npy:", save_scores_path)


if __name__ == "__main__":
    main()
