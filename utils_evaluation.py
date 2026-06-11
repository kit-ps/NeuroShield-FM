#!/usr/bin/env python3
"""
Minimal evaluation helpers used by NeuroShield.

These functions provide the subject split and similarity-score aggregation
needed by the core training/evaluation scripts. This repository keeps a
single optimized scorer that uses chunked GPU computation when CUDA is
available and otherwise falls back to a NumPy implementation.
"""

from typing import Dict

import numpy as np


def get_enrollment_verification_indices(Y_all, S_all):
    """
    Split each subject into enrollment and verification samples.

    Preferred rule:
      - use the first observed session for enrollment
      - use later sessions for verification

    Fallbacks:
      - if a subject has only one session but multiple samples, use the first
        sample for enrollment and the rest for verification
      - if a subject has only one sample, reuse it on both sides
    """
    y_all = np.asarray(Y_all)
    s_all = np.asarray(S_all)
    enroll = []
    verify = []

    for subject_id in np.unique(y_all):
        idx = np.where(y_all == subject_id)[0]
        if idx.size == 0:
            continue

        subj_sessions = s_all[idx]
        seen_sessions = list(dict.fromkeys(subj_sessions.tolist()))

        if len(seen_sessions) >= 2:
            enroll_session = seen_sessions[0]
            enroll_idx = idx[subj_sessions == enroll_session]
            verify_idx = idx[subj_sessions != enroll_session]
        elif idx.size >= 2:
            enroll_idx = idx[:1]
            verify_idx = idx[1:]
        else:
            enroll_idx = idx[:1]
            verify_idx = idx[:1]

        enroll.extend(int(i) for i in enroll_idx.tolist())
        verify.extend(int(i) for i in verify_idx.tolist())

    if len(enroll) == 0:
        raise RuntimeError("No enrollment samples could be constructed.")
    if len(verify) == 0:
        verify = list(enroll)

    return np.asarray(enroll, dtype=np.int64), np.asarray(verify, dtype=np.int64)


def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize one embedding or a batch of embeddings.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    squeeze = False
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        squeeze = True
    if arr.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [D] or [N, D], got {arr.shape}")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    out = arr / norms
    return out[0] if squeeze else out


def build_mean_template(
    enroll_embeddings: np.ndarray,
    normalize_template: bool = True,
) -> np.ndarray:
    """
    Build a single enrollment template by averaging enrollment embeddings.

    If `normalize_template=True`, the averaged template is L2-normalized before
    scoring. If `False`, the raw average is returned, which matches averaging
    all pairwise dot-product scores when both enrollment and query embeddings
    are already L2-normalized.
    """
    enroll = l2_normalize_embeddings(enroll_embeddings)
    if enroll.ndim != 2 or enroll.shape[0] == 0:
        raise ValueError(f"Expected non-empty enrollment embeddings with shape [N, D], got {enroll.shape}")

    template = np.mean(enroll, axis=0).astype(np.float32, copy=False)
    if normalize_template:
        template = l2_normalize_embeddings(template)
    return np.asarray(template, dtype=np.float32)


def score_mean_template(
    enroll_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    normalize_template: bool = True,
) -> np.ndarray:
    """
    Score one or more query embeddings against a mean enrollment template.

    Returns:
      - shape [1] for a single query embedding input [D]
      - shape [N] for batched query embeddings input [N, D]
    """
    template = build_mean_template(
        enroll_embeddings,
        normalize_template=bool(normalize_template),
    ).reshape(1, -1)
    queries = l2_normalize_embeddings(query_embeddings)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.shape[1] != template.shape[1]:
        raise ValueError(
            f"Embedding dims do not match: query dim {queries.shape[1]} vs template dim {template.shape[1]}"
        )
    scores = (queries @ template.T).reshape(-1).astype(np.float32, copy=False)
    return scores


def compute_similarity_scores(
    en_emb,
    y_enroll,
    ve_emb,
    y_verify,
    distance: str = "cd",
    top_k: int = 40,
    chunk_size: int = 2048,
) -> Dict[int, np.ndarray]:
    """
    Build subject-wise genuine/impostor score tables expected by the training
    and evaluation scripts:

      {subject_id: array([[score, is_genuine], ...], dtype=float32)}

    The current implementation supports cosine-style similarity only (`cd`),
    implemented as normalized dot product.

    For each verification embedding and subject:
      - if the subject has more than `top_k` enrollment embeddings, average the
        best `top_k` similarities
      - otherwise average all available similarities for that subject

    When PyTorch with CUDA is available, scoring is computed in chunks on the
    current CUDA device for speed.
    """
    metric = str(distance).strip().lower()
    if metric != "cd":
        raise ValueError(f"Only distance='cd' is supported, got {distance!r}")

    y_enroll = np.asarray(y_enroll)
    y_verify = np.asarray(y_verify)
    en = np.asarray(en_emb, dtype=np.float32)
    ve = np.asarray(ve_emb, dtype=np.float32)

    if en.ndim != 2 or ve.ndim != 2:
        raise ValueError("Embedding arrays must be 2D.")
    if en.shape[1] != ve.shape[1]:
        raise ValueError(f"Embedding dims do not match: {en.shape[1]} vs {ve.shape[1]}")

    en_norm = np.linalg.norm(en, axis=1, keepdims=True)
    ve_norm = np.linalg.norm(ve, axis=1, keepdims=True)
    np.maximum(en_norm, 1e-12, out=en_norm)
    np.maximum(ve_norm, 1e-12, out=ve_norm)
    en = en / en_norm
    ve = ve / ve_norm

    subject_ids = np.unique(y_enroll)
    sort_idx = np.argsort(y_enroll, kind="stable")
    en_sorted = en[sort_idx]
    y_sorted = y_enroll[sort_idx]

    subject_blocks = []
    start = 0
    for subject_id in subject_ids:
        end = start
        while end < y_sorted.shape[0] and y_sorted[end] == subject_id:
            end += 1
        subject_blocks.append((int(subject_id), int(start), int(end)))
        start = end

    n_subjects = int(subject_ids.shape[0])
    n_verify = int(ve.shape[0])
    labels_matrix = (subject_ids[:, None] == y_verify[None, :]).astype(np.float32, copy=False)
    score_matrix = np.empty((n_subjects, n_verify), dtype=np.float32)

    chunk_size = max(1, int(chunk_size))
    top_k = max(1, int(top_k))

    torch = None
    try:
        import torch as _torch
        torch = _torch
    except Exception:
        torch = None

    use_cuda = bool(torch is not None and torch.cuda.is_available())

    if use_cuda:
        device = torch.device("cuda", torch.cuda.current_device())
        en_t = torch.from_numpy(en_sorted).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            for start_idx in range(0, n_verify, chunk_size):
                end_idx = min(start_idx + chunk_size, n_verify)
                ve_chunk_t = torch.from_numpy(ve[start_idx:end_idx]).to(device=device, dtype=torch.float32)
                sim_chunk = ve_chunk_t @ en_t.T
                for subject_rank, (_subject_id, left, right) in enumerate(subject_blocks):
                    block = sim_chunk[:, left:right]
                    block_width = int(right - left)
                    if block_width <= 0:
                        raise RuntimeError("Encountered empty enrollment block for subject.")
                    if block_width <= top_k:
                        subj_scores = block.mean(dim=1)
                    else:
                        subj_scores = torch.topk(
                            block,
                            k=top_k,
                            dim=1,
                            largest=True,
                            sorted=False,
                        ).values.mean(dim=1)
                    score_matrix[subject_rank, start_idx:end_idx] = subj_scores.detach().cpu().numpy()
                del ve_chunk_t, sim_chunk
        del en_t
    else:
        for start_idx in range(0, n_verify, chunk_size):
            end_idx = min(start_idx + chunk_size, n_verify)
            sim_chunk = ve[start_idx:end_idx] @ en_sorted.T
            for subject_rank, (_subject_id, left, right) in enumerate(subject_blocks):
                block = sim_chunk[:, left:right]
                block_width = int(right - left)
                if block_width <= 0:
                    raise RuntimeError("Encountered empty enrollment block for subject.")
                if block_width <= top_k:
                    subj_scores = block.mean(axis=1)
                else:
                    kth = block_width - top_k
                    top_block = np.partition(block, kth=kth, axis=1)[:, kth:]
                    subj_scores = top_block.mean(axis=1)
                score_matrix[subject_rank, start_idx:end_idx] = subj_scores.astype(np.float32, copy=False)

    results: Dict[int, np.ndarray] = {}
    for subject_rank, subject_id in enumerate(subject_ids.tolist()):
        scores = score_matrix[subject_rank]
        labels = labels_matrix[subject_rank]
        results[int(subject_id)] = np.column_stack((scores, labels)).astype(np.float32, copy=False)
    return results
