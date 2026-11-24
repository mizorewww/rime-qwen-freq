"""
Prefix-Trie Exact Teacher-Forcing Scorer & Empirical Anchor Calibrator.

Supports CUDA, Apple Silicon (MPS), and CPU backends with any Hugging Face CausalLM.
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
logging.getLogger("kernels").setLevel(logging.ERROR)


def select_device_and_dtype(requested: str = "auto"):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = requested

    if dev.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif dev == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    return dev, dtype


def batch_tokenize(tok, words: list[str], max_toks: int = 24) -> list[list[int]]:
    all_ids: list[list[int]] = []
    chunk = 100_000
    for i in range(0, len(words), chunk):
        enc = tok(
            words[i : i + chunk],
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        for seq in enc:
            if not seq:
                all_ids.append([0])
            elif len(seq) > max_toks:
                all_ids.append(seq[:max_toks])
            else:
                all_ids.append(seq)
    return all_ids


def build_prefix_trie(token_seqs: list[list[int]]) -> dict:
    """
    Deduplicate all proper token prefixes across `token_seqs` so shared prefixes are
    evaluated by the transformer only once while preserving exact causal log-probabilities.
    """
    try:
        import numpy as np
    except ImportError:
        np = None

    internal_children: list[dict[int, int]] = [{}]
    node_seq: list[list[int]] = [[]]

    for seq in token_seqs:
        u = 0
        for t in seq[:-1]:
            nxt = internal_children[u].get(t)
            if nxt is None:
                nxt = len(internal_children)
                internal_children[u][t] = nxt
                internal_children.append({})
                node_seq.append(node_seq[u] + [t])
            u = nxt

    num_internal = len(internal_children)
    edge_dicts: list[dict[int, int]] = [{} for _ in range(num_internal)]
    num_edges = 0
    word_edges: list[list[int]] = []

    for seq in token_seqs:
        u = 0
        w_eids: list[int] = []
        last_idx = len(seq) - 1
        for j, t in enumerate(seq):
            ed = edge_dicts[u]
            eid = ed.get(t)
            if eid is None:
                eid = num_edges
                ed[t] = eid
                num_edges += 1
            w_eids.append(eid)
            if j < last_idx:
                u = internal_children[u][t]
        word_edges.append(w_eids)

    node_query_toks = [None] * num_internal
    node_query_eids = [None] * num_internal
    for u, ed in enumerate(edge_dicts):
        if ed:
            if np is not None:
                node_query_toks[u] = np.fromiter(ed.keys(), dtype=np.int64, count=len(ed))
                node_query_eids[u] = np.fromiter(ed.values(), dtype=np.int64, count=len(ed))
            else:
                node_query_toks[u] = list(ed.keys())
                node_query_eids[u] = list(ed.values())
    del edge_dicts

    leaves = [u for u in range(1, num_internal) if not internal_children[u]]
    claimed = bytearray(num_internal)
    claimed[0] = 1
    leaf_owned_steps: list[list[tuple[int, int]]] = []

    for leaf in leaves:
        seq_u = node_seq[leaf]
        u = 0
        owned: list[tuple[int, int]] = []
        for s, t in enumerate(seq_u):
            u = internal_children[u][t]
            if not claimed[u]:
                claimed[u] = 1
                if node_query_toks[u] is not None:
                    owned.append((s, u))
        leaf_owned_steps.append(owned)

    return {
        "num_edges": num_edges,
        "num_internal": num_internal,
        "word_edges": word_edges,
        "node_seq": node_seq,
        "node_query_toks": node_query_toks,
        "node_query_eids": node_query_eids,
        "leaves": leaves,
        "leaf_owned_steps": leaf_owned_steps,
    }


def extract_backbone_and_head(full_model):
    if hasattr(full_model, "model") and hasattr(full_model.model, "language_model") and hasattr(full_model, "lm_head"):
        return full_model.model.language_model, full_model.lm_head
    if hasattr(full_model, "model") and hasattr(full_model, "lm_head"):
        return full_model.model, full_model.lm_head
    if hasattr(full_model, "transformer") and hasattr(full_model, "lm_head"):
        return full_model.transformer, full_model.lm_head
    return None, None


def trie_score_pass(
    tok,
    full_model,
    backbone,
    lm_head,
    trie: dict,
    prefix_text: str,
    device: str,
    max_batch: int = 2048,
) -> "np.ndarray":
    import numpy as np
    import torch

    t0 = time.time()
    edge_lp = np.empty(trie["num_edges"], dtype=np.float32)

    p_ids = tok.encode(prefix_text, add_special_tokens=False)
    if not p_ids:
        p_ids = [tok.bos_token_id or tok.eos_token_id or 0]
    m = len(p_ids)
    p_tensor = torch.tensor(p_ids, device=device, dtype=torch.long)

    with torch.inference_mode():
        # 1. Root pass
        if backbone is not None and lm_head is not None:
            hs0 = backbone(input_ids=p_tensor.unsqueeze(0), use_cache=False).last_hidden_state
            z0 = lm_head(hs0[0, m - 1]).float()
        else:
            z0 = full_model(input_ids=p_tensor.unsqueeze(0), use_cache=False).logits[0, m - 1].float()
        lse0 = torch.logsumexp(z0, dim=-1)
        if trie["node_query_toks"][0] is not None:
            q0_dev = torch.from_numpy(trie["node_query_toks"][0]).to(device)
            edge_lp[trie["node_query_eids"][0]] = (z0.index_select(0, q0_dev) - lse0).cpu().numpy()

        # 2. Maximal proper-prefix leaf batches grouped by exact length d (zero padding)
        leaves = trie["leaves"]
        leaf_owned = trie["leaf_owned_steps"]
        node_seq = trie["node_seq"]
        by_len: dict[int, list[int]] = {}
        for idx, leaf_u in enumerate(leaves):
            by_len.setdefault(len(node_seq[leaf_u]), []).append(idx)

        total_leaves = len(leaves)
        done_leaves = 0
        last_log = t0
        token_budget = 24576 if device.startswith("cuda") else (8192 if device == "mps" else 2048)
        eff_max_batch = min(max_batch, 2048 if device.startswith("cuda") else 512)

        for d in sorted(by_len.keys()):
            l_idxs = by_len[d]
            L = m + d
            bsz = min(eff_max_batch, max(32, token_budget // L))

            arr_cpu = np.empty((len(l_idxs), d), dtype=np.int64)
            for j, li in enumerate(l_idxs):
                arr_cpu[j] = node_seq[leaves[li]]

            for s in range(0, len(l_idxs), bsz):
                sub = l_idxs[s : s + bsz]
                b = len(sub)
                u_dev = torch.from_numpy(arr_cpu[s : s + b]).to(device)
                inp = torch.empty((b, L), device=device, dtype=torch.long)
                inp[:, :m] = p_tensor
                inp[:, m:] = u_dev

                rows: list[int] = []
                cols: list[int] = []
                owned_nodes: list[int] = []
                for r, li in enumerate(sub):
                    for step_s, node_u in leaf_owned[li]:
                        rows.append(r)
                        cols.append(m + step_s)
                        owned_nodes.append(node_u)

                if owned_nodes:
                    r_t = torch.tensor(rows, device=device, dtype=torch.long)
                    c_t = torch.tensor(cols, device=device, dtype=torch.long)

                    if backbone is not None and lm_head is not None:
                        hs = backbone(input_ids=inp, use_cache=False).last_hidden_state
                        h_sel = hs[r_t, c_t, :]
                        Q = len(owned_nodes)
                        for qs in range(0, Q, 512):
                            qe = min(Q, qs + 512)
                            z = lm_head(h_sel[qs:qe]).float()
                            lse = torch.logsumexp(z, dim=-1)

                            flat_rows, flat_toks, flat_eids = [], [], []
                            for offset, node_u in enumerate(owned_nodes[qs:qe]):
                                q_toks = trie["node_query_toks"][node_u]
                                q_eids = trie["node_query_eids"][node_u]
                                flat_rows.append(np.full(len(q_toks), offset, dtype=np.int64))
                                flat_toks.append(q_toks)
                                flat_eids.append(q_eids)

                            r_idx_np = np.concatenate(flat_rows)
                            t_idx_np = np.concatenate(flat_toks)
                            e_idx_np = np.concatenate(flat_eids)

                            r_idx_dev = torch.from_numpy(r_idx_np).to(device)
                            t_idx_dev = torch.from_numpy(t_idx_np).to(device)
                            edge_lp[e_idx_np] = (z[r_idx_dev, t_idx_dev] - lse[r_idx_dev]).cpu().numpy()
                    else:
                        logits_all = full_model(input_ids=inp, use_cache=False).logits
                        z_all = logits_all[r_t, c_t, :].float()
                        lse = torch.logsumexp(z_all, dim=-1)
                        for offset, node_u in enumerate(owned_nodes):
                            q_toks = trie["node_query_toks"][node_u]
                            q_eids = trie["node_query_eids"][node_u]
                            q_dev = torch.from_numpy(q_toks).to(device)
                            edge_lp[q_eids] = (z_all[offset].index_select(0, q_dev) - lse[offset]).cpu().numpy()

                done_leaves += b
                now = time.time()
                if now - last_log >= 8.0:
                    rate = done_leaves / (now - t0)
                    eta = (total_leaves - done_leaves) / max(rate, 1.0)
                    print(
                        f"    [{prefix_text!r}] {done_leaves:,}/{total_leaves:,} "
                        f"({done_leaves*100/max(1,total_leaves):.1f}%) | d={d} | {rate:,.0f} paths/s | ETA {eta:.0f}s",
                        flush=True,
                    )
                    last_log = now

    word_edges = trie["word_edges"]
    n_words = len(word_edges)
    feats = np.empty((n_words, 5), dtype=np.float32)
    for i, eids in enumerate(word_edges):
        k = len(eids)
        if k == 1:
            feats[i, :] = edge_lp[eids[0]]
        else:
            s_lp = 0.0
            mn_lp = 1e9
            for eid in eids:
                val = float(edge_lp[eid])
                s_lp += val
                if val < mn_lp:
                    mn_lp = val
            feats[i, 0] = s_lp
            feats[i, 1] = edge_lp[eids[0]]
            feats[i, 2] = edge_lp[eids[-1]]
            feats[i, 3] = s_lp / k
            feats[i, 4] = mn_lp

    dt = time.time() - t0
    print(f"  Finished context {prefix_text!r} in {dt:.1f}s ({n_words/max(dt, 1e-3):,.0f} words/s)", flush=True)
    return feats


def verify_trie_exactness(tok, full_model, backbone, lm_head, device: str) -> float:
    import torch
    import torch.nn.functional as F

    test_words = ["啊", "中国", "人工智能", "深度学习", "中华人民共和国", "5G", "README.md"]
    t_seqs = batch_tokenize(tok, test_words)
    t_trie = build_prefix_trie(t_seqs)
    trie_feats = trie_score_pass(tok, full_model, backbone, lm_head, t_trie, "。\n", device=device)

    p_ids = tok.encode("。\n", add_special_tokens=False) or [0]
    m = len(p_ids)
    max_err = 0.0
    with torch.inference_mode():
        for i, seq in enumerate(t_seqs):
            inp = torch.tensor([p_ids + seq[:-1]], device=device, dtype=torch.long)
            logits = full_model(input_ids=inp, use_cache=False).logits[0, m - 1 :, :].float()
            logp = F.log_softmax(logits, dim=-1)
            ref_sum = sum(logp[j, seq[j]].item() for j in range(len(seq)))
            max_err = max(max_err, abs(ref_sum - float(trie_feats[i, 0])))
    if max_err >= 0.35:
        raise AssertionError(f"Prefix-Trie numerical equivalence check failed: max_err={max_err}")
    return max_err


def compute_domain_lift(logp_domain: list[float] | "np.ndarray", logp_general: list[float] | "np.ndarray"):
    """
    Compute exact Bayesian Domain Log-Odds Lift (in nats):
        Delta_domain(w) = log P(w | C_domain) - log P(w | C_general)
    """
    try:
        import numpy as np
        if isinstance(logp_domain, np.ndarray) and isinstance(logp_general, np.ndarray):
            return (logp_domain - logp_general).astype(np.float32)
    except ImportError:
        pass
    return [float(d) - float(g) for d, g in zip(logp_domain, logp_general)]


def bayesian_domain_log10_tilt(
    delta_domain: list[float] | "np.ndarray",
    threshold: float = 1.3,
    slope: float = 0.135,
    max_boost: float = 0.54,
):
    """
    Convert Bayesian Domain Log-Odds Lift (nats) into a smooth, bounded log10 weight boost:
        tilt(w) = clip((Delta_domain(w) - threshold) * slope, 0.0, max_boost)
    """
    try:
        import numpy as np
        if isinstance(delta_domain, np.ndarray):
            return np.clip((delta_domain - threshold) * slope, 0.0, max_boost).astype(np.float32)
    except ImportError:
        pass
    return [min(max_boost, max(0.0, (float(d) - threshold) * slope)) for d in delta_domain]


def build_single_context_features(words: list[str], token_seqs: list[list[int]], f_gen):
    """Build calibration feature matrix from a single general context so domain bias never skews anchor fitting."""
    import numpy as np

    n = len(words)
    lex = np.empty((n, 6), dtype=np.float32)
    for i, (w, seq) in enumerate(zip(words, token_seqs)):
        lex[i, 0] = len(seq)
        lex[i, 1] = len(w)
        lex[i, 2] = seq[0]
        lex[i, 3] = max(seq)
        lex[i, 4] = sum(seq) / len(seq)
        lex[i, 5] = 1.0 if (not w.isascii() and any(ord(ch) < 128 for ch in w)) else (2.0 if w.isascii() else 0.0)
    return np.hstack([f_gen, lex])


def build_feature_matrix(words: list[str], token_seqs: list[list[int]], f1, f2):
    import numpy as np

    n = len(words)
    cols = [
        f1,
        f2,
        np.maximum(f1[:, 0:1], f2[:, 0:1]),
        np.maximum(f1[:, 3:4], f2[:, 3:4]),
        np.maximum(f1[:, 4:5], f2[:, 4:5]),
        f2[:, 0:1] - f1[:, 0:1],
    ]
    lex = np.empty((n, 6), dtype=np.float32)
    for i, (w, seq) in enumerate(zip(words, token_seqs)):
        lex[i, 0] = len(seq)
        lex[i, 1] = len(w)
        lex[i, 2] = seq[0]
        lex[i, 3] = max(seq)
        lex[i, 4] = sum(seq) / len(seq)
        lex[i, 5] = 1.0 if (not w.isascii() and any(ord(ch) < 128 for ch in w)) else (2.0 if w.isascii() else 0.0)
    cols.append(lex)
    return np.hstack(cols)


def calibrate_weights(
    words: list[str],
    X,
    anchors: dict[str, int],
    log_space: bool = True,
    default_min: int = 1,
    default_max: int = 350_000,
    delta_domain: "np.ndarray | None" = None,
    domain_threshold: float = 1.3,
    domain_slope: float = 0.135,
    domain_max_boost: float = 0.54,
) -> tuple["np.ndarray", dict]:
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    word_to_idx = {w: i for i, w in enumerate(words)}
    anchor_idxs, anchor_vals = [], []
    for w, wt in anchors.items():
        idx = word_to_idx.get(w)
        if idx is not None and wt > 0:
            anchor_idxs.append(idx)
            anchor_vals.append(wt)

    if len(anchor_idxs) < 50:
        # Unsupervised rank-based fallback when fewer than 50 valid empirical anchors match
        raw_score = X[:, 3] if X.shape[1] <= 11 else X[:, 11]  # mean_lp across context(s)
        z = (raw_score - raw_score.mean()) / max(float(raw_score.std()), 1e-6)
        log_w = 2.3 + 0.85 * z
        if delta_domain is not None:
            log_w = log_w + bayesian_domain_log10_tilt(delta_domain, domain_threshold, domain_slope, domain_max_boost)
        weights = np.clip(np.rint(np.power(10.0, log_w) - 1.0), default_min, default_max).astype(np.uint32)
        return weights, {"mode": "unsupervised_prior", "anchor_count": len(anchor_idxs)}

    anchor_idxs_np = np.array(anchor_idxs, dtype=np.int64)
    raw_vals_np = np.array(anchor_vals, dtype=np.float64)

    # Auto-detect compressed linear rank scale (e.g. ECDICT 900,000..999,999) vs Zipfian count scale
    ratio_p99_p10 = float(np.percentile(raw_vals_np, 99)) / max(1.0, float(np.percentile(raw_vals_np, 10)))
    use_log = log_space and (ratio_p99_p10 >= 5.0)

    if use_log:
        y = np.log10(raw_vals_np + 1.0).astype(np.float32)
        w_min = default_min
        w_max = max(default_max, int(np.percentile(raw_vals_np, 99.95)))
    else:
        y = raw_vals_np.astype(np.float32)
        w_min = int(np.percentile(raw_vals_np, 0.5))
        w_max = int(np.percentile(raw_vals_np, 99.95))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X[anchor_idxs_np], y, test_size=0.15, random_state=42
    )
    reg_val = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=25, l2_regularization=1.0, random_state=42
    )
    reg_val.fit(X_tr, y_tr)
    pred_val = reg_val.predict(X_val)
    r2 = float(r2_score(y_val, pred_val))
    mae = float(mean_absolute_error(y_val, pred_val))
    rho = float(spearmanr(y_val, pred_val).statistic)

    reg = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=25, l2_regularization=1.0, random_state=42
    )
    reg.fit(X[anchor_idxs_np], y)
    anchor_pred = reg.predict(X[anchor_idxs_np])
    mu_y, std_y = float(y.mean()), float(y.std())
    mu_p, std_p = float(anchor_pred.mean()), float(anchor_pred.std())
    stretch = min(1.25, max(1.0, std_y / max(std_p, 1e-6)))

    preds = np.empty(len(words), dtype=np.float32)
    for s in range(0, len(words), 250_000):
        p = reg.predict(X[s : s + 250_000])
        preds[s : s + 250_000] = mu_y + stretch * (p - mu_p)

    boosted_count = 0
    if delta_domain is not None:
        if use_log:
            tilt = bayesian_domain_log10_tilt(delta_domain, domain_threshold, domain_slope, domain_max_boost)
            boosted_count = int(np.count_nonzero(tilt > 0))
            preds = preds + tilt
        else:
            span = max(1000.0, float(w_max - w_min))
            lin_tilt = np.clip((delta_domain - max(domain_threshold, 1.5)) * (0.022 * span), 0.0, 0.085 * span).astype(np.float32)
            boosted_count = int(np.count_nonzero(lin_tilt > 0))
            preds = preds + lin_tilt

    if use_log:
        weights = np.clip(np.rint(np.power(10.0, preds) - 1.0), w_min, w_max).astype(np.uint32)
    else:
        weights = np.clip(np.rint(preds), w_min, w_max).astype(np.uint32)

    meta = {
        "mode": "log10_gbdt" if use_log else "linear_gbdt",
        "anchor_count": len(anchor_idxs_np),
        "val_r2": round(r2, 4),
        "val_mae": round(mae, 4),
        "val_spearman": round(rho, 4),
    }
    if delta_domain is not None:
        meta["domain_boosted_count"] = boosted_count
    return weights, meta


def score_payload_file(
    payload_path: Path,
    output_path: Path,
    model_id: str = "Qwen/Qwen3.5-0.8B-Base",
    device: str = "auto",
    max_batch: int = 2048,
    cn_prefixes: tuple[str, str] = ("。\n", "提到"),
    en_prefixes: tuple[str, str] = (".\n", "The "),
    domain_prompt: str | None = None,
    general_prompt: str = "日常与百科常见词条：",
    en_domain_prompt: str | None = None,
    en_general_prompt: str = "Common words and terms: ",
    domain_threshold: float = 1.3,
    domain_slope: float = 0.135,
    domain_max_boost: float = 0.54,
) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    dev, dtype = select_device_and_dtype(device)
    dev_label = torch.cuda.get_device_name(0) if dev.startswith("cuda") else dev
    print(f"Loading {model_id} on {dev_label} ({dtype}) ...", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    load_kwargs = {"dtype": dtype}
    if dev.startswith("cuda"):
        load_kwargs["device_map"] = dev
        try:
            import kernels  # noqa: F401
            full_model = AutoModelForCausalLM.from_pretrained(model_id, use_kernels=True, **load_kwargs)
        except Exception:
            full_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    else:
        full_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(dev)

    full_model.eval()
    backbone, lm_head = extract_backbone_and_head(full_model)
    max_err = verify_trie_exactness(tok, full_model, backbone, lm_head, device=dev)
    print(f"Prefix-Trie equivalence verified (max_abs_diff={max_err:.5f})", flush=True)

    with gzip.open(payload_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    cn_words = payload.get("cn_words", [])
    cn_anchors = payload.get("cn_anchors", {})
    en_words = payload.get("en_words", [])
    en_anchors = payload.get("en_anchors", {})
    del payload

    cn_weights = np.array([], dtype=np.uint32)
    cn_delta_domain: np.ndarray | None = None
    cn_metrics: dict = {}
    if cn_words:
        print(f"\nScoring {len(cn_words):,} CJK/Mixed words ...", flush=True)
        cn_seqs = batch_tokenize(tok, cn_words)
        cn_trie = build_prefix_trie(cn_seqs)
        print(
            f"  Prefix Trie compressed {len(cn_words):,} words -> {len(cn_trie['leaves']):,} leaf paths",
            flush=True,
        )
        if domain_prompt:
            cn_f_gen = trie_score_pass(tok, full_model, backbone, lm_head, cn_trie, general_prompt, device=dev, max_batch=max_batch)
            cn_f_dom = trie_score_pass(tok, full_model, backbone, lm_head, cn_trie, domain_prompt, device=dev, max_batch=max_batch)
            del cn_trie
            cn_delta_domain = compute_domain_lift(cn_f_dom[:, 0], cn_f_gen[:, 0])
            X_cn = build_single_context_features(cn_words, cn_seqs, cn_f_gen)
            cn_weights, cn_metrics = calibrate_weights(
                cn_words,
                X_cn,
                cn_anchors,
                log_space=True,
                delta_domain=cn_delta_domain,
                domain_threshold=domain_threshold,
                domain_slope=domain_slope,
                domain_max_boost=domain_max_boost,
            )
            cn_metrics["general_prompt"] = general_prompt
            cn_metrics["domain_prompt"] = domain_prompt
        else:
            cn_f1 = trie_score_pass(tok, full_model, backbone, lm_head, cn_trie, cn_prefixes[0], device=dev, max_batch=max_batch)
            cn_f2 = trie_score_pass(tok, full_model, backbone, lm_head, cn_trie, cn_prefixes[1], device=dev, max_batch=max_batch)
            del cn_trie
            X_cn = build_feature_matrix(cn_words, cn_seqs, cn_f1, cn_f2)
            cn_weights, cn_metrics = calibrate_weights(cn_words, X_cn, cn_anchors, log_space=True)
        print(f"  CJK calibration: {cn_metrics}", flush=True)

    en_weights = np.array([], dtype=np.uint32)
    en_delta_domain: np.ndarray | None = None
    en_metrics: dict = {}
    if en_words:
        print(f"\nScoring {len(en_words):,} English words ...", flush=True)
        en_seqs = batch_tokenize(tok, en_words)
        en_trie = build_prefix_trie(en_seqs)
        print(
            f"  Prefix Trie compressed {len(en_words):,} words -> {len(en_trie['leaves']):,} leaf paths",
            flush=True,
        )
        eff_en_dom = en_domain_prompt
        if eff_en_dom is None and domain_prompt:
            eff_en_dom = "Computer science and programming terms: "
        if eff_en_dom:
            en_f_gen = trie_score_pass(tok, full_model, backbone, lm_head, en_trie, en_general_prompt, device=dev, max_batch=max_batch)
            en_f_dom = trie_score_pass(tok, full_model, backbone, lm_head, en_trie, eff_en_dom, device=dev, max_batch=max_batch)
            del en_trie
            en_delta_domain = compute_domain_lift(en_f_dom[:, 0], en_f_gen[:, 0])
            X_en = build_single_context_features(en_words, en_seqs, en_f_gen)
            en_weights, en_metrics = calibrate_weights(
                en_words,
                X_en,
                en_anchors,
                log_space=True,
                delta_domain=en_delta_domain,
                domain_threshold=domain_threshold,
                domain_slope=domain_slope,
                domain_max_boost=domain_max_boost,
            )
            en_metrics["general_prompt"] = en_general_prompt
            en_metrics["domain_prompt"] = eff_en_dom
        else:
            en_f1 = trie_score_pass(tok, full_model, backbone, lm_head, en_trie, en_prefixes[0], device=dev, max_batch=max_batch)
            en_f2 = trie_score_pass(tok, full_model, backbone, lm_head, en_trie, en_prefixes[1], device=dev, max_batch=max_batch)
            del en_trie
            X_en = build_feature_matrix(en_words, en_seqs, en_f1, en_f2)
            en_weights, en_metrics = calibrate_weights(en_words, X_en, en_anchors, log_space=True)
        print(f"  EN calibration: {en_metrics}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rime_qwen_scores_") as tmp_str:
        tmp_dir = Path(tmp_str)
        np.save(tmp_dir / "cn_weights.npy", cn_weights)
        np.save(tmp_dir / "en_weights.npy", en_weights)
        extra_arcs: list[str] = []
        if cn_delta_domain is not None:
            np.save(tmp_dir / "cn_delta_domain.npy", np.round(cn_delta_domain, 3).astype(np.float16))
            extra_arcs.append("cn_delta_domain.npy")
        if en_delta_domain is not None:
            np.save(tmp_dir / "en_delta_domain.npy", np.round(en_delta_domain, 3).astype(np.float16))
            extra_arcs.append("en_delta_domain.npy")
        metrics = {
            "device": dev_label,
            "model": model_id,
            "cn_count": len(cn_words),
            "en_count": len(en_words),
            "cn": cn_metrics,
            "en": en_metrics,
            "total_seconds": round(time.time() - t_start, 1),
        }
        (tmp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        with tarfile.open(output_path, "w:gz", compresslevel=6) as tar:
            tar.add(tmp_dir / "cn_weights.npy", arcname="cn_weights.npy")
            tar.add(tmp_dir / "en_weights.npy", arcname="en_weights.npy")
            for arc in extra_arcs:
                tar.add(tmp_dir / arc, arcname=arc)
            tar.add(tmp_dir / "metrics.json", arcname="metrics.json")

    return metrics
