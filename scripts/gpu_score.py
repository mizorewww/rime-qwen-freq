#!/usr/bin/env python3
"""
Step 2 (CUDA GPU / Colab): Exact Prefix-Trie Teacher-Forcing Scorer & Anchor Calibrator
using Qwen/Qwen3.5-0.8B-Base.

Key design points:
1. Prefix-Trie deduplication evaluates shared BPE proper prefixes only once while preserving
   exact causal token log-probabilities (verified automatically before full run).
2. Sparse GPU-side gather extracts only required child token log-probs on device, avoiding
   multi-gigabyte [B, vocab_size=248320] PCIe host transfers.
3. Dual-context scoring ("。\n" sentence start + "提到" entity mention for CN; ".\n" + "The " for EN)
   followed by HistGradientBoostingRegressor + variance-preserving stretch onto Rime's real
   Wanxiang / ECDICT weight scales.
"""
import argparse
import gzip
import json
import logging
import math
import os
import tarfile
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.getLogger("kernels").setLevel(logging.ERROR)

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer


def batch_tokenize(tok, words: list[str], max_toks: int = 24) -> list[list[int]]:
    t0 = time.time()
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
    print(f"  Tokenized {len(words):,} words in {time.time() - t0:.1f}s", flush=True)
    return all_ids


def build_prefix_trie(token_seqs: list[list[int]]) -> dict:
    t0 = time.time()
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
            node_query_toks[u] = np.fromiter(ed.keys(), dtype=np.int64, count=len(ed))
            node_query_eids[u] = np.fromiter(ed.values(), dtype=np.int64, count=len(ed))
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

    dt = time.time() - t0
    print(
        f"  Built Prefix Trie in {dt:.1f}s: {len(token_seqs):,} words -> "
        f"{num_internal:,} prefixes -> {len(leaves):,} maximal leaf paths "
        f"({len(token_seqs)/max(1, len(leaves)):.2f}x compression)",
        flush=True,
    )
    return {
        "num_edges": num_edges,
        "word_edges": word_edges,
        "node_seq": node_seq,
        "node_query_toks": node_query_toks,
        "node_query_eids": node_query_eids,
        "leaves": leaves,
        "leaf_owned_steps": leaf_owned_steps,
    }


@torch.inference_mode()
def trie_score_pass(tok, backbone, lm_head, trie: dict, prefix_text: str, max_batch: int = 2048) -> np.ndarray:
    t0 = time.time()
    edge_lp = np.empty(trie["num_edges"], dtype=np.float32)

    p_ids = tok.encode(prefix_text, add_special_tokens=False)
    m = len(p_ids)
    p_tensor = torch.tensor(p_ids, device="cuda", dtype=torch.long)

    # Root pass
    hs0 = backbone(input_ids=p_tensor.unsqueeze(0), use_cache=False).last_hidden_state
    z0 = lm_head(hs0[0, m - 1]).float()
    lse0 = torch.logsumexp(z0, dim=-1)
    if trie["node_query_toks"][0] is not None:
        q0_gpu = torch.from_numpy(trie["node_query_toks"][0]).to("cuda")
        edge_lp[trie["node_query_eids"][0]] = (z0.index_select(0, q0_gpu) - lse0).cpu().numpy()

    leaves = trie["leaves"]
    leaf_owned = trie["leaf_owned_steps"]
    node_seq = trie["node_seq"]
    by_len: dict[int, list[int]] = {}
    for idx, leaf_u in enumerate(leaves):
        by_len.setdefault(len(node_seq[leaf_u]), []).append(idx)

    total_leaves = len(leaves)
    done_leaves = 0
    last_log = t0

    for d in sorted(by_len.keys()):
        l_idxs = by_len[d]
        L = m + d
        # Keep batch <= 2048 so B * num_linear_heads (2048 * 16 = 32768) stays under CUDA gridDim.z <= 65535
        bsz = min(max_batch, max(64, 24576 // L))

        arr_cpu = np.empty((len(l_idxs), d), dtype=np.int64)
        for j, li in enumerate(l_idxs):
            arr_cpu[j] = node_seq[leaves[li]]

        for s in range(0, len(l_idxs), bsz):
            sub = l_idxs[s : s + bsz]
            b = len(sub)
            u_gpu = torch.from_numpy(arr_cpu[s : s + b]).to("cuda", non_blocking=True)
            inp = torch.empty((b, L), device="cuda", dtype=torch.long)
            inp[:, :m] = p_tensor
            inp[:, m:] = u_gpu

            hs = backbone(input_ids=inp, use_cache=False).last_hidden_state

            rows: list[int] = []
            cols: list[int] = []
            owned_nodes: list[int] = []
            for r, li in enumerate(sub):
                for step_s, node_u in leaf_owned[li]:
                    rows.append(r)
                    cols.append(m + step_s)
                    owned_nodes.append(node_u)

            if owned_nodes:
                r_t = torch.tensor(rows, device="cuda", dtype=torch.long)
                c_t = torch.tensor(cols, device="cuda", dtype=torch.long)
                h_sel = hs[r_t, c_t, :]

                Q = len(owned_nodes)
                for qs in range(0, Q, 512):
                    qe = min(Q, qs + 512)
                    z = lm_head(h_sel[qs:qe]).float()
                    lse = torch.logsumexp(z, dim=-1)

                    # Sparse on-GPU gather: transfer kilobytes instead of gigabytes over PCIe
                    flat_rows = []
                    flat_toks = []
                    flat_eids = []
                    for offset, node_u in enumerate(owned_nodes[qs:qe]):
                        q_toks = trie["node_query_toks"][node_u]
                        q_eids = trie["node_query_eids"][node_u]
                        flat_rows.append(np.full(len(q_toks), offset, dtype=np.int64))
                        flat_toks.append(q_toks)
                        flat_eids.append(q_eids)

                    r_idx_np = np.concatenate(flat_rows)
                    t_idx_np = np.concatenate(flat_toks)
                    e_idx_np = np.concatenate(flat_eids)

                    r_idx_gpu = torch.from_numpy(r_idx_np).to("cuda", non_blocking=True)
                    t_idx_gpu = torch.from_numpy(t_idx_np).to("cuda", non_blocking=True)
                    vals = (z[r_idx_gpu, t_idx_gpu] - lse[r_idx_gpu]).cpu().numpy()
                    edge_lp[e_idx_np] = vals

            done_leaves += b
            now = time.time()
            if now - last_log >= 8.0:
                rate = done_leaves / (now - t0)
                eta = (total_leaves - done_leaves) / max(rate, 1.0)
                print(
                    f"    [{prefix_text!r}] leaves {done_leaves:,}/{total_leaves:,} "
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
            v = edge_lp[eids[0]]
            feats[i, :] = v
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
    print(f"  Finished pass {prefix_text!r} in {dt:.1f}s (effective {n_words/dt:,.0f} words/s)", flush=True)
    return feats


@torch.inference_mode()
def verify_trie_exactness(tok, full_model, backbone, lm_head):
    test_words = ["啊", "中国", "人工智能", "深度学习", "中华人民共和国", "5G", "README.md"]
    t_seqs = batch_tokenize(tok, test_words)
    t_trie = build_prefix_trie(t_seqs)
    trie_feats = trie_score_pass(tok, backbone, lm_head, t_trie, "。\n")

    p_ids = tok.encode("。\n", add_special_tokens=False)
    m = len(p_ids)
    max_err = 0.0
    for i, seq in enumerate(t_seqs):
        inp = torch.tensor([p_ids + seq[:-1]], device="cuda", dtype=torch.long)
        logits = full_model(input_ids=inp, use_cache=False).logits[0, m - 1 :, :].float()
        logp = F.log_softmax(logits, dim=-1)
        ref_sum = sum(logp[j, seq[j]].item() for j in range(len(seq)))
        max_err = max(max_err, abs(ref_sum - float(trie_feats[i, 0])))
    print(f"=== Trie Exactness Verification max_abs_diff = {max_err:.6f} ===", flush=True)
    assert max_err < 0.35, f"Trie verification failed: max_err={max_err}"


def build_features(words: list[str], token_seqs: list[list[int]], f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
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


def main():
    parser = argparse.ArgumentParser(description="Score Rime payload on CUDA GPU using Qwen3.5-0.8B-Base.")
    parser.add_argument("--payload", type=Path, default=Path("/content/payload.json.gz"))
    parser.add_argument("--output", type=Path, default=Path("/content/rime_scores.tar.gz"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--max-batch", type=int, default=2048)
    parser.add_argument("--use-kernels", action="store_true", default=True)
    args = parser.parse_args()

    t_start = time.time()
    assert torch.cuda.is_available(), "CUDA GPU is required!"
    print(f"=== Loading {args.model} on {torch.cuda.get_device_name(0)} ===", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        full_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda", use_kernels=args.use_kernels
        )
    except Exception as e:
        print(f"Note: use_kernels={args.use_kernels} unavailable ({e}), falling back to default kernels.", flush=True)
        full_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda"
        )
    full_model.eval()
    backbone = full_model.model
    lm_head = full_model.lm_head

    verify_trie_exactness(tok, full_model, backbone, lm_head)

    with gzip.open(args.payload, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    cn_words = payload["cn_words"]
    cn_anchors = payload["cn_anchors"]
    en_words = payload["en_words"]
    en_anchors = payload["en_anchors"]
    del payload

    # 1. CN/Mixed Scoring & Calibration
    cn_weights = np.array([], dtype=np.uint32)
    r2_cn = mae_cn = rho_cn = 0.0
    if cn_words:
        print(f"\n=== Scoring {len(cn_words):,} CN/Mixed words ===", flush=True)
        cn_seqs = batch_tokenize(tok, cn_words)
        cn_trie = build_prefix_trie(cn_seqs)
        cn_f1 = trie_score_pass(tok, backbone, lm_head, cn_trie, "。\n", max_batch=args.max_batch)
        cn_f2 = trie_score_pass(tok, backbone, lm_head, cn_trie, "提到", max_batch=args.max_batch)
        del cn_trie
        X_cn = build_features(cn_words, cn_seqs, cn_f1, cn_f2)

        word_to_idx_cn = {w: i for i, w in enumerate(cn_words)}
        anchor_idxs_cn, anchor_y_cn = [], []
        for w, wt in cn_anchors.items():
            idx = word_to_idx_cn.get(w)
            if idx is not None and wt > 0:
                anchor_idxs_cn.append(idx)
                anchor_y_cn.append(math.log10(wt + 1.0))

        anchor_idxs_cn = np.array(anchor_idxs_cn, dtype=np.int64)
        anchor_y_cn = np.array(anchor_y_cn, dtype=np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_cn[anchor_idxs_cn], anchor_y_cn, test_size=0.15, random_state=42
        )
        reg_val = HistGradientBoostingRegressor(
            max_iter=450, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=25, l2_regularization=1.0, random_state=42
        )
        reg_val.fit(X_tr, y_tr)
        pred_val = reg_val.predict(X_val)
        r2_cn = float(r2_score(y_val, pred_val))
        mae_cn = float(mean_absolute_error(y_val, pred_val))
        rho_cn = float(spearmanr(y_val, pred_val).statistic)
        print(f"[CN Validation n={len(y_val):,}] R2={r2_cn:.4f} | MAE(log10)={mae_cn:.4f} | Spearman={rho_cn:.4f}", flush=True)

        cn_reg = HistGradientBoostingRegressor(
            max_iter=450, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=25, l2_regularization=1.0, random_state=42
        )
        cn_reg.fit(X_cn[anchor_idxs_cn], anchor_y_cn)
        anchor_pred = cn_reg.predict(X_cn[anchor_idxs_cn])
        mu_y, std_y = float(anchor_y_cn.mean()), float(anchor_y_cn.std())
        mu_p, std_p = float(anchor_pred.mean()), float(anchor_pred.std())
        stretch_cn = min(1.25, max(1.0, std_y / max(std_p, 1e-6)))

        raw_pred_cn = np.empty(len(cn_words), dtype=np.float32)
        for s in range(0, len(cn_words), 250_000):
            p = cn_reg.predict(X_cn[s : s + 250_000])
            raw_pred_cn[s : s + 250_000] = mu_y + stretch_cn * (p - mu_p)
        cn_weights = np.clip(np.rint(np.power(10.0, raw_pred_cn) - 1.0), 1, 300_000).astype(np.uint32)

    # 2. EN Scoring & Calibration
    en_weights = np.array([], dtype=np.uint32)
    r2_en = rho_en = 0.0
    if en_words:
        print(f"\n=== Scoring {len(en_words):,} EN words ===", flush=True)
        en_seqs = batch_tokenize(tok, en_words)
        en_trie = build_prefix_trie(en_seqs)
        en_f1 = trie_score_pass(tok, backbone, lm_head, en_trie, ".\n", max_batch=args.max_batch)
        en_f2 = trie_score_pass(tok, backbone, lm_head, en_trie, "The ", max_batch=args.max_batch)
        del en_trie
        X_en = build_features(en_words, en_seqs, en_f1, en_f2)

        word_to_idx_en = {w: i for i, w in enumerate(en_words)}
        anchor_idxs_en, anchor_y_en = [], []
        for w, wt in en_anchors.items():
            idx = word_to_idx_en.get(w)
            if idx is not None and wt > 0:
                anchor_idxs_en.append(idx)
                anchor_y_en.append(float(wt))

        anchor_idxs_en = np.array(anchor_idxs_en, dtype=np.int64)
        anchor_y_en = np.array(anchor_y_en, dtype=np.float32)

        X_etr, X_eval, y_etr, y_eval = train_test_split(
            X_en[anchor_idxs_en], anchor_y_en, test_size=0.15, random_state=42
        )
        ereg_val = HistGradientBoostingRegressor(
            max_iter=350, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, random_state=42
        )
        ereg_val.fit(X_etr, y_etr)
        epred_val = ereg_val.predict(X_eval)
        r2_en = float(r2_score(y_eval, epred_val))
        rho_en = float(spearmanr(y_eval, epred_val).statistic)
        print(f"[EN Validation n={len(y_eval):,}] R2={r2_en:.4f} | Spearman={rho_en:.4f}", flush=True)

        en_reg = HistGradientBoostingRegressor(
            max_iter=350, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20, random_state=42
        )
        en_reg.fit(X_en[anchor_idxs_en], anchor_y_en)
        epred_anchor = en_reg.predict(X_en[anchor_idxs_en])
        emu_y, estd_y = float(anchor_y_en.mean()), float(anchor_y_en.std())
        emu_p, estd_p = float(epred_anchor.mean()), float(epred_anchor.std())
        stretch_en = min(1.25, max(1.0, estd_y / max(estd_p, 1e-6)))
        raw_pred_en = emu_y + stretch_en * (en_reg.predict(X_en) - emu_p)
        en_weights = np.clip(np.rint(raw_pred_en), 900_000, 999_995).astype(np.uint32)

    # 3. Package output
    tmp_dir = args.output.parent / "_rime_qwen_out"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.save(tmp_dir / "cn_weights.npy", cn_weights)
    np.save(tmp_dir / "en_weights.npy", en_weights)
    metrics = {
        "gpu": torch.cuda.get_device_name(0),
        "model": args.model,
        "cn_count": len(cn_words),
        "en_count": len(en_words),
        "cn_val_r2": r2_cn,
        "cn_val_mae_log10": mae_cn,
        "cn_val_spearman": rho_cn,
        "en_val_r2": r2_en,
        "en_val_spearman": rho_en,
        "total_seconds": round(time.time() - t_start, 1),
    }
    (tmp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    with tarfile.open(args.output, "w:gz", compresslevel=6) as tar:
        tar.add(tmp_dir / "cn_weights.npy", arcname="cn_weights.npy")
        tar.add(tmp_dir / "en_weights.npy", arcname="en_weights.npy")
        tar.add(tmp_dir / "metrics.json", arcname="metrics.json")

    print(f"\nSaved {args.output} ({args.output.stat().st_size / (1024*1024):.2f} MB) in {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
