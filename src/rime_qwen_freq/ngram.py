"""
Offline CausalLM-to-Octagram (`.gram`) N-gram Distillation & Homophone Typo Audit.

Why PMI instead of raw joint/conditional counts in Rime?
In `librime` (`poet.cc` + `grammar.h`), the word-graph transition score is:
    Score(context -> word) = ln(dict_weight(word)) + Octagram::Query(context, word)
Because `ln(dict_weight(word))` already supplies the unigram log-prior `ln P(word)`,
the exact Markov log-probability `ln P(word | context)` requires `Octagram::Query` to
supply Pointwise Mutual Information (conditional log-gain):
    PMI(context, word) = ln P(word | context) - ln P(word)

This module distills high-PMI collocations and sentence-ending `$` markers directly
from a causal language model's `lm_head` logit distribution (evaluating billions of
single-token transitions in vectorized matrix form plus 1-step multi-token prefix
expansions), and audits existing `.gram` tables to purge scraped homophone typos.
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import time
from pathlib import Path

from .scorer import (
    batch_tokenize,
    build_prefix_trie,
    extract_backbone_and_head,
    select_device_and_dtype,
    trie_score_pass,
)

TONE_MAP = str.maketrans("āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜüńňǹ", "aaaaooooeeeeiiiiuuuuvvvvvnnn")


def _forward_last_logits(full_model, backbone, lm_head, input_ids):
    """Return float32 next-token logits `[batch, vocab]` at the final sequence position."""
    if backbone is not None and lm_head is not None:
        hs = backbone(input_ids=input_ids, use_cache=False).last_hidden_state
        return lm_head(hs[:, -1, :]).float()
    return full_model(input_ids=input_ids, use_cache=False).logits[:, -1, :].float()


def canonical_fuzzy_syllable(syl: str) -> str:
    """Normalize tone, retroflex (zh/ch/sh -> z/c/s), and nasal codas (ing/eng/ang -> in/en/an)."""
    s = syl.translate(TONE_MAP).strip().lower()
    if s.startswith("zh"):
        s = "z" + s[2:]
    elif s.startswith("ch"):
        s = "c" + s[2:]
    elif s.startswith("sh"):
        s = "s" + s[2:]
    if s.endswith("ing"):
        s = s[:-3] + "in"
    elif s.endswith("eng"):
        s = s[:-3] + "en"
    elif s.endswith("ang"):
        s = s[:-3] + "an"
    return s


def pmi_to_octagram_scale(
    cond_lp: float,
    pmi: float,
    is_domain: bool = False,
    collocation_penalty: float = -14.0,
    min_val: int = 154_000,
    max_val: int = 232_000,
) -> int:
    """
    Map conditional log-probability and PMI (in nats) onto `librime-octagram`'s integer scale
    (`stored_int = round((|collocation_penalty| + bonus) * 10000)`).
    """
    fluency_adj = max(-2.0, min(1.8, (cond_lp + 4.8) * 0.55))
    dom_boost = 0.65 if is_domain else 0.0
    eff_bonus = 1.4 + 0.72 * pmi + fluency_adj + dom_boost
    base_offset = abs(collocation_penalty)
    raw_int = round((base_offset + eff_bonus) * 10_000.0)
    return max(min_val, min(max_val, int(raw_int)))


def eos_to_rear_scale(
    eos_lp: float,
    rear_penalty: float = -20.0,
    min_val: int = 196_000,
    max_val: int = 232_000,
) -> int:
    """Map clause-ending log-probability `ln P(EOS/punct | w)` onto Octagram `$` rear scale."""
    base_offset = abs(rear_penalty)
    raw_int = round((base_offset + (eos_lp + 2.6) * 1.45) * 10_000.0)
    return max(min_val, min(max_val, int(raw_int)))


def distill_ngrams_from_lexicon(
    lexicon_words: list[str],
    context_words: list[str],
    output_tsv_gz: Path,
    model_id: str = "Qwen/Qwen3-14B-Base",
    device: str = "auto",
    general_prompt: str = "常见中文语句片段：",
    domain_prompt: str | None = "计算机科学与人工智能技术讨论：",
    domain_seed_words: set[str] | None = None,
    top_k_v1: int = 120,
    top_k_t1: int = 8,
    min_pmi: float = 2.35,
) -> dict:
    """
    Distill 3..6 character multi-word collocations and Rear `$` markers directly from a CausalLM.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    dev, dtype = select_device_and_dtype(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    load_kwargs = {"dtype": dtype}
    if dev.startswith("cuda"):
        load_kwargs["device_map"] = dev
        full_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    else:
        full_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(dev)
    full_model.eval()
    backbone, lm_head = extract_backbone_and_head(full_model)

    domain_seeds = set(domain_seed_words or ())
    unified = sorted(set(lexicon_words) | set(context_words) | domain_seeds)
    w2i = {w: i for i, w in enumerate(unified)}
    uni_seqs = batch_tokenize(tok, unified, max_toks=16)
    uni_trie = build_prefix_trie(uni_seqs)

    feats_gen = trie_score_pass(tok, full_model, backbone, lm_head, uni_trie, general_prompt, device=dev)
    feats_dom = (
        trie_score_pass(tok, full_model, backbone, lm_head, uni_trie, domain_prompt, device=dev)
        if domain_prompt
        else feats_gen
    )
    del uni_trie

    # Root unigram distribution for 1st-token PMI screening
    p_gen_ids = tok.encode(general_prompt, add_special_tokens=False) or [0]
    with torch.inference_mode():
        z0 = _forward_last_logits(full_model, backbone, lm_head, torch.tensor([p_gen_ids], device=dev))[0]
        root_lp_gen = (z0 - torch.logsumexp(z0, dim=-1)).cpu().numpy()

    v1_words, v1_toks, v1_clens, v1_blp = [], [], [], []
    v2_by_t1: dict[int, list[tuple[int, str, int, float]]] = collections.defaultdict(list)
    for w in lexicon_words:
        idx = w2i[w]
        seq = uni_seqs[idx]
        if len(seq) == 1:
            v1_words.append(w)
            v1_toks.append(seq[0])
            v1_clens.append(len(w))
            v1_blp.append(float(feats_gen[idx, 0]))
        elif len(seq) == 2:
            v2_by_t1[seq[0]].append((seq[1], w, len(w), float(feats_gen[idx, 0])))

    v1_tok_dev = torch.tensor(v1_toks, device=dev, dtype=torch.long)
    v1_len_dev = torch.tensor(v1_clens, device=dev, dtype=torch.int32)
    v1_blp_dev = torch.tensor(v1_blp, device=dev, dtype=torch.float32)

    v2_t1_sorted = sorted(v2_by_t1.keys())
    v2_t1_dev = torch.tensor(v2_t1_sorted, device=dev, dtype=torch.long)
    v2_t1_blp_dev = torch.tensor([float(root_lp_gen[t]) for t in v2_t1_sorted], device=dev, dtype=torch.float32)

    eos_ids = []
    for s_eos in ("。", "！", "？", "；", "\n"):
        ids = tok.encode(s_eos, add_special_tokens=False)
        if len(ids) == 1:
            eos_ids.append(ids[0])
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)
    eos_dev = torch.tensor(sorted(set(eos_ids)), device=dev, dtype=torch.long)

    by_d: dict[int, list[str]] = collections.defaultdict(list)
    for w in context_words:
        by_d[len(uni_seqs[w2i[w]])].append(w)

    p_tensor = torch.tensor(p_gen_ids, device=dev, dtype=torch.long)
    m = len(p_gen_ids)

    pure_map: dict[str, int] = {}
    pending_t1: list[tuple[str, int, float]] = []
    rear_cnt = 0

    with torch.inference_mode():
        for d in sorted(by_d.keys()):
            c_list = by_d[d]
            L = m + d
            bsz = min(256, max(32, 8192 // L))
            for s in range(0, len(c_list), bsz):
                batch_w = c_list[s : s + bsz]
                b = len(batch_w)
                arr = np.array([uni_seqs[w2i[w]] for w in batch_w], dtype=np.int64)
                w1_lens = np.array([len(w) for w in batch_w], dtype=np.int32)
                inp = torch.empty((b, L), device=dev, dtype=torch.long)
                inp[:, :m] = p_tensor
                inp[:, m:] = torch.from_numpy(arr).to(dev)

                z = _forward_last_logits(full_model, backbone, lm_head, inp)
                logp = z - torch.logsumexp(z, dim=-1, keepdim=True)

                eos_lp = torch.logsumexp(logp[:, eos_dev], dim=-1).cpu().numpy()
                for r, w1 in enumerate(batch_w):
                    if float(eos_lp[r]) >= -2.85 and len(w1) <= 4:
                        pure_map[f"{w1}$"] = eos_to_rear_scale(float(eos_lp[r]))
                        rear_cnt += 1

                if len(v1_words) > 0:
                    cond_v1 = logp.index_select(1, v1_tok_dev)
                    pmi_v1 = cond_v1 - v1_blp_dev.unsqueeze(0)
                    tot_len = torch.from_numpy(w1_lens).to(dev).unsqueeze(1) + v1_len_dev.unsqueeze(0)
                    mask = (tot_len >= 3) & (tot_len <= 6) & (cond_v1 >= -8.0) & (pmi_v1 >= min_pmi)
                    rk = torch.where(mask, cond_v1 + 0.65 * pmi_v1, torch.tensor(-1e9, device=dev))
                    ks = min(top_k_v1, rk.shape[1])
                    tv, tc = torch.topk(rk, k=ks, dim=1)
                    tv_np, tc_np = tv.cpu().numpy(), tc.cpu().numpy()
                    sc_np = torch.gather(cond_v1, 1, tc).cpu().numpy()
                    sp_np = torch.gather(pmi_v1, 1, tc).cpu().numpy()
                    for r, w1 in enumerate(batch_w):
                        is_dom = (w1 in domain_seeds) or (float(feats_dom[w2i[w1], 0] - feats_gen[w2i[w1], 0]) >= 1.5)
                        for j in range(ks):
                            if tv_np[r, j] < -1e8:
                                break
                            col_str = w1 + v1_words[int(tc_np[r, j])]
                            gv = pmi_to_octagram_scale(float(sc_np[r, j]), float(sp_np[r, j]), is_domain=is_dom)
                            if gv > pure_map.get(col_str, 0):
                                pure_map[col_str] = gv

                if top_k_t1 > 0 and len(v2_t1_sorted) > 0:
                    cond_t1 = logp.index_select(1, v2_t1_dev)
                    pmi_t1 = cond_t1 - v2_t1_blp_dev.unsqueeze(0)
                    tmask = (cond_t1 >= -6.5) & (pmi_t1 >= 1.3)
                    trk = torch.where(tmask, cond_t1 + 0.6 * pmi_t1, torch.tensor(-1e9, device=dev))
                    kt1 = min(top_k_t1, trk.shape[1])
                    tv1, tc1 = torch.topk(trk, k=kt1, dim=1)
                    tv1_np, tc1_np = tv1.cpu().numpy(), tc1.cpu().numpy()
                    sc1_np = torch.gather(cond_t1, 1, tc1).cpu().numpy()
                    for r, w1 in enumerate(batch_w):
                        if len(w1) <= 4:
                            for j in range(kt1):
                                if tv1_np[r, j] < -1e8:
                                    break
                                pending_t1.append((w1, v2_t1_sorted[int(tc1_np[r, j])], float(sc1_np[r, j])))

        if pending_t1:
            ext_by_d: dict[int, list[tuple[str, int, float]]] = collections.defaultdict(list)
            for w1, t1_id, lp1 in pending_t1:
                ext_by_d[len(uni_seqs[w2i[w1]]) + 1].append((w1, t1_id, lp1))
            for d_ext in sorted(ext_by_d.keys()):
                items = ext_by_d[d_ext]
                L = m + d_ext
                bsz = min(512, max(32, 12288 // L))
                for s in range(0, len(items), bsz):
                    chunk = items[s : s + bsz]
                    b = len(chunk)
                    arr = np.empty((b, d_ext), dtype=np.int64)
                    for r, (w1, t1_id, _) in enumerate(chunk):
                        arr[r, :-1] = uni_seqs[w2i[w1]]
                        arr[r, -1] = t1_id
                    inp = torch.empty((b, L), device=dev, dtype=torch.long)
                    inp[:, :m] = p_tensor
                    inp[:, m:] = torch.from_numpy(arr).to(dev)
                    z = _forward_last_logits(full_model, backbone, lm_head, inp)
                    lse = torch.logsumexp(z, dim=-1)
                    fr, ft2, meta = [], [], []
                    for r, (w1, t1_id, lp1) in enumerate(chunk):
                        is_dom = (w1 in domain_seeds) or (float(feats_dom[w2i[w1], 0] - feats_gen[w2i[w1], 0]) >= 1.5)
                        for t2_id, w2_str, l2, w2_blp in v2_by_t1[t1_id]:
                            if 3 <= len(w1) + l2 <= 6:
                                fr.append(r)
                                ft2.append(t2_id)
                                meta.append((w1 + w2_str, lp1, w2_blp, is_dom))
                    if fr:
                        lp2 = (z[torch.tensor(fr, device=dev), torch.tensor(ft2, device=dev)] - lse[torch.tensor(fr, device=dev)]).cpu().numpy()
                        for k_i, (col_str, lp1, w2_blp, is_dom) in enumerate(meta):
                            clp = lp1 + float(lp2[k_i])
                            pmi = clp - w2_blp
                            if clp >= -8.5 and pmi >= (min_pmi + 0.2):
                                gv = pmi_to_octagram_scale(clp, pmi, is_domain=is_dom)
                                if gv > pure_map.get(col_str, 0):
                                    pure_map[col_str] = gv

    output_tsv_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_tsv_gz, "wt", encoding="utf-8", compresslevel=6) as f:
        for k, v in sorted(pure_map.items()):
            f.write(f"{k}\t{v}\n")

    return {
        "model": model_id,
        "total_keys": len(pure_map),
        "rear_markers": rear_cnt,
        "seconds": round(time.time() - t0, 1),
        "output": str(output_tsv_gz),
    }
