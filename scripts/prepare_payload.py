#!/usr/bin/env python3
"""
Step 1 (Local): Extract unweighted / constant-weighted target vocabularies and
ground-truth calibration anchors from a Rime configuration directory into a compact
gzipped JSON payload for GPU scoring.
"""
import argparse
import gzip
import json
import math
import os
import random
from pathlib import Path

DEFAULT_CN_TARGETS = [
    "zhwiki.dict.yaml",
    "zhwiktionary.dict.yaml",
    "moegirl.dict.yaml",
    "web_slang.dict.yaml",
    "wanxiang_mixed.dict.yaml",
]

DEFAULT_EN_TARGETS = [
    "rime_ice.en.dict.yaml",
    "rime_ice.en_ext.dict.yaml",
]

DEFAULT_CN_ANCHOR_DICTS = [
    "wanxiang_jichu.dict.yaml",
    "rime_mint.base.dict.yaml",
    "wanxiang_mingren.dict.yaml",
    "wanxiang_yiren.dict.yaml",
    "rime_mint.chars.dict.yaml",
]


def default_rime_dir() -> Path:
    candidates = [
        Path.home() / "Library" / "Rime",
        Path.home() / ".local" / "share" / "fcitx5" / "rime",
        Path.home() / ".config" / "ibus" / "rime",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


def iter_dict_entries(filepath: Path):
    in_body = False
    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\r\n")
            if not in_body:
                if s == "...":
                    in_body = True
                continue
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t")
            word = parts[0]
            code = parts[1] if len(parts) >= 2 else ""
            weight_str = parts[2].strip() if len(parts) >= 3 else ""
            yield word, code, weight_str


def load_clean_weights(filepath: Path) -> dict[str, int]:
    res: dict[str, int] = {}
    if not filepath.is_file():
        return res
    for w, _, wt in iter_dict_entries(filepath):
        if not wt:
            continue
        try:
            v = int(wt)
        except ValueError:
            continue
        if v > 0 and (w not in res or v > res[w]):
            res[w] = v
    return res


def main():
    parser = argparse.ArgumentParser(description="Prepare Rime vocabulary payload for Qwen GPU scoring.")
    parser.add_argument("--rime-dir", type=Path, default=default_rime_dir(), help="Path to Rime user directory")
    parser.add_argument("--dicts-subdir", default="dicts", help="Subdirectory containing .dict.yaml files")
    parser.add_argument("--cn-targets", nargs="*", default=DEFAULT_CN_TARGETS, help="CN/Mixed target .dict.yaml filenames")
    parser.add_argument("--en-targets", nargs="*", default=DEFAULT_EN_TARGETS, help="EN target .dict.yaml filenames")
    parser.add_argument("--cn-anchors-max", type=int, default=100_000, help="Max stratified CN anchor words")
    parser.add_argument("--en-anchors-max", type=int, default=25_000, help="Max EN anchor words from easy_en.dict.yaml")
    parser.add_argument("--output", type=Path, default=Path("payload.json.gz"), help="Output .json.gz payload path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    dicts_dir = args.rime_dir / args.dicts_subdir
    if not dicts_dir.is_dir():
        raise FileNotFoundError(f"Dicts directory not found: {dicts_dir}")

    cn_words: set[str] = set()
    for fn in args.cn_targets:
        p = dicts_dir / fn
        if not p.is_file():
            print(f"[Skip missing CN target] {p}")
            continue
        cnt = 0
        for w, _, _ in iter_dict_entries(p):
            if w:
                cn_words.add(w)
                cnt += 1
        print(f"[CN Target] {fn:30s}: {cnt:9,d} rows")

    en_words: set[str] = set()
    for fn in args.en_targets:
        p = dicts_dir / fn
        if not p.is_file():
            print(f"[Skip missing EN target] {p}")
            continue
        cnt = 0
        for w, _, _ in iter_dict_entries(p):
            if w:
                en_words.add(w)
                cnt += 1
        print(f"[EN Target] {fn:30s}: {cnt:9,d} rows")

    # Build CN ground-truth calibration pool
    cn_gt_pool: dict[str, int] = {}
    for anchor_fn in DEFAULT_CN_ANCHOR_DICTS:
        p = dicts_dir / anchor_fn
        d = load_clean_weights(p)
        if not d:
            continue
        for k, v in d.items():
            if anchor_fn.endswith("chars.dict.yaml") and (len(k) != 1 or v < 10):
                continue
            if k not in cn_gt_pool:
                cn_gt_pool[k] = v

    buckets: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for w, v in cn_gt_pool.items():
        l_bin = min(len(w), 6)
        v_bin = min(int(math.log10(v + 1) * 2), 12)
        buckets.setdefault((l_bin, v_bin), []).append((w, v))

    cn_anchors: dict[str, int] = {}
    target_per_bucket = max(1, args.cn_anchors_max // max(1, len(buckets)))
    leftovers: list[tuple[str, int]] = []
    for _, items in sorted(buckets.items()):
        random.shuffle(items)
        for w, v in items[:target_per_bucket]:
            cn_anchors[w] = v
        leftovers.extend(items[target_per_bucket:])

    if len(cn_anchors) < args.cn_anchors_max and leftovers:
        random.shuffle(leftovers)
        for w, v in leftovers[: args.cn_anchors_max - len(cn_anchors)]:
            cn_anchors[w] = v

    easy_en = load_clean_weights(dicts_dir / "easy_en.dict.yaml")
    en_items = list(easy_en.items())
    random.shuffle(en_items)
    en_anchors = dict(en_items[: args.en_anchors_max])

    all_cn_list = sorted(cn_words | set(cn_anchors.keys()))
    all_en_list = sorted(en_words | set(en_anchors.keys()))

    payload = {
        "cn_words": all_cn_list,
        "cn_anchors": cn_anchors,
        "en_words": all_en_list,
        "en_anchors": en_anchors,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(payload, f, ensure_ascii=False)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\nSaved {args.output} ({size_mb:.2f} MB)")
    print(f"  CN unique words to score : {len(all_cn_list):,} (anchors: {len(cn_anchors):,})")
    print(f"  EN unique words to score : {len(all_en_list):,} (anchors: {len(en_anchors):,})")


if __name__ == "__main__":
    main()
