#!/usr/bin/env python3
"""
Step 3 (Local, zero external dependencies): Read GPU-generated rime_scores.tar.gz,
back up original .dict.yaml files, write calibrated weights atomically, and optionally
trigger Rime redeployment + live reload.
"""
import argparse
import array
import ast
import gzip
import json
import os
import shutil
import struct
import subprocess
import tarfile
from pathlib import Path

from prepare_payload import DEFAULT_CN_TARGETS, DEFAULT_EN_TARGETS, default_rime_dir


def read_uint32_npy(path: Path) -> array.array:
    with path.open("rb") as f:
        magic = f.read(6)
        assert magic == b"\x93NUMPY", f"Invalid .npy header: {magic!r}"
        major, _ = struct.unpack("<BB", f.read(2))
        hlen = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        hdr = ast.literal_eval(f.read(hlen).decode("latin1").strip())
        assert hdr["descr"] in ("<u4", "|u4", "uint32"), f"Expected uint32 .npy, got {hdr}"
        arr = array.array("I")
        arr.frombytes(f.read())
        return arr


def rewrite_dict(filepath: Path, weight_map: dict[str, int], fallback_weight: int) -> tuple[int, int]:
    tmp_path = filepath.with_suffix(filepath.suffix + ".qwen_tmp")
    in_body = False
    updated = 0
    total_entries = 0

    with filepath.open("r", encoding="utf-8", errors="replace") as fin, \
         tmp_path.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            raw = line.rstrip("\r\n")
            if not in_body:
                if "统一权重 60000" in raw:
                    raw = raw.replace("统一权重 60000", "Qwen3.5-0.8B 离线蒸馏校准词频")
                fout.write(raw + "\n")
                if raw == "...":
                    in_body = True
                continue

            if not raw or raw.startswith("#"):
                fout.write(raw + "\n")
                continue

            total_entries += 1
            parts = raw.split("\t")
            word = parts[0]
            new_w = str(weight_map.get(word, fallback_weight))

            if len(parts) == 1:
                out_line = f"{word}\t{word}\t{new_w}"
            elif len(parts) == 2:
                out_line = f"{parts[0]}\t{parts[1]}\t{new_w}"
            else:
                parts[2] = new_w
                out_line = "\t".join(parts)

            fout.write(out_line + "\n")
            updated += 1

    os.replace(tmp_path, filepath)
    return total_entries, updated


def deploy_squirrel(rime_dir: Path):
    squirrel_macos = Path("/Library/Input Methods/Squirrel.app/Contents/MacOS")
    deployer = squirrel_macos / "rime_deployer"
    shared = Path("/Library/Input Methods/Squirrel.app/Contents/SharedSupport")
    squirrel_bin = squirrel_macos / "Squirrel"

    if deployer.is_file():
        print("Running rime_deployer --build ...")
        subprocess.run([str(deployer), "--build", str(rime_dir), str(shared)], check=True)
    if squirrel_bin.is_file():
        print("Triggering Squirrel --reload ...")
        subprocess.run([str(squirrel_bin), "--reload"], check=True)


def main():
    parser = argparse.ArgumentParser(description="Apply Qwen-calibrated word frequencies to Rime dictionaries.")
    parser.add_argument("--rime-dir", type=Path, default=default_rime_dir())
    parser.add_argument("--dicts-subdir", default="dicts")
    parser.add_argument("--payload", type=Path, default=Path("payload.json.gz"))
    parser.add_argument("--scores", type=Path, default=Path("rime_scores.tar.gz"))
    parser.add_argument("--backup-dir", type=Path, default=Path("backup_dicts"))
    parser.add_argument("--cn-targets", nargs="*", default=DEFAULT_CN_TARGETS)
    parser.add_argument("--en-targets", nargs="*", default=DEFAULT_EN_TARGETS)
    parser.add_argument("--deploy", action="store_true", help="Rebuild and reload Squirrel on macOS after writeback")
    args = parser.parse_args()

    extract_dir = args.scores.parent / "extracted_scores"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.scores, "r:gz") as tar:
        tar.extractall(extract_dir)

    with gzip.open(args.payload, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    cn_words = payload["cn_words"]
    en_words = payload["en_words"]
    cn_weights = read_uint32_npy(extract_dir / "cn_weights.npy")
    en_weights = read_uint32_npy(extract_dir / "en_weights.npy")
    metrics = json.loads((extract_dir / "metrics.json").read_text(encoding="utf-8"))

    assert len(cn_words) == len(cn_weights)
    assert len(en_words) == len(en_weights)
    cn_map = {w: int(v) for w, v in zip(cn_words, cn_weights)}
    en_map = {w: int(v) for w, v in zip(en_words, en_weights)}

    print("Validation metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    dicts_dir = args.rime_dir / args.dicts_subdir
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    for fn in args.cn_targets + args.en_targets:
        src = dicts_dir / fn
        bak = args.backup_dir / fn
        if src.is_file() and not bak.exists():
            shutil.copy2(src, bak)
            print(f"Backed up {fn} -> {bak}")

    for fn in args.cn_targets:
        p = dicts_dir / fn
        if p.is_file():
            tot, upd = rewrite_dict(p, cn_map, fallback_weight=10)
            print(f"[CN] {fn:30s}: updated {upd:9,d} / {tot:9,d}")

    for fn in args.en_targets:
        p = dicts_dir / fn
        if p.is_file():
            tot, upd = rewrite_dict(p, en_map, fallback_weight=930000)
            print(f"[EN] {fn:30s}: updated {upd:9,d} / {tot:9,d}")

    if args.deploy:
        deploy_squirrel(args.rime_dir)


if __name__ == "__main__":
    main()
