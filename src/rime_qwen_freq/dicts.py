"""
Zero-dependency Rime `.dict.yaml` inspection, extraction, atomic writeback, and deployment.

`inspect`, `extract`, and `apply` use only the Python standard library so they can run directly
on lightweight client machines without installing PyTorch or NumPy.
"""
from __future__ import annotations

import array
import ast
import collections
import gzip
import json
import math
import os
import platform
import random
import shutil
import struct
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SPECIAL_SKIP_KEYWORDS = (
    "others",
    "emoji",
    "kaomoji",
    "stroke",
    "wubi",
    "radical",
    "reverse",
    "symbol",
    "chaizi",
    "liangfen",
    "custom",
)


@dataclass
class DictStats:
    rel_path: str
    path: Path
    lang: str  # "zh" or "en"
    total_rows: int
    weighted_rows: int
    zero_rows: int
    missing_rows: int
    unique_weights: int
    top_weight: int | None
    top_weight_ratio: float
    p50: int
    p90: int
    p99: int
    role: str  # "anchor", "needs-weights", "ok", "skip-special"
    reason: str


def default_rime_dir() -> Path:
    system = platform.system()
    candidates: list[Path] = []
    if system == "Darwin":
        candidates.append(Path.home() / "Library" / "Rime")
    elif system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "Rime")
    else:
        candidates.extend([
            Path.home() / ".local" / "share" / "fcitx5" / "rime",
            Path.home() / ".config" / "ibus" / "rime",
            Path.home() / ".config" / "fcitx" / "rime",
        ])
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0] if candidates else Path.cwd()


def discover_dict_files(rime_dir: Path) -> list[Path]:
    results: list[Path] = []
    for root, dirs, files in os.walk(rime_dir):
        dirs[:] = [
            d for d in sorted(dirs)
            if d not in ("build", ".git", "plum", "opencc", "lua") and not d.endswith(".userdb")
        ]
        for fn in sorted(files):
            if fn.endswith(".dict.yaml"):
                results.append(Path(root) / fn)
    return results


def iter_dict_entries(filepath: Path) -> Iterator[tuple[str, str, str]]:
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


def has_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2CEAF:
            return True
    return False


def analyze_dict(rime_dir: Path, filepath: Path) -> DictStats:
    rel = filepath.relative_to(rime_dir).as_posix()
    lower_name = filepath.name.lower()

    total = 0
    missing = 0
    zero = 0
    pos_weights: list[int] = []
    cjk_sample = 0
    ascii_sample = 0

    for word, _, wt in iter_dict_entries(filepath):
        if not word:
            continue
        total += 1
        if total <= 500:
            if has_cjk(word):
                cjk_sample += 1
            elif word.isascii():
                ascii_sample += 1
        if not wt:
            missing += 1
        else:
            try:
                val = int(float(wt))
            except ValueError:
                missing += 1
                continue
            if val <= 0:
                zero += 1
            else:
                pos_weights.append(val)

    lang = "en" if (ascii_sample > cjk_sample * 3 and ascii_sample >= 10) else "zh"

    if total == 0:
        return DictStats(
            rel_path=rel, path=filepath, lang=lang, total_rows=0, weighted_rows=0,
            zero_rows=0, missing_rows=0, unique_weights=0, top_weight=None,
            top_weight_ratio=0.0, p50=0, p90=0, p99=0,
            role="skip-special", reason="import-only or empty table",
        )

    uniq = len(set(pos_weights))
    if pos_weights:
        ws = sorted(pos_weights)
        p50 = ws[len(ws) // 2]
        p90 = ws[int(len(ws) * 0.90)]
        p99 = ws[int(len(ws) * 0.99)]
        top_w, top_cnt = collections.Counter(pos_weights).most_common(1)[0]
        top_ratio = top_cnt / total
    else:
        p50 = p90 = p99 = 0
        top_w = None
        top_ratio = 0.0

    is_special = any(kw in lower_name for kw in SPECIAL_SKIP_KEYWORDS)
    if is_special:
        return DictStats(
            rel_path=rel, path=filepath, lang=lang, total_rows=total,
            weighted_rows=len(pos_weights), zero_rows=zero, missing_rows=missing,
            unique_weights=uniq, top_weight=top_w, top_weight_ratio=top_ratio,
            p50=p50, p90=p90, p99=p99, role="skip-special",
            reason="typo/aux/shape dictionary",
        )

    missing_ratio = missing / total
    if missing_ratio >= 0.50:
        return DictStats(
            rel_path=rel, path=filepath, lang=lang, total_rows=total,
            weighted_rows=len(pos_weights), zero_rows=zero, missing_rows=missing,
            unique_weights=uniq, top_weight=top_w, top_weight_ratio=top_ratio,
            p50=p50, p90=p90, p99=p99, role="needs-weights",
            reason=f"{missing_ratio*100:.0f}% missing weights",
        )

    if total >= 50 and (uniq <= 2 or top_ratio >= 0.95):
        return DictStats(
            rel_path=rel, path=filepath, lang=lang, total_rows=total,
            weighted_rows=len(pos_weights), zero_rows=zero, missing_rows=missing,
            unique_weights=uniq, top_weight=top_w, top_weight_ratio=top_ratio,
            p50=p50, p90=p90, p99=p99, role="needs-weights",
            reason=f"{top_ratio*100:.1f}% constant weight ({top_w})",
        )

    if len(pos_weights) >= 1000 and uniq >= 100:
        return DictStats(
            rel_path=rel, path=filepath, lang=lang, total_rows=total,
            weighted_rows=len(pos_weights), zero_rows=zero, missing_rows=missing,
            unique_weights=uniq, top_weight=top_w, top_weight_ratio=top_ratio,
            p50=p50, p90=p90, p99=p99, role="anchor",
            reason=f"{uniq:,} distinct empirical weights",
        )

    return DictStats(
        rel_path=rel, path=filepath, lang=lang, total_rows=total,
        weighted_rows=len(pos_weights), zero_rows=zero, missing_rows=missing,
        unique_weights=uniq, top_weight=top_w, top_weight_ratio=top_ratio,
        p50=p50, p90=p90, p99=p99, role="ok",
        reason="already weighted",
    )


def load_clean_weights(filepath: Path, single_char_min_weight: int = 0) -> dict[str, int]:
    res: dict[str, int] = {}
    if not filepath.is_file():
        return res
    for w, _, wt in iter_dict_entries(filepath):
        if not w or not wt:
            continue
        try:
            v = int(float(wt))
        except ValueError:
            continue
        if v <= 0:
            continue
        if single_char_min_weight > 0 and len(w) == 1 and v < single_char_min_weight:
            continue
        if w not in res or v > res[w]:
            res[w] = v
    return res


def stratified_sample_anchors(pool: dict[str, int], max_count: int, seed: int = 42) -> dict[str, int]:
    if len(pool) <= max_count:
        return dict(pool)
    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list[tuple[str, int]]] = {}
    for w, v in pool.items():
        l_bin = min(len(w), 6)
        v_bin = min(int(math.log10(v + 1) * 2), 12)
        buckets.setdefault((l_bin, v_bin), []).append((w, v))

    sampled: dict[str, int] = {}
    per_bucket = max(1, max_count // max(1, len(buckets)))
    leftovers: list[tuple[str, int]] = []
    for _, items in sorted(buckets.items()):
        rng.shuffle(items)
        for w, v in items[:per_bucket]:
            sampled[w] = v
        leftovers.extend(items[per_bucket:])

    if len(sampled) < max_count and leftovers:
        rng.shuffle(leftovers)
        for w, v in leftovers[: max_count - len(sampled)]:
            sampled[w] = v
    return sampled


def resolve_dict_paths(rime_dir: Path, specs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    all_dicts = {p.name: p for p in discover_dict_files(rime_dir)}
    for sp in specs:
        cand = Path(sp)
        if cand.is_file():
            resolved.append(cand)
        elif (rime_dir / sp).is_file():
            resolved.append(rime_dir / sp)
        elif sp in all_dicts:
            resolved.append(all_dicts[sp])
        else:
            raise FileNotFoundError(f"Dictionary not found: {sp}")
    return resolved


def extract_payload(
    rime_dir: Path,
    output_path: Path,
    targets: list[str] | None = None,
    anchors: list[str] | None = None,
    cn_anchors_max: int = 100_000,
    en_anchors_max: int = 25_000,
    seed: int = 42,
) -> dict:
    all_stats = [analyze_dict(rime_dir, p) for p in discover_dict_files(rime_dir)]
    stats_by_path = {st.path.resolve(): st for st in all_stats}

    if targets:
        target_paths = resolve_dict_paths(rime_dir, targets)
    else:
        target_paths = [st.path for st in all_stats if st.role == "needs-weights"]

    if not target_paths:
        raise RuntimeError(
            f"No unweighted or constant-weight dictionaries detected in {rime_dir}. "
            "Pass --targets explicitly if you want to re-score specific files."
        )

    if anchors:
        anchor_paths = resolve_dict_paths(rime_dir, anchors)
    else:
        # Sort larger anchor tables first so primary base lexicons take precedence
        anchor_stats = sorted(
            (st for st in all_stats if st.role == "anchor"),
            key=lambda s: s.weighted_rows,
            reverse=True,
        )
        anchor_paths = [st.path for st in anchor_stats]

    cn_targets_rel: list[str] = []
    en_targets_rel: list[str] = []
    cn_words: set[str] = set()
    en_words: set[str] = set()

    for p in target_paths:
        st = stats_by_path.get(p.resolve()) or analyze_dict(rime_dir, p)
        rel = p.resolve().relative_to(rime_dir.resolve()).as_posix() if p.resolve().is_relative_to(rime_dir.resolve()) else str(p)
        if st.lang == "en":
            en_targets_rel.append(rel)
            for w, _, _ in iter_dict_entries(p):
                if w:
                    en_words.add(w)
        else:
            cn_targets_rel.append(rel)
            for w, _, _ in iter_dict_entries(p):
                if w:
                    cn_words.add(w)

    cn_gt_pool: dict[str, int] = {}
    en_gt_pool: dict[str, int] = {}
    for p in anchor_paths:
        st = stats_by_path.get(p.resolve()) or analyze_dict(rime_dir, p)
        min_char_w = 10 if "char" in p.name.lower() else 0
        d = load_clean_weights(p, single_char_min_weight=min_char_w)
        dest = en_gt_pool if st.lang == "en" else cn_gt_pool
        for k, v in d.items():
            if k not in dest:
                dest[k] = v

    cn_anchors = stratified_sample_anchors(cn_gt_pool, cn_anchors_max, seed=seed) if cn_words else {}
    en_anchors = stratified_sample_anchors(en_gt_pool, en_anchors_max, seed=seed) if en_words else {}

    all_cn_list = sorted(cn_words | set(cn_anchors.keys())) if cn_words else []
    all_en_list = sorted(en_words | set(en_anchors.keys())) if en_words else []

    payload = {
        "schema_version": 2,
        "cn_targets": cn_targets_rel,
        "en_targets": en_targets_rel,
        "cn_words": all_cn_list,
        "cn_anchors": cn_anchors,
        "en_words": all_en_list,
        "en_anchors": en_anchors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(payload, f, ensure_ascii=False)

    return payload


def read_uint32_npy(path: Path) -> array.array:
    """Read a 1D little-endian uint32 NumPy `.npy` file using only the standard library."""
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"Invalid .npy header in {path}")
        major, _ = struct.unpack("<BB", f.read(2))
        hlen = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        hdr = ast.literal_eval(f.read(hlen).decode("latin1").strip())
        if hdr.get("descr") not in ("<u4", "|u4", "uint32"):
            raise ValueError(f"Expected uint32 .npy array, got {hdr}")
        arr = array.array("I")
        arr.frombytes(f.read())
        if sys_is_big_endian := (struct.pack("<H", 1) != struct.pack("=H", 1)):
            if sys_is_big_endian:
                arr.byteswap()
        return arr


def read_float16_npy(path: Path) -> list[float]:
    """Read a 1D little-endian float16 NumPy `.npy` file using only the standard library."""
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"Invalid .npy header in {path}")
        major, _ = struct.unpack("<BB", f.read(2))
        hlen = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        hdr = ast.literal_eval(f.read(hlen).decode("latin1").strip())
        if hdr.get("descr") not in ("<f2", "float16"):
            raise ValueError(f"Expected float16 .npy array, got {hdr}")
        raw = f.read()
        if struct.pack("<H", 1) != struct.pack("=H", 1):
            count = len(raw) // 2
            return list(struct.unpack(f"<{count}e", raw))
        return [float(x) for x in memoryview(raw).cast("e")]


def rewrite_dict_file(filepath: Path, weight_map: dict[str, int], fallback_weight: int) -> tuple[int, int]:
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp_freq")
    in_body = False
    updated = 0
    total_entries = 0

    with filepath.open("r", encoding="utf-8", errors="replace") as fin, \
         tmp_path.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            raw = line.rstrip("\r\n")
            if not in_body:
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


def deploy_rime(rime_dir: Path) -> None:
    system = platform.system()
    if system == "Darwin":
        squirrel_macos = Path("/Library/Input Methods/Squirrel.app/Contents/MacOS")
        deployer = squirrel_macos / "rime_deployer"
        shared = Path("/Library/Input Methods/Squirrel.app/Contents/SharedSupport")
        squirrel_bin = squirrel_macos / "Squirrel"
        if deployer.is_file():
            subprocess.run([str(deployer), "--build", str(rime_dir), str(shared)], check=True)
        if squirrel_bin.is_file():
            subprocess.run([str(squirrel_bin), "--reload"], check=True)
            return
    elif system == "Windows":
        for prog_env in ("ProgramFiles(x86)", "ProgramFiles"):
            base = os.environ.get(prog_env)
            if base:
                candidates = list(Path(base).glob("Rime/*/WeaselDeployer.exe"))
                if candidates:
                    subprocess.run([str(candidates[-1]), "/deploy"], check=True)
                    return
    else:
        deployer_bin = shutil.which("rime_deployer")
        if deployer_bin:
            shared_candidates = [
                Path("/usr/share/rime-data"),
                Path("/usr/local/share/rime-data"),
            ]
            shared = next((p for p in shared_candidates if p.is_dir()), rime_dir)
            subprocess.run([deployer_bin, "--build", str(rime_dir), str(shared)], check=True)
        if fcitx_remote := shutil.which("fcitx5-remote"):
            subprocess.run([fcitx_remote, "-r"], check=False)
        elif ibus_bin := shutil.which("ibus"):
            subprocess.run([ibus_bin, "restart"], check=False)


def apply_scores_archive(
    rime_dir: Path,
    payload_path: Path,
    scores_path: Path,
    backup_dir: Path,
    targets: list[str] | None = None,
    deploy: bool = False,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="rime_scores_extract_") as tmp_extract:
        extract_dir = Path(tmp_extract)
        with tarfile.open(scores_path, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
        cn_weights = read_uint32_npy(extract_dir / "cn_weights.npy")
        en_weights = read_uint32_npy(extract_dir / "en_weights.npy")
        metrics = json.loads((extract_dir / "metrics.json").read_text(encoding="utf-8"))

    with gzip.open(payload_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    cn_words = payload["cn_words"]
    en_words = payload["en_words"]
    if len(cn_words) != len(cn_weights) or len(en_words) != len(en_weights):
        raise ValueError("Payload word list length does not match scores archive.")

    cn_map = {w: int(v) for w, v in zip(cn_words, cn_weights)}
    en_map = {w: int(v) for w, v in zip(en_words, en_weights)}

    if targets:
        target_paths = resolve_dict_paths(rime_dir, targets)
        cn_paths, en_paths = [], []
        for p in target_paths:
            st = analyze_dict(rime_dir, p)
            (en_paths if st.lang == "en" else cn_paths).append(p)
    else:
        cn_paths = resolve_dict_paths(rime_dir, payload.get("cn_targets", []))
        en_paths = resolve_dict_paths(rime_dir, payload.get("en_targets", []))

    backup_dir.mkdir(parents=True, exist_ok=True)
    updated_summary: list[tuple[str, int, int]] = []
    rime_root = rime_dir.resolve()

    for p in cn_paths + en_paths:
        p_res = p.resolve()
        rel_sub = p_res.relative_to(rime_root) if p_res.is_relative_to(rime_root) else Path(p.name)
        bak = backup_dir / rel_sub
        if not bak.exists():
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, bak)

    cn_fallback = int(sorted(cn_weights)[len(cn_weights) // 4]) if cn_weights else 10
    en_fallback = int(sorted(en_weights)[len(en_weights) // 4]) if en_weights else 100

    for p in cn_paths:
        tot, upd = rewrite_dict_file(p, cn_map, fallback_weight=cn_fallback)
        updated_summary.append((p.name, upd, tot))

    for p in en_paths:
        tot, upd = rewrite_dict_file(p, en_map, fallback_weight=en_fallback)
        updated_summary.append((p.name, upd, tot))

    if deploy:
        deploy_rime(rime_dir)

    return {"metrics": metrics, "updated": updated_summary}
