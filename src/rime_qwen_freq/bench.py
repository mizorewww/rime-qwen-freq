"""
End-to-end native `librime` + `librime-octagram` benchmark runner.

Compiles `rime_bench.cc` on demand into a user cache directory and evaluates whole-clause
Pinyin decoding, phrase ranking, and contextual next-word disambiguation inside an isolated
sandbox directory so live Rime `*.userdb` state is never locked or mutated.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from .dicts import default_rime_dir


def default_shared_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        for cand in (
            Path("/Library/Input Methods/Squirrel.app/Contents/SharedSupport"),
            Path("/opt/homebrew/share/rime-data"),
            Path("/usr/local/share/rime-data"),
        ):
            if cand.is_dir():
                return cand
    elif system == "Windows":
        for env_key in ("ProgramFiles(x86)", "ProgramFiles"):
            base = os.environ.get(env_key)
            if base:
                matches = sorted(Path(base).glob("Rime/*/data"))
                if matches:
                    return matches[-1]
    else:
        for cand in (Path("/usr/share/rime-data"), Path("/usr/local/share/rime-data")):
            if cand.is_dir():
                return cand
    return default_rime_dir()


def resolve_bench_assets_dir() -> Path:
    bundled = Path(__file__).resolve().parent / "_bench"
    if (bundled / "rime_bench.cc").is_file():
        return bundled
    repo_bench = Path(__file__).resolve().parents[2] / "bench"
    if (repo_bench / "rime_bench.cc").is_file():
        return repo_bench
    raise FileNotFoundError("Benchmark assets (_bench/rime_bench.cc) not found in package or repository.")


def _bench_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    candidates = [
        Path(xdg) / "rime-qwen-freq" if xdg else Path.home() / ".cache" / "rime-qwen-freq",
        Path(tempfile.gettempdir()) / "rime-qwen-freq-cache",
    ]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return c
        except OSError:
            continue
    raise RuntimeError("Could not create a writable cache directory for rime_bench binary.")


def ensure_bench_binary(assets_dir: Path | None = None) -> Path:
    bdir = assets_dir or resolve_bench_assets_dir()
    src_cc = bdir / "rime_bench.cc"
    hdr_h = bdir / "rime_api.h"
    if not src_cc.is_file() or not hdr_h.is_file():
        raise FileNotFoundError(f"Benchmark C++ sources missing in {bdir}")

    digest = hashlib.sha256(src_cc.read_bytes() + b"\0" + hdr_h.read_bytes()).hexdigest()[:12]
    exe_suffix = ".exe" if platform.system() == "Windows" else ""
    bin_path = _bench_cache_dir() / f"rime_bench_{digest}{exe_suffix}"

    if not bin_path.is_file():
        cxx = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
        if not cxx:
            raise RuntimeError("No C++17 compiler (clang++/g++/c++) found in PATH to build rime_bench.cc")
        cmd = [cxx, "-O3", "-std=c++17", f"-I{bdir}", str(src_cc), "-o", str(bin_path)]
        if platform.system() == "Linux":
            cmd.append("-ldl")
        subprocess.run(cmd, check=True)
    return bin_path


def _populate_isolated_user_dir(src_user_dir: Path, dst_dir: Path, include_userdb: bool = False) -> None:
    skip_exact = {"user.yaml", "installation.yaml", "sync", ".rime_bench_last.json", ".DS_Store"}
    for item in src_user_dir.iterdir():
        name = item.name
        if name in skip_exact:
            continue
        is_userdb = name.endswith(".userdb") or ".userdb." in name
        if is_userdb:
            if include_userdb and item.is_dir():
                copied = dst_dir / name
                shutil.copytree(item, copied)
                lock_file = copied / "LOCK"
                if lock_file.exists():
                    lock_file.unlink(missing_ok=True)
            continue
        target = dst_dir / name
        try:
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
        except OSError:
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)


def run_benchmark(
    user_dir: Path,
    cases_path: Path | None = None,
    out_json: Path | None = None,
    schema_id: str = "rime_mint",
    shared_dir: Path | None = None,
    deploy: bool = False,
    include_userdb: bool = False,
    rime_lib: Path | None = None,
    octagram_lib: Path | None = None,
    lua_lib: Path | None = None,
) -> dict:
    assets_dir = resolve_bench_assets_dir()
    bin_path = ensure_bench_binary(assets_dir)
    eff_cases = cases_path or (assets_dir / "bench_cases.tsv")
    eff_shared = shared_dir or default_shared_dir()

    with tempfile.TemporaryDirectory(prefix="rime_bench_sandbox_") as tmp_str:
        sandbox_dir = Path(tmp_str)
        _populate_isolated_user_dir(user_dir.resolve(), sandbox_dir, include_userdb=include_userdb)
        tmp_out = out_json or (sandbox_dir / "bench_result.json")

        cmd = [
            str(bin_path),
            "--user-dir", str(sandbox_dir),
            "--shared-dir", str(eff_shared),
            "--cases", str(eff_cases),
            "--schema", schema_id,
            "--out", str(tmp_out),
        ]
        if rime_lib:
            cmd.extend(["--rime-lib", str(rime_lib)])
        if octagram_lib:
            cmd.extend(["--octagram-lib", str(octagram_lib)])
        if lua_lib:
            cmd.extend(["--lua-lib", str(lua_lib)])
        if deploy:
            cmd.append("--deploy")

        subprocess.run(cmd, check=True)
        data = json.loads(tmp_out.read_text(encoding="utf-8"))

    by_cat: dict[str, list[dict]] = collections.defaultdict(list)
    for r in data.get("details", []):
        by_cat[r["category"]].append(r)

    print("\nCategory Breakdown:")
    print(f"  {'Category':18s} {'Top-1':>12s} {'Top-3':>10s} {'CharAcc':>10s}")
    print("  " + "-" * 54)
    for cat, rows in by_cat.items():
        n = len(rows)
        t1 = sum(1 for r in rows if r["rank"] == 1)
        t3 = sum(1 for r in rows if 1 <= r["rank"] <= 3)
        ca = sum(r["char_acc"] for r in rows) / max(1, n)
        print(f"  {cat:18s} {t1:3d}/{n:<3d} ({t1*100/n:5.1f}%) {t3*100/n:8.1f}% {ca*100:9.1f}%")

    return data
