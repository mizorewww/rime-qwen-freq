"""Unified command-line interface for `rime-qwen-freq`."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from . import __version__
from .dicts import (
    analyze_dict,
    apply_scores_archive,
    default_rime_dir,
    discover_dict_files,
    extract_payload,
)


def cmd_inspect(args: argparse.Namespace) -> int:
    rime_dir: Path = args.rime_dir.expanduser().resolve()
    files = discover_dict_files(rime_dir)
    stats = [analyze_dict(rime_dir, p) for p in files]

    if args.json:
        out = [
            {
                "rel_path": s.rel_path,
                "lang": s.lang,
                "total_rows": s.total_rows,
                "weighted_rows": s.weighted_rows,
                "missing_rows": s.missing_rows,
                "unique_weights": s.unique_weights,
                "p50": s.p50,
                "p99": s.p99,
                "role": s.role,
                "reason": s.reason,
            }
            for s in stats
        ]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    print(f"Rime directory: {rime_dir}\n")
    hdr = f"{'Dictionary':36s} {'Lang':4s} {'Entries':>10s} {'UniqW':>8s} {'P50':>8s} {'P99':>9s}  {'Role':14s} {'Diagnostic'}"
    print(hdr)
    print("-" * len(hdr))
    for s in stats:
        if s.total_rows == 0:
            continue
        print(
            f"{s.rel_path:36s} {s.lang:4s} {s.total_rows:10,d} {s.unique_weights:8,d} "
            f"{s.p50:8,d} {s.p99:9,d}  {s.role:14s} {s.reason}"
        )
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    rime_dir = args.rime_dir.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    payload = extract_payload(
        rime_dir=rime_dir,
        output_path=out_path,
        targets=args.targets,
        anchors=args.anchors,
        cn_anchors_max=args.cn_anchors_max,
        en_anchors_max=args.en_anchors_max,
        seed=args.seed,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved payload: {out_path} ({size_mb:.2f} MB)")
    print(f"  CJK targets ({len(payload['cn_targets'])}): {', '.join(payload['cn_targets']) or '-'}")
    print(f"  EN  targets ({len(payload['en_targets'])}): {', '.join(payload['en_targets']) or '-'}")
    print(f"  CJK unique words: {len(payload['cn_words']):,} (anchors: {len(payload['cn_anchors']):,})")
    print(f"  EN  unique words: {len(payload['en_words']):,} (anchors: {len(payload['en_anchors']):,})")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .scorer import score_payload_file

    metrics = score_payload_file(
        payload_path=args.input.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        model_id=args.model,
        device=args.device,
        max_batch=args.max_batch,
    )
    print("\nCompleted scoring:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    res = apply_scores_archive(
        rime_dir=args.rime_dir.expanduser().resolve(),
        payload_path=args.payload.expanduser().resolve(),
        scores_path=args.scores.expanduser().resolve(),
        backup_dir=args.backup_dir.expanduser().resolve(),
        targets=args.targets,
        deploy=args.deploy,
    )
    print("Applied calibrated frequencies:")
    for fn, upd, tot in res["updated"]:
        print(f"  {fn:32s}: {upd:9,d} / {tot:9,d} entries updated")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .scorer import score_payload_file

    rime_dir = args.rime_dir.expanduser().resolve()
    backup_dir = args.backup_dir.expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="rime_qwen_freq_") as tmp:
        tmp_path = Path(tmp)
        payload_path = tmp_path / "payload.json.gz"
        scores_path = tmp_path / "scores.tar.gz"

        payload = extract_payload(
            rime_dir=rime_dir,
            output_path=payload_path,
            targets=args.targets,
            anchors=args.anchors,
            cn_anchors_max=args.cn_anchors_max,
            en_anchors_max=args.en_anchors_max,
            seed=args.seed,
        )
        print(f"Detected {len(payload['cn_targets'])} CJK target(s) and {len(payload['en_targets'])} EN target(s).")

        score_payload_file(
            payload_path=payload_path,
            output_path=scores_path,
            model_id=args.model,
            device=args.device,
            max_batch=args.max_batch,
        )

        res = apply_scores_archive(
            rime_dir=rime_dir,
            payload_path=payload_path,
            scores_path=scores_path,
            backup_dir=backup_dir,
            targets=args.targets,
            deploy=args.deploy,
        )

    print("\nFinished updating Rime dictionaries:")
    for fn, upd, tot in res["updated"]:
        print(f"  {fn:32s}: {upd:9,d} / {tot:9,d} entries updated")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rime-qwen-freq",
        description="Distill calibrated static word frequencies for Rime dictionaries from a causal language model.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_ins = sub.add_parser("inspect", help="Audit Rime dictionaries and auto-detect targets & calibration anchors.")
    p_ins.add_argument("--rime-dir", type=Path, default=default_rime_dir(), help="Rime user configuration directory")
    p_ins.add_argument("--json", action="store_true", help="Emit JSON output")
    p_ins.set_defaults(func=cmd_inspect)

    # run (all-in-one)
    p_run = sub.add_parser("run", help="Extract, score on local GPU/MPS/CPU, and write back weights in one step.")
    p_run.add_argument("--rime-dir", type=Path, default=default_rime_dir(), help="Rime user configuration directory")
    p_run.add_argument("--targets", nargs="*", default=None, help="Explicit target .dict.yaml files (default: auto-detect)")
    p_run.add_argument("--anchors", nargs="*", default=None, help="Explicit anchor .dict.yaml files (default: auto-detect)")
    p_run.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base", help="Hugging Face CausalLM ID or local path")
    p_run.add_argument("--device", default="auto", help="Compute device: auto, cuda, mps, or cpu")
    p_run.add_argument("--max-batch", type=int, default=2048, help="Maximum leaf batch size")
    p_run.add_argument("--cn-anchors-max", type=int, default=100_000)
    p_run.add_argument("--en-anchors-max", type=int, default=25_000)
    p_run.add_argument("--backup-dir", type=Path, default=Path("./backup_dicts"), help="Directory for original dictionary backups")
    p_run.add_argument("--deploy", action="store_true", help="Rebuild Rime tables and reload input method after writeback")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.set_defaults(func=cmd_run)

    # extract
    p_ext = sub.add_parser("extract", help="Package target words and calibration anchors into a portable .json.gz file.")
    p_ext.add_argument("--rime-dir", type=Path, default=default_rime_dir(), help="Rime user configuration directory")
    p_ext.add_argument("-o", "--output", type=Path, default=Path("payload.json.gz"), help="Output payload .json.gz")
    p_ext.add_argument("--targets", nargs="*", default=None, help="Explicit target .dict.yaml files (default: auto-detect)")
    p_ext.add_argument("--anchors", nargs="*", default=None, help="Explicit anchor .dict.yaml files (default: auto-detect)")
    p_ext.add_argument("--cn-anchors-max", type=int, default=100_000)
    p_ext.add_argument("--en-anchors-max", type=int, default=25_000)
    p_ext.add_argument("--seed", type=int, default=42)
    p_ext.set_defaults(func=cmd_extract)

    # score
    p_scr = sub.add_parser("score", help="Score a payload .json.gz on CUDA/MPS/CPU and output calibrated weights.")
    p_scr.add_argument("-i", "--input", type=Path, default=Path("payload.json.gz"), help="Input payload .json.gz")
    p_scr.add_argument("-o", "--output", type=Path, default=Path("scores.tar.gz"), help="Output calibrated scores .tar.gz")
    p_scr.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base", help="Hugging Face CausalLM ID or local path")
    p_scr.add_argument("--device", default="auto", help="Compute device: auto, cuda, mps, or cpu")
    p_scr.add_argument("--max-batch", type=int, default=2048, help="Maximum leaf batch size")
    p_scr.set_defaults(func=cmd_score)

    # apply
    p_app = sub.add_parser("apply", help="Write calibrated weights back into Rime dictionaries (stdlib only).")
    p_app.add_argument("--rime-dir", type=Path, default=default_rime_dir(), help="Rime user configuration directory")
    p_app.add_argument("-p", "--payload", type=Path, default=Path("payload.json.gz"), help="Payload .json.gz path")
    p_app.add_argument("-s", "--scores", type=Path, default=Path("scores.tar.gz"), help="Scores .tar.gz path")
    p_app.add_argument("--backup-dir", type=Path, default=Path("./backup_dicts"), help="Backup directory")
    p_app.add_argument("--targets", nargs="*", default=None, help="Override target dictionary list")
    p_app.add_argument("--deploy", action="store_true", help="Rebuild Rime tables and reload input method")
    p_app.set_defaults(func=cmd_apply)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
