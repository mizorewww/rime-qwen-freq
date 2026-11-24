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
    iter_dict_entries,
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
        domain_prompt=args.domain_prompt,
        general_prompt=args.general_prompt,
        en_domain_prompt=args.en_domain_prompt,
        en_general_prompt=args.en_general_prompt,
        domain_threshold=args.domain_threshold,
        domain_slope=args.domain_slope,
        domain_max_boost=args.domain_max_boost,
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
            domain_prompt=args.domain_prompt,
            general_prompt=args.general_prompt,
            en_domain_prompt=args.en_domain_prompt,
            en_general_prompt=args.en_general_prompt,
            domain_threshold=args.domain_threshold,
            domain_slope=args.domain_slope,
            domain_max_boost=args.domain_max_boost,
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


def cmd_distill_ngram(args: argparse.Namespace) -> int:
    from .ngram import distill_ngrams_from_lexicon

    rime_dir = args.rime_dir.expanduser().resolve()
    word_weights: dict[str, int] = {}
    for p in discover_dict_files(rime_dir):
        st = analyze_dict(rime_dir, p)
        if st.lang != "zh" or st.role == "skip-special":
            continue
        for w, _, wt in iter_dict_entries(p):
            if w and 1 <= len(w) <= 5 and all("\u4e00" <= c <= "\u9fff" for c in w):
                v = int(float(wt)) if wt and wt.replace(".", "", 1).isdigit() else 50
                if v > word_weights.get(w, 0):
                    word_weights[w] = v

    ranked = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)
    ctx_words = [w for w, _ in ranked[: args.max_contexts] if 1 <= len(w) <= 4]
    lex_words = [w for w, _ in ranked[: args.max_lexicon]]
    res = distill_ngrams_from_lexicon(
        lexicon_words=lex_words,
        context_words=ctx_words,
        output_tsv_gz=args.output.expanduser().resolve(),
        model_id=args.model,
        device=args.device,
        general_prompt=args.general_prompt,
        domain_prompt=args.domain_prompt,
        min_pmi=args.min_pmi,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmark

    run_benchmark(
        user_dir=args.rime_dir.expanduser().resolve(),
        cases_path=args.cases.expanduser().resolve() if args.cases else None,
        out_json=args.out.expanduser().resolve() if args.out else None,
        schema_id=args.schema,
        shared_dir=args.shared_dir.expanduser().resolve() if args.shared_dir else None,
        deploy=args.deploy,
        include_userdb=args.include_userdb,
        rime_lib=args.rime_lib.expanduser().resolve() if args.rime_lib else None,
        octagram_lib=args.octagram_lib.expanduser().resolve() if args.octagram_lib else None,
        lua_lib=args.lua_lib.expanduser().resolve() if args.lua_lib else None,
    )
    return 0


def _add_domain_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--domain-prompt",
        default=None,
        help="Optional CJK domain context C_domain (e.g. '计算机科学与人工智能术语：') for Bayesian Log-Odds Lift",
    )
    p.add_argument(
        "--general-prompt",
        default="日常与百科常见词条：",
        help="Frame-matched CJK general context C_general paired with --domain-prompt",
    )
    p.add_argument(
        "--en-domain-prompt",
        default=None,
        help="Optional EN domain context (e.g. 'Computer science and programming terms: ')",
    )
    p.add_argument(
        "--en-general-prompt",
        default="Common words and terms: ",
        help="Frame-matched EN general context paired with --en-domain-prompt",
    )
    p.add_argument(
        "--domain-threshold",
        type=float,
        default=1.3,
        help="Minimum log-odds lift in nats before domain boost activates (default: 1.3)",
    )
    p.add_argument(
        "--domain-slope",
        type=float,
        default=0.135,
        help="Log10 weight gain per nat above --domain-threshold (default: 0.135)",
    )
    p.add_argument(
        "--domain-max-boost",
        type=float,
        default=0.54,
        help="Maximum log10 weight boost cap (~3.47x at 0.54, default: 0.54)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rime-qwen-freq",
        description="Distill calibrated unigram word frequencies and PMI N-gram (.gram) models for Rime from a CausalLM.",
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
    _add_domain_args(p_run)
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
    _add_domain_args(p_scr)
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

    # distill-ngram
    p_ng = sub.add_parser("distill-ngram", help="Distill PMI collocations and Rear-$ markers from a CausalLM for Octagram (.gram).")
    p_ng.add_argument("--rime-dir", type=Path, default=default_rime_dir(), help="Rime user configuration directory")
    p_ng.add_argument("-o", "--output", type=Path, default=Path("distilled_ngrams.tsv.gz"), help="Output .tsv.gz for tools/gram_tool")
    p_ng.add_argument("--model", default="Qwen/Qwen3-14B-Base", help="Hugging Face CausalLM ID or local path")
    p_ng.add_argument("--device", default="auto", help="Compute device: auto, cuda, mps, or cpu")
    p_ng.add_argument("--max-contexts", type=int, default=50_000, help="Number of top context words w1 to expand")
    p_ng.add_argument("--max-lexicon", type=int, default=90_000, help="Size of continuation vocabulary V")
    p_ng.add_argument("--general-prompt", default="常见中文语句片段：")
    p_ng.add_argument("--domain-prompt", default="计算机科学与人工智能技术讨论：")
    p_ng.add_argument("--min-pmi", type=float, default=2.35, help="Minimum PMI threshold in nats (default: 2.35)")
    p_ng.set_defaults(func=cmd_distill_ngram)

    # bench
    p_bn = sub.add_parser("bench", help="Run native librime + librime-octagram end-to-end Pinyin decoding benchmark.")
    p_bn.add_argument("--rime-dir", type=Path, default=default_rime_dir(), help="Rime user configuration directory to benchmark")
    p_bn.add_argument("--shared-dir", type=Path, default=None, help="Rime shared data directory (auto-detected if omitted)")
    p_bn.add_argument("--cases", type=Path, default=None, help="Custom benchmark TSV (default: bundled bench_cases.tsv)")
    p_bn.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    p_bn.add_argument("--schema", default="rime_mint", help="Target schema ID (default: rime_mint)")
    p_bn.add_argument("--deploy", action="store_true", help="Run Rime maintenance/deploy inside isolated sandbox before benchmarking")
    p_bn.add_argument("--include-userdb", action="store_true", help="Copy personal *.userdb into sandbox (default: clean 0-userdb evaluation)")
    p_bn.add_argument("--rime-lib", type=Path, default=None, help="Explicit path to librime (.dylib / .so.1 / rime.dll)")
    p_bn.add_argument("--octagram-lib", type=Path, default=None, help="Explicit path to librime-octagram plugin")
    p_bn.add_argument("--lua-lib", type=Path, default=None, help="Explicit path to librime-lua plugin")
    p_bn.set_defaults(func=cmd_bench)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
