# rime-qwen-freq

Offline unigram word-frequency calibration, Bayesian domain log-odds boosting, and PMI `.gram` distillation for [Rime](https://rime.im/) from causal language models. Zero runtime inference latency.

**English** · [简体中文](README.zh-CN.md) · [Releases (Pre-built Dicts & `.gram`)](https://github.com/mizorewww/rime-qwen-freq/releases/latest) · [Design Post](https://mizore.blog/posts/rime-llm-word-frequency/)

## Pre-built Assets

Download ready-to-use `.dict.yaml.gz` tables (`cs_ai`, `zhwiki`, `zhwiktionary`, `moegirl`, `web_slang`, `wanxiang_mixed`, `rime_ice.en`, `rime_ice.en_ext`) and the standalone `qwen14b-zh-hans.gram.gz` (`9.4 MiB`, `1,031,842` keys distilled from `Qwen/Qwen3-14B-Base`) from [Releases](https://github.com/mizorewww/rime-qwen-freq/releases/latest).

## Benchmark (`bench/bench_cases.tsv`, 142 inputs, 0-`userdb`)

| Setup | Top-1 | CS Clauses (`40`) | CS Terms (`35`) | CharAcc | `.gram` |
|---|---:|---:|---:|---:|---:|
| Baseline (uncalibrated dicts + stock `.gram`) | `38.03%` | `22.5%` | `40.0%` | `65.38%` | 400.8 MiB |
| + 14B Dicts & CS Lift | `88.73%` | `82.5%` | `100.0%` | `96.28%` | 400.8 MiB |
| + Pure 14B `.gram` (`qwen14b-zh-hans`) | `90.85%` | `85.0%` | `100.0%` | `97.52%` | **11.4 MiB** |
| + Hybrid Cleaned + 14B `.gram` | **`92.25%`** | **`95.0%`** | **`100.0%`** | **`97.79%`** | 385.5 MiB |

## Usage

```bash
pip install -e ".[score]"              # or `pip install --no-deps -e .` for stdlib-only client
c++ -O3 -std=c++17 tools/gram_tool.cc -o gram_tool

# 1. Inspect & calibrate Rime dictionaries
rime-qwen-freq inspect
rime-qwen-freq run --model Qwen/Qwen3-14B-Base \
  --domain-prompt "计算机科学与人工智能术语：" \
  --general-prompt "日常与百科常见词条：" --deploy

# 2. Distill & build Octagram .gram model
rime-qwen-freq distill-ngram -o ngrams.tsv.gz --model Qwen/Qwen3-14B-Base
gunzip -k ngrams.tsv.gz && ./gram_tool build ngrams.tsv qwen14b-zh-hans.gram

# 3. End-to-end native librime benchmark
rime-qwen-freq bench --rime-dir ~/Library/Rime --schema rime_mint
```

For remote GPU scoring, split step 1 into `extract -o payload.json.gz` → `score -i payload.json.gz -o scores.tar.gz` → `apply -p payload.json.gz -s scores.tar.gz --deploy`.

## License

[MIT](LICENSE) (`tools/darts.h`: BSD-licensed Darts-clone v0.32).
