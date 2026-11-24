# rime-qwen-freq

**Distill calibrated unigram word frequencies and PMI N-gram (`.gram`) language models for Rime from open-weight causal language models.**

[![Newest release](https://img.shields.io/github/v/release/mizorewww/rime-qwen-freq?sort=semver&label=release)](https://github.com/mizorewww/rime-qwen-freq/releases)
[![License: MIT](https://img.shields.io/github/license/mizorewww/rime-qwen-freq)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![macOS · Linux · Windows](https://img.shields.io/badge/rime-macOS%20%C2%B7%20Linux%20%C2%B7%20Windows-informational)

**English** · [简体中文](README.zh-CN.md) · [Pre-built Assets](#pre-built-dictionaries--n-gram-model) · [Benchmark](#end-to-end-librime-benchmark)

`rime-qwen-freq` upgrades [Rime](https://rime.im/) input method dictionaries and [`librime-octagram`](https://github.com/lotem/librime-octagram) `.gram` language models completely offline:

1. **Unigram Dictionary Calibration (`inspect` / `run` / `extract` / `score` / `apply`):** Finds `.dict.yaml` tables with missing or placeholder weights (e.g. `60000` on every Wikipedia row), evaluates exact token log-probabilities via a deduplicated **Prefix Trie**, applies optional zero-corpus **Bayesian Domain Log-Odds Lift** (`--domain-prompt`), and calibrates onto the empirical scale of your base dictionary (`rime-wanxiang`, `rime-ice`, `rime-mint`, `luna_pinyin`, or `easy_en`).
2. **Direct-Logit PMI N-gram Distillation (`distill-ngram` + `tools/gram_tool.cc`):** Distills multi-word collocations and sentence-ending `$` markers directly from a CausalLM's `lm_head` conditional distribution into Darts Double-Array `.gram` files, and audits existing `.gram` models to remove scraped web-corpus homophone typos.
3. **Native `librime` End-to-End Benchmark (`bench`):** Links directly against `librime` + `librime-octagram` (`bench/rime_bench.cc`) to measure real whole-clause Pinyin conversion, phrase ranking, and contextual next-word accuracy.

All neural computation happens offline. At runtime, Rime uses native memory-mapped `.table.bin` and `.gram` lookup with **0 ms neural inference latency**.

## Pre-built Dictionaries & N-gram Model

Ready-to-use `.dict.yaml` tables and a standalone `qwen14b-zh-hans.gram` model distilled from `Qwen/Qwen3-14B-Base` are available on the [Releases](https://github.com/mizorewww/rime-qwen-freq/releases/latest) page — no Python or GPU required:

| Asset | Entries / Keys | Size (`.gz`) | Description |
|---|---:|---:|---|
| `qwen14b-zh-hans.gram.gz` | 1,031,842 | 9.4 MiB | Standalone `Qwen3-14B-Base` PMI N-gram + Rear-`$` model (`11.4 MiB` uncompressed, 35× smaller than `400.8 MiB` corpus `.gram`) |
| `cs_ai.dict.yaml.gz` | 1,986 | 26 KiB | Curated + Bayesian-mined CS / AI / ML / Systems terms with polyphone-verified Pinyin |
| `zhwiki.dict.yaml.gz` | 1,464,285 | 17.4 MiB | Chinese Wikipedia (`fcitx5-pinyin-zhwiki`, deduplicated & calibrated to Wanxiang `1–350,000`) |
| `zhwiktionary.dict.yaml.gz` | 165,545 | 1.6 MiB | Chinese Wiktionary (`fcitx5-pinyin-zhwiki`, deduplicated & calibrated) |
| `moegirl.dict.yaml.gz` | 105,076 | 1.9 MiB | Moegirlpedia (`mw2fcitx`, deduplicated & calibrated) |
| `web_slang.dict.yaml.gz` | 423 | 6.5 KiB | Web slang (`fcitx5-pinyin-zhwiki`, calibrated) |
| `wanxiang_mixed.dict.yaml.gz` | 1,817 | 19 KiB | CJK-Latin mixed terms (`rime-wanxiang` toneless, polyphone-disambiguated) |
| `rime_ice.en.dict.yaml.gz` | 21,263 | 194 KiB | Core English vocabulary (`rime-ice`, calibrated to ECDICT `900,000–999,995`) |
| `rime_ice.en_ext.dict.yaml.gz` | 2,598 | 29 KiB | Extended English & tech terms (`rime-ice`, calibrated to ECDICT) |

## End-to-End `librime` Benchmark

Evaluated in a clean 0-`userdb` sandbox with `rime-qwen-freq bench` (`bench/bench_cases.tsv`, 142 continuous Pinyin inputs $\le 25$ keys across CS/AI whole clauses, technical compounds, CJK-Latin mixed terms, contextual disambiguation, homophone typo traps, and daily conversational guard sentences) using native `librime` + `librime-octagram`:

| Configuration | Overall Top-1 | Target Suite (`125`) | CS Clauses (`40`) | CS Terms (`35`) | CJK-Latin (`20`) | Contextual (`12`) | Daily Guard (`17`) | CharAcc | `.gram` Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Original Baseline** (`60000` wiki + stock `.gram`) | `38.03%` | `29.60%` | `22.5%` | `40.0%` | `5.0%` | `41.7%` | `100.0%` | `65.38%` | 400.8 MiB |
| **+ 14B Unigram Dicts & CS Lift** | `88.73%` | `87.20%` | `82.5%` | `100.0%` | `100.0%` | `75.0%` | `100.0%` | `96.28%` | 400.8 MiB |
| **+ Pure 14B `.gram`** (`qwen14b-zh-hans.gram`) | `90.85%` | `90.40%` | `85.0%` | `100.0%` | `100.0%` | `83.3%` | `94.1%` | `97.52%` | **11.4 MiB** |
| **+ Hybrid Cleaned + 14B `.gram`** | **`92.25%`** | **`91.20%`** | **`95.0%`** | **`100.0%`** | **`100.0%`** | **`83.3%`** | **`100.0%`** | **`97.79%`** | 385.5 MiB |

Deduplicating redundant base pairs and zero-weight English entries also reduced compiled Rime binary tables (`build/*.bin`) from **296.0 MiB to 170.4 MiB (`-42.4%`)**.

## Why PMI for Rime N-grams, and Why Naive LLM Approaches Fail

### 1. Rime's `Poet` + `Octagram` equation requires Pointwise Mutual Information (PMI)
In `librime` (`poet.cc` + `dict_compiler.cc`), candidate path scoring computes:

$$\text{Score}(\text{context} \to \text{word}) = \ln\big(\text{dict\_weight}(\text{word})\big) + \text{Octagram::Query}(\text{context}, \text{word})$$

Because the dictionary term $\ln(\text{weight}(\text{word}))$ **already includes the unigram log-prior $\ln P(\text{word})$**, matching the true log-probability $\ln P(\text{word} \mid \text{context})$ requires `Octagram::Query` to supply the **conditional log-gain (PMI)**:

$$\text{PMI}(\text{context}, \text{word}) = \ln P(\text{word} \mid \text{context}) - \ln P(\text{word})$$

Storing raw corpus co-occurrence counts double-counts high-frequency words, while corpus-segmented KenLM pipelines suffer from web misspellings (e.g. `再接再励 = 200,616` outranking `再接再厉 = 176,835`, or `穿流不息 = 180,687` while `川流不息 = -1` was dropped by 1-gram segmentation underflow). Evaluating $\ln P(w_2 \mid w_1) - \ln P(w_2)$ directly from `Qwen3-14B-Base`'s `lm_head` eliminates web-corpus typos and scores billions of single-token word transitions in vectorized matrix form.

### 2. Frame-Matched Bayesian Domain Log-Odds Lift
Prompting a model with a bare prefix like `"在计算机科学中"` boosts generic glue words (`"的"`, `"问题"`). Pairing two structurally matched prompts — $C_{\text{domain}} = \text{"计算机科学与人工智能术语："}$ and $C_{\text{general}} = \text{"日常与百科常见词条："}$ — isolates exact domain concentration:

$$\Delta_{\text{domain}}(w) = \log P(w \mid C_{\text{domain}}) - \log P(w \mid C_{\text{general}})$$

General words cancel near zero or negative (`原神 = -3.50`), while genuine technical terms (`算子融合 = +7.56`, `前缀树 = +6.12`, `推测解码 = +4.24`) receive a smooth, bounded weight boost.

## Installation

```bash
git clone https://github.com/mizorewww/rime-qwen-freq.git
cd rime-qwen-freq

# Full environment (CUDA / Apple Silicon MPS / CPU scoring + distillation):
pip install -e ".[score]"

# Lightweight client (Python stdlib only — for inspect / extract / apply / bench):
pip install --no-deps -e .
```

To compile the standalone C++ `.gram` builder/merger (`tools/gram_tool.cc`):

```bash
c++ -O3 -std=c++17 tools/gram_tool.cc -o gram_tool
```

## Quick Start

### 1. Audit and calibrate Rime dictionaries
```bash
# Inspect your Rime dictionaries (macOS / Linux / Windows auto-detected)
rime-qwen-freq inspect

# One-shot local scoring + atomic writeback + Rime redeploy
rime-qwen-freq run --model Qwen/Qwen3.5-0.8B-Base --deploy

# Or split across laptop (stdlib) and GPU server with CS/AI domain lift:
rime-qwen-freq extract -o payload.json.gz
rime-qwen-freq score -i payload.json.gz -o scores.tar.gz \
  --model Qwen/Qwen3-14B-Base \
  --domain-prompt "计算机科学与人工智能术语：" \
  --general-prompt "日常与百科常见词条："
rime-qwen-freq apply -p payload.json.gz -s scores.tar.gz --deploy
```

### 2. Distill and compile a `.gram` N-gram model
```bash
# Distill high-PMI collocations & Rear-$ markers directly from Qwen3-14B-Base
rime-qwen-freq distill-ngram -o distilled_ngrams.tsv.gz --model Qwen/Qwen3-14B-Base
gunzip -k distilled_ngrams.tsv.gz

# Compile a standalone .gram file:
./gram_tool build distilled_ngrams.tsv qwen14b-zh-hans.gram

# Or clean homophone typos + merge into an existing base .gram:
./gram_tool merge wanxiang-lts-zh-hans.gram typo_blacklist.txt distilled_ngrams.tsv 140000 wanxiang-qwen14b-zh-hans.gram
```

### 3. Run the native `librime` benchmark
```bash
rime-qwen-freq bench --rime-dir ~/Library/Rime --schema rime_mint
```

## CLI Reference

| Command | Dependencies | Purpose |
|---|---|---|
| `rime-qwen-freq inspect` | Python stdlib | Audit `.dict.yaml` weight distributions and auto-detected roles (`--json` supported) |
| `rime-qwen-freq run` | `[score]` | End-to-end local extract, score, backup, atomic rewrite, and optional `--deploy` |
| `rime-qwen-freq extract` | Python stdlib | Export deduplicated target words and stratified anchors to `.json.gz` |
| `rime-qwen-freq score` | `[score]` | Score `.json.gz` on `cuda`/`mps`/`cpu` with optional `--domain-prompt` lift |
| `rime-qwen-freq apply` | Python stdlib | Back up originals, apply `.tar.gz` weights atomically, and `--deploy` |
| `rime-qwen-freq distill-ngram` | `[score]` | Distill PMI collocations and Rear-`$` markers from a CausalLM `.tsv.gz` |
| `rime-qwen-freq bench` | Python stdlib + C++17 | Compile `bench/rime_bench.cc` and benchmark a Rime folder via native `librime` |

## Testing

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## License

[MIT](LICENSE) (`tools/darts.h` is BSD-licensed Darts-clone v0.32 by Susumu Yata).
