# rime-qwen-freq

**Distill calibrated static word frequencies for Rime input method dictionaries from open-weight language models.**

[![Newest release](https://img.shields.io/github/v/release/mizorewww/rime-qwen-freq?sort=semver&label=release)](https://github.com/mizorewww/rime-qwen-freq/releases)
[![License: MIT](https://img.shields.io/github/license/mizorewww/rime-qwen-freq)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![macOS · Linux · Windows](https://img.shields.io/badge/rime-macOS%20%C2%B7%20Linux%20%C2%B7%20Windows-informational)

**English** · [简体中文](README.zh-CN.md) · [Pre-built Dictionaries](#pre-built-dictionaries)

`rime-qwen-freq` audits a [Rime](https://rime.im/) configuration directory, finds `.dict.yaml` dictionaries that have missing or constant placeholder weights, scores their entries offline with a causal language model (`Qwen/Qwen3.5-0.8B-Base` by default), and calibrates the predicted weights onto the exact empirical scale of your existing base dictionary (`rime-ice`, `rime-wanxiang`, `rime-mint`, `luna_pinyin`, or `easy_en`).

All computation happens offline. Rime continues to use native `.table.bin` lookup at runtime with **zero inference latency**.

## Pre-built Dictionaries

Ready-to-use `.dict.yaml` tables pre-scored with `Qwen/Qwen3.5-0.8B-Base` and calibrated to the **Wanxiang (`wanxiang_jichu`)** / **ECDICT (`easy_en`)** scales (2,170,186 entries total) are available on the [Releases](https://github.com/mizorewww/rime-qwen-freq/releases/latest) page — no Python or GPU setup required:

| Dictionary | Entries | Upstream Source | Scale |
|---|---:|---|---|
| `zhwiki.dict.yaml` | 1,728,419 | Chinese Wikipedia (`fcitx5-pinyin-zhwiki`) | Wanxiang (`1` – `300,000`) |
| `zhwiktionary.dict.yaml` | 285,853 | Chinese Wiktionary (`fcitx5-pinyin-zhwiki`) | Wanxiang (`1` – `300,000`) |
| `moegirl.dict.yaml` | 129,813 | Moegirlpedia (`mw2fcitx`) | Wanxiang (`1` – `300,000`) |
| `web_slang.dict.yaml` | 423 | Web slang (`fcitx5-pinyin-zhwiki`) | Wanxiang (`1` – `300,000`) |
| `wanxiang_mixed.dict.yaml` | 1,817 | CJK-Latin mixed terms (`rime-wanxiang` toneless) | Wanxiang (`1` – `300,000`) |
| `rime_ice.en.dict.yaml` | 21,263 | Core English vocabulary (`rime-ice`) | ECDICT (`900,000` – `999,995`) |
| `rime_ice.en_ext.dict.yaml` | 2,598 | Extended English & tech terms (`rime-ice`) | ECDICT (`900,000` – `999,995`) |

Download [`rime-qwen-dicts-v0.2.0.tar.gz`](https://github.com/mizorewww/rime-qwen-freq/releases/download/v0.2.0/rime-qwen-dicts-v0.2.0.tar.gz) (or [`.zip`](https://github.com/mizorewww/rime-qwen-freq/releases/download/v0.2.0/rime-qwen-dicts-v0.2.0.zip)), extract into your Rime `dicts/` directory, reference the desired tables under `import_tables:` in your `.dict.yaml`, and redeploy Rime.

## Why not read BPE ranks or raw log-probabilities?

Third-party Rime extension dictionaries (Wikipedia titles, Wiktionary, ACG/fandom lexicons, slang lists, bilingual supplements) frequently omit the weight column or assign a flat placeholder constant (such as `60000` on every row), causing obscure multi-word stubs to outrank everyday words on short Pinyin prefixes.

Two common shortcuts fail in practice:

1. **Tokenizer BPE merge ranks are not word frequencies.** Byte-level BPE vocabularies contain non-lexical glue fragments (`"的时"`, `"了一"`), omit pronunciation disambiguation, and flatten Zipfian frequency gaps into linear merge indices.
2. **Raw sequence log-likelihood $\log P(w)$ is length-biased and uncalibrated.** A 2-character word naturally has a much higher joint probability than a 6-character proper noun, and raw log-probabilities cannot be merged directly with corpus-counted weights in an existing Rime schema.

`rime-qwen-freq` solves both by combining **exact Prefix-Trie teacher-forcing** across complementary contexts with **empirical anchor regression** trained directly on the weighted base dictionaries already present in your Rime folder.

## Installation

Requires **Python 3.10+**.

### Full installation (scoring + dictionary management)

Install in a virtual environment on any machine with **Apple Silicon (`mps`)**, an **NVIDIA GPU (`cuda`)**, or CPU:

```bash
git clone https://github.com/mizorewww/rime-qwen-freq.git
cd rime-qwen-freq
pip install -e ".[score]"
```

On Linux with an NVIDIA GPU, optional fused linear-attention kernels for `Qwen3.5` can be installed via:

```bash
pip install -e ".[score,cuda-kernels]"
```

### Lightweight client installation (stdlib only)

The `inspect`, `extract`, and `apply` subcommands use **only the Python standard library** (including a pure-stdlib `.npy` reader). If you extract words on a lightweight laptop and run `score` on a separate GPU box, the laptop needs no third-party packages:

```bash
pip install --no-deps -e .
```

## Usage

### 1. Audit your Rime dictionaries

Scan your Rime user directory (`~/Library/Rime` on macOS, `~/.local/share/fcitx5/rime` or `~/.config/ibus/rime` on Linux, `%APPDATA%\Rime` on Windows) to see which tables have real empirical distributions (`anchor`), which lack real weights (`needs-weights`), and which auxiliary typo/symbol tables are intentionally skipped (`skip-special`):

```bash
rime-qwen-freq inspect
```

Example output:

```text
Rime directory: /Users/alice/Library/Rime

Dictionary                           Lang    Entries    UniqW      P50       P99  Role            Diagnostic
------------------------------------------------------------------------------------------------------------------------------------
dicts/wanxiang_jichu.dict.yaml       zh    1,493,800   11,352      279     5,256  anchor          11,352 distinct empirical weights
dicts/zhwiki.dict.yaml               zh    1,728,419        1   60,000    60,000  needs-weights   100.0% constant weight (60000)
dicts/moegirl.dict.yaml              zh      129,813        1   60,000    60,000  needs-weights   100.0% constant weight (60000)
dicts/wanxiang_mixed.dict.yaml       zh        1,817        0        0         0  needs-weights   100% missing weights
dicts/rime_ice.others.dict.yaml      zh          584        0        0         0  skip-special    typo/aux/shape dictionary
dicts/easy_en.dict.yaml              en      745,397   40,770  973,299   998,120  anchor          40,770 distinct empirical weights
dicts/rime_ice.en.dict.yaml          en       21,263        0        0         0  needs-weights   100% missing weights
```

### 2. One-shot local pipeline (`run`)

If your machine has Apple Silicon (`mps`) or an NVIDIA GPU (`cuda`), run extraction, scoring, atomic dictionary update (with automatic backups in `./backup_dicts`), and Rime redeployment in a single command:

```bash
rime-qwen-freq run --deploy
```

To score only specific dictionaries or use another Hugging Face CausalLM checkpoint:

```bash
rime-qwen-freq run \
  --targets dicts/zhwiki.dict.yaml dicts/wanxiang_mixed.dict.yaml \
  --model Qwen/Qwen3-0.6B-Base \
  --deploy
```

### 3. Split workflow across two machines (`extract` → `score` → `apply`)

When your Rime setup lives on a thin laptop and you want to run scoring on a separate GPU workstation or server:

```bash
# Step A (on laptop, stdlib only): package target words + stratified anchor sample
rime-qwen-freq extract -o payload.json.gz

# Step B (on GPU machine): run Prefix-Trie scoring & calibration
rime-qwen-freq score -i payload.json.gz -o scores.tar.gz --model Qwen/Qwen3.5-0.8B-Base

# Step C (on laptop, stdlib only): back up originals, write weights atomically, and reload Rime
rime-qwen-freq apply -p payload.json.gz -s scores.tar.gz --deploy
```

## How It Works

1. **Prefix-Trie Exact Teacher-Forcing:** All target words and anchor words are tokenized into BPE sequences and inserted into a proper-prefix Trie. Shared prefixes are evaluated by the transformer backbone **only once**, and a sparse on-device gather extracts only the required child token log-probabilities (avoiding multi-gigabyte `[batch, vocab_size]` host transfers). A built-in numerical equivalence check verifies exactness against full-sequence `log_softmax` before each run.
2. **Dual-Context Feature Extraction:** Each entry is scored under two complementary contexts — clause-initial (`"。\n"` / `".\n"`) for general vocabulary and mention-context (`"提到"` / `"The "`) for named entities, technical terms, and mixed CJK-ASCII tokens — extracting total log-probability, entry/exit transitions, weakest-link token log-probability (which penalizes malformed encyclopedia titles), token count, character length, and BPE rank features.
3. **Empirical Anchor Calibration:** Stratified ground-truth anchors sampled from your existing base dictionaries train a regularized `HistGradientBoostingRegressor` in $\log_{10}(w + 1)$ space (or linear rank space for ECDICT-style tables), followed by a variance-preserving stretch so high-frequency words and rare long-tail entries match your schema's native Zipfian distribution.

### Sample Calibration Results (`Qwen/Qwen3.5-0.8B-Base`, 2.16M unique entries)

Evaluated on 15,001 held-out ground-truth entries from `wanxiang_jichu`: **$R^2 = 0.637$**, **Spearman $\rho = 0.735$** ($\text{MAE}_{\log_{10}} = 0.465$).

| Entry | Source Dictionary | Previous Weight | Calibrated Weight | Reference Ground Truth |
|---|---|---:|---:|---:|
| `深度学习` | Held-out validation | — | `14,714` | `15,502` |
| `原神` | Held-out validation | — | `13,914` | `12,425` |
| `3D` / `5G` | `wanxiang_mixed` | *(missing)* | `4,338` / `2,301` | — |
| `B站` / `A股` | `web_slang` / `zhwiktionary` | `60,000` | `2,053` / `840` | — |
| `女奴` | `moegirl` | `60,000` | `374` | `455` |
| `三角力量` / `滨边美波` | `moegirl` | `60,000` | `106` / `52` | — |
| `-真天地开辟集团-慈愚挫愚` | `zhwiki` | `60,000` | `3` | — |

## CLI Reference

| Command | Dependencies | Purpose |
|---|---|---|
| `rime-qwen-freq inspect` | Python stdlib | Audit `.dict.yaml` weight distributions and auto-detected roles (`--json` supported) |
| `rime-qwen-freq run` | `[score]` | End-to-end local extract, score, backup, atomic rewrite, and optional `--deploy` |
| `rime-qwen-freq extract` | Python stdlib | Export deduplicated target words and stratified anchors to `.json.gz` |
| `rime-qwen-freq score` | `[score]` | Score a `.json.gz` bundle on `cuda`, `mps`, or `cpu` and emit `.tar.gz` weights + metrics |
| `rime-qwen-freq apply` | Python stdlib | Back up original `.dict.yaml` files, apply `.tar.gz` weights atomically, and `--deploy` |

`--deploy` automatically invokes Squirrel (`rime_deployer` + `Squirrel --reload`) on macOS, `rime_deployer` + `fcitx5-remote -r` / `ibus restart` on Linux, and `WeaselDeployer.exe /deploy` on Windows.

## Development & Testing

Run the unit test suite (covers dictionary classification, extraction, atomic rewriting, and the pure-stdlib `.npy` parser):

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Contributing

Issues and pull requests are welcome. For major changes to the feature extractor or calibration objective, please open an issue first with before/after Spearman $\rho$ metrics on a held-out anchor split.

## License

[MIT](LICENSE)
