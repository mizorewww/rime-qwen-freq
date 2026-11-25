# rime-qwen-freq

用开源因果语言模型为 [Rime](https://rime.im/) 离线校准词表词频、提升领域术语，并直接从 `lm_head` 蒸馏 `librime-octagram`（`.gram`）语法模型。运行时纯静态查表，0 毫秒推理延迟。

[English](README.md) · **简体中文** · [Releases（成品词库与 `.gram` 下载）](https://github.com/mizorewww/rime-qwen-freq/releases/latest) · [原理与实测博客](https://mizore.blog/posts/rime-llm-word-frequency/)

## 成品下载

无需配置 Python 或 GPU，可直接在 [Releases](https://github.com/mizorewww/rime-qwen-freq/releases/latest) 下载基于 `Qwen/Qwen3-14B-Base` 构建的词表（`cs_ai`、`zhwiki`、`zhwiktionary`、`moegirl`、`web_slang`、`wanxiang_mixed`、`rime_ice.en`、`rime_ice.en_ext`）与独立语法模型 `qwen14b-zh-hans.gram.gz`（压缩包 `9.4 MiB`，解压 `11.4 MiB`，含 `1,031,842` 条高 PMI 搭配与句尾 `$` 标记）。

## 基准评测（`bench/bench_cases.tsv`，142 条，零 `userdb` 沙箱）

| 配置 | 总体首选率 | CS 整句 (`40`) | CS 术语 (`35`) | 逐字准确率 | `.gram` 体积 |
|---|---:|---:|---:|---:|---:|
| 原方案基线（未校准扩展词表 + 原版 `.gram`） | `38.03%` | `22.5%` | `40.0%` | `65.38%` | 400.8 MiB |
| + 14B 词表校准与 CS 增益 | `88.73%` | `82.5%` | `100.0%` | `96.28%` | 400.8 MiB |
| + 纯 14B `.gram`（`qwen14b-zh-hans`） | `90.85%` | `85.0%` | `100.0%` | `97.52%` | **11.4 MiB** |
| + 同音错搭清洗 + 14B 融合 `.gram` | **`92.25%`** | **`95.0%`** | **`100.0%`** | **`97.79%`** | 385.5 MiB |

## 用法

```bash
pip install -e ".[score]"              # 仅客户端提取/回写/评测可用 `pip install --no-deps -e .`
c++ -O3 -std=c++17 tools/gram_tool.cc -o gram_tool

# 1. 扫描并校准本地 Rime 词表
rime-qwen-freq inspect
rime-qwen-freq run --model Qwen/Qwen3-14B-Base \
  --domain-prompt "计算机科学与人工智能术语：" \
  --general-prompt "日常与百科常见词条：" --deploy

# 2. 蒸馏并编译 .gram 语法模型
rime-qwen-freq distill-ngram -o ngrams.tsv.gz --model Qwen/Qwen3-14B-Base
gunzip -k ngrams.tsv.gz && ./gram_tool build ngrams.tsv qwen14b-zh-hans.gram

# 3. 运行原生 librime 端到端评测
rime-qwen-freq bench --rime-dir ~/Library/Rime --schema rime_mint
```

跨机器在远端 GPU 打分可拆分为：`extract -o payload.json.gz` → `score -i payload.json.gz -o scores.tar.gz` → `apply -p payload.json.gz -s scores.tar.gz --deploy`。

## 许可

[MIT](LICENSE)（`tools/darts.h` 为 BSD 协议 Darts-clone v0.32）。
