# rime-qwen-freq

**利用开源因果语言模型，为 Rime（中州韵）离线蒸馏静态词频与 PMI 语法模型（`.gram`）。**

[![Newest release](https://img.shields.io/github/v/release/mizorewww/rime-qwen-freq?sort=semver&label=release)](https://github.com/mizorewww/rime-qwen-freq/releases)
[![License: MIT](https://img.shields.io/github/license/mizorewww/rime-qwen-freq)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![macOS · Linux · Windows](https://img.shields.io/badge/rime-macOS%20%C2%B7%20Linux%20%C2%B7%20Windows-informational)

[English](README.md) · **简体中文** · [预构建词表与语法模型下载](#预构建词表与语法模型下载) · [端到端评测](#端到端-librime-基准评测)

`rime-qwen-freq` 在**纯离线阶段**用开源基座语言模型升级 [Rime](https://rime.im/) 的两层核心数据：

1. **词表词频校准与领域增益（Unigram）**：自动扫描缺词频或填死常数（如全部 `60000`）的 `.dict.yaml`，通过 **BPE 精确前缀树（Prefix Trie）** 前向打分、**经验锚点分布回归** 与可选的 **贝叶斯领域对数几率提升（`--domain-prompt`）**，把词频精准对齐到你现有主力词库（万象、雾凇、薄荷、朙月或 `easy_en`）的尺度上。
2. **直接对数几率 PMI 语法模型蒸馏（N-gram / `.gram`）**：直接读取语言模型 `lm_head` 的条件概率分布，按点互信息（PMI）蒸馏多词搭配与句尾 `$` 收敛标记，通过内置 C++ 工具 `tools/gram_tool.cc` 编译为 `librime-octagram` 双数组前缀树 `.gram` 文件，并支持扫描清洗传统爬虫 `.gram` 中的同音错别字。
3. **原生 `librime` 端到端基准评测（`bench`）**：直接动态链接系统 Rime 引擎（`librime` + `librime-octagram`），实测连续全拼整句盲打、专业术语首选率与上下文联想消歧准确率。

所有计算在离线一次性完成；日常打字依然是纯静态 `.table.bin` + `.gram` 内存映射查表，**0 毫秒推理延迟、零后台驻留**。

## 预构建词表与语法模型下载

不想本地跑模型的用户，可直接在 [Releases](https://github.com/mizorewww/rime-qwen-freq/releases/latest) 下载基于 `Qwen/Qwen3-14B-Base` 蒸馏好的成品文件：

| 文件名 | 词条 / 键数量 | 压缩体积 | 说明 |
|---|---:|---:|---|
| `qwen14b-zh-hans.gram.gz` | 1,031,842 | 9.4 MiB | 纯 14B 蒸馏开源 `.gram` 语法模型（解压后 **11.4 MiB**，体积约为 `400.8 MiB` 语料版 `.gram` 的 1/35，零爬虫错别字） |
| `cs_ai.dict.yaml.gz` | 1,986 | 26 KiB | 精选 + 贝叶斯领域挖掘的 CS / AI / ML / 系统工程术语词表（多音字读音已校对） |
| `zhwiki.dict.yaml.gz` | 1,464,285 | 17.4 MiB | 中文维基百科（`fcitx5-pinyin-zhwiki`，已与基础词库去重降噪，万象尺度 `1–350,000`） |
| `zhwiktionary.dict.yaml.gz` | 165,545 | 1.6 MiB | 维基词典（已去重，万象尺度） |
| `moegirl.dict.yaml.gz` | 105,076 | 1.9 MiB | 萌娘百科（`mw2fcitx`，已去重，万象尺度） |
| `web_slang.dict.yaml.gz` | 423 | 6.5 KiB | 网络流行语（万象尺度） |
| `wanxiang_mixed.dict.yaml.gz` | 1,817 | 19 KiB | 中英混输词条（去声调兼容版，已做多音字消歧） |
| `rime_ice.en.dict.yaml.gz` | 21,263 | 194 KiB | 雾凇英文主词库（ECDICT 尺度 `900,000–999,995`） |
| `rime_ice.en_ext.dict.yaml.gz` | 2,598 | 29 KiB | 雾凇英文扩展词库（ECDICT 尺度） |

## 端到端 `librime` 基准评测

在隔离的零用户词典（0-`userdb`）沙箱中运行 `rime-qwen-freq bench`（`bench/bench_cases.tsv`，共 142 条长度 $\le 25$ 键的无截断真实测试：覆盖 CS/AI 连续整句、专业复合词、中英混输、上下文消歧、百科 `60000` 抢权与同音错别字陷阱，以及日常口语防退化护栏集）调用原生 `librime` + `librime-octagram` 实测：

| 测试环境 | 总体首选率 (Top-1) | 目标集首选率 (`125` 条) | CS 整句 (`40`) | CS 术语 (`35`) | 中英混输 (`20`) | 上下文消歧 (`12`) | 日常护栏 (`17`) | 逐字准确率 | `.gram` 体积 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **原方案基线**（百科 `60000` + 原版 `.gram`） | `38.03%` | `29.60%` | `22.5%` | `40.0%` | `5.0%` | `41.7%` | `100.0%` | `65.38%` | 400.8 MiB |
| **+ 14B 词表校准与 CS 领域增益** | `88.73%` | `87.20%` | `82.5%` | `100.0%` | `100.0%` | `75.0%` | `100.0%` | `96.28%` | 400.8 MiB |
| **+ 纯 14B 开源 `.gram`**（`qwen14b-zh-hans`） | `90.85%` | `90.40%` | `85.0%` | `100.0%` | `100.0%` | `83.3%` | `94.1%` | `97.52%` | **11.4 MiB** |
| **+ 同音错搭清洗 + 14B 融合 `.gram`** | **`92.25%`** | **`91.20%`** | **`95.0%`** | **`100.0%`** | **`100.0%`** | **`83.3%`** | **`100.0%`** | **`97.79%`** | 385.5 MiB |

同时，剔除基础词表重复项与英文零权重废词后，Rime 编译二进制词表（`build/*.bin`）从 **296.0 MiB 降至 170.4 MiB（缩减 42.4%）**。

## 核心原理

### 1. 为什么 Rime 的 `.gram` 应该蒸馏点互信息（PMI），而不是数共现频次？
在 `librime` 源码（`poet.cc` + `dict_compiler.cc`）中，整句搜索从上文 `context` 走到候选词 `word` 的转移得分是：

$$\text{Score}(\text{context} \to \text{word}) = \ln\big(\text{dict\_weight}(\text{word})\big) + \text{Octagram::Query}(\text{context}, \text{word})$$

注意第一项**已经加过一次候选词自身的对数先验 $\ln P(\text{word})$**。要让整条路径得分严格对应真实条件对数概率 $\ln P(\text{word} \mid \text{context})$，语法模型 `Octagram::Query` 真正需要给出的是**点互信息（条件对数增益）**：

$$\text{PMI}(\text{context}, \text{word}) = \ln P(\text{word} \mid \text{context}) - \ln P(\text{word})$$

传统基于网页语料分词 + KenLM 计数的流程有两个固有缺陷：一是爬虫语料中的高频同音错别字被结巴切碎后在多阶 N-gram 中拿到虚高分数（例如实测原版 `.gram` 中 `再接再励 = 200,616` 反超正字 `再接再厉 = 176,835`，`穿流不息 = 180,687` 而正字 `川流不息 = -1` 漏收，`整行医院 = 231,925`、`强联通分量 = 223,327`）；二是新领域搭配（如 `推测解码`、`缓存命中`）因频次不足被剪枝丢弃。直接从 `Qwen3-14B-Base` 的 `lm_head` 矩阵读取精确条件分布计算 PMI，一次前向即可同时评估数万个单 token 后续词，天然不含错别字噪声。

### 2. 句式匹配的贝叶斯领域对数几率增益（`--domain-prompt`）
如果直接拼接 `"在计算机科学中"` 计算概率，`"的"`、`"问题"` 等通用虚词也会跟着涨分。我们采用结构对称的双提示词做差：

$$\Delta_{\text{domain}}(w) = \log P(w \mid \text{“计算机科学与人工智能术语：”}) - \log P(w \mid \text{“日常与百科常见词条：”})$$

基础词频严格在通用语境下回归校准以保护日常词序；只有领域富集度显著为正的词条（如 `算子融合 = +7.56`、`前缀树 = +6.12`、`推测解码 = +4.24`）才获得平滑有界的权重加成，而日常词与非本领域词（如 `原神 = -3.50`）增益精确归零。

## 安装与快速上手

```bash
git clone https://github.com/mizorewww/rime-qwen-freq.git
cd rime-qwen-freq

# 完整安装（支持 CUDA / Apple Silicon MPS / CPU 模型打分与蒸馏）：
pip install -e ".[score]"

# 轻量客户端安装（仅依赖 Python 标准库，用于 inspect / extract / apply / bench）：
pip install --no-deps -e .

# 编译独立 C++ .gram 构建与清洗合并工具：
c++ -O3 -std=c++17 tools/gram_tool.cc -o gram_tool
```

### 常用命令

```bash
# 1. 扫描诊断本地 Rime 词库
rime-qwen-freq inspect

# 2. 跨机器分步生成带 CS/AI 领域增强的词频
rime-qwen-freq extract -o payload.json.gz
rime-qwen-freq score -i payload.json.gz -o scores.tar.gz \
  --model Qwen/Qwen3-14B-Base \
  --domain-prompt "计算机科学与人工智能术语：" \
  --general-prompt "日常与百科常见词条："
rime-qwen-freq apply -p payload.json.gz -s scores.tar.gz --deploy

# 3. 直接从语言模型蒸馏 PMI 语法模型 (.gram)
rime-qwen-freq distill-ngram -o distilled_ngrams.tsv.gz --model Qwen/Qwen3-14B-Base
gunzip -k distilled_ngrams.tsv.gz
./gram_tool build distilled_ngrams.tsv qwen14b-zh-hans.gram

# 4. 运行原生 librime 端到端基准测试
rime-qwen-freq bench --rime-dir ~/Library/Rime --schema rime_mint
```

## 测试与协议

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

本项目采用 [MIT License](LICENSE) 开源（`tools/darts.h` 为 Susumu Yata 开发的 BSD 协议 Darts-clone v0.32）。
