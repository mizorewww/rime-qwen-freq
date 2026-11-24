# rime-qwen-freq

**利用开源因果语言模型，为 Rime（中州韵）输入法词库离线蒸馏并校准静态词频权重。**

[![License: MIT](https://img.shields.io/github/license/mizorewww/rime-qwen-freq)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![macOS · Linux · Windows](https://img.shields.io/badge/rime-macOS%20%C2%B7%20Linux%20%C2%B7%20Windows-informational)

[English](README.md) · **简体中文**

`rime-qwen-freq` 会自动扫描你的 [Rime](https://rime.im/) 配置目录，识别**缺失词频列**或**被统一填充假常数权重**的 `.dict.yaml` 扩展词库，在离线环境下利用开源语言模型（默认 `Qwen/Qwen3.5-0.8B-Base`，支持任意 Hugging Face `CausalLM`）提取序列概率特征，并将其精准校准到你现有主力词库（如雾凇拼音 `rime-ice`、万象拼音 `rime-wanxiang`、薄荷拼音 `rime-mint`、朙月拼音或 `easy_en`）的真实词频尺度上。

所有模型计算均在**离线阶段一次性完成**；打字运行时依然使用 Rime 原生 `.table.bin` 查表，**零推理延迟、零后台常驻进程**。

## 为什么不能直接抠 Tokenizer 词表或裸算对数概率？

许多第三方导入的 Rime 扩展词库（中文维基百科、维基词典、萌娘百科、网络热词、中英混输补充词表等）要么完全没有第 3 列权重，要么在转换时被一刀切填成固定高数值（例如全部设为 `60000`，高于 99.9% 的日常词），导致敲简拼或短码时被冷门百科词条霸屏。

常见的两种捷径在实际使用中都会翻车：

1. **BPE 词表的合并次序（Merge Rank）不等于词频**：Byte-level BPE 充斥 `"的时"`、`"了一"` 等跨词胶着片段，没有多音字注音，且抹平了齐夫定律（Zipf's law）下的真实频次差距。
2. **原始对数似然 $\log P(w)$ 存在严重词长偏置且尺度不兼容**：2 字词的联合概率天然远高于 6 字专有名词，且裸概率无法直接与现有词库的语料统计频数混排。

`rime-qwen-freq` 通过 **精确前缀树（Prefix Trie）双语境前向传播** 提取多维序列特征，并以你本地已有的高质量基础词库作为监督锚点进行 **经验分布回归校准**，一步解决长短词偏置与跨词库尺度对齐问题。

## 安装

需要 **Python 3.10+**。

### 完整安装（含模型打分与词库管理）

适用于配备 **Apple Silicon (`mps`)**、**NVIDIA 显卡 (`cuda`)** 或 CPU 的机器：

```bash
git clone https://github.com/mizorewww/rime-qwen-freq.git
cd rime-qwen-freq
pip install -e ".[score]"
```

若在 Linux + NVIDIA GPU 上运行 `Qwen3.5` 系列模型，可额外安装线性注意力融合算子加速包：

```bash
pip install -e ".[score,cuda-kernels]"
```

### 轻量客户端安装（零第三方依赖）

`inspect`、`extract`、`apply` 三个子命令**仅使用 Python 标准库**（内置纯标准库 `.npy` 解析器）。如果你在轻薄本上提取/回写词库、在另一台 GPU 机器上跑 `score`，轻薄本无需安装 PyTorch 或 NumPy：

```bash
pip install --no-deps -e .
```

## 使用方法

### 1. 诊断本地 Rime 词库（`inspect`）

自动识别 macOS 鼠须管（`~/Library/Rime`）、Linux Fcitx5 / IBus（`~/.local/share/fcitx5/rime`、`~/.config/ibus/rime`）及 Windows 小狼毫（`%APPDATA%\Rime`），输出所有词典的权重分布与自动判定角色：

```bash
rime-qwen-freq inspect
```

示例输出：

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

*(注：错音纠错表 `*others*`、符号/Emoji、五笔/笔画/拆字反查表会被自动标记为 `skip-special` 跳过，防止错音压过正音。)*

### 2. 本机一键生成并部署（`run`）

如果本机自带 Apple Silicon (`mps`) 或 NVIDIA GPU (`cuda`)，一条命令即可完成提取、打分、备份原词典（默认存于 `./backup_dicts`）、原子回写并重新编译重载 Rime：

```bash
rime-qwen-freq run --deploy
```

也可以手动指定目标词典或切换任意 Hugging Face 基座模型：

```bash
rime-qwen-freq run \
  --targets dicts/zhwiki.dict.yaml dicts/wanxiang_mixed.dict.yaml \
  --model Qwen/Qwen3-0.6B-Base \
  --deploy
```

### 3. 跨机器分步工作流（`extract` → `score` → `apply`）

当 Rime 安装在无独立显卡的轻薄本上，而你想把计算放到远程 GPU 工作站或服务器执行时：

```bash
# 第一步（本地机器，零依赖）：抽取去重待打分词表 + 分层锚点
rime-qwen-freq extract -o payload.json.gz

# 第二步（GPU 机器）：运行前缀树双语境打分与分布校准
rime-qwen-freq score -i payload.json.gz -o scores.tar.gz --model Qwen/Qwen3.5-0.8B-Base

# 第三步（本地机器，零依赖）：自动备份原词库、原子替换权重并触发 Rime 重新部署
rime-qwen-freq apply -p payload.json.gz -s scores.tar.gz --deploy
```

## 工作原理与实测指标

1. **BPE 精确前缀树（Prefix Trie）去重加速**：将百万级词条的所有真前缀合并为一棵前缀树，共享前缀在 Transformer 主干上**仅计算一次**，并在设备端直接完成稀疏索引收集（避免将 `[batch, vocab_size]` 大张量拷回内存）；每次运行前自动与完整序列 `log_softmax` 做数值等价校验。
2. **双语境特征提取**：同时在句首语境（`"。\n"` / `".\n"`）与实体提及语境（`"提到"` / `"The "`）下提取总对数似然、首尾转移概率、最弱 token 概率（自动惩罚残缺词条）及 BPE 合并秩特征。
3. **经验锚点回归与方差还原**：使用主力词库分层抽样的真实权重训练正则化梯度提升树（`HistGradientBoostingRegressor`），并通过方差还原拉伸恢复齐夫长尾分布，自动兼容对数频数尺度（万象/雾凇）与线性秩尺度（ECDICT `easy_en`）。

在 216.5 万去重词条上的实测结果（`Qwen/Qwen3.5-0.8B-Base`，15,001 条万象留出验证集）：**$R^2 = 0.637$**，**Spearman 秩相关系数 $\rho = 0.735$**。

| 词条 | 来源词典 | 修复前权重 | 校准后权重 | 万象参考真值 |
|---|---|---:|---:|---:|
| `深度学习` | 留出集验证 | — | `14,714` | `15,502` |
| `原神` | 留出集验证 | — | `13,914` | `12,425` |
| `3D` / `5G` | `wanxiang_mixed` | *(缺失)* | `4,338` / `2,301` | — |
| `B站` / `A股` | `web_slang` / `zhwiktionary` | `60,000` | `2,053` / `840` | — |
| `女奴` | `moegirl` | `60,000` | `374` | `455` |
| `三角力量` / `滨边美波` | `moegirl` | `60,000` | `106` / `52` | — |
| `-真天地开辟集团-慈愚挫愚` | `zhwiki` | `60,000` | `3` | — |

## 开发与测试

运行内置单元测试（覆盖词库自动分类、锚点抽样、原子回写与标准库 `.npy` 读取器）：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 贡献与开源协议

欢迎提交 Issue 与 Pull Request。涉及特征提取或校准目标函数的改动，请附上留出锚点验证集前后的 Spearman $\rho$ 对比数据。

本项目基于 [MIT License](LICENSE) 开源。
