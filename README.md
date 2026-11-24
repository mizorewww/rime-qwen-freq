# rime-qwen-freq

用 **Qwen3.5-0.8B-Base** 在离线 GPU 上为缺失真实词频（或被填充统一假常数权重）的 **Rime 输入法词库** 生成对齐万象（Wanxiang）/ ECDICT 尺度的静态词频权重，**打字运行时零推理开销**。

## 解决什么痛点？

很多第三方扩展词库（如维基百科 `zhwiki`、维基词典 `zhwiktionary`、萌娘百科 `moegirl`、网络流行语 `web_slang`、中英混输 `wanxiang_mixed`、雾凇英文 `rime_ice.en`）要么**完全缺失词频列**，要么在转换时被一刀切填成固定高权重（如统一 `60000`，高于 99.9% 的日常词条），导致输入法简拼/短码首选被生僻百科词霸占。

为什么不能直接抠大模型的 `tokenizer.json` 当词频？
1. **BPE 词表只有合并次序（Merge Rank），没有线性词频**，且充斥 `"的时"`、`"了一"` 等跨词边界碎片；
2. **原始对数似然 $\log P(w)$ 有严重的词长偏置**（2 字词概率天然远高于 6 字专有名词），且数值无法直接与现有主力词库（如 `wanxiang_jichu`）融合。

## 核心方法

```
本地 Rime 词库 ──▶ [1. prepare_payload.py] 抽取待打分去重词表 + 10万万象/Easy-En真实词频锚点
                                │
                                ▼
                   [2. gpu_score.py (CUDA)] Qwen/Qwen3.5-0.8B-Base
                     ├─ 构建 BPE 精确前缀树（Prefix Trie），共享前缀仅前向计算一次
                     ├─ 双语境 Teacher-Forcing（句首 "。\n" + 实体提及 "提到"）提取序列概率特征
                     └─ HistGradientBoostingRegressor 拟合锚点对数词频 + 方差还原校准
                                │
                                ▼
本地 Rime 词库 ◀── [3. apply_scores.py] 零外部依赖读取 .npy，自动备份原词典、原子替换并触发 Squirrel 热重载
```

### 实测校准效果（NVIDIA L4, 216.5 万去重词条）

在 15,001 条从未参与训练的万象留出验证集上：
- **$R^2 = 0.637$**，**Spearman 秩相关系数 $\rho = 0.735$**（对数词频 MAE `0.465`）

| 词条 | 来源词库 | 原权重 | Qwen3.5-0.8B 校准权重 | 万象参考真值 |
| :--- | :--- | ---: | ---: | ---: |
| `深度学习` | 锚点验证 | - | `14,714` | `15,502` |
| `原神` | 锚点验证 | - | `13,914` | `12,425` |
| `3D` / `5G` | `wanxiang_mixed` | *(无权重)* | `4,338` / `2,301` | - |
| `B站` / `A股` | `web_slang` / `zhwiktionary` | `60,000` | `2,053` / `840` | - |
| `女奴` | `moegirl` | `60,000` | `374` | `455` |
| `三角力量` / `滨边美波` | `moegirl` | `60,000` | `106` / `52` | - |
| `-真天地开辟集团-慈愚挫愚` | `zhwiki` | `60,000` | `3` | - |

## 快速开始

### 1. 本地打包待打分词表与校准锚点

默认自动识别 macOS `~/Library/Rime` 或 Linux `~/.local/share/fcitx5/rime`：

```bash
python3 scripts/prepare_payload.py \
  --rime-dir ~/Library/Rime \
  --output out/payload.json.gz
```

### 2. 在 CUDA GPU（如 Google Colab L4 / A100）上计算校准词频

```bash
pip install -r requirements-gpu.txt

python3 scripts/gpu_score.py \
  --payload out/payload.json.gz \
  --output out/rime_scores.tar.gz \
  --model Qwen/Qwen3.5-0.8B-Base
```

### 3. 本地备份、回写词典并热重载 Rime（仅依赖 Python 标准库）

```bash
python3 scripts/apply_scores.py \
  --rime-dir ~/Library/Rime \
  --payload out/payload.json.gz \
  --scores out/rime_scores.tar.gz \
  --backup-dir out/backup_dicts \
  --deploy
```

## License

MIT
