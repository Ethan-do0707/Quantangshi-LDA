# 《全唐诗》四路分词与比较

本目录把 `tang_poetry.sql` 中的 43,030 条诗歌正文导出为统一语料，分别运行：

- sikufenci 0.0.28 的官方 SikuBERT 分词微调权重；
- Jihuai/bert-ancient-chinese 的官方 WordPiece tokenizer；
- Jiayan 0.0.21 的 `WordNgramTokenizer`；
- jieba 0.42.1 精确模式（启用 HMM）。

四套全量结果与比较报告均已生成。总体结论见 `comparison/comparison_report.md`。

## 重要口径

`bert-ancient-chinese` 仓库发布的是预训练 Fill-Mask 模型，未发布可直接推理的 CWS 任务头。这里保留其官方 tokenizer 的子词边界作为基线，不将它冒充为经过分词微调的模型结果。

Jiayan 官方推荐的 `CharHMMTokenizer` 需要 KenLM 和单独下载的 `jiayan.klm`。当前 Python 3.11/Windows 环境没有 KenLM 轮子，本机也无 MSVC/WSL 编译工具链，因此使用同一项目官方提供、无需 KenLM 的 `WordNgramTokenizer`。完整限制记录在每套结果旁的 `*.metadata.json`。

sikufenci README 的 Google Drive 文件经权重键实测是 `BertForMaskedLM`，缺少 `classifier.weight` 与 `classifier.bias`，不能生成 B/I/E/S 标签。本次使用 README 所列百度网盘中的 415.6 MB 分词权重；程序会在加载时检查分类头，避免把预训练权重误当作分词结果。

## 运行

```powershell
python segmentation/pipeline.py export
python segmentation/pipeline.py run --method jieba
python segmentation/pipeline.py run --method jiayan
python segmentation/pipeline.py run --method bert_ancient
python segmentation/pipeline.py run --method sikufenci --batch-size 8
python segmentation/pipeline.py compare
```

sikufenci 是 CPU 上最慢的一路，使用 `segmentation/results/sikufenci.jsonl.part` 保存完整行断点；重新执行相同命令会从断点继续。

运行 sikufenci 前，请将含 `classifier.weight` 和 `classifier.bias` 的官方分词文件放到 `models/sikufenci/pytorch_model.bin`。官方百度网盘链接为 <https://pan.baidu.com/s/1ePPlCpoZ4UTsUaQumMpZTQ>，提取码 `c9hb`。

## 输出

- `data/poems.jsonl`：统一语料，含 ID、诗人、题目和原文；
- `results/*.jsonl`：四套完整分词，每行含 ID 与词数组；
- `results/*.metadata.json`：版本、算法口径、限制和耗时；
- `results/sikufenci_lda.jsonl`：依照 LDA 分词规范对 sikufenci 作确定性覆盖后的另存结果；
- `results/sikufenci_lda.metadata.json`：覆盖规则、哈希、统计和验证信息；
- `results/sikufenci_lda.audit.csv`：词界发生变化的逐诗审计；
- `results/sikufenci_manual_gold.jsonl`：依据人工金标准对原始 sikufenci 作局部覆盖后的全量另存结果；
- `results/sikufenci_manual_gold.metadata.json`：人工覆盖范围、来源哈希、异文和验证统计；
- `results/sikufenci_manual_gold.audit.csv`：485 行人工标注的逐条覆盖审计；
- `comparison/comparison_report.md`：中文比较报告；
- `comparison/*.csv`：总体统计、两两边界一致性、高频词和差异样例。

比较只考察原文中相邻两个汉字之间的词界。由于没有人工标注金标准，报告只描述粒度和方法间一致性，不把一致率解释成准确率。

## LDA 人工覆盖口径

四路结果是基础分词比较，不直接等同于最终 LDA 词界。最终口径应在选定的基础结果上应用 [`lda_standard/README.md`](lda_standard/README.md) 所规定的保护词、典故词、命名实体与禁合并规则；其中 `来仪`、`折腰`、`倾城`、`难再逢`、`良可惜`、`千钟` 为最高优先级强制合并项。

## 人工金标准覆盖

[`manual_gold/README.md`](manual_gold/README.md) 说明如何把 `tang_poetry 人工金标准.xlsx` 的人工词界覆盖到原始 sikufenci。当前工作簿含 485 首：426 首全文标注，59 首只标注省略号前的前缀；另有 4 首存在字面异文，只投射词界而不改写基础字符。未标诗与前缀之外的正文保留原分词，因此该派生结果不能表述为 43,030 首全量人工金标准。

## 官方来源

- https://github.com/hsc748NLP/sikufenci
- https://github.com/Jihuai-wpy/bert-ancient-chinese
- https://github.com/jiaeyan/Jiayan
- https://github.com/fxsjy/jieba
