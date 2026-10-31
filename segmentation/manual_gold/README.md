# 《全唐诗》人工金标准覆盖

本目录把 `tang_poetry 人工金标准.xlsx` 中的人工词界覆盖到原始
`segmentation/results/sikufenci.jsonl`，但不改写基础结果的任何字符。

## 覆盖口径

- 以“诗人 + 篇名”定位诗歌；同名诗必须再以标注文本唯一消歧。
- 人工表中的 `/` 表示词界，单元格内格式空白不属于原文。
- 以 `…` 结束的行只覆盖省略号之前的已标前缀；其余正文保留原分词。
- 人工表与基础结果存在少量字面异文时，只在相似度不低于 0.95、且词界不落在异文内部时投射词界；输出仍保留基础结果字符。
- 未被人工表覆盖的诗歌原样保留。因此，派生文件是“人工金标准覆盖版”，不是 43,030 首全部人工标注的金标准。

## 输出

- `results/sikufenci_manual_gold.jsonl`：人工覆盖后的全量分词；
- `results/sikufenci_manual_gold.metadata.json`：来源哈希、覆盖范围、异文和验证统计；
- `results/sikufenci_manual_gold.audit.csv`：485 行人工标注的逐条覆盖审计。
- `全唐诗.txt`：由全量人工覆盖版导出的无 BOM UTF-8 文本；词元以单个半角空格分隔，诗篇以空行分隔，不写入 ID、诗人或篇名，并按原顺序保留 43,030 首（含 11 条空正文）。

`manual_gold.annotations.jsonl` 与旁边的元数据是从工作簿主表抽取的机器可读副本；应用脚本会同时核验工作簿、抽取文件和元数据哈希，避免在来源变化后静默复用旧标注。

## 预处理资源

`resources/` 根据人工覆盖后的全量结果提供五类可审计资源：异体字/词形规范、核心与扩展停用词、专名/典故/固定表达保护词、TF/DF 过滤规则和可选语义映射。运行以下命令可以重建：

```powershell
python segmentation/manual_gold/build_preprocessing_resources.py
```

具体阈值、证据边界和使用顺序见 `resources/README.md`。其中词形与语义映射均为旁表，不覆盖原诗字符。
