"""汇总留出困惑度结果，生成CSV、图、报告和最终审计。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model_selection/perplexity_diagnostic"
DATA = OUT / "data"
FIGURES = OUT / "figures"
FONT = ROOT / "historical_analysis/publication_figures/qa/fonts/QTS_Noto_Regular.ttf"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""): digest.update(chunk)
    return digest.hexdigest()


def f(value, digits=10): return f"{float(value):.{digits}f}"


def summarize(runs):
    groups = {}
    for row in runs: groups.setdefault((row["corpus"], int(row["K"])), []).append(row)
    output = []
    for (corpus, k), rows in sorted(groups.items()):
        test = np.array([float(r["test_perplexity"]) for r in rows]); train = np.array([float(r["train_perplexity"]) for r in rows])
        output.append({"corpus": corpus, "K": k, "seed_count": len(rows), "mean_test_perplexity": f(test.mean()),
                       "median_test_perplexity": f(np.median(test)), "std_test_perplexity": f(test.std(ddof=1)),
                       "min_test_perplexity": f(test.min()), "max_test_perplexity": f(test.max()), "mean_train_perplexity": f(train.mean())})
    by_corpus = {}
    for row in output: by_corpus.setdefault(row["corpus"], []).append(row)
    for rows in by_corpus.values():
        rows.sort(key=lambda x: x["K"]); previous = None
        for row in rows:
            current = float(row["mean_test_perplexity"])
            row["previous_K"] = "" if previous is None else previous["K"]
            row["absolute_improvement"] = "" if previous is None else f(float(previous["mean_test_perplexity"]) - current)
            row["relative_improvement_percent"] = "" if previous is None else f((float(previous["mean_test_perplexity"]) - current) / float(previous["mean_test_perplexity"]) * 100)
            previous = row
    return output


def combined(summary):
    original = {(r["corpus"], int(r["K"])): r for r in read_csv(ROOT / "reports/lda_sensitivity/configuration_summary.csv")}
    human = {(r["corpus"], int(r["K"])): r for r in read_csv(ROOT / "outputs/01a05061-9ee3-7610-a401-abfec81de1b2/model_selection/model_comprehensive_evaluation.csv")}
    rows = []
    for item in summary:
        key = item["corpus"], item["K"]; auto = original[key]; review = human.get(key)
        rows.append({"corpus": key[0], "K": key[1], "NPMI": auto["mean_npmi"], "Topic_Diversity": auto["mean_topic_diversity"],
                     "Topic_Stability": auto["mean_topic_stability"], "Human_Score": "" if review is None else review["mean_human_score"],
                     "Human_Score_Status": "NOT_REVIEWED_FOR_THIS_K" if review is None else "AVAILABLE",
                     "Test_Perplexity": item["mean_test_perplexity"], "Test_Perplexity_SD": item["std_test_perplexity"]})
    return rows


def c_diagnostics(summary, merged):
    c_rows = [r for r in summary if r["corpus"] == "C"]
    c_simple = [{"K": r["K"], "mean_test_perplexity": r["mean_test_perplexity"], "std": r["std_test_perplexity"],
                 "relative_improvement": r["relative_improvement_percent"]} for r in c_rows]
    metrics = {(r["corpus"], r["K"]): r for r in merged}
    multi = [{"K": r["K"], "NPMI": metrics[("C", r["K"])]["NPMI"], "Topic_Diversity": metrics[("C", r["K"])]["Topic_Diversity"],
              "Topic_Stability": metrics[("C", r["K"])]["Topic_Stability"], "Test_Perplexity": r["mean_test_perplexity"],
              "human_review_summary": ("mean_human_score=" + metrics[("C", r["K"])]["Human_Score"]) if metrics[("C", r["K"])]["Human_Score"] else "NOT_REVIEWED_FOR_THIS_K"} for r in c_rows]
    return c_simple, multi


def preferences(summary, merged):
    result = {}
    for corpus in "ABCD":
        rows = [r for r in summary if r["corpus"] == corpus]
        ppl_best = min(rows, key=lambda r: float(r["mean_test_perplexity"]))
        metric_rows = [r for r in merged if r["corpus"] == corpus]
        result[corpus] = {"perplexity_k": ppl_best["K"], "perplexity": float(ppl_best["mean_test_perplexity"]),
                          "npmi_k": max(metric_rows, key=lambda r: float(r["NPMI"]))["K"],
                          "stability_k": max(metric_rows, key=lambda r: float(r["Topic_Stability"]))["K"]}
    return result


def classify_c(c_rows, prefs):
    values = {r["K"]: float(r["mean_test_perplexity"]) for r in c_rows}; best_k = prefs["C"]["perplexity_k"]
    rel = [float(r["relative_improvement_percent"]) for r in c_rows if r["relative_improvement_percent"] != ""]
    loss_20 = (values[20] - min(values.values())) / values[20] * 100
    if best_k == 20: status = "SUPPORTS_K20"; why = "K=20取得C语料最低平均测试集困惑度。"
    elif loss_20 <= 1: status = "NEUTRAL_TO_K20"; why = f"K=20距最低值仅{loss_20:.2f}%，差异很小。"
    else: status = "TENSIONS_WITH_K20"; why = f"K=20比最低值高{loss_20:.2f}%，构成反向证据。"
    positive = [x for x in rel if x > 0]
    if len(positive) == len(rel) and len(rel) >= 2:
        ratios = [rel[i] / rel[i-1] if rel[i-1] else math.inf for i in range(1, len(rel))]
        elbow = "存在边际改善递减迹象" if any(x < .5 for x in ratios) else "持续改善，但没有足够清楚的肘部"
    elif not positive: elbow = "没有改善型肘部；增加K反而提高困惑度"
    else: elbow = "曲线非单调，未识别出稳健而清楚的肘部"
    return status, why, elbow


def style():
    if FONT.exists(): fm.fontManager.addfont(str(FONT)); plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})


def plots(summary, merged):
    FIGURES.mkdir(parents=True, exist_ok=True); style(); colors = dict(A="#6B7280", B="#2563EB", C="#C2410C", D="#059669")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for corpus in "ABCD":
        rows = [r for r in summary if r["corpus"] == corpus]; x = [r["K"] for r in rows]; y = [float(r["mean_test_perplexity"]) for r in rows]; e = [float(r["std_test_perplexity"]) for r in rows]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=corpus, color=colors[corpus])
    ax.set_yscale("log")
    ax.set(title="《全唐诗》LDA：不同主题数 K 的测试集困惑度", xlabel="主题数 K", ylabel="Held-out Perplexity（对数刻度；越低越好）"); ax.legend(title="语料"); ax.grid(axis="y", which="both", alpha=.2)
    fig.text(.5, .01, "困惑度越低，表示模型对未见文本的词语分布预测越好；不同词表的绝对值不宜直接判优。", ha="center", fontsize=9); fig.tight_layout(rect=[0, .04, 1, 1]); fig.savefig(FIGURES / "perplexity_by_k_all_corpora.png"); plt.close(fig)
    c = [r for r in summary if r["corpus"] == "C"]; x = [r["K"] for r in c]; y = [float(r["mean_test_perplexity"]) for r in c]; e = [float(r["std_test_perplexity"]) for r in c]
    fig, ax = plt.subplots(figsize=(8, 5.4)); ax.errorbar(x, y, yerr=e, marker="o", capsize=4, color=colors["C"]); ax.scatter([20], [y[0]], s=110, facecolors="none", edgecolors="#111827", zorder=4); ax.annotate("Primary model: C_K20", (20, y[0]), xytext=(27, y[0] + max(max(y)-min(y), 1)*.08), arrowprops={"arrowstyle": "->", "color": "#374151"}); ax.set(xlabel="主题数 K", ylabel="Held-out Perplexity（越低越好）", title="C 语料：不同 K 的测试集困惑度"); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "C_corpus_perplexity_by_k.png"); plt.close(fig)
    cm = [r for r in merged if r["corpus"] == "C"]
    for metric, filename, title, ylabel in [("NPMI", "C_K_perplexity_vs_npmi.png", "C 语料：NPMI 与留出困惑度", "NPMI（越高越好）"), ("Topic_Stability", "C_K_perplexity_vs_stability.png", "C 语料：稳定性与留出困惑度", "Topic Stability（越高越好）")]:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True); axes[0].plot(x, [float(r[metric]) for r in cm], marker="o", color="#2563EB"); axes[0].set_ylabel(ylabel); axes[0].grid(axis="y", alpha=.2); axes[1].errorbar(x, y, yerr=e, marker="o", capsize=4, color=colors["C"]); axes[1].set(xlabel="主题数 K", ylabel="困惑度（越低越好）"); axes[1].grid(axis="y", alpha=.2); fig.suptitle(title); fig.tight_layout(); fig.savefig(FIGURES / filename); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.6)); ax.axis("off"); ax.text(.5, .9, "怎样理解 LDA 的困惑度？", ha="center", fontsize=24, weight="bold", color="#1F2937");
    for i, k in enumerate([20, 30, 40, 50, 60]): ax.text(.1+i*.2, .62, f"K={k}", ha="center", fontsize=17, bbox={"boxstyle":"round,pad=.45", "facecolor":"#E0F2FE", "edgecolor":"#0284C7"})
    ax.annotate("↓  困惑度越低 = 模型面对新诗歌的词语分布越“不困惑”", (.5, .4), ha="center", fontsize=17, color="#075985"); ax.text(.5, .2, "但：最低困惑度  ≠  最有文学解释力", ha="center", fontsize=20, weight="bold", color="#B91C1C"); ax.text(.5, .09, "它衡量预测表现，不是模型有没有“读懂”诗歌。", ha="center", fontsize=14, color="#4B5563"); fig.tight_layout(); fig.savefig(FIGURES / "perplexity_explainer_for_students.png"); plt.close(fig)


def report(summary, merged, prefs, status, why, elbow):
    c = [r for r in summary if r["corpus"] == "C"]; split_audits = {label: json.loads((OUT / f"audit/split_{label}.json").read_text(encoding="utf-8")) for label in "ABCD"}
    c_lines = "\n".join(f"| {r['K']} | {float(r['mean_test_perplexity']):,.3f} | {float(r['std_test_perplexity']):,.3f} | {r['relative_improvement_percent'] or '—'} |" for r in c)
    corpus_lines = "\n".join(f"| {label} | {prefs[label]['perplexity_k']} | {prefs[label]['perplexity']:,.3f} | {prefs[label]['npmi_k']} | {prefs[label]['stability_k']} |" for label in "ABCD")
    compare = []
    base = float(c[0]["mean_test_perplexity"])
    for r in c[1:]:
        difference = float(r["mean_test_perplexity"]) - base
        compare.append(f"- C_K20 比 C_K{r['K']} 低 {difference:,.3f}，即后者比 K20 高 {difference/base*100:.2f}%。")
    text = f"""# QTS LDA K-Value Perplexity Diagnostic Report

## 1. 为什么补做 Perplexity？

本诊断从“对未见诗歌的词语分布预测得怎样”这一角度，补充既有 NPMI、Topic Diversity、跨随机种子稳定性与人工可解释性评价。它不重新加权、不重排模型，也不单独改选正式模型。

## 2. 什么是 Perplexity？

困惑度可以理解为：模型面对没有参与训练的新诗歌时，对这些诗歌的词语分布感到有多“困惑”。**数值越低越好**。较低值表示学到的 Topic 较能解释未见文本中的词语分布。不过它不是“模型有没有看懂诗歌”，也不等于 Semantic Coherence、Human Interpretability 或 Literary Meaning。

## 3. 为什么必须使用 Test Set？

只让模型在训练材料上“自己考试”，容易偏向更复杂的模型。本诊断把每套语料固定分为 80% 训练与 20% 测试，只在训练子集拟合，再用完全未进入 `fit` 的测试诗歌评分。主结论使用 test perplexity；train perplexity 只供审计。

## 4. Experimental Design

- 文档单位：一首诗。
- A/B/C/D 分别沿用既有 BoW、词表、有效文档索引，不重新分词或清洗。
- K = 20、30、40、50、60；seed = 42、2026、20260830，共 60 次诊断模型。
- 每套语料单独使用 `random_state=20260904` 固定 80/20 拆分，同一语料所有 K 和 seed 共用完全相同的训练/测试行。
- 训练严格复用 scikit-learn Batch LDA 原参数，含 `max_iter=30`、训练时 `n_jobs=-1`。只读评分改用 `n_jobs=1`；与并行评分对照的 perplexity 和 score 差值均为 0。
- 报告的 SD 是三个预设随机种子的样本标准差（ddof=1），描述算法随机性，不是总体置信区间。
- 主实验没有执行可选的共同文档重训；这不影响各语料内部的 K 比较。

## 5. Results

| Corpus | 最低 mean test perplexity 的 K | 最低值 | NPMI 偏好 K | Stability 偏好 K |
|---|---:|---:|---:|---:|
{corpus_lines}

图 1 见 `figures/perplexity_by_k_all_corpora.png`。由于 A/B/C/D 的词表大小不同，绝对困惑度还受到词汇空间影响，四条线主要并列展示，**不能据其高低直接选择词表**。

## 6. C Corpus

| K | mean test perplexity | SD | 相邻 K 改善率（%） |
|---:|---:|---:|---:|
{c_lines}

{chr(10).join(compare)}

对肘部的描述：**{elbow}**。这里只把边际变化当作参考，不机械宣称 elbow 等于 optimal K。

## 7. Perplexity 与 NPMI

Perplexity 问“模型预测新文本词语分布的能力如何”；NPMI 问“同一 Topic 的关键词是否经常合理地共同出现”。C 中两者偏好的 K {'相同' if prefs['C']['perplexity_k']==prefs['C']['npmi_k'] else '不同'}（分别为 {prefs['C']['perplexity_k']} 与 {prefs['C']['npmi_k']}）。某个 K 可以预测得更好却更难解释，这并不矛盾。

## 8. Perplexity 与 Stability

Stability 衡量不同随机种子得到的 Topic 能否相对稳定地对应。C 中 perplexity 与 stability 偏好的 K {'相同' if prefs['C']['perplexity_k']==prefs['C']['stability_k'] else '不同'}（分别为 {prefs['C']['perplexity_k']} 与 {prefs['C']['stability_k']}）。预测表现与重复训练的一致性仍是不同问题。

## 9. 是否改变 K=20 的选择？

诊断标签：**{status}**。{why} 这一结论只说明预测诊断与 K20 的关系；不覆盖既有综合评价。综合 NPMI、Topic Diversity、Stability、人工评价与本次 Perplexity，{'继续保留 C_K20 为 primary model，不需要重新审议。' if status != 'TENSIONS_WITH_K20' else '建议标记 MODEL_SELECTION_REVIEW_RECOMMENDED，交由人工复核；本报告不自动改选模型。'}

## 10. Limitations

1. 测试诗没有参与模型 `fit`，但 A/B/C/D 词表原先按全语料 TF/DF 建立；因此这是**冻结词表条件下的 held-out document diagnostic**，不是把分词、TF/DF 和词表建立也一并留出的完全独立流水线。
2. A/B/C/D 训练子集中从未出现、但测试集中出现的词次分别为 {split_audits['A']['test_tokens_unseen_in_train']:,}、{split_audits['B']['test_tokens_unseen_in_train']:,}、{split_audits['C']['test_tokens_unseen_in_train']:,}、{split_audits['D']['test_tokens_unseen_in_train']:,}；A 的宽词表尤其受此影响，但同一 A 拆分中的各 K 仍面对同样词项。
3. scikit-learn 的 `score` 是变分下界近似值，perplexity 是其按词数规范化后的指数形式；两者不是外部金标准。
4. 一次固定 80/20 拆分不能描述拆分随机性；SD 只来自三个模型种子。
5. 本诊断不是 document-completion perplexity，也没有为测试文档遮蔽部分词语。
6. 更大的 K 可能降低预测困惑度，同时造成 Topic 分裂、重复或解释困难；不能只凭最低困惑度选 K。

## 图注（人文学者友好版）

困惑度衡量模型在未参与训练的新文本上预测词语分布的能力，越低越好；它不是模型是否“看懂”诗歌。误差条为三个预设随机种子的 ±1 SD。最低困惑度不等于最有文学解释力。
"""
    (OUT / "QTS_LDA_K_Perplexity_Diagnostic_Report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.parse_args(); runs = read_csv(DATA / "perplexity_all_runs.csv")
    if len(runs) != 60: raise ValueError(f"需要60次完整运行，当前{len(runs)}")
    summary = summarize(runs); summary_fields = ["corpus", "K", "seed_count", "mean_test_perplexity", "median_test_perplexity", "std_test_perplexity", "min_test_perplexity", "max_test_perplexity", "mean_train_perplexity", "previous_K", "absolute_improvement", "relative_improvement_percent"]
    write_csv(DATA / "perplexity_summary_by_k.csv", summary_fields, summary)
    merged = combined(summary); merged_fields = ["corpus", "K", "NPMI", "Topic_Diversity", "Topic_Stability", "Human_Score", "Human_Score_Status", "Test_Perplexity", "Test_Perplexity_SD"]
    write_csv(DATA / "model_selection_with_perplexity.csv", merged_fields, merged)
    c_simple, multi = c_diagnostics(summary, merged)
    write_csv(DATA / "C_perplexity_k_diagnostic.csv", ["K", "mean_test_perplexity", "std", "relative_improvement"], c_simple)
    write_csv(DATA / "C_K_multimetric_diagnostic.csv", ["K", "NPMI", "Topic_Diversity", "Topic_Stability", "Test_Perplexity", "human_review_summary"], multi)
    prefs = preferences(summary, merged); status, why, elbow = classify_c([r for r in summary if r["corpus"] == "C"], prefs)
    plots(summary, merged); report(summary, merged, prefs, status, why, elbow)
    readme = """# Perplexity diagnostic\n\n本目录为独立的留出困惑度补充实验，不属于 frozen models。主结果为测试集困惑度（越低越好），只作 K 值辅助诊断。运行：`python src/run_perplexity_diagnostic.py`；汇总：`python src/build_perplexity_outputs.py`。方法、限制和结论见 `QTS_LDA_K_Perplexity_Diagnostic_Report.md`。\n"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    deliverables = [p for p in OUT.rglob("*") if p.is_file() and "checkpoints" not in p.parts and p.name not in {"output_manifest.json", "runner.lock"}]
    manifest = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(deliverables)}
    (OUT / "audit/output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"preferences": prefs, "status": status, "why": why, "elbow": elbow, "outputs": len(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
