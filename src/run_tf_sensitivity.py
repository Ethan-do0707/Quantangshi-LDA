"""只改变 LOW_TF 阈值，生成《全唐诗》四组可复核对照语料。"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import clean_corpus as cleaner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "tf_sensitivity_config.yaml"
REQUIRED_THRESHOLDS = (1, 2, 3, 5)
REFERENCE_PROFILE_FIELDS = {
    "retained_type_count",
    "removed_type_count",
    "retained_token_count",
    "retained_token_rate",
}


@dataclass
class ThresholdResult:
    """保存一组 TF 阈值语料的统计与判断。"""

    label: str
    threshold: int
    decisions: dict[str, tuple[bool, str, tuple[str, ...], bool]]
    vocabulary: set[str]
    document_lengths: list[int]
    statistics: dict[str, Any]
    corpus_path: Path
    tokens_path: Path
    statistics_path: Path


def percentile(values: Sequence[int], probability: float) -> float:
    """使用线性插值计算可复现的百分位数。"""

    if not values:
        return 0.0
    if not 0 <= probability <= 1:
        raise cleaner.CleaningError("百分位概率必须位于 0 与 1 之间")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def is_subsequence(smaller: Sequence[str], larger: Sequence[str]) -> bool:
    """检查词序是否在更宽松语料中原样保留。"""

    position = 0
    for token in larger:
        if position < len(smaller) and smaller[position] == token:
            position += 1
    return position == len(smaller)


def build_threshold_profile(
    frozen_profile: Mapping[str, Any], threshold: int
) -> dict[str, Any]:
    """复制固定档位并只修改 min_tf。"""

    profile = dict(frozen_profile)
    profile["min_tf"] = threshold
    return profile


def validate_threshold_profiles(
    profiles: Mapping[str, Mapping[str, Any]], frozen_profile: Mapping[str, Any]
) -> None:
    """确认四个实验档位之间只有 min_tf 不同。"""

    frozen_keys = set(frozen_profile) - {"min_tf"}
    for label, profile in profiles.items():
        if set(profile) != set(frozen_profile):
            raise cleaner.CleaningError(f"{label} 档位字段与固定规则不一致")
        for key in frozen_keys:
            if profile[key] != frozen_profile[key]:
                raise cleaner.CleaningError(
                    f"{label} 除 LOW_TF 外改变了固定规则：{key}"
                )


def build_decisions(
    token_tf: Counter[str],
    token_df: Counter[str],
    effective_poem_count: int,
    protected_words: set[str],
    stopwords: set[str],
    profile: Mapping[str, Any],
) -> dict[str, tuple[bool, str, tuple[str, ...], bool]]:
    """使用现有清洗器计算一个阈值下的全部词型判断。"""

    min_tf = int(profile["min_tf"])
    min_df = int(profile["min_df"])
    automatic_max_df = profile["automatic_max_df"]
    return {
        token: cleaner.token_decision(
            token,
            tf,
            token_df[token],
            cleaner.ratio(token_df[token], effective_poem_count),
            protected_words,
            stopwords,
            min_tf,
            min_df,
            automatic_max_df,
        )
        for token, tf in token_tf.items()
    }


def document_statistics(lengths: Sequence[int]) -> dict[str, float | int]:
    """计算诗歌长度统计。"""

    total = sum(lengths)
    count = len(lengths)
    return {
        "min": min(lengths, default=0),
        "p10": percentile(lengths, 0.10),
        "p25": percentile(lengths, 0.25),
        "p50": percentile(lengths, 0.50),
        "mean": total / count if count else 0.0,
        "median": percentile(lengths, 0.50),
        "p75": percentile(lengths, 0.75),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths, default=0),
    }


def build_group(
    *,
    label: str,
    threshold: int,
    profile: Mapping[str, Any],
    records: Sequence[cleaner.PoemRecord],
    token_tf: Counter[str],
    token_df: Counter[str],
    effective_poem_count: int,
    protected_words: set[str],
    stopwords: set[str],
    token_separator: str,
    output_root: Path,
) -> ThresholdResult:
    """生成一组完整语料及统计，所有诗歌均保留原顺序。"""

    decisions = build_decisions(
        token_tf,
        token_df,
        effective_poem_count,
        protected_words,
        stopwords,
        profile,
    )
    cleaned_by_poem = [
        [token for token in record.normalized_tokens if decisions[token][0]]
        for record in records
    ]
    document_lengths = [len(tokens) for tokens in cleaned_by_poem]
    vocabulary = {token for token, decision in decisions.items() if decision[0]}
    total_tokens = sum(document_lengths)
    valid_poems = sum(length > 0 for length in document_lengths)
    low_tf_tokens = {
        token for token, decision in decisions.items() if decision[1] == "LOW_TF"
    }
    protected_vocabulary = vocabulary & protected_words
    length_stats = document_statistics(document_lengths)
    tf_bins = {
        "tf_eq_1_types": sum(token_tf[token] == 1 for token in vocabulary),
        "tf_eq_2_types": sum(token_tf[token] == 2 for token in vocabulary),
        "tf_eq_3_types": sum(token_tf[token] == 3 for token in vocabulary),
        "tf_eq_4_types": sum(token_tf[token] == 4 for token in vocabulary),
        "tf_ge_5_types": sum(token_tf[token] >= 5 for token in vocabulary),
    }
    statistics = {
        "threshold": label,
        "min_tf": threshold,
        "fixed_min_df": int(profile["min_df"]),
        "fixed_automatic_max_df": profile["automatic_max_df"],
        "poem_count": len(records),
        "valid_poem_count": valid_poems,
        "empty_poem_count": len(records) - valid_poems,
        "total_tokens": total_tokens,
        "vocabulary_size": len(vocabulary),
        "document_length": length_stats,
        "vocabulary_frequency_bands": tf_bins,
        "protected_vocabulary_size": len(protected_vocabulary),
        "protected_token_occurrences": sum(
            token_tf[token] for token in protected_vocabulary
        ),
        "low_tf_removed_vocabulary": len(low_tf_tokens),
        "low_tf_removed_occurrences": sum(token_tf[token] for token in low_tf_tokens),
        "poems_at_or_below_5_tokens": sum(length <= 5 for length in document_lengths),
        "poems_at_or_below_10_tokens": sum(length <= 10 for length in document_lengths),
        "poems_above_p99": sum(
            length > float(length_stats["p99"]) for length in document_lengths
        ),
        "percentile_method": "linear interpolation at (n-1)*p",
    }

    group_dir = output_root / label
    group_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = group_dir / "lda_corpus.csv"
    tokens_path = group_dir / "lda_tokens.txt"
    statistics_path = group_dir / "corpus_statistics.json"
    cleaner.write_csv(
        corpus_path,
        [
            "poem_id",
            "author",
            "title",
            "original_text",
            "cleaned_tokens",
            "token_count",
            "empty_after_filter",
        ],
        (
            {
                "poem_id": record.poem_id,
                "author": record.author,
                "title": record.title,
                "original_text": record.original_text,
                "cleaned_tokens": token_separator.join(tokens),
                "token_count": len(tokens),
                "empty_after_filter": not tokens,
            }
            for record, tokens in zip(records, cleaned_by_poem)
        ),
    )
    with tokens_path.open("w", encoding="utf-8", newline="\n") as handle:
        for tokens in cleaned_by_poem:
            handle.write(token_separator.join(tokens) + "\n")
    statistics["outputs"] = {
        "lda_corpus": {
            "path": cleaner.display_path(corpus_path),
            "sha256": cleaner.file_sha256(corpus_path),
        },
        "lda_tokens": {
            "path": cleaner.display_path(tokens_path),
            "sha256": cleaner.file_sha256(tokens_path),
        },
    }
    cleaner.write_json(statistics_path, statistics)
    return ThresholdResult(
        label=label,
        threshold=threshold,
        decisions=decisions,
        vocabulary=vocabulary,
        document_lengths=document_lengths,
        statistics=statistics,
        corpus_path=corpus_path,
        tokens_path=tokens_path,
        statistics_path=statistics_path,
    )


def comparison_rows(results: Sequence[ThresholdResult]) -> list[dict[str, Any]]:
    """生成相对于 TF1 的阈值比较表。"""

    baseline = results[0].statistics
    rows: list[dict[str, Any]] = []
    for result in results:
        stats = result.statistics
        lengths = stats["document_length"]
        bands = stats["vocabulary_frequency_bands"]
        rows.append(
            {
                "threshold": result.label,
                "poems": stats["poem_count"],
                "valid_poems": stats["valid_poem_count"],
                "empty_poems": stats["empty_poem_count"],
                "total_tokens": stats["total_tokens"],
                "vocabulary_size": stats["vocabulary_size"],
                "mean_doc_length": f"{float(lengths['mean']):.8f}",
                "median_doc_length": f"{float(lengths['median']):.8f}",
                "min_doc_length": lengths["min"],
                "p10_doc_length": f"{float(lengths['p10']):.8f}",
                "p25_doc_length": f"{float(lengths['p25']):.8f}",
                "p50_doc_length": f"{float(lengths['p50']):.8f}",
                "p75_doc_length": f"{float(lengths['p75']):.8f}",
                "p90_doc_length": f"{float(lengths['p90']):.8f}",
                "p95_doc_length": f"{float(lengths['p95']):.8f}",
                "p99_doc_length": f"{float(lengths['p99']):.8f}",
                "max_doc_length": lengths["max"],
                "tf_eq_1_types": bands["tf_eq_1_types"],
                "tf_eq_2_types": bands["tf_eq_2_types"],
                "tf_eq_3_types": bands["tf_eq_3_types"],
                "tf_eq_4_types": bands["tf_eq_4_types"],
                "tf_ge_5_types": bands["tf_ge_5_types"],
                "protected_vocabulary_size": stats["protected_vocabulary_size"],
                "protected_token_occurrences": stats["protected_token_occurrences"],
                "removed_vocab": stats["low_tf_removed_vocabulary"],
                "removed_occurrences": stats["low_tf_removed_occurrences"],
                "vocabulary_retention_rate": f"{cleaner.ratio(int(stats['vocabulary_size']), int(baseline['vocabulary_size'])):.8f}",
                "token_retention_rate": f"{cleaner.ratio(int(stats['total_tokens']), int(baseline['total_tokens'])):.8f}",
                "poem_retention_rate": f"{cleaner.ratio(int(stats['valid_poem_count']), int(baseline['valid_poem_count'])):.8f}",
                "poems_le_5_tokens": stats["poems_at_or_below_5_tokens"],
                "poems_le_10_tokens": stats["poems_at_or_below_10_tokens"],
                "poems_above_p99": stats["poems_above_p99"],
            }
        )
    return rows


def removal_rows(
    result: ThresholdResult,
    token_tf: Counter[str],
    token_df: Counter[str],
    effective_poem_count: int,
    protected_words: set[str],
) -> list[dict[str, Any]]:
    """列出在指定档位由 LOW_TF 作为主原因删除的词型。"""

    rows = []
    for token, decision in result.decisions.items():
        keep, reason, all_reasons, _rescued = decision
        if keep or reason != "LOW_TF":
            continue
        rows.append(
            {
                "token": token,
                "TF": token_tf[token],
                "DF": token_df[token],
                "DF_ratio": f"{cleaner.ratio(token_df[token], effective_poem_count):.8f}",
                "protected": str(token in protected_words).lower(),
                "removal_reason": reason,
                "all_removal_reasons": "|".join(all_reasons),
            }
        )
    rows.sort(key=lambda row: (int(row["TF"]), int(row["DF"]), str(row["token"])))
    return rows


def difference_rows(
    previous: ThresholdResult,
    current: ThresholdResult,
    token_tf: Counter[str],
    token_df: Counter[str],
    effective_poem_count: int,
    protected_words: set[str],
) -> list[dict[str, Any]]:
    """列出相邻档位间实际从主语料消失的词型。"""

    removed = previous.vocabulary - current.vocabulary
    rows = [
        {
            "token": token,
            "TF": token_tf[token],
            "DF": token_df[token],
            "DF_ratio": f"{cleaner.ratio(token_df[token], effective_poem_count):.8f}",
            "protected": str(token in protected_words).lower(),
            "removal_reason": current.decisions[token][1],
            "all_removal_reasons": "|".join(current.decisions[token][2]),
            "previous_threshold": previous.label,
            "current_threshold": current.label,
        }
        for token in removed
    ]
    rows.sort(key=lambda row: (int(row["TF"]), int(row["DF"]), str(row["token"])))
    return rows


def case_rows(
    case_tokens: Sequence[str],
    results: Sequence[ThresholdResult],
    token_tf: Counter[str],
    token_df: Counter[str],
    protected_words: set[str],
) -> list[dict[str, Any]]:
    """追踪指定案例词在四组语料中的状态。"""

    rows: list[dict[str, Any]] = []
    for token in case_tokens:
        row: dict[str, Any] = {
            "token": token,
            "TF": token_tf[token],
            "DF": token_df[token],
            "protected": str(token in protected_words).lower(),
        }
        for result in results:
            decision = result.decisions.get(token)
            row[result.label] = "KEEP" if decision and decision[0] else "REMOVE"
            row[f"{result.label}_reason"] = decision[1] if decision else "NOT_IN_CORPUS"
        rows.append(row)
    return rows


def literary_candidate_rows(
    token_tf: Counter[str],
    token_df: Counter[str],
    effective_poem_count: int,
    protected_words: set[str],
    stopwords: set[str],
    results: Sequence[ThresholdResult],
) -> list[dict[str, Any]]:
    """只按 TF=1–4 生成低频人工审核候选，不做语义判断。"""

    rows: list[dict[str, Any]] = []
    for token, tf in token_tf.items():
        if not 1 <= tf <= 4 or not cleaner.is_han_token(token) or token in stopwords:
            continue
        row: dict[str, Any] = {
            "token": token,
            "TF": tf,
            "DF": token_df[token],
            "DF_ratio": f"{cleaner.ratio(token_df[token], effective_poem_count):.8f}",
            "protected": str(token in protected_words).lower(),
            "candidate_basis": "TF_1_TO_4_ONLY_NO_SEMANTIC_JUDGMENT",
        }
        for result in results:
            row[result.label] = "KEEP" if result.decisions[token][0] else "REMOVE"
            row[f"{result.label}_reason"] = result.decisions[token][1]
        rows.append(row)
    rows.sort(key=lambda row: (int(row["TF"]), int(row["DF"]), str(row["token"])))
    return rows


def extreme_length_rows(
    records: Sequence[cleaner.PoemRecord], results: Sequence[ThresholdResult]
) -> list[dict[str, Any]]:
    """汇总任一版本超 P99 或不超过 10 token 的诗。"""

    rows: list[dict[str, Any]] = []
    p99_values = {
        result.label: float(result.statistics["document_length"]["p99"])
        for result in results
    }
    for index, record in enumerate(records):
        flags: list[str] = []
        counts = {
            result.label: result.document_lengths[index] for result in results
        }
        for result in results:
            count = counts[result.label]
            if count > p99_values[result.label]:
                flags.append(f"{result.label}_ABOVE_P99")
            if count <= 5:
                flags.append(f"{result.label}_LE5")
            if count <= 10:
                flags.append(f"{result.label}_LE10")
        if flags:
            rows.append(
                {
                    "poem_id": record.poem_id,
                    "author": record.author,
                    "title": record.title,
                    "original_text": record.original_text,
                    **{
                        f"{result.label}_token_count": counts[result.label]
                        for result in results
                    },
                    "extreme_categories": "|".join(flags),
                }
            )
    return rows


def write_svg_bar_chart(
    path: Path,
    title: str,
    y_label: str,
    labels: Sequence[str],
    values: Sequence[float],
    value_formatter: Callable[[float], str],
) -> None:
    """在 matplotlib 不可用时生成确定性的简洁 SVG 柱状图。"""

    width, height = 1000, 650
    left, right, top, bottom = 115, 45, 80, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(values, default=1.0)
    y_max = maximum * 1.12 if maximum > 0 else 1.0
    bar_slot = plot_width / max(len(labels), 1)
    bar_width = bar_slot * 0.48
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">Bar chart comparing {html.escape(y_label)} across TF1, TF2, TF3, and TF5.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="600" fill="#111111">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        y = top + plot_height * (1 - fraction)
        value = y_max * fraction
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#222222">{html.escape(value_formatter(value))}</text>'
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#222222" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#222222" stroke-width="1.5"/>',
        ]
    )
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + bar_slot * index + (bar_slot - bar_width) / 2
        bar_height = plot_height * value / y_max if y_max else 0
        y = top + plot_height - bar_height
        elements.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#4c78a8"/>',
                f'<text x="{x+bar_width/2:.2f}" y="{max(y-10, top+14):.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111111">{html.escape(value_formatter(value))}</text>',
                f'<text x="{x+bar_width/2:.2f}" y="{top+plot_height+32}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111111">{html.escape(label)}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{width/2}" y="{height-25}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#111111">LOW_TF threshold</text>',
            f'<text x="26" y="{top+plot_height/2}" text-anchor="middle" transform="rotate(-90 26 {top+plot_height/2})" font-family="Arial, sans-serif" font-size="15" fill="#111111">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def generate_figures(
    figures_dir: Path,
    results: Sequence[ThresholdResult],
    figure_config: Mapping[str, Any],
) -> tuple[str, str | None, list[Path]]:
    """优先用 matplotlib 作图；不可用时写出确定性 SVG。"""

    if not bool(figure_config.get("enabled", True)):
        return "disabled", None, []
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [result.label for result in results]
    baseline_vocab = int(results[0].statistics["vocabulary_size"])
    datasets = [
        (
            "vocabulary_size",
            "TF Threshold vs Vocabulary Size",
            "Vocabulary size (types)",
            [float(result.statistics["vocabulary_size"]) for result in results],
            lambda value: f"{value:,.0f}",
        ),
        (
            "total_tokens",
            "TF Threshold vs Total Tokens",
            "Total tokens (occurrences)",
            [float(result.statistics["total_tokens"]) for result in results],
            lambda value: f"{value:,.0f}",
        ),
        (
            "mean_document_length",
            "TF Threshold vs Mean Document Length",
            "Mean tokens per poem",
            [
                float(result.statistics["document_length"]["mean"])
                for result in results
            ],
            lambda value: f"{value:.2f}",
        ),
        (
            "vocabulary_retention_rate",
            "Vocabulary Retention Rate",
            "Retention relative to TF1 (%)",
            [
                100 * cleaner.ratio(
                    int(result.statistics["vocabulary_size"]), baseline_vocab
                )
                for result in results
            ],
            lambda value: f"{value:.1f}%",
        ),
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        paths: list[Path] = []
        for filename, title, y_label, values, formatter in datasets:
            figure, axis = plt.subplots(figsize=(7.2, 4.8))
            bars = axis.bar(labels, values, color="#4c78a8", width=0.52)
            axis.set_title(title)
            axis.set_xlabel("LOW_TF threshold")
            axis.set_ylabel(y_label)
            axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.bar_label(bars, labels=[formatter(value) for value in values], padding=3)
            axis.set_ylim(0, max(values) * 1.14 if max(values) else 1)
            figure.tight_layout()
            path = figures_dir / f"{filename}.png"
            figure.savefig(path, dpi=int(figure_config.get("dpi", 200)))
            plt.close(figure)
            paths.append(path)
        return "matplotlib", matplotlib.__version__, paths
    except ImportError:
        paths = []
        for filename, title, y_label, values, formatter in datasets:
            path = figures_dir / f"{filename}.svg"
            write_svg_bar_chart(path, title, y_label, labels, values, formatter)
            paths.append(path)
        return "deterministic_svg", None, paths


def write_report(
    path: Path,
    results: Sequence[ThresholdResult],
    comparisons: Sequence[Mapping[str, Any]],
    incremental: Sequence[tuple[str, list[Mapping[str, Any]]]],
    cases: Sequence[Mapping[str, Any]],
    protected_all_retained: bool,
    protected_resource_types: int,
    protected_present_types: int,
    protected_occurrences: int,
    protected_absent_surfaces: Sequence[str],
    figure_backend: str,
    figure_paths: Sequence[Path],
    fixed_min_df: int,
) -> None:
    """生成完整的 TF 阈值敏感性实验报告。"""

    comparison_table = [
        (
            row["threshold"],
            f"{int(row['vocabulary_size']):,}",
            f"{int(row['total_tokens']):,}",
            row["valid_poems"],
            row["empty_poems"],
            f"{float(row['mean_doc_length']):.3f}",
            f"{float(row['median_doc_length']):.3f}",
            f"{float(row['p95_doc_length']):.3f}",
            f"{float(row['p99_doc_length']):.3f}",
        )
        for row in comparisons
    ]
    retention_table = [
        (
            row["threshold"],
            f"{float(row['vocabulary_retention_rate']):.6%}",
            f"{float(row['token_retention_rate']):.6%}",
            f"{float(row['poem_retention_rate']):.6%}",
            f"{int(row['removed_vocab']):,}",
            f"{int(row['removed_occurrences']):,}",
        )
        for row in comparisons
    ]
    incremental_table = [
        (
            label,
            f"{len(rows):,}",
            f"{sum(int(row['TF']) for row in rows):,}",
        )
        for label, rows in incremental
    ]
    document_table = [
        (
            row["threshold"],
            row["min_doc_length"],
            f"{float(row['p10_doc_length']):.3f}",
            f"{float(row['p25_doc_length']):.3f}",
            f"{float(row['p50_doc_length']):.3f}",
            f"{float(row['mean_doc_length']):.3f}",
            f"{float(row['p75_doc_length']):.3f}",
            f"{float(row['p90_doc_length']):.3f}",
            f"{float(row['p95_doc_length']):.3f}",
            f"{float(row['p99_doc_length']):.3f}",
            row["max_doc_length"],
            row["poems_le_5_tokens"],
            row["poems_le_10_tokens"],
        )
        for row in comparisons
    ]
    case_table = [
        (
            row["token"],
            row["TF"],
            row["DF"],
            row["protected"],
            row["TF1"],
            row["TF2"],
            row["TF3"],
            row["TF5"],
        )
        for row in cases
    ]
    figure_links = [
        f"- `{cleaner.display_path(figure_path)}`" for figure_path in figure_paths
    ]
    lines = [
        "# 《全唐诗》LOW_TF 阈值敏感性实验报告",
        "",
        "## 1. Experimental Design",
        "",
        "本实验由同一份人工金标准覆盖版分词语料出发，只把 `min_tf` 依次设为 1、2、3、5。词形规范、核心停用词、保护词、token 顺序、"
        f"`min_df={fixed_min_df}`、自动最大 DF、非汉字规则和语义映射状态全部固定。原有 `output/` TF5 主结果未被覆盖。",
        "",
        f"由于固定 `DF≥{fixed_min_df}`，任何 `TF=1` 或 `TF=2` 的非保护词都必然同时不满足 DF；因此 TF1、TF2、TF3 的最终语料相同。"
        "这不是程序错误，而是冻结 DF 阈值后的数学结果。报告分别给出规则主原因统计与相邻语料的实际新增损失。",
        "",
        "## 2. Corpus Statistics",
        "",
        cleaner.markdown_table(
            ["Corpus", "Vocabulary", "Tokens", "Valid poems", "Empty", "Mean", "Median", "P95", "P99"],
            comparison_table,
        ),
        "",
        "完整字段见 `reports/tf_sensitivity/tf_threshold_comparison.csv`。",
        "",
        "## 3. Vocabulary Loss",
        "",
        cleaner.markdown_table(
            ["Corpus", "Vocab retention", "Token retention", "Poem retention", "LOW_TF attributed types", "LOW_TF attributed occurrences"],
            retention_table,
        ),
        "",
        "相邻阈值的实际新增损失：",
        "",
        cleaner.markdown_table(
            ["Transition", "Newly lost vocabulary", "Newly lost occurrences"],
            incremental_table,
        ),
        "",
        "`removed_by_TF*.csv` 是该档位以 LOW_TF 为主删除原因的累计归因表；`TF*_vs_TF*_removed.csv` 才表示相邻主语料之间真正新增消失的词。",
        "",
        "## 4. Token Loss",
        "",
        "Vocabulary loss 统计消失的不同词型；token occurrence loss 统计这些词在全库中的实际出现次数。两者不能互相替代。"
        "低频词可占 vocabulary 的很大比例，但只占 token occurrence 的较小比例。",
        "",
        "## 5. Document Length",
        "",
        cleaner.markdown_table(
            ["Corpus", "Min", "P10", "P25", "P50", "Mean", "P75", "P90", "P95", "P99", "Max", "≤5", "≤10"],
            document_table,
        ),
        "",
        "百分位数使用 `(n-1)×p` 位置的线性插值；所有空诗和极端长度诗均保留。详见 `extreme_length_poems.csv`。",
        "",
        "## 6. Case Study Tokens",
        "",
        cleaner.markdown_table(
            ["token", "TF", "DF", "protected", "TF1", "TF2", "TF3", "TF5"],
            case_table,
        ),
        "",
        "`黄金台、玉衡、景祚、历象`是 TF=2、DF=2 的非保护词，因冻结 LOW_DF 而在 TF1 中已不存在；"
        "`乾道、朝彩、金阵`为 TF=3、DF=3，到 TF5 才实际消失。",
        "",
        "## 7. Protected Words",
        "",
        f"- 保护词资源词面数：{protected_resource_types:,}",
        f"- 语料中现成 token 形式出现的保护词型：{protected_present_types:,}",
        f"- 保护词出现次数：{protected_occurrences:,}",
        f"- 四组全部正确保留：{str(protected_all_retained).lower()}",
        f"- 未作为完整 token 出现的保护词面：{'、'.join(protected_absent_surfaces) if protected_absent_surfaces else '无'}",
        "- 上述未出现词面不是被 TF 规则删除，而是在冻结的输入分词中没有完整 token；本实验没有重新分词。",
        "- 未重新分词，也未自动合并疑似切分的保护词。",
        "",
        "## Figures",
        "",
        f"- 绘图后端：`{figure_backend}`",
        "- 当前 Python 环境未安装 matplotlib，因此按配置使用确定性 SVG 回退；没有为作图安装或更改项目依赖。" if figure_backend == "deterministic_svg" else "- 图表由 matplotlib 生成。",
        *figure_links,
        "",
        "## 8. Recommendation",
        "",
        "当前统计只能说明各阈值的词型、词次和文档长度变化，不能仅凭 vocabulary size 宣称某个阈值最佳。"
        "最终选择仍须等待下一阶段在完全相同参数和随机种子下比较 Topic Coherence、Topic Diversity、Topic Stability、主题关键词质量、代表诗歌和人工可解释性。",
        "",
        "本阶段没有运行 LDA，也没有搜索最佳 K。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_tf5_against_existing(
    result: ThresholdResult, existing_corpus: Path, existing_tokens: Path
) -> dict[str, Any]:
    """确认独立 TF5 与既有结果内容一致，且既有文件未被改写。"""

    mismatch_count = 0
    with result.corpus_path.open("r", encoding="utf-8-sig", newline="") as new_handle, existing_corpus.open(
        "r", encoding="utf-8-sig", newline=""
    ) as old_handle:
        new_reader = csv.DictReader(new_handle)
        old_reader = csv.DictReader(old_handle)
        new_count = 0
        old_count = 0
        for new_row, old_row in zip(new_reader, old_reader):
            new_count += 1
            old_count += 1
            if any(
                new_row[field] != old_row[field]
                for field in ("poem_id", "author", "title", "original_text", "cleaned_tokens", "token_count")
            ):
                mismatch_count += 1
        new_count += sum(1 for _ in new_reader)
        old_count += sum(1 for _ in old_reader)
    return {
        "row_count_new": new_count,
        "row_count_existing": old_count,
        "row_content_mismatch_count": mismatch_count,
        "lda_tokens_sha256_new": cleaner.file_sha256(result.tokens_path),
        "lda_tokens_sha256_existing": cleaner.file_sha256(existing_tokens),
        "matches": (
            new_count == old_count
            and mismatch_count == 0
            and cleaner.file_sha256(result.tokens_path)
            == cleaner.file_sha256(existing_tokens)
        ),
    }


def run_experiment(
    config_path: Path = DEFAULT_CONFIG, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """执行四组 TF 阈值敏感性实验。"""

    started_clock = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    experiment_config = cleaner.load_config(config_path)
    thresholds = tuple(int(value) for value in experiment_config.get("thresholds", ()))
    if thresholds != REQUIRED_THRESHOLDS:
        raise cleaner.CleaningError(
            f"本实验阈值必须严格为 {REQUIRED_THRESHOLDS}，当前为 {thresholds}"
        )
    base_config_path = cleaner.resolve_path(
        str(experiment_config["base_cleaning_config"]), project_root
    )
    base_config = cleaner.load_config(base_config_path)
    paths = cleaner.prepare_paths(base_config, project_root)
    output_root = cleaner.resolve_path(str(experiment_config["output_root"]), project_root)
    reports_root = cleaner.resolve_path(str(experiment_config["reports_root"]), project_root)
    log_path = cleaner.resolve_path(str(experiment_config["log_path"]), project_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = cleaner.setup_logger(log_path)
    logger.info("开始 TF 阈值敏感性实验：%s", config_path)

    cleaning_config = cleaner.require_mapping(base_config.get("cleaning"), "cleaning")
    profile_name = str(cleaning_config["profile"])
    token_separator = str(cleaning_config.get("token_separator", " "))
    automatic_only = bool(cleaning_config.get("apply_automatic_normalization_only", True))
    rules_document = json.loads(paths["filtering_rules"].read_text(encoding="utf-8-sig"))
    frozen_profile = cleaner.require_mapping(rules_document[profile_name], profile_name)
    stopword_tier = str(frozen_profile["stopword_tier"])
    stopword_path = paths["stopwords"][stopword_tier]
    profiles = {
        f"TF{threshold}": build_threshold_profile(frozen_profile, threshold)
        for threshold in thresholds
    }
    validate_threshold_profiles(profiles, frozen_profile)

    existing_corpus = paths["output_dir"] / "lda_corpus.csv"
    existing_tokens = paths["output_dir"] / "lda_tokens.txt"
    if not existing_corpus.is_file() or not existing_tokens.is_file():
        raise cleaner.CleaningError("既有 TF5 主输出不存在，无法执行不覆盖验证")
    existing_hashes_before = {
        "lda_corpus": cleaner.file_sha256(existing_corpus),
        "lda_tokens": cleaner.file_sha256(existing_tokens),
    }
    input_paths = [
        Path(__file__).resolve(),
        (SRC_DIR / "clean_corpus.py").resolve(),
        config_path,
        base_config_path,
        paths["corpus"],
        paths["tokenized_corpus"],
        paths["normalization"],
        stopword_path,
        paths["protected_words"],
        paths["filtering_rules"],
        paths["semantic_mapping"],
    ]
    input_hashes_before = {str(path.resolve()): cleaner.file_sha256(path) for path in input_paths}

    corpus_rows = cleaner.read_jsonl(paths["corpus"])
    token_rows = cleaner.read_jsonl(paths["tokenized_corpus"])
    records, data_issues = cleaner.align_records(corpus_rows, token_rows)
    normalization_rules = cleaner.load_normalization_rules(paths["normalization"])
    normalization_events = []
    for record in records:
        normalized, events = cleaner.apply_normalization(
            record.poem_id, record.original_tokens, normalization_rules, automatic_only
        )
        record.normalized_tokens = normalized
        normalization_events.extend(events)
    stopwords = set(cleaner.read_terms(stopword_path))
    protected_rows = cleaner.read_tsv(paths["protected_words"])
    protected_words = {row["surface"].strip() for row in protected_rows}

    token_tf: Counter[str] = Counter()
    token_df: Counter[str] = Counter()
    for record in records:
        token_tf.update(record.normalized_tokens)
        token_df.update(set(record.normalized_tokens))
    effective_poem_count = sum(
        any(
            token in protected_words
            or (cleaner.is_han_token(token) and token not in stopwords)
            for token in record.normalized_tokens
        )
        for record in records
    )
    logger.info(
        "固定输入完成：%d 首、%d 词型、DF 分母 %d、规范替换 %d 次",
        len(records),
        len(token_tf),
        effective_poem_count,
        len(normalization_events),
    )

    results: list[ThresholdResult] = []
    for threshold in thresholds:
        label = f"TF{threshold}"
        result = build_group(
            label=label,
            threshold=threshold,
            profile=profiles[label],
            records=records,
            token_tf=token_tf,
            token_df=token_df,
            effective_poem_count=effective_poem_count,
            protected_words=protected_words,
            stopwords=stopwords,
            token_separator=token_separator,
            output_root=output_root,
        )
        results.append(result)
        logger.info(
            "%s 完成：vocabulary=%d，tokens=%d，empty=%d",
            label,
            result.statistics["vocabulary_size"],
            result.statistics["total_tokens"],
            result.statistics["empty_poem_count"],
        )

    comparisons = comparison_rows(results)
    comparison_headers = list(comparisons[0].keys())
    cleaner.write_csv(
        reports_root / "tf_threshold_comparison.csv", comparison_headers, comparisons
    )

    removal_headers = [
        "token",
        "TF",
        "DF",
        "DF_ratio",
        "protected",
        "removal_reason",
        "all_removal_reasons",
    ]
    removal_outputs: dict[str, list[dict[str, Any]]] = {}
    for result in results[1:]:
        rows = removal_rows(
            result,
            token_tf,
            token_df,
            effective_poem_count,
            protected_words,
        )
        removal_outputs[result.label] = rows
        cleaner.write_csv(
            reports_root / f"removed_by_{result.label}.csv", removal_headers, rows
        )

    difference_headers = removal_headers + ["previous_threshold", "current_threshold"]
    incremental: list[tuple[str, list[dict[str, Any]]]] = []
    for previous, current in zip(results, results[1:]):
        rows = difference_rows(
            previous,
            current,
            token_tf,
            token_df,
            effective_poem_count,
            protected_words,
        )
        transition = f"{previous.label}_vs_{current.label}"
        incremental.append((transition, rows))
        cleaner.write_csv(
            reports_root / f"{transition}_removed.csv", difference_headers, rows
        )

    cases = case_rows(
        [str(token) for token in experiment_config["case_tokens"]],
        results,
        token_tf,
        token_df,
        protected_words,
    )
    case_headers = ["token", "TF", "DF", "protected"]
    for result in results:
        case_headers.extend([result.label, f"{result.label}_reason"])
    cleaner.write_csv(
        reports_root / "case_token_tracking.csv", case_headers, cases
    )

    candidates = literary_candidate_rows(
        token_tf,
        token_df,
        effective_poem_count,
        protected_words,
        stopwords,
        results,
    )
    candidate_headers = [
        "token",
        "TF",
        "DF",
        "DF_ratio",
        "protected",
        "candidate_basis",
    ]
    for result in results:
        candidate_headers.extend([result.label, f"{result.label}_reason"])
    cleaner.write_csv(
        reports_root / "literary_low_frequency_candidates.csv",
        candidate_headers,
        candidates,
    )

    extremes = extreme_length_rows(records, results)
    extreme_headers = ["poem_id", "author", "title", "original_text"] + [
        f"{result.label}_token_count" for result in results
    ] + ["extreme_categories"]
    cleaner.write_csv(
        reports_root / "extreme_length_poems.csv", extreme_headers, extremes
    )

    figure_config = cleaner.require_mapping(experiment_config.get("figures"), "figures")
    figure_backend, matplotlib_version, figure_paths = generate_figures(
        reports_root / "figures", results, figure_config
    )

    protected_present = protected_words & token_tf.keys()
    protected_all_retained = all(
        result.decisions[token][0]
        for result in results
        for token in protected_present
    )
    if not protected_all_retained:
        raise cleaner.CleaningError("至少一个现成保护 token 在敏感性语料中被删除")

    sequence_monotonic = True
    for record in records:
        sequences = [
            [
                token
                for token in record.normalized_tokens
                if result.decisions[token][0]
            ]
            for result in results
        ]
        if any(
            not is_subsequence(current, previous)
            for previous, current in zip(sequences, sequences[1:])
        ):
            sequence_monotonic = False
            break
    if not sequence_monotonic:
        raise cleaner.CleaningError("阈值提高后出现 token 顺序或子序列异常")

    tf5_validation = validate_tf5_against_existing(
        results[-1], existing_corpus, existing_tokens
    )
    if not tf5_validation["matches"]:
        raise cleaner.CleaningError("独立 TF5 与既有 TF5 主结果不一致")

    existing_hashes_after = {
        "lda_corpus": cleaner.file_sha256(existing_corpus),
        "lda_tokens": cleaner.file_sha256(existing_tokens),
    }
    input_hashes_after = {str(path.resolve()): cleaner.file_sha256(path) for path in input_paths}
    if existing_hashes_before != existing_hashes_after:
        raise cleaner.CleaningError("既有 TF5 输出在实验过程中发生变化")
    if input_hashes_before != input_hashes_after:
        raise cleaner.CleaningError("只读输入或规则文件在实验过程中发生变化")

    write_report(
        reports_root / "tf_sensitivity_report.md",
        results,
        comparisons,
        incremental,
        cases,
        protected_all_retained,
        len(protected_words),
        len(protected_present),
        sum(token_tf[token] for token in protected_present),
        sorted(protected_words - token_tf.keys()),
        figure_backend,
        figure_paths,
        int(frozen_profile["min_df"]),
    )

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    runtime_seconds = time.perf_counter() - started_clock
    output_files = sorted(
        [path for path in output_root.rglob("*") if path.is_file()]
        + [
            path
            for path in reports_root.rglob("*")
            if path.is_file() and path.name != "run_metadata.json"
        ],
        key=lambda path: str(path),
    )
    metadata = {
        "schema_version": 1,
        "method": "tf_threshold_sensitivity_only",
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": round(runtime_seconds, 6),
        "inputs": [
            {
                "path": cleaner.display_path(path),
                "sha256": input_hashes_before[str(path.resolve())],
            }
            for path in input_paths
        ],
        "experimental_variable": {
            "name": "LOW_TF threshold",
            "thresholds": list(thresholds),
        },
        "frozen_rules": {
            key: value
            for key, value in frozen_profile.items()
            if key != "min_tf" and key not in REFERENCE_PROFILE_FIELDS
        },
        "base_profile_reference_statistics": {
            key: value
            for key, value in frozen_profile.items()
            if key in REFERENCE_PROFILE_FIELDS
        },
        "resource_counts": {
            "poems": len(records),
            "normalization_rules": len(normalization_rules),
            "normalization_occurrences": len(normalization_events),
            "stopwords": len(stopwords),
            "protected_words": len(protected_words),
            "protected_present_token_types": len(protected_present),
            "protected_token_occurrences": sum(
                token_tf[token] for token in protected_present
            ),
            "data_quality_issue_rows": len(data_issues),
            "literary_candidate_rows": len(candidates),
            "extreme_length_rows": len(extremes),
        },
        "groups": {result.label: result.statistics for result in results},
        "incremental_losses": {
            transition: {
                "vocabulary": len(rows),
                "occurrences": sum(int(row["TF"]) for row in rows),
            }
            for transition, rows in incremental
        },
        "figures": {
            "preferred_backend": figure_config.get("preferred_backend"),
            "actual_backend": figure_backend,
            "matplotlib_version": matplotlib_version,
            "paths": [cleaner.display_path(path) for path in figure_paths],
        },
        "environment": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "validation": {
            "only_min_tf_varies": True,
            "input_hashes_unchanged": input_hashes_before == input_hashes_after,
            "existing_tf5_hashes_unchanged": existing_hashes_before == existing_hashes_after,
            "protected_tokens_retained_in_all_groups": protected_all_retained,
            "token_order_monotonic_subsequence": sequence_monotonic,
            "poem_count_and_order_constructed_from_one_source": True,
            "tf5_matches_existing_output": tf5_validation,
            "tf1_tf2_tf3_tokens_identical": len(
                {
                    cleaner.file_sha256(result.tokens_path) for result in results[:3]
                }
            )
            == 1,
            "semantic_mapping_applied": False,
            "lda_executed": False,
        },
        "outputs": [
            {
                "path": cleaner.display_path(path),
                "sha256": cleaner.file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        ],
    }
    cleaner.write_json(reports_root / "run_metadata.json", metadata)
    logger.info("TF 阈值敏感性实验完成；耗时 %.3f 秒", runtime_seconds)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="只改变 LOW_TF 阈值，生成 TF1、TF2、TF3、TF5 对照语料"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="TF 阈值敏感性实验配置文件",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

    args = parse_args(argv)
    try:
        run_experiment(args.config.resolve())
    except Exception as exc:
        logging.getLogger("clean_corpus").exception("TF 敏感性实验失败：%s", exc)
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
