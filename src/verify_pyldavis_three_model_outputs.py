#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""静态复核三模型 pyLDAvis 输出；不读取模型做推断，也不改动源工件。"""

from __future__ import annotations

import ast
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "visualization/pyldavis_C_K12_K15_K20"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def check(name: str, condition: bool, evidence: str, results: list[dict]) -> None:
    results.append({"qa_item": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})
    if not condition:
        raise AssertionError(f"{name}: {evidence}")


def main() -> int:
    data = OUTPUT / "data"
    audit = rows(data / "pyldavis_three_model_comparability_audit.csv")
    amap = {int(row["K"]): row for row in audit}
    results: list[dict] = []
    check("QA01_K12_topic_term_shape", amap[12]["topic_term_shape"] == "12x35063", amap[12]["topic_term_shape"], results)
    check("QA02_K15_topic_term_shape", amap[15]["topic_term_shape"] == "15x35063", amap[15]["topic_term_shape"], results)
    check("QA03_K20_topic_term_shape", amap[20]["topic_term_shape"] == "20x35063", amap[20]["topic_term_shape"], results)
    check("QA04_all_doc_topic_43015_rows", all(row["doc_topic_shape"] == f"43015x{row['K']}" for row in audit), "; ".join(row["doc_topic_shape"] for row in audit), results)
    check("QA05_same_vocab_hash", len({row["vocabulary_sha256"] for row in audit}) == 1, audit[0]["vocabulary_sha256"], results)
    check("QA06_same_document_order", len({row["document_order_sha256"] for row in audit}) == 1, audit[0]["document_order_sha256"], results)

    config = json.loads((OUTPUT / "config/pyldavis_three_model_config.yaml").read_text(encoding="utf-8"))
    row_sums = []
    theta_sums = []
    for model_id, cfg in config["models"].items():
        K = cfg["K"]
        if cfg["topic_term_key"] == "scipy_sparse":
            phi = load_npz(ROOT / cfg["topic_term"]).toarray()
        else:
            with np.load(ROOT / cfg["topic_term"], allow_pickle=False) as archive:
                phi = np.asarray(archive[cfg["topic_term_key"]])
        phi = phi / phi.sum(axis=1, keepdims=True)
        row_sums.append(float(np.max(np.abs(phi.sum(axis=1) - 1))))
        if cfg["doc_topic_key"] == "scipy_sparse":
            theta = load_npz(ROOT / cfg["doc_topic"])
            sums = np.asarray(theta.sum(axis=1)).ravel()
        else:
            with np.load(ROOT / cfg["doc_topic"], allow_pickle=False) as archive:
                sums = np.asarray(archive[cfg["doc_topic_key"]].sum(axis=1))
        theta_sums.append(float(np.max(np.abs(sums - 1))))
    check("QA07_topic_term_row_sum", max(row_sums) < 1e-10, f"max_abs_error={max(row_sums):.3g}", results)
    check("QA08_doc_topic_row_sum", max(theta_sums) < 1e-10, f"max_abs_error={max(theta_sums):.3g}", results)
    check("QA09_sort_topics_false", config["pyldavis"]["sort_topics"] is False, "config sort_topics=false", results)
    topic_map = rows(data / "pyldavis_topic_id_map.csv")
    expected = [(K, f"T{i:02d}", i) for K in (12, 15, 20) for i in range(1, K + 1)]
    actual = [(int(row["K"]), row["original_topic_id"], int(row["pyldavis_topic_id"])) for row in topic_map]
    check("QA10_all_47_topic_ids_preserved", actual == expected and all(row["mapping_status"] == "PRESERVED" for row in topic_map), f"{len(actual)}/47", results)

    html_paths = [OUTPUT / f"html/C_K{K}_pyLDAvis.html" for K in (12, 15, 20)]
    html_text = [path.read_text(encoding="utf-8") for path in html_paths]
    html_ok = all(len(text) > 500_000 and "new LDAvis" in text and not re.search(r'<script\b[^>]*\bsrc\s*=|<link\b[^>]*\bhref\s*=', text, re.I) for text in html_text)
    check("QA11_three_html_openable_self_contained", html_ok, "; ".join(f"{p.name}:{p.stat().st_size}" for p in html_paths), results)
    chinese_ok = all(f"《全唐诗》LDA Topic Visualization — C_K{K}" in text for K, text in zip((12, 15, 20), html_text))
    check("QA12_chinese_utf8_and_fallback_fonts", chinese_ok and all("Microsoft YaHei" in text and "Noto Sans CJK SC" in text for text in html_text), "UTF-8 标题及中文字体回退存在", results)
    comparison = (OUTPUT / "html/C_K12_K15_K20_pyLDAvis_comparison.html").read_text(encoding="utf-8")
    links = re.findall(r'href="([^"]+_pyLDAvis\.html)"', comparison)
    check("QA13_comparison_three_links", links == [f"C_K{K}_pyLDAvis.html" for K in (12, 15, 20)] and all((OUTPUT / "html" / link).is_file() for link in links), repr(links), results)

    source = (ROOT / "src/generate_pyldavis_three_model_comparison.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform", "transform"}]
    check("QA14_no_fit_or_transform_calls", not forbidden, f"forbidden_calls={forbidden}", results)
    hashes = rows(OUTPUT / "audit/source_artifact_hash_qc.csv")
    check("QA15_source_artifact_hashes_unchanged", bool(hashes) and all(row["unchanged"] == "True" and row["sha256_before"] == row["sha256_after"] for row in hashes), f"{len(hashes)} files unchanged", results)

    # 附加输出完整性检查，覆盖 PreparedData、相关词、图、报告和研究者表。
    for K in (12, 15, 20):
        with (OUTPUT / f"prepared/C_K{K}_pyLDAvis_prepared.pkl").open("rb") as handle:
            vis = pickle.load(handle)
        check(f"EXTRA_prepared_K{K}", list(vis.topic_order) == list(range(1, K + 1)), f"topic_order 1..{K}", results)
        rel = rows(data / f"C_K{K}_pyldavis_relevant_terms.csv")
        check(f"EXTRA_relevant_terms_K{K}", len(rel) == K * 3 * 30, f"{len(rel)} rows", results)
    figures = [
        "C_K12_topic_distance_map.png", "C_K15_topic_distance_map.png", "C_K20_topic_distance_map.png",
        "C_K12_K15_K20_topic_prevalence_comparison.png", "C_K12_K15_K20_topic_pair_similarity_distribution.png",
        "C_K12_K15_K20_nearest_topic_similarity.png", "C_K12_K15_K20_high_similarity_pair_counts.png",
        "C_K12_K15_K20_resolution_diagnostic.png",
    ]
    dimensions = []
    for filename in figures:
        path = OUTPUT / "figures" / filename
        with Image.open(path) as image: dimensions.append((filename, image.size))
    check("EXTRA_eight_figures", len(dimensions) == 8 and all(w >= 1500 and h >= 1000 for _, (w, h) in dimensions), repr(dimensions), results)
    alignment = rows(data / "K12_K15_K20_pyldavis_alignment_reference.csv")
    check("EXTRA_adjacent_alignment", any(row["source_K"] == "12" and row["target_K"] == "15" for row in alignment) and any(row["source_K"] == "15" and row["target_K"] == "20" for row in alignment), f"{len(alignment)} rows", results)
    report = (OUTPUT / "QTS_C_K12_K15_K20_pyLDAvis_Comparative_Report.md").read_text(encoding="utf-8")
    check("EXTRA_report_17_sections", all(f"# {i}." in report for i in range(1, 18)), "17/17 sections", results)
    xlsx = OUTPUT / "researcher/K12_K15_K20_pyldavis_researcher_checklist.xlsx"
    check("EXTRA_researcher_workbook", xlsx.is_file() and xlsx.stat().st_size > 10_000, f"{xlsx.stat().st_size} bytes", results)

    with (data / "pyldavis_three_model_qa.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    (OUTPUT / "audit/run_status.json").write_text(json.dumps({"status": "PYLDAVIS_THREE_MODEL_READY", "qa_passed": len(results), "qa_failed": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"QA 完成：{len(results)} 项通过，0 项失败。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
