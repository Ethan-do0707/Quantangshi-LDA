"""最终困惑度诊断产物的独立验收测试。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model_selection/perplexity_diagnostic"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""): value.update(chunk)
    return value.hexdigest()


class FinalArtifacts(unittest.TestCase):
    def test_run_grid_and_parameters(self):
        runs = rows(OUT / "data/perplexity_all_runs.csv")
        self.assertEqual(len(runs), 60)
        grid = {(r["corpus"], int(r["K"]), int(r["seed"])) for r in runs}
        expected = {(c, k, s) for c in "ABCD" for k in [20, 30, 40, 50, 60] for s in [42, 2026, 20260830]}
        self.assertEqual(grid, expected)
        for row in runs:
            self.assertEqual((row["n_iter_"], row["fit_n_jobs"], row["fit_backend"], row["evaluation_n_jobs"]), ("30", "-1", "loky", "1"))
            self.assertTrue(math.isfinite(float(row["test_perplexity"])) and float(row["test_perplexity"]) > 0)
            self.assertTrue(math.isfinite(float(row["train_perplexity"])) and float(row["train_perplexity"]) > 0)

    def test_splits_are_complete(self):
        expected = {"A": (43019, 8604), "B": (43016, 8604), "C": (43015, 8603), "D": (43014, 8603)}
        for corpus, (total, tests) in expected.items():
            split = rows(OUT / f"splits/perplexity_split_{corpus}.csv")
            self.assertEqual((len(split), sum(r["split"] == "TEST" for r in split)), (total, tests))
            self.assertEqual(set(r["split"] for r in split), {"TRAIN", "TEST"})
            self.assertEqual(len({r["poem_id"] for r in split}), total)

    def test_summary_recomputes_from_raw(self):
        raw = rows(OUT / "data/perplexity_all_runs.csv"); summary = rows(OUT / "data/perplexity_summary_by_k.csv")
        self.assertEqual(len(summary), 20)
        for row in summary:
            values = [float(r["test_perplexity"]) for r in raw if r["corpus"] == row["corpus"] and r["K"] == row["K"]]
            self.assertAlmostEqual(float(row["mean_test_perplexity"]), statistics.mean(values), places=8)
            self.assertAlmostEqual(float(row["std_test_perplexity"]), statistics.stdev(values), places=8)

    def test_required_outputs_and_report(self):
        required = [
            "README.md", "QTS_LDA_K_Perplexity_Diagnostic_Report.md",
            "data/perplexity_all_runs.csv", "data/perplexity_summary_by_k.csv", "data/model_selection_with_perplexity.csv",
            "data/C_perplexity_k_diagnostic.csv", "data/C_K_multimetric_diagnostic.csv",
            "figures/perplexity_by_k_all_corpora.png", "figures/C_corpus_perplexity_by_k.png",
            "figures/C_K_perplexity_vs_npmi.png", "figures/C_K_perplexity_vs_stability.png", "figures/perplexity_explainer_for_students.png",
        ]
        self.assertTrue(all((OUT / path).is_file() for path in required))
        report = (OUT / required[1]).read_text(encoding="utf-8")
        for heading in range(1, 11): self.assertIn(f"## {heading}.", report)
        self.assertIn("SUPPORTS_K20", report)
        for path in required[7:]:
            with Image.open(OUT / path) as image: self.assertGreaterEqual(min(image.size), 1500)

    def test_manifests_and_integrity_checks(self):
        checks = json.loads((OUT / "audit/integrity_checks.json").read_text(encoding="utf-8"))
        self.assertTrue(checks["complete"] and checks["n_runs"] == 60)
        self.assertTrue(all(item["passed"] for item in checks["checks"]))
        manifest = json.loads((OUT / "audit/output_manifest.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(manifest), 30)
        self.assertTrue(all(digest(ROOT / path) == value for path, value in manifest.items()))


if __name__ == "__main__": unittest.main()
