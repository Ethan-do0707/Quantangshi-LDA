"""对量化候选验证中被触发的 K 做 max_iter=60 收敛敏感性复核。"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from joblib.externals.loky import get_reusable_executor


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_quantitative_candidate_validation as validation


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "model_selection/quantitative_candidate_validation"
AFFECTED_K = [12, 13, 15, 20]
ORIGINAL_SEEDS = [42, 2026, 20260830]
REGIMES = ["ORIGINAL_PRIOR", "FIXED_PRIOR_K20"]
MAX_ITER_BASELINE = 30
MAX_ITER_SENSITIVITY = 60
RESULT_FIELDS = validation.ALL_RUN_FIELDS + [
    "max_iter_configured",
    "unique_fit",
    "reused_from_prior_regime",
]


def unique_fit_plan() -> list[tuple[str, int, int]]:
    """列出实际拟合计划；K20 两套先验数学等价，因此不重复拟合。"""

    plan: list[tuple[str, int, int]] = []
    for regime, topic_count, seed in itertools.product(REGIMES, AFFECTED_K, ORIGINAL_SEEDS):
        if regime == "FIXED_PRIOR_K20" and topic_count == 20:
            continue
        plan.append((regime, topic_count, seed))
    return plan


def checkpoint_path(regime: str, topic_count: int, seed: int) -> Path:
    """返回迭代敏感性检查点路径。"""

    return OUTPUT / "iteration_sensitivity/checkpoints" / f"{regime}_K{topic_count}_seed_{seed}.json"


def checkpoint_signature(context: Mapping[str, Any], regime: str, topic_count: int, seed: int) -> str:
    """生成与主实验身份和 max_iter 绑定的检查点签名。"""

    return validation.scan.stable_hash(
        {
            "main_identity": context["identity"],
            "prior_regime": regime,
            "K": topic_count,
            "seed": seed,
            "max_iter": MAX_ITER_SENSITIVITY,
            "script_sha256": validation.scan.file_sha256(Path(__file__).resolve()),
        }
    )


def validate_trigger_scope() -> dict[str, Any]:
    """从已完成的主实验表中复核预注册触发条件及受影响范围。"""

    summary = pd.read_csv(OUTPUT / "data/quantitative_candidate_summary.csv")
    prior = pd.read_csv(OUTPUT / "data/candidate_prior_sensitivity.csv")
    trigger_a: list[dict[str, Any]] = []
    for regime in REGIMES:
        current = summary[summary["prior_regime"] == regime].set_index("K")
        for other in [15, 20]:
            difference = float(current.loc[12, "NPMI_mean"] - current.loc[other, "NPMI_mean"])
            if abs(difference) <= 0.001:
                trigger_a.append({"prior_regime": regime, "comparison": f"K12-K{other}", "difference": difference})
    trigger_b = sorted(
        summary.loc[summary["quantitative_status"].str.contains("SEED_SENSITIVE", na=False), "K"]
        .astype(int)
        .unique()
        .tolist()
    )
    trigger_c = sorted(
        prior.loc[prior["rank_change"].abs() >= 2, "K"].astype(int).unique().tolist()
    )
    affected = sorted(
        set(trigger_b)
        | set(trigger_c)
        | {12}
        | {int(item["comparison"].split("K")[-1]) for item in trigger_a}
    )
    if affected != AFFECTED_K:
        raise ValueError(f"触发范围应为 {AFFECTED_K}，实际为 {affected}")
    return {
        "trigger_A_practical_equivalence": trigger_a,
        "trigger_B_seed_sensitive_K": trigger_b,
        "trigger_C_rank_change_abs_ge_2_K": trigger_c,
        "affected_K": affected,
    }


def load_checkpoint(context: Mapping[str, Any], regime: str, topic_count: int, seed: int) -> dict[str, Any]:
    """读取并验证一个迭代敏感性检查点。"""

    path = checkpoint_path(regime, topic_count, seed)
    item = validation.scan.read_json(path)
    expected = checkpoint_signature(context, regime, topic_count, seed)
    if item.get("checkpoint_signature") != expected:
        raise ValueError(f"迭代敏感性检查点签名不符：{path}")
    if item.get("result_sha256") != validation.scan.stable_hash(item.get("result")):
        raise ValueError(f"迭代敏感性检查点结果哈希不符：{path}")
    return item["result"]


def expand_reused_k20(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """为 K20 显式补登记等价的固定先验行，并保留复用审计字段。"""

    expanded = [dict(row) for row in rows]
    existing = {(row["prior_regime"], int(row["K"]), int(row["seed"])) for row in expanded}
    completed_k20_seeds = sorted(
        {
            int(row["seed"])
            for row in expanded
            if row["prior_regime"] == "ORIGINAL_PRIOR" and int(row["K"]) == 20
        },
        key=ORIGINAL_SEEDS.index,
    )
    for seed in completed_k20_seeds:
        key = ("FIXED_PRIOR_K20", 20, seed)
        if key in existing:
            continue
        source = next(
            row
            for row in expanded
            if row["prior_regime"] == "ORIGINAL_PRIOR" and int(row["K"]) == 20 and int(row["seed"]) == seed
        )
        reused = dict(source)
        reused["prior_regime"] = "FIXED_PRIOR_K20"
        reused["unique_fit"] = False
        reused["reused_from_prior_regime"] = "ORIGINAL_PRIOR"
        expanded.append(reused)
    return sorted(expanded, key=lambda row: (REGIMES.index(str(row["prior_regime"])), AFFECTED_K.index(int(row["K"])), ORIGINAL_SEEDS.index(int(row["seed"]))))


def write_runs(rows: Sequence[Mapping[str, Any]]) -> None:
    """原子写出当前已完成的敏感性逐 run 数据。"""

    validation.scan.atomic_frame(
        OUTPUT / "data/iteration_sensitivity_all_runs.csv",
        pd.DataFrame(rows, columns=RESULT_FIELDS),
    )


def build_comparison(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """按 K×prior 汇总 30→60 次迭代的配对变化。"""

    baseline = pd.read_csv(OUTPUT / "data/quantitative_candidate_all_runs.csv")
    baseline = baseline[
        baseline["K"].isin(AFFECTED_K)
        & baseline["seed"].isin(ORIGINAL_SEEDS)
        & baseline["prior_regime"].isin(REGIMES)
    ].copy()
    merged = baseline.merge(
        sensitivity,
        on=["K", "seed", "prior_regime"],
        suffixes=("_30", "_60"),
        validate="one_to_one",
    )
    metrics = [
        "NPMI",
        "test_perplexity",
        "topic_diversity",
        "mean_pairwise_topic_cosine",
        "mean_pairwise_top20_jaccard",
    ]
    rows: list[dict[str, Any]] = []
    for (regime, topic_count), group in merged.groupby(["prior_regime", "K"], sort=False):
        item: dict[str, Any] = {
            "K": int(topic_count),
            "prior_regime": regime,
            "seed_count": int(len(group)),
        }
        for metric in metrics:
            values_30 = group[f"{metric}_30"]
            values_60 = group[f"{metric}_60"]
            difference = values_60 - values_30
            item[f"{metric}_mean_30"] = float(values_30.mean())
            item[f"{metric}_mean_60"] = float(values_60.mean())
            item[f"{metric}_mean_delta_60_minus_30"] = float(difference.mean())
            item[f"{metric}_max_abs_seed_delta"] = float(difference.abs().max())
        rows.append(item)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["prior_regime", "K"], key=lambda col: col.map({name: i for i, name in enumerate(REGIMES)}) if col.name == "prior_regime" else col).reset_index(drop=True)


def write_report(comparison: pd.DataFrame, triggers: Mapping[str, Any], unique_fit_count: int) -> None:
    """生成人工可审计的迭代敏感性简报。"""

    lines = [
        "# ITERATION_SENSITIVITY",
        "",
        "## 1. 触发与范围",
        "",
        "主实验全部保持 `max_iter=30`。依据预注册条件，K12–K15/K20 的实用等价、K20 的种子敏感标签，以及固定先验后的明显排名变化，触发本项复核。",
        "",
        f"受影响 K：{', '.join(f'K{k}' for k in triggers['affected_K'])}；种子：{', '.join(map(str, ORIGINAL_SEEDS))}；比较 `max_iter=30` 与 `max_iter=60`。",
        "",
        f"共登记 {len(comparison) * len(ORIGINAL_SEEDS)} 个 K×prior×seed 条目，实际新增拟合 {unique_fit_count} 次。K20 在两套 prior 下的有效 alpha/eta 均为 0.05/0.05，因此每个 seed 只拟合一次并复用结果。",
        "",
        "## 2. NPMI 与 held-out perplexity",
        "",
        "| Prior | K | NPMI@30 | NPMI@60 | ΔNPMI | PPL@30 | PPL@60 | ΔPPL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.prior_regime} | {int(row.K)} | {row.NPMI_mean_30:.8f} | {row.NPMI_mean_60:.8f} | "
            f"{row.NPMI_mean_delta_60_minus_30:+.8f} | {row.test_perplexity_mean_30:.2f} | "
            f"{row.test_perplexity_mean_60:.2f} | {row.test_perplexity_mean_delta_60_minus_30:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## 3. 排名与结论",
            "",
        ]
    )
    for regime in REGIMES:
        current = comparison[comparison["prior_regime"] == regime]
        rank_30 = current.sort_values("NPMI_mean_30", ascending=False)["K"].astype(int).tolist()
        rank_60 = current.sort_values("NPMI_mean_60", ascending=False)["K"].astype(int).tolist()
        lines.append(
            f"- {regime}：受影响候选的 NPMI 排序由 `{' > '.join(f'K{k}' for k in rank_30)}` 变为 `{' > '.join(f'K{k}' for k in rank_60)}`。"
        )
    max_delta = float(comparison["NPMI_mean_delta_60_minus_30"].abs().max())
    lines.extend(
        [
            "",
            f"最大组均值 |ΔNPMI| = {max_delta:.8f}。本表仅用于判断主结论是否依赖 30 次迭代上限，不据此创建新的正式模型或覆盖冻结 K20。",
            "",
        ]
    )
    target = OUTPUT / "ITERATION_SENSITIVITY.md"
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(target)


def finalize(context: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], triggers: Mapping[str, Any]) -> None:
    """写出完整数据、比较表、简报和完整性审计。"""

    expanded = expand_reused_k20(rows)
    if len(expanded) != len(AFFECTED_K) * len(ORIGINAL_SEEDS) * len(REGIMES):
        raise ValueError("迭代敏感性条目数不等于 24")
    frame = pd.DataFrame(expanded, columns=RESULT_FIELDS)
    if not (frame["max_iter_configured"] == MAX_ITER_SENSITIVITY).all():
        raise ValueError("迭代敏感性 max_iter 配置不一致")
    if not (frame["n_iter"] == MAX_ITER_SENSITIVITY).all():
        raise ValueError("至少一个敏感性模型未运行至 60 次迭代")
    write_runs(expanded)
    comparison = build_comparison(frame)
    validation.scan.atomic_frame(OUTPUT / "data/iteration_sensitivity_summary.csv", comparison)
    write_report(comparison, triggers, len(rows))
    artifacts = [
        OUTPUT / "data/iteration_sensitivity_all_runs.csv",
        OUTPUT / "data/iteration_sensitivity_summary.csv",
        OUTPUT / "ITERATION_SENSITIVITY.md",
    ]
    audit = {
        "main_identity": context["identity"],
        "triggers": triggers,
        "registered_rows": len(frame),
        "unique_new_fits": len(rows),
        "k20_prior_equivalence_reused_rows": int((~frame["unique_fit"]).sum()),
        "all_finite": bool(frame.select_dtypes(include="number").apply(lambda col: col.map(pd.notna).all()).all()),
        "artifacts": {str(path.relative_to(ROOT)).replace("\\", "/"): validation.scan.file_sha256(path) for path in artifacts},
    }
    validation.scan.atomic_json(OUTPUT / "audit/iteration_sensitivity_integrity.json", audit)
    validation.scan.atomic_json(
        OUTPUT / "audit/iteration_sensitivity_progress.json",
        {"state": "COMPLETE", "unique_new_fits": len(rows), "registered_rows": len(frame)},
    )


def main() -> None:
    """执行可续跑的迭代敏感性复核。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    triggers = validate_trigger_scope()
    context = validation.prepare()
    if int(context["base_lda"]["max_iter"]) != MAX_ITER_BASELINE:
        raise ValueError("主实验基准 max_iter 必须为 30")
    context_60 = dict(context)
    context_60["base_lda"] = dict(context["base_lda"])
    context_60["base_lda"]["max_iter"] = MAX_ITER_SENSITIVITY
    (OUTPUT / "iteration_sensitivity/checkpoints").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for regime, topic_count, seed in unique_fit_plan():
        path = checkpoint_path(regime, topic_count, seed)
        if path.is_file():
            rows.append(load_checkpoint(context, regime, topic_count, seed))
    if args.verify_only:
        if len(rows) != len(unique_fit_plan()):
            raise ValueError(f"敏感性检查点仅完成 {len(rows)}/{len(unique_fit_plan())}")
        finalize(context, rows, triggers)
        print("迭代敏感性核验通过：21/21 unique fits，24/24 registered rows")
        return

    completed = {(row["prior_regime"], int(row["K"]), int(row["seed"])) for row in rows}
    lock = OUTPUT / "audit/iteration_sensitivity_runner.lock"
    if lock.exists():
        raise RuntimeError(f"已有迭代敏感性运行锁：{lock}")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    new_runs = 0
    try:
        for regime, topic_count, seed in unique_fit_plan():
            key = (regime, topic_count, seed)
            if key in completed:
                continue
            if args.max_new_runs is not None and new_runs >= args.max_new_runs:
                break
            validation.scan.atomic_json(
                OUTPUT / "audit/iteration_sensitivity_progress.json",
                {
                    "state": "RUNNING",
                    "completed_unique_fits": len(rows),
                    "expected_unique_fits": len(unique_fit_plan()),
                    "current": {"prior_regime": regime, "K": topic_count, "seed": seed},
                },
            )
            print(
                f"开始 ITER60 {regime} K={topic_count} seed={seed}（{len(rows)}/{len(unique_fit_plan())}）",
                flush=True,
            )
            result, audit, _arrays = validation.evaluate_one(context_60, regime, topic_count, seed)
            result["max_iter_configured"] = MAX_ITER_SENSITIVITY
            result["unique_fit"] = True
            result["reused_from_prior_regime"] = ""
            item = {
                "checkpoint_signature": checkpoint_signature(context, regime, topic_count, seed),
                "result": result,
                "result_sha256": validation.scan.stable_hash(result),
                "audit": audit,
            }
            validation.scan.atomic_json(checkpoint_path(regime, topic_count, seed), item)
            rows.append(result)
            del _arrays
            completed.add(key)
            new_runs += 1
            write_runs(expand_reused_k20(rows) if any(int(row["K"]) == 20 for row in rows) else rows)
            print(
                f"完成 ITER60 {len(rows)}/{len(unique_fit_plan())} {regime} K={topic_count} seed={seed} "
                f"NPMI={result['NPMI']:.8f} testPPL={result['test_perplexity']:.4f} fit={result['fit_time']:.1f}s",
                flush=True,
            )
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
            gc.collect()
        if len(rows) == len(unique_fit_plan()):
            finalize(context, rows, triggers)
            print("ITERATION_SENSITIVITY 完成：21/21 unique fits，24/24 registered rows", flush=True)
        else:
            validation.scan.atomic_json(
                OUTPUT / "audit/iteration_sensitivity_progress.json",
                {"state": "PARTIAL", "completed_unique_fits": len(rows), "expected_unique_fits": len(unique_fit_plan())},
            )
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
