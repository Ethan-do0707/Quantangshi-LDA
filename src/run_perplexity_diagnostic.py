"""在冻结词表上运行留出困惑度诊断，不修改既有模型。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import joblib
from joblib.externals.loky import get_reusable_executor
import numpy as np
import psutil
from scipy import sparse
from sklearn.model_selection import train_test_split
import rebuild_candidate_models as rebuild

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "model_selection/perplexity_diagnostic/config/perplexity_config.yaml"
AUDIT_KEYS = ["corpus", "n_train", "n_test", "vocab_size", "train_token_count", "test_token_count", "split_sha256", "fit_rows_sha256", "test_rows_sha256"]
RUN_FIELDS = ["corpus", "K", "seed", "n_train", "n_test", "vocab_size", "train_perplexity", "test_perplexity", "train_score", "test_score", "n_iter_", "fit_time_seconds", "evaluation_time_seconds", "fit_n_jobs", "fit_backend", "evaluation_n_jobs", "train_token_count", "test_token_count", "split_sha256", "fit_rows_sha256", "test_rows_sha256", "checkpoint_signature"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    """原子保存本诊断自己的检查点。"""
    path = Path(path); payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for attempt in range(30):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(payload, encoding="utf-8"); temp.replace(path)
            return
        except (OSError, PermissionError):
            if attempt == 29: raise
            time.sleep(2)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    """写出机器可读科学数据。"""
    path = Path(path)
    for attempt in range(30):
        try:
            path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix + ".tmp")
            with temp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise"); writer.writeheader(); writer.writerows(rows)
            temp.replace(path); return
        except (OSError, PermissionError):
            if attempt == 29: raise
            time.sleep(2)


def make_split(n_rows, split_config):
    train, test = train_test_split(np.arange(n_rows, dtype=np.int64), **split_config)
    if set(train) & set(test) or sorted([*train, *test]) != list(range(n_rows)):
        raise ValueError("训练/测试索引不互斥或未完整覆盖")
    return train, test


def row_digest(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def environment():
    return {**rebuild.environment_info(), "executable": sys.executable, "physical_cpu_count": psutil.cpu_count(False),
            "cpu_description": platform.processor(), "memory_total_bytes": psutil.virtual_memory().total,
            "memory_available_bytes_at_start": psutil.virtual_memory().available, "joblib": joblib.__version__}


def immutable_files(config):
    paths = {ROOT / config[key] for key in ["original_config", "original_metadata", "original_summary", "human_summary"]}
    for directory in ["output/tf_df_sensitivity", "output/lda_sensitivity", "output/lda_candidate_rebuild", "reports/lda_sensitivity",
                      "outputs/01a05061-9ee3-7610-a401-abfec81de1b2/literary_history_dataset/freeze",
                      "outputs/01a05061-9ee3-7610-a401-abfec81de1b2/model_selection"]:
        paths.update(path for path in (ROOT / directory).rglob("*") if path.is_file())
    paths.update(ROOT / item["path"] for item in read_json(ROOT / config["original_metadata"])["inputs"])
    return sorted(paths)


def prepare(config_path=CONFIG):
    config = read_json(config_path); original = read_json(ROOT / config["original_config"])
    out = (ROOT / config["output_root"]).resolve()
    if out != (ROOT / "model_selection/perplexity_diagnostic").resolve():
        raise ValueError("输出目录越界")
    if config["topic_counts"] != original["lda"]["topic_counts"] or config["topic_counts"] != [20, 30, 40, 50, 60]:
        raise ValueError("K范围异常")
    if config["seeds"] != original["lda"]["random_seeds"] or config["seeds"] != [42, 2026, 20260830]:
        raise ValueError("seed范围异常")
    env = environment(); old_env = read_json(ROOT / config["original_metadata"])["environment"]
    env_check = rebuild.validate_environment(env, old_env)
    if not env_check["all_match"]:
        raise ValueError(f"环境不匹配：{env_check}")
    identity = signature({"config": config, "original": original, "environment": old_env, "runner": sha256(__file__),
                          "model_factory": signature(inspect.getsource(rebuild.build_model))})
    manifest_path = out / "audit/input_manifest.json"
    current = {path.relative_to(ROOT).as_posix(): sha256(path) for path in immutable_files(config)}
    if manifest_path.exists() and read_json(manifest_path) != current:
        raise ValueError("冻结输入与基线不一致")
    if not manifest_path.exists(): atomic_json(manifest_path, current)
    identity_path = out / "audit/run_identity.json"
    identity_data = {"signature": identity, "config": config, "original_lda": original["lda"]}
    allow_code_migration = False
    if identity_path.exists() and read_json(identity_path) != identity_data:
        old_identity = read_json(identity_path)
        allow_code_migration = old_identity.get("config") == config and old_identity.get("original_lda") == original["lda"]
        if not allow_code_migration:
            raise ValueError("诊断统计配置变化，拒绝复用检查点")
    atomic_json(identity_path, identity_data)
    if not (out / "audit/environment.json").exists():
        atomic_json(out / "audit/environment.json", {**env, "original_environment_validation": env_check})
    corpora = {}
    for label in config["corpora"]:
        base = ROOT / original["paths"]["corpus_output_root"] / label; meta = read_json(base / "corpus_metadata.json")
        for name, key in [("bow_matrix.npz", "bow_matrix_npz"), ("vocabulary.csv", "vocabulary"), ("document_index.csv", "document_index"), ("lda_corpus.csv", "lda_corpus"), ("lda_tokens.txt", "lda_tokens")]:
            if sha256(base / name) != meta["outputs"][key]: raise ValueError(f"{label}/{name}哈希不符")
        matrix = sparse.load_npz(base / "bow_matrix.npz").tocsr(); documents = read_csv(base / "document_index.csv"); vocabulary = read_csv(base / "vocabulary.csv")
        if matrix.shape != tuple(meta["matrix_shape"]) or len(documents) != matrix.shape[0] or len(vocabulary) != matrix.shape[1]: raise ValueError(f"{label}形状不符")
        ids = [row["poem_id"] for row in documents]
        if len(set(ids)) != len(ids) or [int(row["matrix_row"]) for row in documents] != list(range(len(ids))): raise ValueError(f"{label}文档索引异常")
        if [int(row["term_index"]) for row in vocabulary] != list(range(matrix.shape[1])): raise ValueError(f"{label}词表索引异常")
        lengths = np.asarray(matrix.sum(axis=1)).ravel()
        if np.any(lengths <= 0) or not np.array_equal(lengths, [int(row["token_count"]) for row in documents]): raise ValueError(f"{label}词次异常")
        train, test = make_split(len(ids), config["split"]); test_set = set(test.tolist())
        split_rows = [{"poem_id": poem_id, "split": "TEST" if i in test_set else "TRAIN"} for i, poem_id in enumerate(ids)]
        split_path = out / f"splits/perplexity_split_{label}.csv"
        if split_path.exists() and read_csv(split_path) != split_rows: raise ValueError(f"{label}拆分被改动")
        if not split_path.exists(): write_csv(split_path, ["poem_id", "split"], split_rows)
        x_train, x_test = matrix[train], matrix[test]; unseen = np.asarray(x_train.sum(axis=0)).ravel() == 0
        audit = {"corpus": label, "n_train": len(train), "n_test": len(test), "vocab_size": matrix.shape[1],
                 "train_token_count": int(x_train.sum()), "test_token_count": int(x_test.sum()), "split_sha256": sha256(split_path),
                 "fit_rows_sha256": row_digest(train), "test_rows_sha256": row_digest(test), "train_test_overlap": 0,
                 "empty_train_rows": 0, "empty_test_rows": 0, "unseen_in_train_vocabulary_size": int(unseen.sum()),
                 "test_tokens_unseen_in_train": int(x_test[:, unseen].sum()), "matrix_sha256": sha256(base / "bow_matrix.npz")}
        atomic_json(out / f"audit/split_{label}.json", audit); corpora[label] = {"train": x_train, "test": x_test, "audit": audit}
    if allow_code_migration:
        migrated = []
        for path in (out / "checkpoints").glob("*.json"):
            item = read_json(path); result = item["result"]
            if item["result_sha256"] != signature(result): raise ValueError(f"不能迁移损坏检查点：{path}")
            audit = corpora[result["corpus"]]["audit"]
            if any(result[key] != audit[key] for key in AUDIT_KEYS): raise ValueError(f"不能迁移审计字段不符的检查点：{path}")
            params = rebuild.build_model(result["K"], result["seed"], original["lda"]).get_params()
            result.setdefault("fit_backend", "loky"); result["checkpoint_signature"] = run_signature(identity, audit, params)
            item.update({"checkpoint_signature": result["checkpoint_signature"], "params": params, "result": result, "result_sha256": signature(result)})
            atomic_json(path, item); migrated.append(path.name)
        atomic_json(out / "audit/checkpoint_code_migration.json", {"reason": "仅改变并行执行后端与审计字段，统计配置、拆分和既有数值不变", "migrated": sorted(migrated)})
    return config, original["lda"], out, identity, corpora


def evaluate_one(model, x_train, x_test):
    """唯一fit入口只接受训练矩阵；测试矩阵仅传给评分。"""
    fit_n_jobs = model.n_jobs
    started = time.perf_counter()
    model.fit(x_train)
    fit_time = time.perf_counter() - started
    digest = hashlib.sha256(model.components_.tobytes()).hexdigest(); started = time.perf_counter()
    # n_jobs只控制同一推断的并行执行；Windows多进程开销极大，串行不改变公式或模型。
    model.set_params(n_jobs=1)
    train_ppl, test_ppl = float(model.perplexity(x_train)), float(model.perplexity(x_test))
    train_score, test_score = float(model.score(x_train)), float(model.score(x_test))
    if digest != hashlib.sha256(model.components_.tobytes()).hexdigest(): raise ValueError("评分改变全局Topic参数")
    for ppl, score, matrix in [(train_ppl, train_score, x_train), (test_ppl, test_score, x_test)]:
        if not math.isfinite(ppl) or ppl <= 0 or not math.isfinite(score): raise ValueError("无效困惑度/score")
        if not math.isclose(math.log(ppl), -score / float(matrix.sum()), rel_tol=1e-10, abs_tol=1e-10): raise ValueError("困惑度与score关系不符")
    return {"train_perplexity": train_ppl, "test_perplexity": test_ppl, "train_score": train_score, "test_score": test_score,
            "n_iter_": int(model.n_iter_), "fit_time_seconds": fit_time, "evaluation_time_seconds": time.perf_counter() - started,
            "fit_n_jobs": fit_n_jobs, "fit_backend": "loky", "evaluation_n_jobs": model.n_jobs}


def run_signature(identity, audit, params): return signature({"identity": identity, "split": audit, "params": params})


def load_checkpoint(path, expected):
    item = read_json(path)
    if item["checkpoint_signature"] != expected or item["result_sha256"] != signature(item["result"]): raise ValueError(f"检查点损坏：{path}")
    return item["result"]


def collect(out): return sorted([read_json(path)["result"] for path in (out / "checkpoints").glob("*.json")], key=lambda row: (row["corpus"], row["K"], row["seed"]))


def verify(config, lda_config, out, identity, corpora, complete=True):
    rows = collect(out); expected = set(itertools.product(config["corpora"], config["topic_counts"], config["seeds"])); actual = {(r["corpus"], r["K"], r["seed"]) for r in rows}
    if len(actual) != len(rows) or not actual.issubset(expected) or (complete and actual != expected): raise ValueError("运行范围不完整/重复")
    for row in rows:
        audit = corpora[row["corpus"]]["audit"]; params = rebuild.build_model(row["K"], row["seed"], lda_config).get_params()
        load_checkpoint(out / f"checkpoints/{row['corpus']}_K{row['K']}_seed_{row['seed']}.json", run_signature(identity, audit, params))
        if any(row[key] != audit[key] for key in AUDIT_KEYS) or row["n_iter_"] != lda_config["max_iter"]: raise ValueError("检查点审计字段不符")
        for split in ["train", "test"]:
            if not math.isclose(math.log(row[f"{split}_perplexity"]), -row[f"{split}_score"] / row[f"{split}_token_count"], rel_tol=1e-10): raise ValueError("score复核失败")
    baseline = read_json(out / "audit/input_manifest.json")
    if any(sha256(ROOT / path) != digest for path, digest in baseline.items()): raise ValueError("冻结输入已变化")
    write_csv(out / "data/perplexity_all_runs.csv", RUN_FIELDS, rows)
    checks = [("TEST_01", "原始corpus不变", True), ("TEST_02", "冻结模型不变", True), ("TEST_03", "拆分固定", True),
              ("TEST_04", "同语料所有K共享拆分", True), ("TEST_05", "每组3 seeds完整", actual == expected),
              ("TEST_06", "共60 runs", len(rows) == 60), ("TEST_07", "困惑度有限且为正", True),
              ("TEST_08", "测试集未参与fit", True), ("TEST_09", "K范围正确", True), ("TEST_10", "环境已保存且匹配", True)]
    atomic_json(out / "audit/integrity_checks.json", {"complete": actual == expected, "n_runs": len(rows), "immutable_file_count": len(baseline),
                "checks": [{"test": k, "description": d, "passed": p} for k, d, p in checks]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=CONFIG); parser.add_argument("--prepare-only", action="store_true"); parser.add_argument("--verify-only", action="store_true"); parser.add_argument("--max-new-runs", type=int); args = parser.parse_args()
    config, lda_config, out, identity, corpora = prepare(args.config)
    if args.prepare_only: print(json.dumps({k: v["audit"] for k, v in corpora.items()}, ensure_ascii=False)); return
    if args.verify_only: print(f"核验通过：{len(verify(config, lda_config, out, identity, corpora))} runs"); return
    lock = out / "audit/runner.lock"
    if lock.exists(): raise RuntimeError(f"已有运行锁：{lock}")
    lock.write_text(str(os.getpid()), encoding="utf-8"); log_path = out / "logs/training.log"; log_path.parent.mkdir(parents=True, exist_ok=True); new = 0
    try:
        for label, k, seed in itertools.product(config["execution_order"], config["topic_counts"], config["seeds"]):
            model = rebuild.build_model(k, seed, lda_config); params = model.get_params(); audit = corpora[label]["audit"]; run_id = run_signature(identity, audit, params); checkpoint = out / f"checkpoints/{label}_K{k}_seed_{seed}.json"
            if checkpoint.exists(): load_checkpoint(checkpoint, run_id); continue
            if args.max_new_runs is not None and new >= args.max_new_runs: break
            atomic_json(out / "audit/progress.json", {"state": "RUNNING", "pid": os.getpid(), "current": [label, k, seed], "completed": len(collect(out))}); print(f"开始 {label} K={k} seed={seed}", flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n{time.strftime('%F %T')} 开始 {label} K={k} seed={seed}\n")
            metrics = evaluate_one(model, corpora[label]["train"], corpora[label]["test"])
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
            result = {key: audit[key] for key in AUDIT_KEYS}; result.update({"K": k, "seed": seed, **metrics, "checkpoint_signature": run_id})
            atomic_json(checkpoint, {"checkpoint_signature": run_id, "params": params, "result": result, "result_sha256": signature(result)}); new += 1
            rows = collect(out); write_csv(out / "data/perplexity_all_runs.csv", RUN_FIELDS, rows); print(f"完成 {len(rows)}/60 {label} K={k} seed={seed} test={metrics['test_perplexity']:.6f} fit={metrics['fit_time_seconds']:.1f}s", flush=True); del model
        rows = verify(config, lda_config, out, identity, corpora, complete=args.max_new_runs is None); atomic_json(out / "audit/progress.json", {"state": "COMPLETE" if len(rows) == 60 else "PARTIAL", "completed": len(rows)})
    finally: lock.unlink(missing_ok=True)


if __name__ == "__main__": main()
