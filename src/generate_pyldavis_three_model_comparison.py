#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用既有 C_K12/C_K15/C_K20 全语料工件生成 pyLDAvis 比较诊断。

本脚本只读模型、主题词分布、文档主题分布、DTM 和词表；不调用 fit、
transform、分词或词表构建。若语料、词表或文档顺序不一致，会在审计后停止。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import inspect
import io
import json
import math
import pickle
import re
import sys
import warnings
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.urls
import sklearn
from matplotlib import font_manager
from scipy.sparse import issparse, load_npz
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "visualization/pyldavis_C_K12_K15_K20"
CONFIG = OUTPUT / "config/pyldavis_three_model_config.yaml"
MODEL_IDS = ("C_K12", "C_K15", "C_K20")
COLORS = {12: "#2B6F8A", 15: "#B46B3B", 20: "#5B7456"}


class ComparabilityBlock(RuntimeError):
    """比较性硬条件不满足时停止。"""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparabilityBlock(message)


def save_csv(rows: list[dict], path: Path, fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        require(bool(rows), f"没有可写入的数据：{path}")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def matrix_fingerprint(matrix) -> str:
    """使用原训练代码的 CSR 逻辑指纹算法。"""
    canonical = matrix.copy().tocsr()
    canonical.sort_indices()
    digest = hashlib.sha256()
    for array in (canonical.shape, canonical.indptr, canonical.indices, canonical.data):
        digest.update(np.asarray(array, dtype=np.int64).tobytes())
    return digest.hexdigest()


def load_saved_array(path: Path, key: str) -> np.ndarray:
    """读取保存的概率数组；兼容 scipy.sparse.save_npz。"""
    if key == "scipy_sparse":
        return load_npz(path).toarray()
    with np.load(path, allow_pickle=False) as archive:
        require(key in archive.files, f"{path} 缺少键 {key}；实际键：{archive.files}")
        return np.asarray(archive[key], dtype=np.float64)


def configure_fonts() -> str:
    candidates = ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"]
    available = {item.name for item in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": chosen, "axes.unicode_minus": False, "figure.dpi": 140, "savefig.dpi": 250})
    return chosen


def artifact_output_hash(meta: dict, filename: str) -> str | None:
    for row in meta.get("outputs", []):
        if Path(row.get("path", "")).name == filename:
            return row.get("sha256")
    return None


def load_and_audit(config: dict) -> tuple[dict, dict, list[dict], list[dict]]:
    """核对共同 C 语料、三个模型及保存概率，不做任何概率推断。"""
    corpus_cfg = config["corpus"]
    corpus_paths = {key: ROOT / corpus_cfg[key] for key in ("dtm", "vocabulary", "document_index", "metadata")}
    for path in corpus_paths.values():
        require(path.is_file(), f"缺少冻结 C corpus 工件：{path}")
    corpus_meta = read_json(corpus_paths["metadata"])
    dtm = load_npz(corpus_paths["dtm"]).tocsr()
    vocab_rows = sorted(read_csv(corpus_paths["vocabulary"]), key=lambda row: int(row["term_index"]))
    document_rows = sorted(read_csv(corpus_paths["document_index"]), key=lambda row: int(row["matrix_row"]))
    vocab = np.asarray([row["token"] for row in vocab_rows], dtype=object)
    doc_lengths = np.asarray(dtm.sum(axis=1)).ravel().astype(np.int64)
    term_frequency = np.asarray(dtm.sum(axis=0)).ravel().astype(np.int64)
    corpus_hashes = {key: sha256(path) for key, path in corpus_paths.items()}
    require(list(dtm.shape) == [43015, 35063], f"冻结 C DTM shape={dtm.shape}，预期 (43015, 35063)")
    require(corpus_meta["label"] == "C" and corpus_meta["min_tf"] == 3 and corpus_meta["min_df"] == 3, "冻结语料不是 C（TF>=3, DF>=3）")
    require(matrix_fingerprint(dtm) == corpus_meta["matrix_fingerprint"], "冻结 DTM 逻辑指纹与 corpus_metadata.json 不一致")
    require(corpus_hashes["dtm"] == corpus_meta["outputs"]["bow_matrix_npz"], "冻结 DTM 文件哈希不一致")
    require(corpus_hashes["vocabulary"] == corpus_meta["outputs"]["vocabulary"], "冻结 vocabulary 文件哈希不一致")
    require(corpus_hashes["document_index"] == corpus_meta["outputs"]["document_index"], "冻结 document_index 文件哈希不一致")
    require([int(row["term_index"]) for row in vocab_rows] == list(range(dtm.shape[1])), "vocabulary term_index 不连续")
    require([int(row["matrix_row"]) for row in document_rows] == list(range(dtm.shape[0])), "document_index matrix_row 不连续")
    require(np.array_equal(term_frequency, np.asarray([int(row["TF"]) for row in vocab_rows])), "DTM 列词频与 vocabulary TF 不一致")
    require(np.array_equal(doc_lengths, np.asarray([int(row["token_count"]) for row in document_rows])), "DTM 行长度与 document_index token_count 不一致")
    require((doc_lengths > 0).all(), "冻结 DTM 含零长度文档")

    bundles: dict[str, dict] = {}
    audits: list[dict] = []
    source_hash_rows: list[dict] = []
    common_model_params = None
    for model_id in MODEL_IDS:
        model_cfg = config["models"][model_id]
        paths = {key: ROOT / model_cfg[key] for key in ("model", "topic_term", "doc_topic", "metadata", "document_order_evidence")}
        for path in paths.values():
            require(path.is_file(), f"{model_id} 缺少工件：{path}")
        path_hashes = {key: sha256(path) for key, path in paths.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            lda = joblib.load(paths["model"])
        require(isinstance(lda, LatentDirichletAllocation), f"{paths['model']} 不是 sklearn LDA")
        topic_term_saved = load_saved_array(paths["topic_term"], model_cfg["topic_term_key"])
        theta = load_saved_array(paths["doc_topic"], model_cfg["doc_topic_key"])
        K = int(model_cfg["K"])
        meta = read_json(paths["metadata"])
        require(lda.components_.shape == (K, dtm.shape[1]), f"{model_id} LDA components shape={lda.components_.shape}")
        require(topic_term_saved.shape == (K, dtm.shape[1]), f"{model_id} topic-term shape={topic_term_saved.shape}")
        require(theta.shape == (dtm.shape[0], K), f"{model_id} doc-topic shape={theta.shape}")
        require(lda.n_components == K and int(lda.random_state) == int(model_cfg["seed"]), f"{model_id} 模型 K 或 seed 与配置不一致")
        require(lda.learning_method == "batch" and lda.max_iter == 30 and lda.n_iter_ == 30, f"{model_id} 不是 max_iter=30 的 Batch LDA")
        require(np.isfinite(topic_term_saved).all() and (topic_term_saved >= 0).all(), f"{model_id} topic-term 含非法值")
        require(np.isfinite(theta).all() and (theta >= 0).all() and np.allclose(theta.sum(axis=1), 1, atol=1e-10), f"{model_id} doc-topic 行和不为 1")
        normalized_components = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        if np.allclose(topic_term_saved.sum(axis=1), 1, atol=1e-10):
            phi = topic_term_saved
        else:
            require(np.array_equal(topic_term_saved, lda.components_), f"{model_id} 保存的 components 与模型不一致")
            phi = topic_term_saved / topic_term_saved.sum(axis=1, keepdims=True)
        require(np.allclose(phi, normalized_components, atol=2e-12, rtol=2e-12), f"{model_id} 保存的 topic-term 与模型 components 不一致")
        params = lda.get_params()
        comparable_params = {key: params[key] for key in ("learning_method", "max_iter", "evaluate_every", "perp_tol", "mean_change_tol", "max_doc_update_iter", "doc_topic_prior", "topic_word_prior")}
        if common_model_params is None:
            common_model_params = comparable_params
        require(comparable_params == common_model_params, f"{model_id} prior/max_iter convention 与其他模型不一致")

        # K20 必须是冻结副本；其清单哈希须逐项吻合。
        if model_id == "C_K20":
            require(meta.get("freeze_status") == "passed", "C_K20 未标记为冻结通过")
            require(path_hashes["model"] == meta["frozen_sha256"]["model.joblib"], "C_K20 frozen model 哈希不一致")
            require(path_hashes["topic_term"] == meta["frozen_sha256"]["topic_word_probabilities.npz"], "C_K20 frozen topic-term 哈希不一致")
            require(path_hashes["doc_topic"] == meta["frozen_sha256"]["document_topic_probabilities.npz"], "C_K20 frozen theta 哈希不一致")
            doc_order_hash = sha256(paths["document_order_evidence"])
            require(doc_order_hash == corpus_hashes["document_index"], "C_K20 document order 与 C corpus 不一致")
        elif model_id == "C_K12":
            require(meta["fit_scope"] == "FULL_CORPUS", "C_K12 不是 FULL_CORPUS")
            require(meta["DTM_file_sha256"] == corpus_hashes["dtm"] and meta["vocab_sha256"] == corpus_hashes["vocabulary"], "C_K12 corpus/vocab 哈希不一致")
            require(path_hashes["model"] == meta["model_sha256"], "C_K12 model 哈希不一致")
            require(path_hashes["topic_term"] == artifact_output_hash(meta, "components.npz"), "C_K12 components 文件哈希不一致")
            require(path_hashes["doc_topic"] == artifact_output_hash(meta, "document_topic_full.npz"), "C_K12 theta 文件哈希不一致")
            map_rows = sorted(read_csv(paths["document_order_evidence"]), key=lambda row: int(row["matrix_row"]))
            require([row["poem_id"] for row in map_rows] == [row["poem_id"] for row in document_rows], "C_K12 document order 与 C corpus 不一致")
            doc_order_hash = corpus_hashes["document_index"]
        else:
            require(meta["run_type"] == "full_corpus_model_selection_diagnostic" and int(meta["documents"]) == dtm.shape[0], "C_K15 不是 FULL_CORPUS")
            require(path_hashes["model"] == artifact_output_hash(meta, "model.joblib"), "C_K15 model 哈希不一致")
            require(path_hashes["topic_term"] == artifact_output_hash(meta, "topic_word_probabilities.npz"), "C_K15 topic-term 文件哈希不一致")
            require(path_hashes["doc_topic"] == artifact_output_hash(meta, "document_topic_probabilities.npz"), "C_K15 theta 文件哈希不一致")
            manifest = [row for row in read_csv(paths["document_order_evidence"]) if int(row["K"]) == 15][0]
            require(manifest["fit_scope"] == "FULL_CORPUS" and manifest["corpus_hash"] == corpus_meta["matrix_fingerprint"] and manifest["vocab_hash"] == corpus_hashes["vocabulary"], "C_K15 共同语料/词表证据不一致")
            doc_order_hash = corpus_hashes["document_index"]

        for kind, path in paths.items():
            source_hash_rows.append({"model_id": model_id, "artifact_type": kind, "path": str(path.resolve()), "sha256_before": path_hashes[kind], "sha256_after": "", "unchanged": ""})
        audits.append({
            "model_id": model_id, "K": K, "role": model_cfg["role"], "fit_scope": "FULL_CORPUS",
            "corpus": "C", "preprocessing": "冻结 C corpus；TF>=3；DF>=3；non-empty poems",
            "dtm_shape": f"{dtm.shape[0]}x{dtm.shape[1]}", "lda_components_shape": f"{lda.components_.shape[0]}x{lda.components_.shape[1]}",
            "topic_term_shape": f"{phi.shape[0]}x{phi.shape[1]}", "doc_topic_shape": f"{theta.shape[0]}x{theta.shape[1]}",
            "topic_term_probability_source": "保存概率矩阵" if np.allclose(topic_term_saved.sum(axis=1), 1, atol=1e-10) else "保存 components 逐行归一化（无 fit/transform）",
            "dtm_sha256": corpus_hashes["dtm"], "dtm_fingerprint": corpus_meta["matrix_fingerprint"],
            "vocabulary_sha256": corpus_hashes["vocabulary"], "document_order_sha256": doc_order_hash,
            "prior_regime": "ORIGINAL_PRIOR / sklearn defaults", "effective_alpha": 1 / K, "effective_eta": 1 / K,
            "max_iter_convention": "max_iter=30; n_iter_=30; learning_method=batch", "seed": model_cfg["seed"],
            "representative_seed_selection_method": model_cfg["representative_seed_method"],
            "comparability_status": "COMPARABLE_WITH_EXPECTED_K_DEPENDENT_DEFAULT_PRIOR",
            "comparability_warning": "REPRESENTATIVE_SEED_AND_SELECTION_METHOD_DIFFER；有效默认先验为 1/K",
        })
        bundles[model_id] = {"K": K, "model": lda, "phi": phi, "theta": theta, "paths": paths, "hashes": path_hashes, "config": model_cfg}

    require(len({row["dtm_sha256"] for row in audits}) == 1, "BLOCK：三模型 DTM 不一致")
    require(len({row["vocabulary_sha256"] for row in audits}) == 1, "BLOCK：三模型 vocabulary 不一致")
    require(len({row["document_order_sha256"] for row in audits}) == 1, "BLOCK：三模型 document order 不一致")
    corpus = {"dtm": dtm, "vocab": vocab, "vocab_rows": vocab_rows, "documents": document_rows,
              "doc_lengths": doc_lengths, "term_frequency": term_frequency, "paths": corpus_paths,
              "hashes": corpus_hashes, "metadata": corpus_meta}
    return corpus, bundles, audits, source_hash_rows


def make_document_alignment_qc(corpus: dict) -> list[dict]:
    rows = []
    for row in corpus["documents"]:
        rows.append({
            "matrix_row": int(row["matrix_row"]), "corpus_row": int(row["corpus_row"]), "poem_id": row["poem_id"],
            "C_K12_poem_id": row["poem_id"], "C_K15_poem_id": row["poem_id"], "C_K20_poem_id": row["poem_id"],
            "all_models_same_document": True,
            "alignment_evidence": "K12 行映射逐行核验；K15 full-corpus manifest 与共同 C DTM 指纹核验；K20 frozen document_index 逐文件哈希核验",
        })
    return rows


def make_prevalence(theta: np.ndarray, lengths: np.ndarray, K: int) -> list[dict]:
    mean_prob = theta.mean(axis=0)
    token_mass = theta.T @ lengths
    token_share = token_mass / token_mass.sum()
    dominant = np.bincount(theta.argmax(axis=1), minlength=K)
    return [{
        "K": K, "topic_id": f"T{topic + 1:02d}", "mean_topic_probability": float(mean_prob[topic]),
        "percentage_topic_mass": float(token_share[topic] * 100), "dominant_poem_count": int(dominant[topic]),
        "dominant_poem_share": float(dominant[topic] / theta.shape[0]),
    } for topic in range(K)]


def relevance_rows(phi: np.ndarray, term_frequency: np.ndarray, vocab: np.ndarray, K: int) -> list[dict]:
    corpus_prob = term_frequency / term_frequency.sum()
    rows = []
    tiny = np.finfo(np.float64).tiny
    for topic in range(K):
        log_phi = np.log(np.maximum(phi[topic], tiny))
        lift = log_phi - np.log(np.maximum(corpus_prob, tiny))
        for lambda_value in (1.0, 0.6, 0.2):
            score = lambda_value * log_phi + (1 - lambda_value) * lift
            order = np.argsort(-score, kind="stable")[:30]
            for rank, index in enumerate(order, 1):
                rows.append({"K": K, "topic_id": f"T{topic + 1:02d}", "lambda": lambda_value, "rank": rank,
                             "term": vocab[index], "topic_probability": float(phi[topic, index]),
                             "corpus_probability": float(corpus_prob[index]), "relevance_score": float(score[index])})
    return rows


def similarity_rows(phi: np.ndarray, K: int) -> tuple[list[dict], list[dict], dict]:
    norms = np.linalg.norm(phi, axis=1, keepdims=True)
    cosine = (phi / norms) @ (phi / norms).T
    top20 = [set(np.argsort(-phi[i], kind="stable")[:20]) for i in range(K)]
    top30 = [set(np.argsort(-phi[i], kind="stable")[:30]) for i in range(K)]
    pairs = []
    for left in range(K):
        for right in range(left + 1, K):
            pairs.append({"K": K, "topic_A": f"T{left + 1:02d}", "topic_B": f"T{right + 1:02d}",
                          "cosine_similarity": float(cosine[left, right]),
                          "top20_jaccard": len(top20[left] & top20[right]) / len(top20[left] | top20[right]),
                          "top30_jaccard": len(top30[left] & top30[right]) / len(top30[left] | top30[right])})
    nearest = []
    for topic in range(K):
        values = cosine[topic].copy(); values[topic] = -np.inf
        neighbor = int(np.argmax(values))
        nearest.append({"K": K, "topic_id": f"T{topic + 1:02d}", "nearest_topic_id": f"T{neighbor + 1:02d}",
                        "nearest_topic_cosine": float(cosine[topic, neighbor])})
    values = np.asarray([row["cosine_similarity"] for row in pairs])
    stats = {"mean_pairwise_cosine": float(values.mean()), "median_pairwise_cosine": float(np.median(values)),
             "max_pairwise_cosine": float(values.max()), "p90_pairwise_cosine": float(np.quantile(values, .90)),
             "p95_pairwise_cosine": float(np.quantile(values, .95)),
             "pairs_cos_ge_0_80": int((values >= .80).sum()), "pairs_cos_ge_0_85": int((values >= .85).sum()),
             "pairs_cos_ge_0_90": int((values >= .90).sum())}
    return pairs, nearest, stats


def topic_labels(config: dict) -> list[dict]:
    sources = {key: ROOT / value for key, value in config["semantic_sources"].items()}
    mapping = {row["blind_topic_id"]: row for row in read_csv(sources["blind_topic_map"])}
    rows = []
    for review in read_csv(sources["topic_ai_review"]):
        mapped = mapping[review["blind_topic_id"]]
        K = int(mapped["true_K"])
        if K in (12, 15):
            rows.append({"K": K, "topic_id": f"T{int(mapped['true_topic_id']):02d}", "topic_label": review["AI_provisional_label"],
                         "label_status": "AI_PROVISIONAL", "source": str(sources["topic_ai_review"].resolve()),
                         "researcher_label_if_available": "", "review_status": review["AI_review_status"]})
    for taxonomy in read_csv(sources["k20_taxonomy"]):
        topic = int(taxonomy["topic_number"])
        rows.append({"K": 20, "topic_id": f"T{topic:02d}", "topic_label": taxonomy["final_topic_label"],
                     "label_status": "RESEARCHER_CONFIRMED_FROZEN_TAXONOMY", "source": str(sources["k20_taxonomy"].resolve()),
                     "researcher_label_if_available": taxonomy["final_topic_label"], "review_status": "RESEARCHER_CONFIRMED"})
    rows.sort(key=lambda row: (row["K"], row["topic_id"]))
    require(len(rows) == 47, f"Topic label reference 应有 47 行，实际 {len(rows)}")
    return rows


def alignment_reference(config: dict, bundles: dict) -> list[dict]:
    sources = {key: ROOT / value for key, value in config["semantic_sources"].items()}
    mapping = {row["blind_topic_id"]: row for row in read_csv(sources["blind_topic_map"])}
    events = read_csv(sources["split_merge"])

    def event_for(source_blind: str, source_topic: str, target_blind: str, target_topic: str) -> str:
        for event in events:
            if event["source_blind_model_id"] == source_blind and event["target_blind_model_id"] == target_blind:
                if source_topic in event["source_blind_topic_ids"].split("|") and target_topic in event["target_blind_topic_ids"].split("|"):
                    return event["AI_semantic_judgment"]
            if event["source_blind_model_id"] == target_blind and event["target_blind_model_id"] == source_blind:
                if target_topic in event["source_blind_topic_ids"].split("|") and source_topic in event["target_blind_topic_ids"].split("|"):
                    return event["AI_semantic_judgment"]
        return ""

    rows = []
    for alignment in read_csv(sources["alignment"]):
        if not alignment["source_blind_topic_id"] or not alignment["target_blind_topic_id"]:
            continue
        if alignment["source_blind_topic_id"] not in mapping or alignment["target_blind_topic_id"] not in mapping:
            continue
        s = mapping[alignment["source_blind_topic_id"]]
        t = mapping[alignment["target_blind_topic_id"]]
        sK, tK = int(s["true_K"]), int(t["true_K"])
        if {sK, tK} not in ({12, 15}, {15, 20}):
            continue
        source_blind, target_blind = alignment["source_blind_model_id"], alignment["target_blind_model_id"]
        source_bt, target_bt = alignment["source_blind_topic_id"], alignment["target_blind_topic_id"]
        source_topic, target_topic = int(s["true_topic_id"]), int(t["true_topic_id"])
        relation = alignment["relation_type"]
        if sK > tK:
            sK, tK = tK, sK
            source_topic, target_topic = target_topic, source_topic
            source_blind, target_blind = target_blind, source_blind
            source_bt, target_bt = target_bt, source_bt
            relation = {"SPLIT": "MERGE", "MERGE": "SPLIT"}.get(relation, relation)
        left = bundles[f"C_K{sK}"]["phi"][source_topic - 1]
        right = bundles[f"C_K{tK}"]["phi"][target_topic - 1]
        cosine = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))
        li = set(np.argsort(-left, kind="stable")[:20]); ri = set(np.argsort(-right, kind="stable")[:20])
        jaccard = len(li & ri) / len(li | ri)
        rows.append({"source_K": sK, "source_topic": f"T{source_topic:02d}", "target_K": tK,
                     "target_topic": f"T{target_topic:02d}", "relation_type": relation,
                     "topic_word_cosine": cosine, "top20_jaccard": jaccard,
                     "semantic_judgment": event_for(source_blind, source_bt, target_blind, target_bt),
                     "review_status": alignment["AI_review_status"]})
    rows.sort(key=lambda row: (row["source_K"], row["source_topic"], row["target_topic"], row["relation_type"]))
    require(any(row["source_K"] == 12 and row["target_K"] == 15 for row in rows), "缺少既有 K12↔K15 alignment")
    require(any(row["source_K"] == 15 and row["target_K"] == 20 for row in rows), "缺少既有 K15↔K20 alignment")
    return rows


def save_offline_html(vis, model_id: str, destination: Path, documents: int, vocabulary: int) -> None:
    """用官方模板生成后，仅内嵌发行包资源并增加标题、说明和字体。"""
    assets = {"d3": Path(pyLDAvis.urls.D3_LOCAL), "ldavis": Path(pyLDAvis.urls.LDAVIS_LOCAL), "css": Path(pyLDAvis.urls.LDAVIS_CSS_LOCAL)}
    for path in assets.values():
        require(path.is_file(), f"缺少 pyLDAvis 本地资源：{path}")
    fragment = io.StringIO()
    pyLDAvis.save_html(vis, fragment, template_type="simple", visid=f"qts_{model_id}",
                       d3_url="qts_d3.js", ldavis_url="qts_ldavis.js", ldavis_css_url="qts_ldavis.css")
    content = fragment.getvalue()
    for kind, url in (("d3", "qts_d3.js"), ("ldavis", "qts_ldavis.js")):
        source = re.sub(r"</script", r"<\\/script", assets[kind].read_text(encoding="utf-8"), flags=re.I)
        tag = f'<script type="text/javascript" src="{url}"></script>'
        require(content.count(tag) == 1, "pyLDAvis HTML 模板变化，无法安全内嵌 JS")
        content = content.replace(tag, f'<script type="text/javascript">\n{source}\n</script>')
    css_tag = '<link rel="stylesheet" type="text/css" href="qts_ldavis.css">'
    require(content.count(css_tag) == 1, "pyLDAvis HTML 模板变化，无法安全内嵌 CSS")
    content = content.replace(css_tag, "<style>\n" + assets["css"].read_text(encoding="utf-8") + "\n</style>")
    K = int(model_id.split("K")[1])
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:">
<title>《全唐诗》LDA Topic Visualization — {model_id}</title>
<style>body{{font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","SimHei",Arial,sans-serif;margin:22px;color:#213547}}h1{{font-size:24px}}p{{line-height:1.65}}.chart{{overflow-x:auto}}footer{{color:#667085;font-size:13px}}</style>
</head><body><h1>《全唐诗》LDA Topic Visualization — {model_id}</h1>
<p>保留原模型 T01–T{K:02d}：pyLDAvis Topic 1–{K}；<strong>sort_topics=False</strong>。点击圆圈，再比较 λ=1.0、约0.6、约0.2 的 Top 30 relevant terms。</p>
<p>{documents:,} 首非空诗，{vocabulary:,} 个词项。圆面积表示主题 prevalence，并不表示文学重要性或模型质量；圆间二维距离是模型内部高维主题距离的 PCoA 投影。</p>
<div class="chart">{content}</div>
<footer>每个 K 独立投影，PC 坐标不可跨模型直接比较。主题接近、分离或重叠均建议结合代表诗与既有语义审核人工复核。</footer>
</body></html>'''
    require(not re.search(r'<script\b[^>]*\bsrc\s*=|<link\b[^>]*\bhref\s*=', page, re.I), "HTML 仍包含外部脚本或样式引用")
    destination.write_text(page, encoding="utf-8")


def make_comparison_html(summaries: list[dict], destination: Path) -> None:
    cards = []
    roles = {12: "低分辨率强候选模型", 15: "中分辨率模型", 20: "冻结高分辨率参考模型"}
    for row in summaries:
        K = int(row["K"])
        cards.append(f'''<article><h2>C_K{K}</h2><p>{roles[K]}</p>
<dl><dt>最大/最小/中位 prevalence</dt><dd>{row['largest_topic_prevalence']:.2f}% / {row['smallest_topic_prevalence']:.2f}% / {row['median_topic_prevalence']:.2f}%</dd>
<dt>mean/max nearest cosine</dt><dd>{row['mean_nearest_neighbor_cosine']:.4f} / {row['max_nearest_neighbor_cosine']:.4f}</dd></dl>
<a href="C_K{K}_pyLDAvis.html">打开 C_K{K} pyLDAvis</a></article>''')
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>《全唐诗》C_K12 / C_K15 / C_K20 pyLDAvis 比较入口</title><style>
body{{font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC","SimHei",Arial,sans-serif;margin:0;background:#f3f5f3;color:#1f2937}}main{{max-width:1100px;margin:auto;padding:34px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}}article,.guide,.note{{background:white;border:1px solid #d8ded8;border-radius:12px;padding:20px;box-shadow:0 4px 14px #2333}}a{{display:inline-block;background:#315f72;color:white;padding:10px 14px;border-radius:7px;text-decoration:none}}dt{{font-weight:700;margin-top:10px}}dd{{margin-left:0}}li{{margin:8px 0;line-height:1.5}}.note{{margin:18px 0;border-left:5px solid #B46B3B}}</style></head>
<body><main><h1>《全唐诗》C_K12 / C_K15 / C_K20 pyLDAvis 比较诊断</h1>
<div class="note">这是视觉诊断入口，不给 K 排名，也不提供“最优 K”结论。三幅 PCoA 图各自独立，坐标不可跨模型直接比较；圆面积只表示 prevalence。</div>
<section class="cards">{''.join(cards)}</section>
<section class="guide"><h2>阅读指南</h2><ol><li>分别打开三个模型，先确认 Topic ID 与原模型一致。</li><li>观察 bubble 的面积，了解各 Topic 的相对 prevalence。</li><li>只在同一模型内观察圆的距离、邻近和重叠。</li><li>点击 Topic，先看 λ=1.0 的高概率词。</li><li>再看 λ≈0.6 和 λ≈0.2，比较区分度较高的词。</li><li>结合代表诗、既有语义审核和研究者标签理解；高相似只记为“建议人工复核”。</li></ol></section>
</main></body></html>'''
    destination.write_text(page, encoding="utf-8")


def make_figures(coords: dict, prevalence: dict, pair_rows: list[dict], nearest_rows: list[dict], summaries: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for K in (12, 15, 20):
        rows = coords[K]
        x = np.asarray([r["PC1"] for r in rows]); y = np.asarray([r["PC2"] for r in rows]); sizes = np.asarray([r["topic_prevalence"] for r in rows])
        fig, ax = plt.subplots(figsize=(8.2, 7))
        ax.scatter(x, y, s=80 + sizes * 130, c=COLORS[K], alpha=.68, edgecolor="white", linewidth=1.2)
        for row in rows: ax.text(row["PC1"], row["PC2"], row["topic_id"], ha="center", va="center", fontsize=8)
        ax.axhline(0, color="#c7cec9", lw=.8); ax.axvline(0, color="#c7cec9", lw=.8)
        ax.set(title=f"C_K{K} Topic Distance Map（独立 PCoA）", xlabel="PC1", ylabel="PC2")
        ax.text(.01, -.11, "圆面积 = topic prevalence；坐标仅用于本模型内部诊断", transform=ax.transAxes, fontsize=9, color="#5f6b66")
        fig.tight_layout(); fig.savefig(output / f"C_K{K}_topic_distance_map.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, K in zip(axes, (12, 15, 20)):
        vals = sorted(prevalence[K], key=lambda r: r["percentage_topic_mass"], reverse=True)
        ax.bar(range(K), [r["percentage_topic_mass"] for r in vals], color=COLORS[K]); ax.set_title(f"C_K{K}")
        ax.set_xticks(range(K)); ax.set_xticklabels([r["topic_id"] for r in vals], rotation=90, fontsize=7); ax.set_xlabel("Topic（按 prevalence 降序展示）")
    axes[0].set_ylabel("Topic mass (%)"); fig.suptitle("C_K12 / C_K15 / C_K20 Topic Prevalence Comparison")
    fig.tight_layout(); fig.savefig(output / "C_K12_K15_K20_topic_prevalence_comparison.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [[r["cosine_similarity"] for r in pair_rows if r["K"] == K] for K in (12, 15, 20)]
    ax.violinplot(data, showmeans=True, showmedians=True); ax.set_xticks((1, 2, 3), ("K12", "K15", "K20")); ax.set_ylabel("Topic-word cosine")
    ax.set_title("Within-model Topic-pair Similarity Distribution"); fig.tight_layout(); fig.savefig(output / "C_K12_K15_K20_topic_pair_similarity_distribution.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    offset = 0
    ticks, labels = [], []
    for K in (12, 15, 20):
        subset = [r for r in nearest_rows if r["K"] == K]
        xs = np.arange(K) + offset; ax.scatter(xs, [r["nearest_topic_cosine"] for r in subset], color=COLORS[K], label=f"K{K}", s=28)
        ticks.extend(xs); labels.extend([r["topic_id"] for r in subset]); offset += K + 2
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=90, fontsize=6.5); ax.set_ylabel("Nearest-topic cosine"); ax.legend(); ax.set_title("Nearest Topic Similarity")
    fig.tight_layout(); fig.savefig(output / "C_K12_K15_K20_nearest_topic_similarity.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2)); xs = np.arange(3); width = .23
    for i, threshold in enumerate(("pairs_cos_ge_0_80", "pairs_cos_ge_0_85", "pairs_cos_ge_0_90")):
        positions = xs + (i - 1) * width
        values = [r[threshold] for r in summaries]
        ax.bar(positions, values, width, label=threshold[-4:].replace("_", "."))
        for position, value in zip(positions, values):
            ax.text(position, max(value, 0) + .03, str(value), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs, ("K12", "K15", "K20")); ax.set_ylabel("Topic pair count"); ax.legend(title="cos ≥"); ax.set_title("High-similarity Topic Pair Counts")
    maximum = max(r[key] for r in summaries for key in ("pairs_cos_ge_0_80", "pairs_cos_ge_0_85", "pairs_cos_ge_0_90"))
    ax.set_ylim(0, max(1, maximum * 1.2))
    fig.tight_layout(); fig.savefig(output / "C_K12_K15_K20_high_similarity_pair_counts.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8)); Ks = [12, 15, 20]
    axes[0].plot(Ks, [r["mean_nearest_neighbor_cosine"] for r in summaries], marker="o", label="Mean nearest")
    axes[0].plot(Ks, [r["max_nearest_neighbor_cosine"] for r in summaries], marker="o", label="Max nearest"); axes[0].set(xlabel="K", ylabel="Cosine", title="Nearest-neighbor structure"); axes[0].legend()
    axes[1].plot(Ks, [r["pairs_cos_ge_0_85"] for r in summaries], marker="o", color="#8F4E38", label="pairs cos≥0.85")
    axes[1].plot(Ks, [r["small_topics_below_2pct"] for r in summaries], marker="s", color="#596B86", label="topics <2%")
    axes[1].set(xlabel="K", ylabel="Count", title="Resolution diagnostic signals"); axes[1].legend()
    fig.suptitle("C_K12 / C_K15 / C_K20 Resolution Diagnostic（不构成 K 评分）")
    fig.tight_layout(); fig.savefig(output / "C_K12_K15_K20_resolution_diagnostic.png", bbox_inches="tight"); plt.close(fig)


def make_report(audits: list[dict], summaries: list[dict], diagnostic: list[dict], alignments: list[dict], status: str) -> str:
    by_k = {int(row["K"]): row for row in summaries}
    def top_terms(K: int, topic: str, column: str) -> str:
        return next(row[column] for row in diagnostic if row["K"] == K and row["topic_id"] == topic)
    def model_facts(K: int) -> str:
        row = by_k[K]
        return (f"最大 prevalence {row['largest_topic_id']}={row['largest_topic_prevalence']:.3f}%；最小 {row['smallest_topic_id']}={row['smallest_topic_prevalence']:.3f}%；"
                f"中位数 {row['median_topic_prevalence']:.3f}%。mean/max nearest cosine={row['mean_nearest_neighbor_cosine']:.4f}/{row['max_nearest_neighbor_cosine']:.4f}。")
    nn = [by_k[K]["mean_nearest_neighbor_cosine"] for K in (12, 15, 20)]
    high = [by_k[K]["pairs_cos_ge_0_85"] for K in (12, 15, 20)]
    if nn[0] <= nn[1] <= nn[2] and high[0] <= high[1] <= high[2] and (nn[2] > nn[0] or high[2] > high[0]): signal = "YES"
    elif nn[0] >= nn[1] >= nn[2] and high[0] >= high[1] >= high[2]: signal = "NO"
    else: signal = "MIXED"
    event_counts = {}
    for row in alignments:
        if row["semantic_judgment"]:
            event_counts[row["semantic_judgment"]] = event_counts.get(row["semantic_judgment"], 0) + 1
    warning = audits[0]["comparability_warning"]
    return f'''# 1. Objective

本任务比较 C_K12、C_K15 与冻结 C_K20 的 pyLDAvis 视觉诊断，用于检查主题占比、模型内部邻近/重叠和不同 λ 下的 relevant terms。它不选择 K，不生成综合评分，也不改写既有文学判断。

# 2. Model Comparability

三模型均为 `FULL_CORPUS`，共用冻结 C corpus（TF≥3、DF≥3）、43,015 行 DTM、35,063 词项及相同文档顺序。Batch LDA 和 max_iter=30 约定一致。审计警告：`{warning}`。K12 seed=101；K15/K20 seed=2026。K20 读取冻结 reference。

# 3. What pyLDAvis Shows

左侧 bubble 面积表示词次加权 prevalence；圆间位置是本模型 topic-term 距离的二维 PCoA 投影。右侧显示点击 Topic 的 relevant terms。λ=1.0 偏重主题内概率，λ≈0.6 作为主要观察点，λ≈0.2 偏重相对区分度；0.6 不是“最优 λ”。三页均设 `sort_topics=False`。

# 4. C_K12 Visualization

{model_facts(12)}这些是概率与相似度事实；圆的视觉重叠需由研究者打开 HTML 后复核。

# 5. C_K15 Visualization

{model_facts(15)}这些事实不构成对 K15 的优劣判断。

# 6. C_K20 Visualization

{model_facts(20)}该页使用冻结 C_K20 工件，没有重新训练或重新推断 theta。

# 7. Topic Prevalence Comparison

K12/K15/K20 的最大 Topic mass 分别为 {by_k[12]['largest_topic_prevalence']:.3f}%、{by_k[15]['largest_topic_prevalence']:.3f}%、{by_k[20]['largest_topic_prevalence']:.3f}%。小于 1% 的 Topic 数为 {by_k[12]['small_topics_below_1pct']}、{by_k[15]['small_topics_below_1pct']}、{by_k[20]['small_topics_below_1pct']}；小于 2% 为 {by_k[12]['small_topics_below_2pct']}、{by_k[15]['small_topics_below_2pct']}、{by_k[20]['small_topics_below_2pct']}。Prevalence 不是文学重要性。

# 8. High-Dimensional Topic Separation

模型内所有 Topic pair 的 mean/median/max cosine：K12={by_k[12]['mean_pairwise_cosine']:.4f}/{by_k[12]['median_pairwise_cosine']:.4f}/{by_k[12]['max_pairwise_cosine']:.4f}；K15={by_k[15]['mean_pairwise_cosine']:.4f}/{by_k[15]['median_pairwise_cosine']:.4f}/{by_k[15]['max_pairwise_cosine']:.4f}；K20={by_k[20]['mean_pairwise_cosine']:.4f}/{by_k[20]['median_pairwise_cosine']:.4f}/{by_k[20]['max_pairwise_cosine']:.4f}。Top20/Top30 Jaccard 已逐 pair 输出。

# 9. Nearest-Neighbor Structure

mean nearest-neighbor cosine 按 K12→K15→K20 为 {nn[0]:.4f}→{nn[1]:.4f}→{nn[2]:.4f}。最高 pair 分别为 {by_k[12]['max_nearest_pair']}、{by_k[15]['max_nearest_pair']}、{by_k[20]['max_nearest_pair']}。

# 10. K12 → K15 Resolution Change

本报告引用既有 blind semantic audit 的 K12↔K15 alignment，并保留其 relation_type 和 AI_PROVISIONAL 状态。二维坐标各自独立，因此没有用 PC 坐标计算跨模型距离。研究者应结合 alignment、cosine、λ=0.6 terms 与代表诗填写 checklist。

# 11. K15 → K20 Resolution Change

K15↔K20 alignment 同样直接引用既有审核。已有 `MEANINGFUL_SPLIT`、`MINOR_REFINEMENT`、`FRAGMENTATION` 或其他事件时，pyLDAvis 仅提供辅助视觉证据，不覆盖原判断。

# 12. Potential Fragmentation Signals

诊断结论为 **{signal}**。量化依据：mean nearest cosine 为 {nn[0]:.4f}→{nn[1]:.4f}→{nn[2]:.4f}；cos≥0.85 pair 数为 {high[0]}→{high[1]}→{high[2]}。这只记为 `POTENTIAL_FRAGMENTATION_SIGNAL`，不等同文学意义上的碎片化。

# 13. Relation to Humanities Semantic Audit

alignment 来源为现有四模型盲审数据；K12/K15 标签保持 `AI_PROVISIONAL`，K20 标签来自已冻结的人工 taxonomy。事件在输出 alignment 中按原状态保留。带判断的 pair 计数为：{json.dumps(event_counts, ensure_ascii=False)}。

# 14. Visual–Semantic Agreements

当前自动流程没有替研究者判断二维 bubble 是否构成视觉 agreement。高维 cosine、Top-word overlap 和现有语义判断已并列输出，供人工确认。

# 15. Visual–Semantic Disagreements

当前没有把二维距离自动解释为 `VISUAL_SEMANTIC_DISAGREEMENT`。如研究者观察到 FRAGMENTATION pair 的 bubble 较远，或 MEANINGFUL_SPLIT pair 高度邻近，应在 checklist 记录，且保留原语义判断。

# 16. Limitations

1. pyLDAvis 二维位置是高维距离投影。
2. 不同 K 的 PC 坐标不能直接跨模型比较。
3. bubble 大小只是 prevalence，不是质量或文学重要性。
4. Topic 接近不等于文学意义重复。
5. Topic 分离不等于文学意义必然不同。
6. relevant terms 不能替代代表诗阅读。
7. pyLDAvis 不能单独决定最优 K。

# 17. Researcher Review Checklist

研究者核查表位于 `researcher/K12_K15_K20_pyldavis_researcher_checklist.xlsx`。视觉分离、重叠、词项区分度和 notes 字段保持空白，等待研究者填写。运行状态：`{status}`。
'''


def make_readme() -> str:
    return '''# 《全唐诗》C_K12 / C_K15 / C_K20 pyLDAvis 比较诊断

先打开 `html/C_K12_K15_K20_pyLDAvis_comparison.html`，再分别进入 K12、K15、K20。三个模型页面均为单文件离线 HTML。

阅读 relevant terms 时，建议先看 λ=1.0，再看 λ≈0.6，最后看 λ≈0.2。每个模型的 PCoA 坐标独立，不可跨模型直接比较。圆面积仅表示 prevalence；主题邻近、重叠或分离需要结合代表诗和既有语义审核人工复核。

复现命令：

```powershell
python src/generate_pyldavis_three_model_comparison.py
```

脚本仅读取配置列明的冻结/既有工件，不执行 LDA fit、transform、重新分词或词表构建。生成后请运行 `python src/verify_pyldavis_three_model_outputs.py` 做静态 QA，并查看 `data/pyldavis_three_model_qa.csv`。
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    config = read_json(args.config.resolve())
    for subdir in ("html", "prepared", "data", "figures", "researcher", "audit"):
        (OUTPUT / subdir).mkdir(parents=True, exist_ok=True)
    font = configure_fonts()
    print("[1/8] 审计共同语料与三套既有模型工件……", flush=True)
    corpus, bundles, audits, source_hash_rows = load_and_audit(config)
    save_csv(audits, OUTPUT / "data/pyldavis_three_model_comparability_audit.csv")
    save_csv(make_document_alignment_qc(corpus), OUTPUT / "data/pyldavis_three_model_document_alignment_qc.csv")

    labels = topic_labels(config)
    save_csv(labels, OUTPUT / "data/topic_label_reference.csv")
    label_lookup = {(row["K"], row["topic_id"]): row for row in labels}
    all_pair, all_nearest, summaries, diagnostics, topic_map = [], [], [], [], []
    coords_by_k, prevalence_by_k = {}, {}
    relevant_by_k = {}

    for model_id in MODEL_IDS:
        bundle = bundles[model_id]; K = bundle["K"]
        print(f"[2/8] {model_id}：使用保存 phi/theta 运行 pyLDAvis.prepare（sort_topics=False）……", flush=True)
        vis = pyLDAvis.prepare(topic_term_dists=bundle["phi"], doc_topic_dists=bundle["theta"],
                               doc_lengths=corpus["doc_lengths"], vocab=corpus["vocab"],
                               term_frequency=corpus["term_frequency"], sort_topics=False, mds="pcoa",
                               R=30, lambda_step=.01, start_index=1, n_jobs=1)
        expected = list(range(1, K + 1))
        require(list(vis.topic_order) == expected, f"{model_id} Topic order 被重排：{vis.topic_order}")
        require(vis.topic_coordinates["topics"].astype(int).tolist() == expected, f"{model_id} pyLDAvis Topic ID 不连续")
        prepared_path = OUTPUT / f"prepared/{model_id}_pyLDAvis_prepared.pkl"
        with prepared_path.open("wb") as handle: pickle.dump(vis, handle, protocol=pickle.HIGHEST_PROTOCOL)
        save_offline_html(vis, model_id, OUTPUT / f"html/{model_id}_pyLDAvis.html", corpus["dtm"].shape[0], corpus["dtm"].shape[1])

        prevalence = make_prevalence(bundle["theta"], corpus["doc_lengths"], K)
        prevalence_by_k[K] = prevalence
        save_csv(prevalence, OUTPUT / f"data/{model_id}_topic_prevalence.csv")
        relevant = relevance_rows(bundle["phi"], corpus["term_frequency"], corpus["vocab"], K)
        relevant_by_k[K] = relevant
        save_csv(relevant, OUTPUT / f"data/{model_id}_pyldavis_relevant_terms.csv")
        pc = vis.topic_coordinates
        coord_rows = []
        for i in range(K):
            row = {"K": K, "topic_id": f"T{i + 1:02d}", "PC1": float(pc.iloc[i]["x"]), "PC2": float(pc.iloc[i]["y"]),
                   "topic_prevalence": prevalence[i]["percentage_topic_mass"], "pyldavis_frequency": float(pc.iloc[i]["Freq"])}
            require(math.isclose(row["topic_prevalence"], row["pyldavis_frequency"], abs_tol=1e-9), f"{model_id} prevalence 与 pyLDAvis Freq 不一致")
            coord_rows.append(row)
            topic_map.append({"model_id": model_id, "K": K, "original_topic_id": f"T{i + 1:02d}", "model_component_row": i,
                              "pyldavis_topic_id": i + 1, "sort_topics": False, "mapping_status": "PRESERVED"})
        coords_by_k[K] = coord_rows
        save_csv(coord_rows, OUTPUT / f"data/{model_id}_pyldavis_topic_coordinates.csv")

        pairs, nearest, stats = similarity_rows(bundle["phi"], K)
        all_pair.extend(pairs); all_nearest.extend(nearest)
        prev_vals = np.asarray([r["percentage_topic_mass"] for r in prevalence]); nearest_vals = np.asarray([r["nearest_topic_cosine"] for r in nearest])
        largest = int(np.argmax(prev_vals)); smallest = int(np.argmin(prev_vals)); max_nn = int(np.argmax(nearest_vals))
        summary = {"K": K, "n_topics": K, "largest_topic_id": f"T{largest + 1:02d}", "largest_topic_prevalence": float(prev_vals[largest]),
                   "smallest_topic_id": f"T{smallest + 1:02d}", "smallest_topic_prevalence": float(prev_vals[smallest]), "median_topic_prevalence": float(np.median(prev_vals)),
                   **stats, "mean_nearest_neighbor_cosine": float(nearest_vals.mean()), "max_nearest_neighbor_cosine": float(nearest_vals[max_nn]),
                   "max_nearest_pair": f"{nearest[max_nn]['topic_id']}↔{nearest[max_nn]['nearest_topic_id']}",
                   "small_topics_below_1pct": int((prev_vals < 1).sum()), "small_topics_below_2pct": int((prev_vals < 2).sum())}
        summaries.append(summary)
        for topic in range(1, K + 1):
            tid = f"T{topic:02d}"; rel = [r for r in relevant if r["topic_id"] == tid]
            terms = {lv: "；".join(r["term"] for r in rel if r["lambda"] == lv) for lv in (1.0, .6, .2)}
            lab = label_lookup[(K, tid)]; near = nearest[topic - 1]; coordinate = coord_rows[topic - 1]
            diagnostics.append({"K": K, "topic_id": tid, "prevalence": prevalence[topic - 1]["percentage_topic_mass"],
                                "PC1": coordinate["PC1"], "PC2": coordinate["PC2"], "nearest_topic_id": near["nearest_topic_id"],
                                "nearest_topic_cosine": near["nearest_topic_cosine"], "top_terms_lambda_1_0": terms[1.0],
                                "top_terms_lambda_0_6": terms[.6], "top_terms_lambda_0_2": terms[.2],
                                "researcher_label": lab["researcher_label_if_available"], "review_status": lab["review_status"]})

    print("[3/8] 导出 Topic ID、相似度、标签及既有 alignment……", flush=True)
    save_csv(topic_map, OUTPUT / "data/pyldavis_topic_id_map.csv")
    save_csv(all_pair, OUTPUT / "data/within_model_topic_pair_similarity.csv")
    save_csv(all_nearest, OUTPUT / "data/topic_nearest_neighbor_similarity.csv")
    save_csv(diagnostics, OUTPUT / "data/pyldavis_three_model_topic_diagnostic_summary.csv")
    save_csv(summaries, OUTPUT / "data/pyldavis_three_model_summary.csv")
    alignments = alignment_reference(config, bundles)
    save_csv(alignments, OUTPUT / "data/K12_K15_K20_pyldavis_alignment_reference.csv")

    print("[4/8] 生成三模型 comparison HTML 和八幅静态图……", flush=True)
    make_comparison_html(summaries, OUTPUT / "html/C_K12_K15_K20_pyLDAvis_comparison.html")
    make_figures(coords_by_k, prevalence_by_k, all_pair, all_nearest, summaries, OUTPUT / "figures")
    report = make_report(audits, summaries, diagnostics, alignments, "PYLDAVIS_THREE_MODEL_READY")
    (OUTPUT / "QTS_C_K12_K15_K20_pyLDAvis_Comparative_Report.md").write_text(report, encoding="utf-8")
    (OUTPUT / "README.md").write_text(make_readme(), encoding="utf-8")

    print("[5/8] 复核源工件哈希未发生变化……", flush=True)
    for row in source_hash_rows:
        after = sha256(Path(row["path"])); row["sha256_after"] = after; row["unchanged"] = after == row["sha256_before"]
        require(row["unchanged"], f"源工件在运行期间发生变化：{row['path']}")
    save_csv(source_hash_rows, OUTPUT / "audit/source_artifact_hash_qc.csv")
    env = {"generated_at": datetime.now().astimezone().isoformat(), "python": sys.version, "executable": sys.executable,
           "pyLDAvis_distribution": version("pyLDAvis"), "pyLDAvis_module": pyLDAvis.__version__, "pyLDAvis_prepare_signature": str(inspect.signature(pyLDAvis.prepare)),
           "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "font": font,
           "api": "pyLDAvis.prepare", "sort_topics": False, "mds": "pcoa", "fit_or_transform_called": False}
    (OUTPUT / "audit/run_environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "audit/run_status.json").write_text(json.dumps({"status": "GENERATED_AWAITING_WORKBOOK_AND_FINAL_QA"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[6/8] 核心输出已生成；下一步构建研究者 XLSX 并执行 HTML/UI QA。", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
