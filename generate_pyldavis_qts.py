#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只读既有《全唐诗》模型、DTM 与词表，生成保留 Topic 编号的离线 HTML。

运行：python generate_pyldavis_qts.py
仅检查：python generate_pyldavis_qts.py --check-only
默认读取文学史数据集配置中的冻结模型；可用 --model-root / --corpus-root
显式指定同结构的原始产物目录。所有新文件均位于 outputs/pyldavis/。
依赖：pyLDAvis、scikit-learn、numpy、scipy、pandas、joblib。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import re
import sys
import warnings
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pyLDAvis
import pyLDAvis.lda_model
import pyLDAvis.urls
import sklearn
from scipy.sparse import load_npz
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parent
MODELS = {"C_K20": (3, 3, 2026), "B_K20": (2, 2, 20260830), "D_K20": (5, 3, 42)}
ROLES = {"C_K20": "主模型", "B_K20": "宽词表对照", "D_K20": "严格过滤对照"}


class ConsistencyError(ValueError):
    """输入不对应时停止该模型；禁止用重建数据或重训来修复。"""


class SavedVocabulary:
    """仅暴露已保存的词项顺序，不创建或拟合 CountVectorizer。

    本项目训练代码直接构建 CSR BoW，未保存 CountVectorizer。
    get_feature_names_out() 仅返回原始词项，供维度核验和矩阵接口使用。
    """

    def __init__(self, terms):
        self._terms = np.asarray(terms, dtype=object)
        self._terms.flags.writeable = False

    def get_feature_names_out(self):
        return self._terms.copy()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ConsistencyError(message)


def matrix_fingerprint(matrix):
    """与原训练脚本相同的逻辑指纹；只排序内存副本。"""
    canonical = matrix.copy()
    canonical.sort_indices()
    digest = hashlib.sha256()
    for array in (canonical.shape, canonical.indptr, canonical.indices, canonical.data):
        digest.update(np.asarray(array, dtype=np.int64).tobytes())
    return digest.hexdigest()


def load_inputs(model_id, model_root, corpus_root, report):
    """先核对实际 shape，再核对原始哈希、行列顺序与保存的概率矩阵。"""
    model_dir = model_root / model_id
    corpus_dir = corpus_root / model_id[0]
    paths = {
        "model": model_dir / "model.joblib",
        "dtm": corpus_dir / "bow_matrix.npz",
        "vocabulary": model_dir / "vocabulary.csv",
        "corpus_vocabulary": corpus_dir / "vocabulary.csv",
        "documents": model_dir / "document_index.csv",
        "corpus_documents": corpus_dir / "document_index.csv",
        "doc_topic": model_dir / "document_topic_probabilities.npz",
        "components": model_dir / "topic_word_components.npz",
        "matrix_metadata": model_dir / "matrix_metadata.json",
        "artifact_manifest": model_dir / "artifact_manifest.json",
        "corpus_metadata": corpus_dir / "corpus_metadata.json",
        "top_words": model_dir / "topics_top30.csv",
    }
    report["inputs"] = {key: str(path.resolve()) for key, path in paths.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少：" + "；".join(missing))

    # 版本不一致时要求找回原运行环境，避免悄悄加载不兼容的序列化对象。
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(paths["model"])
    require(isinstance(model, LatentDirichletAllocation), f"{paths['model']} 不是 sklearn LDA")
    dtm = load_npz(paths["dtm"]).tocsr()
    rows = sorted(read_csv(paths["vocabulary"]), key=lambda row: int(row["term_index"]))
    terms = [row["token"] for row in rows]
    vocabulary = SavedVocabulary(terms)
    report["shapes"] = {
        "lda_components": list(model.components_.shape),
        "dtm": list(dtm.shape),
        "vocabulary_terms": len(vocabulary.get_feature_names_out()),
    }
    diagnostic = (
        f"LDA {paths['model']} shape={model.components_.shape}；"
        f"DTM {paths['dtm']} shape={dtm.shape}；"
        f"vocabulary {paths['vocabulary']} 词项数={len(terms)}。"
        f"应寻找 {model_id}（seed={MODELS[model_id][2]}）训练时同一套原始文件。"
    )
    require(model.components_.shape[1] == dtm.shape[1] == len(terms), "词项维度不匹配：" + diagnostic)
    print(f"{model_id}：LDA={model.components_.shape}，DTM={dtm.shape}，词表={len(terms)}", flush=True)
    require([int(row["term_index"]) for row in rows] == list(range(len(terms))), "词表 term_index 不连续：" + diagnostic)
    require(len(set(terms)) == len(terms), "词表包含重复词项：" + diagnostic)

    metadata = read_json(paths["matrix_metadata"])
    manifest = read_json(paths["artifact_manifest"])
    corpus_metadata = read_json(paths["corpus_metadata"])
    tf, df, seed = MODELS[model_id]
    require(metadata["model_id"] == manifest["model_id"] == model_id, "模型标识不匹配：" + diagnostic)
    require(metadata["corpus"] == corpus_metadata["label"] == model_id[0], "语料标识不匹配：" + diagnostic)
    require((corpus_metadata["min_tf"], corpus_metadata["min_df"]) == (tf, df), "TF/DF 不匹配：" + diagnostic)
    require(model.components_.shape[0] == model.n_components == metadata["K"] == 20, "Topic 数量不匹配：" + diagnostic)
    require(model.random_state == metadata["selected_seed"] == seed, "随机种子不匹配：" + diagnostic)
    require(model.learning_method == "batch", "模型不是 Batch LDA：" + diagnostic)
    require(manifest["reproduction_validation_passed"], "模型未通过既有复现审计")

    hashes = {key: sha256(path) for key, path in paths.items()}
    report["input_sha256"] = hashes
    expected = {item["path"]: item["sha256"] for item in manifest["outputs"]}
    for key in ("model", "vocabulary", "documents", "doc_topic", "components", "matrix_metadata", "top_words"):
        require(hashes[key] == expected.get(paths[key].name), f"{paths[key]} 与 {paths['artifact_manifest']} 登记的 SHA-256 不匹配。" + diagnostic)
    for key, recorded in (("dtm", "bow_matrix_npz"), ("corpus_vocabulary", "vocabulary"), ("corpus_documents", "document_index")):
        require(hashes[key] == corpus_metadata["outputs"][recorded], f"{paths[key]} 与 {paths['corpus_metadata']} 登记的 SHA-256 不匹配。" + diagnostic)
    require(hashes["vocabulary"] == hashes["corpus_vocabulary"], "模型词表与 DTM 词表内容/顺序不匹配：" + diagnostic)
    require(hashes["documents"] == hashes["corpus_documents"], "模型文档索引与 DTM 行顺序不匹配：" + diagnostic)
    require(matrix_fingerprint(dtm) == manifest["corpus_matrix_fingerprint"] == corpus_metadata["matrix_fingerprint"], "DTM 逻辑指纹与模型登记不匹配：" + diagnostic)
    require(list(dtm.shape) == corpus_metadata["matrix_shape"], "DTM 与元数据 shape 不匹配：" + diagnostic)
    require(list(model.components_.shape) == metadata["topic_word_components"]["shape"], "LDA 与元数据 shape 不匹配：" + diagnostic)
    with np.load(paths["components"], allow_pickle=False) as archive:
        require(np.array_equal(model.components_, archive["components"]), "模型与原始 components 不一致：" + diagnostic)
    with np.load(paths["doc_topic"], allow_pickle=False) as archive:
        doc_topic = archive["probabilities"].copy()
    report["shapes"]["document_topic"] = list(doc_topic.shape)
    require(doc_topic.shape == (dtm.shape[0], 20), f"{paths['doc_topic']} shape={doc_topic.shape} 与 DTM 行数/Topic 数不匹配：" + diagnostic)
    require(np.isfinite(doc_topic).all() and (doc_topic >= 0).all() and np.allclose(doc_topic.sum(axis=1), 1, atol=1e-10, rtol=0), "保存的 Document×Topic 概率不合法")
    require(np.isfinite(model.components_).all() and (model.components_ > 0).all(), "LDA components 包含非法值")
    require(np.isfinite(dtm.data).all() and (dtm.data >= 0).all() and np.equal(dtm.data, np.floor(dtm.data)).all(), "DTM 必须为非负原始词频")
    lengths = np.asarray(dtm.sum(axis=1)).ravel()
    frequencies = np.asarray(dtm.sum(axis=0)).ravel()
    documents = sorted(read_csv(paths["documents"]), key=lambda row: int(row["matrix_row"]))
    require([int(row["matrix_row"]) for row in documents] == list(range(dtm.shape[0])), "文档 matrix_row 不连续或行数不匹配")
    require(np.array_equal(lengths, [int(row["token_count"]) for row in documents]), "DTM 行词次与原始文档索引不匹配")
    require(np.array_equal(frequencies, [int(row["TF"]) for row in rows]), "DTM 列词频与原始词表不匹配")
    require((lengths > 0).all() and (frequencies > 0).all(), "DTM 包含零行/零列；请找回原始训练矩阵，不自动过滤")
    report.update({"shape_match": True, "source_alignment_verified": True, "selected_seed": seed,
                   "vocabulary_interface": "SavedVocabulary：原始 CSV 的只读接口，无 CountVectorizer 拟合",
                   "documents": dtm.shape[0], "vocabulary_size": len(terms), "topics": 20,
                   "excluded_empty_documents_in_original_training": corpus_metadata["empty_poem_count"]})
    return model, dtm, vocabulary, doc_topic, lengths, paths


def verify_topic_ids(vis, model, vocabulary, doc_topic, lengths, paths):
    """同时检查编号、词项概率与词次加权占比，避免仅验证标签连续。"""
    expected_ids = list(range(1, 21))
    require(list(vis.topic_order) == expected_ids, f"Topic 被重新排序：{vis.topic_order}")
    require(vis.topic_coordinates["topics"].tolist() == expected_ids, "可视化圆圈编号错位")
    terms = vocabulary.get_feature_names_out()
    lookup = {term: index for index, term in enumerate(terms)}
    phi = model.components_ / model.components_.sum(axis=1, keepdims=True)
    prevalence = doc_topic.T @ lengths / lengths.sum() * 100
    require(np.allclose(vis.topic_coordinates["Freq"], prevalence, atol=1e-10, rtol=0), "主题占比与原始概率的词次加权结果不一致")
    coordinates = vis.topic_coordinates[["x", "y"]].to_numpy()
    require(np.isfinite(coordinates).all(), "主题距离图包含非有限坐标")
    audit = []
    original_words = read_csv(paths["top_words"])
    for topic_id in expected_ids:
        info = vis.topic_info[vis.topic_info["Category"] == f"Topic{topic_id}"]
        indices = [lookup[term] for term in info["Term"]]
        require(len(info) >= 30, f"Topic {topic_id} 的候选词不足30个")
        require(np.allclose(info["logprob"], np.round(np.log(phi[topic_id - 1, indices]), 4), atol=1e-12, rtol=0), f"Topic {topic_id} 的词项概率与原模型对应行不符")
        original = sorted((row for row in original_words if int(row["topic_id"]) == topic_id), key=lambda row: int(row["rank"]))
        top_indices = np.argsort(model.components_[topic_id - 1])[::-1][:30]
        require([row["token"] for row in original] == terms[top_indices].tolist(), f"Topic {topic_id} 与原审核 top-30 词序不符")
        require(set(terms[top_indices]).issubset(set(info["Term"])), f"Topic {topic_id} 未包含原始 top-30 词")
        audit.append({"original_topic_id": f"T{topic_id:02d}", "model_component_row": topic_id - 1,
                      "pyldavis_topic_id": topic_id, "token_weighted_prevalence_percent": float(prevalence[topic_id - 1]),
                      "original_top_word": terms[top_indices[0]], "keyword_probability_match": True})
    return audit


def save_offline_html(vis, model_id, destination, report):
    """先调用官方 save_html，再内嵌本地发行包自带资源，移除网络依赖。"""
    assets = {
        "d3": Path(pyLDAvis.urls.D3_LOCAL),
        "ldavis": Path(pyLDAvis.urls.LDAVIS_LOCAL),
        "css": Path(pyLDAvis.urls.LDAVIS_CSS_LOCAL),
    }
    for path in assets.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少 pyLDAvis 离线资源：{path}")
    fragment = io.StringIO()
    pyLDAvis.save_html(vis, fragment, template_type="simple", visid=f"qts_{model_id}",
                      d3_url="qts_d3.js", ldavis_url="qts_ldavis.js", ldavis_css_url="qts_ldavis.css")
    content = fragment.getvalue()
    for key, url in (("d3", "qts_d3.js"), ("ldavis", "qts_ldavis.js")):
        source = assets[key].read_text(encoding="utf-8")
        source = re.sub(r"</script", r"<\\/script", source, flags=re.IGNORECASE)
        tag = f'<script type="text/javascript" src="{url}"></script>'
        require(content.count(tag) == 1, "save_html 模板发生变化，无法安全内嵌资源")
        content = content.replace(tag, f'<script type="text/javascript">\n{source}\n</script>')
    css_tag = '<link rel="stylesheet" type="text/css" href="qts_ldavis.css">'
    require(content.count(css_tag) == 1, "save_html CSS 模板发生变化")
    content = content.replace(css_tag, '<style>\n' + assets["css"].read_text(encoding="utf-8") + '\n</style>')
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:">
<title>《全唐诗》{model_id} · pyLDAvis</title>
<style>body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px;color:#243247;background:#fff}}h1{{font-size:24px}}p{{line-height:1.7}}.chart{{overflow-x:auto}}footer{{font-size:13px;color:#64748b}}</style>
</head><body><h1>《全唐诗》{model_id} · pyLDAvis</h1>
<p>原模型 T01–T20 对应 Topic 1–20。点击圆圈选择主题；调节 λ 比较 1.0、0.6、0.2 下的 Top 30 相关词。</p>
<p>{report['documents']:,} 首有效诗 · {report['vocabulary_size']:,} 个词项。圆面积表示按文档词次加权的主题占比；二维距离用于探索主题相似程度。</p>
<div class="chart">{content}</div>
<footer>保留原始 Topic 编号及模型概率。重叠主题：建议人工复核。人工主题名称优先。资源已内嵌，可离线打开。</footer>
</body></html>'''
    require(not re.search(r'<script\b[^>]*\bsrc\s*=|<link\b[^>]*\bhref\s*=', page, re.I), "HTML 仍引用外部脚本或样式")
    destination.write_text(page, encoding="utf-8")
    report["offline_assets_sha256"] = {key: sha256(path) for key, path in assets.items()}
    report["html"] = str(destination.resolve())
    report["html_sha256"] = sha256(destination)
    report["offline_self_contained"] = True


def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def print_report(reports, check_only=False):
    lines = ["=" * 35, "《全唐诗》pyLDAvis " + ("输入检查完成" if check_only else "可视化运行报告"), "=" * 35]
    if not check_only and all(row["status"] == "成功" for row in reports):
        lines[1] = "《全唐诗》pyLDAvis 可视化完成"
    lines += [f"pyLDAvis: {version('pyLDAvis')}（模块 __version__: {pyLDAvis.__version__}）"]
    for row in reports:
        model_id = row["model_id"]
        lines += ["", f"{ROLES[model_id]}：", f"{model_id}：{row['status']}"]
        if row["status"] in ("成功", "检查通过"):
            lines += [f"Topics: {row['topics']}", f"Documents: {row['documents']}", f"Vocabulary: {row['vocabulary_size']}", "三者词项维度：一致"]
            if "html" in row:
                lines += [f"HTML: {row['html']}", "Topic编号：T01–T20 = Topic 1–20（逐主题核验通过）"]
        else:
            lines.append(row.get("reason", ""))
    lines.append("=" * 35)
    result = "\n".join(lines)
    print(result, flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, help="含 C_K20/B_K20/D_K20 子目录的冻结模型目录")
    parser.add_argument("--corpus-root", type=Path, help="含 C/B/D 子目录的原始 DTM 目录")
    parser.add_argument("--check-only", action="store_true", help="仅做输入一致性检查")
    args = parser.parse_args(argv)
    config_path = ROOT / "config/literary_history_dataset_config.json"
    config = read_json(config_path)
    model_root = (args.model_root or ROOT / config["output_root"] / "freeze").resolve()
    corpus_root = (args.corpus_root or ROOT / config["corpus_root"]).resolve()
    destination = ROOT / "outputs/pyldavis"
    destination.mkdir(parents=True, exist_ok=True)
    environment = {"python": sys.version, "executable": sys.executable, "pyLDAvis_distribution": version("pyLDAvis"),
                   "pyLDAvis_module": pyLDAvis.__version__, "scikit_learn": sklearn.__version__,
                   "numpy": np.__version__, "scipy": version("scipy"), "pandas": version("pandas"),
                   "api": "pyLDAvis.prepare", "api_signature": str(inspect.signature(pyLDAvis.prepare)),
                   "sklearn_helper_checked": str(inspect.signature(pyLDAvis.lda_model.prepare)),
                   "api_reason": "当前 sklearn 便捷接口无条件重复 transform；采用官方矩阵接口直接复用原始 Document×Topic，无需再次推断"}
    reports = []
    stop_after_main_mismatch = False
    for model_id in MODELS:
        report = {"model_id": model_id}
        reports.append(report)
        if stop_after_main_mismatch:
            report.update(status="未运行", reason="C_K20 一致性检查失败；请先找回主模型的原始对应文件。")
            continue
        try:
            model, dtm, vocabulary, doc_topic, lengths, paths = load_inputs(model_id, model_root, corpus_root, report)
            if not args.check_only:
                print(f"{model_id}：正在生成 pyLDAvis，sort_topics=False，R=30……", flush=True)
                # 已检查 lda_model.prepare：它无条件执行 transform，且读取模型的 n_jobs。
                # 本项目已保存完整概率，改用同一官方核心接口，避免重复推断和多进程开销。
                # 这里只归一化模型 components；不改变模型内存，也不保存或覆盖原始模型。
                vis = pyLDAvis.prepare(
                    topic_term_dists=model.components_ / model.components_.sum(axis=1, keepdims=True),
                    doc_topic_dists=doc_topic, doc_lengths=lengths,
                    vocab=vocabulary.get_feature_names_out(),
                    term_frequency=np.asarray(dtm.sum(axis=0)).ravel(),
                    sort_topics=False, start_index=1, R=30,
                    lambda_step=0.01, mds="pcoa", n_jobs=1,
                )
                audit = verify_topic_ids(vis, model, vocabulary, doc_topic, lengths, paths)
                report.update(topic_order=list(vis.topic_order), topic_ids_verified=True,
                              sort_topics=False, start_index=1, R=30, lambda_step=0.01,
                              doc_topic_source="原始 document_topic_probabilities.npz",
                              prevalence_definition="sum_d(theta[d,k] * 原始DTM行词次[d]) / 全部词次")
                save_offline_html(vis, model_id, destination / f"pyLDAvis_{model_id}.html", report)
                with (destination / f"topic_id_audit_{model_id}.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
                    writer.writeheader()
                    writer.writerows(audit)
                if in_notebook():
                    from IPython.display import display
                    display(pyLDAvis.display(vis))
            # 再读哈希，确认本次处理没有修改任何原始输入。
            require(all(sha256(path) == report["input_sha256"][key] for key, path in paths.items()), "运行期间原始输入文件发生变化")
            report.update(status="检查通过" if args.check_only else "成功", source_inputs_unchanged=True)
        except FileNotFoundError as exc:
            report.update(status="缺少文件", reason=str(exc))
        except ConsistencyError as exc:
            report.update(status="一致性检查失败", reason=str(exc))
            if model_id == "C_K20":
                stop_after_main_mismatch = True
        except Exception as exc:
            report.update(status="失败", reason=f"{type(exc).__name__}: {exc}")
        print(f"{model_id}：{report['status']} {report.get('reason', '')}", flush=True)
    text_report = print_report(reports, args.check_only)
    stem = "input_check" if args.check_only else "run_report"
    (destination / f"{stem}.txt").write_text(text_report + "\n", encoding="utf-8")
    payload = {"generated_at": datetime.now().astimezone().isoformat(), "environment": environment,
               "script_sha256": sha256(Path(__file__)), "models": reports}
    (destination / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["status"] in ("成功", "检查通过") for row in reports) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
