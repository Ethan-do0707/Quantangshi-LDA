"""用无停用词探索性 LDA 估计词语的主题分布熵。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

if __package__:
    from stopwords.build_stopwords import DEFAULT_RESULTS, ROOT, is_han_token
else:
    from build_stopwords import DEFAULT_RESULTS, ROOT, is_han_token


DEFAULT_OUTPUT = ROOT / "stopwords" / "pilot_topic_metrics.csv"
DEFAULT_METADATA = ROOT / "stopwords" / "pilot_topic_metadata.json"


def identity_analyzer(tokens: Sequence[str]) -> Sequence[str]:
    """让词袋向量器直接接收已分词的词序列。"""
    return tokens


def load_documents(result_path: Path) -> list[list[str]]:
    """读取 sikufenci 结果并移除标点与非汉字 token。"""
    documents: list[list[str]] = []
    with result_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            documents.append(
                [str(token) for token in record["tokens"] if is_han_token(str(token))]
            )
    return documents


def calculate_topic_metrics(
    documents: Sequence[Sequence[str]],
    topic_count: int,
    minimum_document_frequency: int,
    maximum_features: int,
    iterations: int,
    random_state: int,
    jobs: int,
):
    """拟合探索性 LDA 并计算每个词的归一化主题熵。"""
    vectorizer = CountVectorizer(
        analyzer=identity_analyzer,
        lowercase=False,
        min_df=minimum_document_frequency,
        max_features=maximum_features,
    )
    matrix = vectorizer.fit_transform(documents)
    model = LatentDirichletAllocation(
        n_components=topic_count,
        max_iter=iterations,
        learning_method="online",
        batch_size=512,
        random_state=random_state,
        n_jobs=jobs,
        evaluate_every=-1,
    )
    document_topics = model.fit_transform(matrix)
    topic_prevalence = document_topics.sum(axis=0)
    topic_prevalence /= topic_prevalence.sum()
    topic_word = model.components_ / model.components_.sum(axis=1, keepdims=True)
    joint_distribution = topic_word * topic_prevalence[:, None]
    topic_given_word = joint_distribution / joint_distribution.sum(axis=0, keepdims=True)
    entropy = -(
        topic_given_word * np.log(topic_given_word + np.finfo(float).tiny)
    ).sum(axis=0) / math.log(topic_count)
    top_topic_share = topic_given_word.max(axis=0)
    dominant_topic = topic_given_word.argmax(axis=0) + 1
    frequency = np.asarray(matrix.sum(axis=0)).ravel()
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    tokens = vectorizer.get_feature_names_out()
    rows = [
        {
            "token": str(tokens[index]),
            "frequency": int(frequency[index]),
            "document_frequency": int(document_frequency[index]),
            "topic_entropy": float(entropy[index]),
            "top_topic_share": float(top_topic_share[index]),
            "dominant_topic": int(dominant_topic[index]),
        }
        for index in range(len(tokens))
    ]
    rows.sort(key=lambda row: (-row["frequency"], row["token"]))
    return rows, matrix.shape, matrix.nnz


def write_metrics(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """写出探索性主题指标。"""
    fieldnames = [
        "token",
        "frequency",
        "document_frequency",
        "topic_entropy",
        "top_topic_share",
        "dominant_topic",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["topic_entropy"] = f"{float(output['topic_entropy']):.6f}"
            output["top_topic_share"] = f"{float(output['top_topic_share']):.6f}"
            writer.writerow(output)


def parse_args() -> argparse.Namespace:
    """解析探索性 LDA 参数。"""
    parser = argparse.ArgumentParser(description="计算 sikufenci 词语的探索性主题熵")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--topics", type=int, default=20)
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    """运行探索性 LDA 并保存可复核参数。"""
    args = parse_args()
    documents = load_documents(args.results)
    rows, matrix_shape, nonzero_count = calculate_topic_metrics(
        documents,
        args.topics,
        args.min_df,
        args.max_features,
        args.iterations,
        args.random_state,
        args.jobs,
    )
    write_metrics(args.output, rows)
    metadata = {
        "purpose": "仅用于停用词候选审查，不作为最终主题模型",
        "source_result": str(args.results.relative_to(ROOT)),
        "document_count": len(documents),
        "matrix_shape": list(matrix_shape),
        "matrix_nonzero_count": nonzero_count,
        "topic_count": args.topics,
        "minimum_document_frequency": args.min_df,
        "maximum_features": args.max_features,
        "iterations": args.iterations,
        "learning_method": "online",
        "random_state": args.random_state,
        "jobs": args.jobs,
        "sklearn_version": sklearn.__version__,
    }
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
