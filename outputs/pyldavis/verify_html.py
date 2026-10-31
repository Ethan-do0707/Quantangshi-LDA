# -*- coding: utf-8 -*-
"""静态检查独立 HTML 的内嵌数据、离线资源和 JavaScript 语法；不冒充浏览器交互验收。"""

import hashlib
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path


class Inspector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.scripts = []
        self.external = []
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.in_script = True
            self.scripts.append("")
            if attrs.get("src"):
                self.external.append(attrs["src"])
        if tag == "link" and attrs.get("href"):
            self.external.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.scripts[-1] += data


def main():
    directory = Path(__file__).resolve().parent
    report = json.loads((directory / "run_report.json").read_text(encoding="utf-8"))
    results = []
    node = shutil.which("node")
    for model in report["models"]:
        model_id = model["model_id"]
        path = directory / f"pyLDAvis_{model_id}.html"
        raw = path.read_bytes()
        parser = Inspector()
        parser.feed(raw.decode("utf-8"))
        assert not parser.external, parser.external
        assert len(parser.scripts) == 3
        match = re.search(r'new LDAvis\("#" \+ "qts_' + model_id + r'", (\{.*\})\);', parser.scripts[-1], re.S)
        assert match is not None
        data = json.loads(match[1])
        assert data["topic.order"] == data["mdsDat"]["topics"] == list(range(1, 21))
        assert data["R"] == 30 and data["lambda.step"] == 0.01
        assert abs(sum(data["mdsDat"]["Freq"]) - 100) < 1e-8
        for topic_id in range(1, 21):
            assert data["tinfo"]["Category"].count(f"Topic{topic_id}") >= 30
        assert hashlib.sha256(raw).hexdigest() == model["html_sha256"]
        syntax = "未检查：Node 不可用"
        if node:
            # 仅解析语法，不执行页面脚本，也不启动浏览器。
            process = subprocess.run([node, "--check", "-"], input="\n".join(parser.scripts),
                                     text=True, encoding="utf-8", capture_output=True, timeout=30)
            assert process.returncode == 0, process.stderr
            syntax = "通过"
        results.append({"model_id": model_id, "embedded_data": "通过", "topic_ids": "通过",
                        "external_asset_references": len(parser.external), "javascript_syntax": syntax,
                        "browser_interaction": "未实测：浏览器工具安全策略禁止本地 file:// 页面"})
        print(model_id, results[-1])
    (directory / "static_verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
