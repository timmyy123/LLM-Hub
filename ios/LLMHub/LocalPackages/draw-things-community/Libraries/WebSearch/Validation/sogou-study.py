#!/usr/bin/env python3
"""Run sequential Sogou transport comparisons with an already built WebSearchCLI."""
import argparse
import datetime
import json
import pathlib
import re
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cli", default="bazel-bin/Apps/WebSearchCLI")
parser.add_argument("--output", required=True)
args = parser.parse_args()

# Keep the built-in ten-query corpus unchanged for comparison with DuckDuckGo.
# Supplement it with Chinese queries and explicit option/edge-case checks.
supplemental = [
    ("chinese-official-site", "故宫博物院 官网", "dpm.org.cn", []),
    ("chinese-documentation", "Swift 中文教程", "swift", []),
    ("chinese-sogou", "搜狗输入法 官网", "shurufa.sogou.com", []),
    ("literal-plus", "C++ 参考手册", "cppreference.com", []),
    ("pagination", "SwiftSoup GitHub", "github.com/scinfu/SwiftSoup", ["--pages", "3", "--max-results", "25"]),
    ("freshness", "苹果 发布会", "apple.com", ["--time", "month"]),
    ("unlikely-query", '"qzxvnonexistent178891sitecheck"', None, []),
]


def scrub(value):
    """Avoid persisting provider challenge identifiers in error URLs."""
    if isinstance(value, str):
        return re.sub(r"(https?://[^ /]+/antispider/?)\?\S+", r"\1[parameters omitted]", value)
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


report = {
    "startedAtUTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "platform": "macOS on the requesting computer",
    "method": "Sequential CLI queries; persistent default WebKit data; no manual verification or result-page fetching; success requires parsed results or explicit no-results markup. Batch order is recorded; provider throttling and session state can affect later batches.",
    "benchmarks": [],
    "supplemental": [],
}
output = pathlib.Path(args.output)


def save():
    output.write_text(json.dumps(scrub(report), indent=2, ensure_ascii=False) + "\n")


for transport, batch in [("http", 1), ("browser", 1), ("automatic", 1), ("automatic", 2), ("automatic", 3)]:
    command = [args.cli, "benchmark", "--provider", "sogou", "--transport", transport, "--pretty"]
    run = subprocess.run(command, capture_output=True, text=True, check=True)
    metrics = json.loads(run.stdout)
    metrics["nonemptyQueryCount"] = sum(r["resultCount"] > 0 and "error" not in r for r in metrics["results"])
    metrics["explicitEmptyQueryCount"] = sum(r["resultCount"] == 0 and "error" not in r for r in metrics["results"])
    report["benchmarks"].append({"transport": transport, "batch": batch, "metrics": metrics})
    save()
    print(transport, batch, "valid", metrics["successfulQueryCount"], "nonempty", metrics["nonemptyQueryCount"], "MRR", metrics["meanReciprocalRank"], flush=True)

for transport in ["http", "browser", "automatic"]:
    for name, query, expected, options in supplemental:
        command = [args.cli, "search", query, "--provider", "sogou", "--transport", transport, "--pretty", *options]
        started = time.monotonic()
        run = subprocess.run(command, capture_output=True, text=True)
        entry = {"name": name, "query": query, "transport": transport, "options": options, "elapsedSeconds": time.monotonic() - started, "expectedURLContains": expected}
        if run.returncode:
            entry["error"] = run.stderr.strip()
        else:
            response = json.loads(run.stdout)
            entry["response"] = response
            # Expected-site checks are only a relevance proxy; preserve full output for review.
            urls = re.findall(r"^\d+\. \[.*\]\((.*)\)$", response["output"], re.MULTILINE)
            entry["uniqueURLCount"] = len(set(urls))
            entry["firstExpectedRank"] = next((i + 1 for i, url in enumerate(urls) if expected and expected.lower() in url.lower()), None)
        report["supplemental"].append(entry)
        save()
        print(transport, name, "error" if run.returncode else response["metadata"]["resultCount"], flush=True)
report["finishedAtUTC"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
save()
