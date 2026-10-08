#!/usr/bin/env python3
"""Compare the working tokenizer with a Git revision, in one optimized Swift process.

python3 Libraries/Tokenizer/Tests/benchmark_tiktoken.py --reference-ref <commit>
Optional --corpus PATH arguments add UTF-8 text files to the repository corpus.
Generated reference sources, executable, corpus and results stay in --output-dir.
"""

import argparse
import json
from pathlib import Path
import random
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ref", required=True)
    parser.add_argument("--corpus", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--corpus-bytes", type=int, default=2 * 1024 * 1024)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = (args.output_dir or Path(tempfile.mkdtemp(prefix="tiktoken-parity-"))).resolve()
    if args.corpus_bytes <= 0:
        parser.error("--corpus-bytes must be positive")
    reference = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.reference_ref}^{{commit}}"], cwd=repo
    ).decode().strip()
    output.mkdir(parents=True, exist_ok=True)

    def baseline(path):
        return subprocess.check_output(
            ["git", "show", f"{reference}:{path}"], cwd=repo
        ).decode()

    def write(name, text):
        path = output / name
        path.write_text(text)
        return str(path)

    tokenizer_path = "Libraries/Tokenizer/Sources/Tiktoken.swift"
    sections_path = "Apps/LocalCode/Sources/Agent/SystemPromptTokenization.swift"
    sources = [str(repo / tokenizer_path)]
    sources.append(write("ReferenceTiktoken.swift", baseline(tokenizer_path).replace(
        "TiktokenTokenizer", "ReferenceTiktokenTokenizer")))
    # Use the actual protocol declarations without unrelated CLIP helpers.
    protocols = (repo / "Libraries/Tokenizer/Sources/Tokenizer.swift").read_text()
    sources.append(write("Protocols.swift", protocols.split(
        "extension Tokenizer {\n  public func automaticLineBreak")[0]))
    sources.append(str(repo / "Apps/LocalCode/Sources/Agent/SystemPromptSection.swift"))
    sources.append(write("Sections.swift", (repo / sections_path).read_text().replace(
        "import Tokenizer\n", "")))
    original_sections = baseline(sections_path).replace("import Tokenizer\n", "")
    sources.append(write("BaselineSections.swift", original_sections.replace(
        "func tokenizeSystemPromptSections(", "func baselineTokenizeSystemPromptSections("
    ).replace("TiktokenTokenizer", "ReferenceTiktokenTokenizer")))
    sources.append(str(repo / "Apps/LocalCode/Sources/Agent/SystemPrompt.swift"))
    # Extract the production token tables, including every reserved spelling.
    model = (repo / "Libraries/SwiftLLM/Sources/Models/DeepSeek4_1.swift").read_text()
    start = model.index("  public static let specialTokens:")
    end = model.index("\n  public init()", start)
    sources.append(write("SpecialTokens.swift", "import Foundation\nenum DeepSeekModel {\n"
                         + model[start:end] + "\n}\n"))

    qwen = (repo / "Apps/LocalCode/Sources/Models/Qwen3_5TextGenerator.swift").read_text()
    start = qwen.index("  private static let specialTokens:")
    end = qwen.index("\n  static let unsafeTokenizer", start)
    sources.append(write("QwenSpecialTokens.swift", "import Foundation\nenum QwenModel {\n"
                         + qwen[start:end].replace("private static", "static") + "\n}\n"))

    corpus = []
    size = 0
    files = subprocess.check_output(["git", "ls-files"], cwd=repo).decode().splitlines()
    candidates = [p for p in files if p.startswith(("Apps/", "Libraries/", "Scripts/"))
                  and Path(p).suffix in {".swift", ".md", ".py", ".json", ".txt"}]
    random.Random(42).shuffle(candidates)
    # Include the actual prompt sources before sampling the wider repository.
    prioritized = ["AGENTS.md", "Apps/LocalCode/Sources/Agent/SystemPrompt.swift"]
    for name in dict.fromkeys(prioritized + candidates):
        if size >= args.corpus_bytes:
            break
        try:
            text = (repo / name).read_text()
        except (UnicodeError, OSError):
            continue
        text = text[:32768]
        if text:
            corpus.append({"name": name, "text": text})
            size += len(text.encode())
    for path in args.corpus:
        corpus.append({"name": str(path), "text": path.read_text()})
    corpus_path = write("corpus.json", json.dumps(corpus, ensure_ascii=False))
    sources.append(str(repo / "Libraries/Tokenizer/Tests/TiktokenBenchmark.swift"))
    executable = output / "benchmark"
    subprocess.run(["xcrun", "swiftc", "-O", "-whole-module-optimization",
                    "-module-cache-path", str(output / "module-cache")] + sources
                   + ["-o", str(executable)], check=True, cwd=repo)
    print(f"Reference: {reference}\nArtifacts: {output}", flush=True)
    (output / "reference.txt").write_text(reference + "\n")
    with (output / "results.txt").open("w") as log:
        process = subprocess.Popen([str(executable), str(repo), corpus_path], cwd=repo,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
