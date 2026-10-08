# Special-token regex benchmark

The tokenizer retains the original special-token alternation and its ordering. A
lookahead checks the possible first characters before entering that alternation.
For DeepSeek4.1, this skips the 1,283 alternatives unless the current character is
`<` or `｜`. Single-spelling and empty-spelling cases retain the original pattern.
The ordinary-text regex, BPE implementation, and pre-iOS-16/macOS-13 fallback are
unchanged.

## Reproduce

From the repository root on macOS with Xcode installed:

```sh
python3 Libraries/Tokenizer/Tests/benchmark_tiktoken.py \
  --reference-ref d77b4c0afc4f313d99ee7d2bf02371af822c4ed3 \
  --output-dir /tmp/tiktoken-benchmark
```

The runner compiles the reference tokenizer from Git and the working tokenizer
into one executable using `swiftc -O -whole-module-optimization`. It uses the
production DeepSeek4.1 and Qwen3.5 special-token tables and vocabulary files.
The reference already includes the separately committed single-pass prompt
boundary change, so this comparison isolates the regex optimization.

The default corpus contains roughly 2 MiB of repository text selected with a
fixed seed: Swift, Python, Markdown, JSON, and text files. Each sampled file is
capped at 32,768 characters. The benchmark adds the built-in prompt templates,
all special spellings both separated and adjacent, partial spellings, empty text,
and multilingual text with combining marks, emoji, and control characters.
Use `--corpus-bytes N` to increase the sample, or repeat `--corpus PATH` to add
UTF-8 files. The exact selected corpus is saved as `corpus.json`.

Every sample is checked with both safe and unsafe DeepSeek4.1 and Qwen3.5
tokenizers. The benchmark asserts equality of every `Int32` token ID and the
UTF-8 bytes of every returned token piece, including calls with and without
start-token insertion. It exits unsuccessfully on a mismatch.

Reported corpus times exclude initialization, warm-up, and equality assertions.
Reference and updated calls alternate order across samples. Separate five-run
means measure special-token matching and the 18,327-byte built-in DeepSeek4.1
prompt template plus a fixed workspace section. These are local optimized
benchmarks, not end-to-end app-open measurements.

## Results

2026-09-25, macOS 27.0 / arm64, Apple Swift 6.4, optimized build. Reference:
`d77b4c0afc4f313d99ee7d2bf02371af822c4ed3`. All 318 samples per configuration
matched exactly: **2,559,676 token IDs and their token-piece UTF-8 bytes** in total.

| Tokenizer | Special spellings | Input bytes | Token IDs | Reference | Updated | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek4.1 unsafe | 1,283 | 2,304,072 | 649,156 | 111.850 s | 6.184 s | 18.09× |
| DeepSeek4.1 safe | 3 | 2,171,158 | 626,170 | 2.370 s | 2.284 s | 1.04× |
| Qwen3.5 unsafe | 33 | 2,172,586 | 642,101 | 4.092 s | 2.169 s | 1.89× |
| Qwen3.5 safe | 1 | 2,170,956 | 642,249 | 2.031 s | 2.024 s | 1.00× |

The five-run mean for special-token regex matching on the built-in DeepSeek4.1
template was **874.700 ms → 5.023 ms**. Tokenizing the template and fixed workspace
section took **904.899 ms → 37.367 ms**, with exactly identical token IDs.

The single-spelling case uses the original pattern; its small timing difference
is measurement variation. Corpus totals are one pass over many samples, not
statistical guarantees for every workload. The largest benefit comes from avoiding
the large alternation on ordinary text; candidate-heavy inputs still enter it.

## Regression tests

```sh
bazel test //Libraries/Tokenizer:TiktokenIncrementalDecoderTests \
  //Apps/LocalCode:SystemPromptSectionTests --test_output=errors
```

The regex regression compares exact match ranges against the original pattern,
including overlapping spellings in different orders, regex metacharacters,
Unicode normalization, empty/single spellings, and a 1,300-spelling set. The
existing tokenization and prompt trust-boundary tests also run.
