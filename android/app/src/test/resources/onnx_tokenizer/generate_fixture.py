"""Regenerates the OnnxTokenizerTest fixture from HuggingFace `tokenizers` (the reference implementation).

Usage (paths are directories holding the downloaded tokenizer.json, tokenizer_config.json,
generation_config.json and chat_template.jinja of each repo):
    python3 -I generate_fixture.py <LiquidAI/LFM2.5-1.2B-Instruct-ONNX dir> <mistralai/Ministral-3-3B-Instruct-2512-ONNX dir> <out dir>
Needs `pip install tokenizers jinja2`. Writes fixture.json and the gzipped tokenizer.json files.
"""
import gzip
import json
import os
import random
import shutil
import sys
from datetime import datetime

import jinja2
from tokenizers import Tokenizer

lfm_dir, mistral_dir, out_dir = sys.argv[1:4]

TEXTS = [
    "Hello, world! How are you doing today?\nI'm fine; thanks.\n\nNew paragraph:\tTabbed.",
    "Grüße aus München! Die Straße ist schön, Äpfel und Öl kosten 3,50 €. Übermäßig große Bären.",
    "こんにちは、世界。今日はいい天気ですね。東京タワーに行きましょう！カタカナとひらがな。",
    "Party 🎉🎉 time 👨‍👩‍👧‍👦 flags 🇩🇪🇯🇵 ok 👍🏽 🤖✨",
    "Pi is 3.14159265358979, and 1234567890 + 42 = 1234567932. Year 2026, 1/2, 10%, -7.5e-3.",
    "def foo(x):\n    return x ** 2  # square\n\nfun main() {\n\tprintln(\"hi\")\n}\n",
    "I'M sure they'LL say it's John's; DON'T you think we've done it? You'd, she'd, I'Ve.",
    "  leading spaces\r\nwindows line nbsp　ideographic   multiple    spaces   \n\n\n",
    "Привет мир. 你好，世界。안녕하세요 مرحبا بالعالم שלום हिन्दी ไทย Ελληνικά",
    "<|im_start|>user\nhi<|im_end|>\n<think>reasoning here</think>answer [INST]x[/INST] <s>a</s>",
    "Mathias likes python and pythonic code; Python too.",
    "See https://example.com/a?b=c&d=e#f and **bold** _it_ `code` | table | --- |",
    "=" * 50 + "aaaa" * 20 + "\n" + "-" * 33,
    "",
    "é café naïve résumé",
    "∑_{i=1}^{n} x² ≥ √2 ∞ ≠ ≈ ± → ← “quotes” ‘single’ … — –",
    "The quick brown fox jumps over the lazy dog. THE QUICK BROWN FOX! 0123456789",
    "x'\u017fy z'\u017ft it'S HE'LL \u212a'\u212ay",
]

# Seeded random mixes of the cases that tend to break tokenizer ports: Unicode white space (\\s in
# Oniguruma is Unicode-aware), letter/number categories, combining marks, contractions, surrogate pairs,
# CRLF, and fragments of special tokens.
SNIPPETS = [
    "a", "Z", "hello", " world", "ABC", "x1", "42", "12345", "3.5", " ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", "\r",
    "\u00a0", "\u2009", "\u3000", "\u200b", "\u0085", "\u2028", "\u202f", "\u1680", "\u000b", "\u000c",
    "'s", "'S", "'t", "'re", "'VE", "'m", "'ll", "'D", "\u017f", "\u212a", "'",
    "e\u0301", "\u0301", "\u01c5", "\u02b0", "\u4e2d\u6587", "\u0627\u0644\u0639", "\u0391\u03b2", "\u0416\u0436",
    "\u0663", "\u216b", "\u00bd", "\u00b2", "\u0e44\u0e17\u0e22", "\u0939\u093f",
    "\U0001F600", "\U0001F44D\U0001F3FD", "\U0001F468\u200d\U0001F469", "\U0001F1E9\U0001F1EA", "\u2764\ufe0f",
    "!", "?!", "...", "--", "//", "/*", "*/", "{", "}", "()", "[]", "<", ">", "|", "<|", "|>", "<|im_end|>", "<|im_start|>",
    "[INST]", "[/INST]", "<s>", "</s>", "<think>", "</think>", "python", "Mathias", "\u00e4\u00f6\u00fc\u00df", "\u20ac", "$",
]
rng = random.Random(20261128)
FUZZ = ["".join(rng.choice(SNIPPETS) for _ in range(rng.randint(1, 24))) for _ in range(400)]

# An app transcript (ChatViewModel's role-labelled format) and the conversation it stands for.
APP_PROMPT = ("system: You are a helpful assistant. Answer briefly.\n\n"
              "user: Hallo! Wie geht's?\nassistant: Gut, danke. Und dir?\n\n"
              "user: Mir geht's gut 😊 Was ist 2+2?\n\nZweite Zeile.\nassistant:")
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
    {"role": "user", "content": "Hallo! Wie geht's?"},
    {"role": "assistant", "content": "Gut, danke. Und dir?"},
    {"role": "user", "content": "Mir geht's gut 😊 Was ist 2+2?\n\nZweite Zeile."},
]
APP_PROMPT_NO_SYSTEM = "user: Tell me a joke.\nassistant:"
MESSAGES_NO_SYSTEM = [{"role": "user", "content": "Tell me a joke."}]


def render(template_text, cfg, messages):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise RuntimeError(msg)

    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda f: datetime.now().strftime(f)
    # HF's {% generation %} tag only marks assistant spans; it renders its body unchanged.
    template_text = template_text.replace("{%- generation -%}", "").replace("{%- endgeneration -%}", "")
    tpl = env.from_string(template_text)

    def tok(name):
        v = cfg.get(name)
        return v["content"] if isinstance(v, dict) else v

    return tpl.render(messages=messages, add_generation_prompt=True, bos_token=tok("bos_token"), eos_token=tok("eos_token"))


def load(d):
    tk = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    cfg = json.load(open(os.path.join(d, "tokenizer_config.json"), encoding="utf-8"))
    gen = json.load(open(os.path.join(d, "generation_config.json"), encoding="utf-8"))
    tpl = open(os.path.join(d, "chat_template.jinja"), encoding="utf-8").read()
    return tk, cfg, gen, tpl


def cases(tk, texts):
    out = []
    for t in texts:
        ids = tk.encode(t, add_special_tokens=False).ids
        out.append({
            "text": t,
            "ids": ids,
            "decoded": tk.decode(ids, skip_special_tokens=False),
            "decoded_skip_special": tk.decode(ids, skip_special_tokens=True),
        })
    return out


fixture = {}
for key, d, gz_name in [("lfm", lfm_dir, "lfm25_tokenizer.json.gz"), ("ministral", mistral_dir, "ministral3_tokenizer.json.gz")]:
    tk, cfg, gen, tpl = load(d)
    rendered = render(tpl, cfg, MESSAGES)
    templates = [{"app_prompt": APP_PROMPT, "rendered": rendered}]
    if key == "lfm":
        # Ministral's template adds a long default system prompt when none is given; the app does not.
        templates.append({"app_prompt": APP_PROMPT_NO_SYSTEM, "rendered": render(tpl, cfg, MESSAGES_NO_SYSTEM)})
    texts = TEXTS + [t["rendered"] for t in templates]
    fixture[key] = {
        "tokenizer": gz_name,
        "tokenizer_config": {k: cfg[k] for k in ("bos_token", "eos_token") if k in cfg},
        "generation_config": gen,
        "templates": templates,
        "cases": cases(tk, texts),
        "fuzz": cases(tk, FUZZ),
    }
    with open(os.path.join(d, "tokenizer.json"), "rb") as src, gzip.open(os.path.join(out_dir, gz_name), "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)

with open(os.path.join(out_dir, "fixture.json"), "w", encoding="utf-8") as f:
    json.dump(fixture, f, ensure_ascii=False, separators=(",", ":"))
print({k: (len(v["cases"]), len(v["fuzz"])) for k, v in fixture.items()})
