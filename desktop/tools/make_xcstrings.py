"""Builds the macOS string catalog from the shared table.

core/LLMHub.Core/strings.json is the single source for all seven languages (Windows and Linux embed it directly).
Run after editing it:  python3 desktop/tools/make_xcstrings.py
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "core", "LLMHub.Core", "strings.json")
TARGET = os.path.join(ROOT, "macos", "LLMHub", "Localizable.xcstrings")
CODES = {"en-US": "en", "es-ES": "es", "pt-BR": "pt-BR", "fr-FR": "fr", "de-DE": "de", "it-IT": "it", "ru-RU": "ru"}

table = json.load(open(SOURCE, encoding="utf-8"))
keys = list(table["en-US"])
for lang, strings in table.items():
    assert lang in CODES, lang
    assert list(strings) == keys, f"{lang} keys differ from en-US"
    for key, value in strings.items():
        # Store copy style: no em or en dashes in any language.
        assert "—" not in value and "–" not in value, (lang, key)
        assert set(re.findall(r"\{\d+\}", value)) == set(re.findall(r"\{\d+\}", table["en-US"][key])), (lang, key)

def swift_format(value):
    return re.sub(r"\{(\d+)\}", lambda m: f"%{int(m.group(1)) + 1}$@", value)

catalog = {"sourceLanguage": "en", "version": "1.0", "strings": {
    key: {"extractionState": "manual", "localizations": {
        CODES[lang]: {"stringUnit": {"state": "translated", "value": swift_format(strings[key])}} for lang, strings in table.items()}}
    for key in sorted(keys)}}
with open(TARGET, "w", encoding="utf-8") as f:
    json.dump(catalog, f, ensure_ascii=False, indent=2)
print(f"{len(keys)} keys x {len(table)} languages -> {os.path.relpath(TARGET, ROOT)}")
