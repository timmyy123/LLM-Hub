#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ASPECT_ORDER = ["1x1", "1x2", "2x1", "3x4", "4x3"]
SPECIAL_DIRECTORIES = {"Scores", "EditTest"}
CATEGORY_DIRECTORY_PATTERN = re.compile(r"^C\d+_.+")
SCORES_FILENAME_PATTERN = re.compile(
    r"^(?:opus_4\.6_scores|gpt_5_4_scores|gpt_5_5_polished_scores)(?:_(1x2|2x1|3x4|4x3))?\.csv$"
)


@dataclass(frozen=True)
class Variant:
    aspect: str
    filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ~/TestSet into a Cloudflare-friendly viewer bundle."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "TestSet",
        help="Source TestSet directory. Defaults to ~/TestSet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for manifest.json, images/, and thumbs/.",
    )
    parser.add_argument(
        "--thumb-max-size",
        type=int,
        default=448,
        help="Maximum width/height for generated thumbnails.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy source images into output/images. Without this flag only manifest and thumbs are written.",
    )
    parser.add_argument(
        "--skip-thumbnails",
        action="store_true",
        help="Do not generate output/thumbs.",
    )
    return parser.parse_args()


def iter_category_dirs(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.iterdir()
        if path.is_dir()
        and path.name not in SPECIAL_DIRECTORIES
        and not path.name.startswith(".")
        and CATEGORY_DIRECTORY_PATTERN.match(path.name)
    )


def display_title(category_id: str) -> str:
    return category_id.replace("_", " ")


def parse_variant(filename: str) -> Variant:
    stem = Path(filename).stem
    for aspect in ASPECT_ORDER[1:]:
        suffix = f"_{aspect}"
        if stem.endswith(suffix):
            return Variant(aspect=aspect, filename=filename)
    return Variant(aspect="1x1", filename=filename)


def parse_scores(csv_path: Path) -> dict[str, dict[str, Any]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        result: dict[str, dict[str, Any]] = {}
        for row in reader:
            model = (row.get("model") or "").strip()
            if not model:
                continue
            score_text = (row.get("score") or "").strip()
            score: int | float | None
            if score_text:
                try:
                    score_value = float(score_text)
                    score = int(score_value) if score_value.is_integer() else score_value
                except ValueError:
                    score = None
            else:
                score = None
            result[model] = {
                "score": score,
                "note": (row.get("note") or "").strip(),
            }
        return result


def parse_scores_by_aspect(category_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for csv_path in sorted(category_dir.glob("*scores*.csv")):
        match = SCORES_FILENAME_PATTERN.match(csv_path.name)
        if not match:
            continue
        aspect = match.group(1) or "1x1"
        result[aspect] = parse_scores(csv_path)
    return result


def read_prompt(category_dir: Path) -> str:
    prompt_path = category_dir / "prompt.txt"
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8").strip()


def read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_metadata(category_dir: Path) -> dict[str, Any]:
    metadata_path = category_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_data_json(source: Path) -> dict[str, dict[str, str]]:
    data_json_path = source / "data.json"
    if not data_json_path.exists():
        return {}
    try:
        entries = json.loads(data_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    result: dict[str, dict[str, str]] = {}
    for entry in entries:
        category_id = str(entry.get("dir") or "").strip()
        if not category_id:
            continue
        result[category_id] = {
            "prompt": str(entry.get("prompt") or "").strip(),
            "promptZh": str(entry.get("prompt_zh") or "").strip(),
        }
    return result


def collect_images(category_dir: Path) -> dict[str, list[Variant]]:
    models: dict[str, list[Variant]] = {}
    for path in sorted(category_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        variant = parse_variant(path.name)
        model_id = Path(variant.filename).stem
        if variant.aspect != "1x1":
            model_id = model_id[: -(len(variant.aspect) + 1)]
        models.setdefault(model_id, []).append(variant)
    return models


def aspect_sort_key(aspect: str) -> int:
    try:
        return ASPECT_ORDER.index(aspect)
    except ValueError:
        return len(ASPECT_ORDER)


def ensure_clean_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "thumbs").mkdir(parents=True, exist_ok=True)


def copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def create_thumbnail(source: Path, destination: Path, max_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        thumbnail = image.copy()
        thumbnail.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        save_kwargs: dict[str, Any] = {}
        if destination.suffix.lower() in {".jpg", ".jpeg"}:
            if thumbnail.mode not in {"RGB", "L"}:
                thumbnail = thumbnail.convert("RGB")
            save_kwargs.update({"quality": 85, "optimize": True, "progressive": True})
        elif destination.suffix.lower() == ".png":
            save_kwargs.update({"optimize": True})
        elif destination.suffix.lower() == ".webp":
            save_kwargs.update({"quality": 85, "method": 6})
        thumbnail.save(destination, **save_kwargs)


def build_manifest(source: Path, output: Path, copy_images: bool, skip_thumbnails: bool, thumb_max_size: int) -> dict[str, Any]:
    categories: list[dict[str, Any]] = []
    total_images = 0
    prompt_overrides = load_data_json(source)
    for category_dir in iter_category_dirs(source):
        category_id = category_dir.name
        metadata = load_metadata(category_dir)
        prompt = read_prompt(category_dir)
        prompt_override = prompt_overrides.get(category_id, {})
        if prompt_override.get("prompt"):
            prompt = prompt_override["prompt"]
        elif not prompt:
            prompt = str(metadata.get("prompt") or "").strip()
        prompt_zh = (
            prompt_override.get("promptZh")
            or read_optional_text(category_dir / "prompt_zh.txt")
            or str(metadata.get("prompt_zh") or "").strip()
        )
        level = str(metadata.get("level") or "").strip()
        title = str(metadata.get("title") or "").strip() or display_title(category_id)
        scores_by_aspect = parse_scores_by_aspect(category_dir)
        image_groups = collect_images(category_dir)

        models: list[dict[str, Any]] = []
        ordered_model_ids: list[str] = []
        for aspect_scores in scores_by_aspect.values():
            for model_id in aspect_scores.keys():
                if model_id not in ordered_model_ids:
                    ordered_model_ids.append(model_id)
        for model_id in sorted(image_groups.keys()):
            if model_id not in ordered_model_ids:
                ordered_model_ids.append(model_id)

        for model_id in ordered_model_ids:
            variants = sorted(image_groups.get(model_id, []), key=lambda item: aspect_sort_key(item.aspect))
            if not variants:
                continue
            default_score_entry = scores_by_aspect.get("1x1", {}).get(model_id, {})
            variant_entries: list[dict[str, Any]] = []
            for variant in variants:
                source_image = category_dir / variant.filename
                image_key = f"images/{category_id}/{variant.filename}"
                thumb_key = f"thumbs/{category_id}/{variant.filename}"
                score_entry = scores_by_aspect.get(variant.aspect, {}).get(model_id, default_score_entry)
                if copy_images:
                    copy_image(source_image, output / image_key)
                if not skip_thumbnails:
                    create_thumbnail(source_image, output / thumb_key, thumb_max_size)
                variant_entries.append(
                    {
                        "aspect": variant.aspect,
                        "filename": variant.filename,
                        "imageKey": image_key,
                        "thumbKey": None if skip_thumbnails else thumb_key,
                        "score": score_entry.get("score"),
                        "note": score_entry.get("note", ""),
                    }
                )
                total_images += 1

            models.append(
                {
                    "id": model_id,
                    "label": model_id,
                    "score": default_score_entry.get("score"),
                    "note": default_score_entry.get("note", ""),
                    "variants": variant_entries,
                }
            )

        categories.append(
            {
                "id": category_id,
                "title": title,
                "level": level,
                "prompt": prompt,
                "promptZh": prompt_zh,
                "models": models,
            }
        )

    return {
        "version": "v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            "categoryCount": len(categories),
            "imageCount": total_images,
        },
        "availableAspects": ASPECT_ORDER,
        "categories": categories,
    }


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source directory does not exist: {source}")

    ensure_clean_output(output)
    manifest = build_manifest(
        source=source,
        output=output,
        copy_images=args.copy_images,
        skip_thumbnails=args.skip_thumbnails,
        thumb_max_size=args.thumb_max_size,
    )
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_hash_full = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    manifest_hash = manifest_hash_full[:16]
    manifest_name = f"manifest_{manifest_hash}.json"
    manifest_path = output / manifest_name
    manifest_path.write_text(manifest_json, encoding="utf-8")
    (output / "manifest_name.txt").write_text(manifest_name + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Manifest filename: {manifest_name}")
    print(f"Categories: {manifest['summary']['categoryCount']}")
    print(f"Images: {manifest['summary']['imageCount']}")
    if args.copy_images:
        print(f"Images copied into: {output / 'images'}")
    if not args.skip_thumbnails:
        print(f"Thumbnails written into: {output / 'thumbs'}")


if __name__ == "__main__":
    main()
