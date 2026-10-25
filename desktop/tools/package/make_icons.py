#!/usr/bin/env python3
"""Regenerates the MSIX logo images (desktop/windows/LLMHub/Assets/Package) from desktop/shared/icons/llmhub-1024.png.

Needs Pillow. The other installers use desktop/shared/icons directly: llmhub.ico (Windows exe, Inno Setup),
llmhub.png and llmhub-512.png (Linux hicolor theme, AppImage); macOS uses its asset catalog.
"""
from pathlib import Path

from PIL import Image

DESKTOP = Path(__file__).resolve().parents[2]
SOURCE = DESKTOP / "shared" / "icons" / "llmhub-1024.png"
OUT = DESKTOP / "windows" / "LLMHub" / "Assets" / "Package"

SCALES = {100: 1.0, 125: 1.25, 150: 1.5, 200: 2.0, 400: 4.0}


def square(icon: Image.Image, size: int) -> Image.Image:
    return icon.resize((size, size), Image.Resampling.LANCZOS)


def banner(icon: Image.Image, width: int, height: int) -> Image.Image:
    """The icon centred on a canvas of its own edge colour (wide tile, splash screen)."""
    canvas = Image.new("RGB", (width, height), icon.getpixel((0, 0)))
    side = height
    canvas.paste(square(icon, side), ((width - side) // 2, 0))
    return canvas


def main() -> None:
    icon = Image.open(SOURCE).convert("RGB")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    images: dict[str, Image.Image] = {}
    for scale, factor in SCALES.items():
        images[f"Square44x44Logo.scale-{scale}.png"] = square(icon, round(44 * factor))
        images[f"StoreLogo.scale-{scale}.png"] = square(icon, round(50 * factor))
        # The App Certification Kit rejects logo files of 200 KB or more, which this artwork reaches at 600 px.
        if scale <= 200:
            images[f"Square150x150Logo.scale-{scale}.png"] = square(icon, round(150 * factor))
    for scale in (100, 200):
        images[f"Wide310x150Logo.scale-{scale}.png"] = banner(icon, 310 * scale // 100, 150 * scale // 100)
    # Required by the manifest schema; desktop apps never show it.
    images["SplashScreen.scale-100.png"] = banner(icon, 620, 300)
    # Taskbar, Start menu list and Alt+Tab sizes; "unplated" draws them without a tile plate.
    for size in (16, 24, 32, 48, 256):
        images[f"Square44x44Logo.targetsize-{size}.png"] = square(icon, size)
        images[f"Square44x44Logo.targetsize-{size}_altform-unplated.png"] = square(icon, size)
    for name, image in images.items():
        image.save(OUT / name, optimize=True)
    print(f"{len(images)} images in {OUT}")


if __name__ == "__main__":
    main()
