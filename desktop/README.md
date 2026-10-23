# LLM Hub Desktop

Native desktop apps for LLM Hub: WinUI 3 on Windows (x64 and Arm64), SwiftUI on macOS, GTK 4 + libadwaita on Linux.
Each is built on its own OS.

```
desktop/
  shared/catalog.json      Curated model list. Names Hugging Face repos only; files and sizes are read from the Hub at runtime.
  core/LLMHub.Core/        Shared C# library for Windows and Linux: engines, downloads, chats, local API, view models.
                           strings.json holds every UI string in all seven languages.
  windows/LLMHub/          WinUI 3 app (C#, .NET 10, Windows App SDK, unpackaged)
  windows/LLMHub.UITests/  FlaUI end-to-end tests for the Windows app
  linux/LLMHub.Gtk/        GTK 4 / libadwaita app (C#, .NET 10, Gir.Core); UiTests.cs is its end-to-end suite
  macos/                   SwiftUI app (XcodeGen project.yml) and LLMHubUITests (XCUITest)
  tools/make_xcstrings.py  Builds the macOS string catalog from core/LLMHub.Core/strings.json
```

## Windows

Requirements: .NET 10 SDK, Windows 11 SDK (10.0.26100). Visual Studio is optional.

```powershell
cd desktop\windows\LLMHub
dotnet build -c Debug -p:Platform=x64      # or -p:Platform=ARM64 for Snapdragon
.\bin\x64\Debug\net10.0-windows10.0.26100.0\win-x64\LLMHub.exe
```

UI tests drive the real app on the interactive desktop (not over SSH), so close windows that could take focus first.
Arguments: the app to test, a folder for results and screenshots, and an optional test name filter.

```powershell
dotnet run --project desktop\windows\LLMHub.UITests -- <path\to\LLMHub.exe> C:\temp\uitest [filter]
```

## Linux

Build requirements: .NET 10 SDK. Runtime packages (Ubuntu names): `libgtk-4-1`, `libadwaita-1-0`, and `libgomp1`
(llama.cpp's Linux builds link OpenMP; without it `llama-server` exits with code 127).

```bash
cd desktop/linux/LLMHub.Gtk
dotnet build -c Debug
dotnet bin/Debug/net10.0/llmhub.dll
```

Under WSL, where WSLg's compositor is unreliable, run it on a virtual display: `Xvfb :99 & DISPLAY=:99 GDK_BACKEND=x11 dotnet ...`.

## macOS

Requirements: Xcode 26+, XcodeGen. macOS 26 or later (Liquid Glass; Apple Intelligence where available).

```bash
cd desktop/macos
xcodegen generate
xcodebuild -project LLMHub.xcodeproj -scheme LLMHub -derivedDataPath build build
xcodebuild -project LLMHub.xcodeproj -scheme LLMHub -derivedDataPath build-tests -destination 'platform=macOS' test
```

## Testing flags (Debug builds, all three apps)

```bash
# Render the window to a PNG after it loads (pages: chat, settings, models, api)
llmhub --capture shot.png --page models --wait 9000
# Download a catalog model if needed, send a message through the real UI, log the reply and speed, then capture
llmhub --selftest gemma-4-12b [--format litertlm] [--quant "LiteRT GPU"] [--prompt "..."] --capture shot.png --exit
# Linux only: run the built-in UI suite (optionally a comma-separated subset); exits with the failure count
llmhub --uitest [ChatReplyWithFooter,MtpToggle]
```

Results and timings go to the daily log (see Data below).

## How it runs models

- **llama.cpp**: on first use the app downloads the newest official llama.cpp release build for the machine
  (Windows: CUDA, ROCm, SYCL, Vulkan, Adreno OpenCL or CPU; Linux: CUDA, ROCm, SYCL, Vulkan or CPU; macOS: Metal)
  and runs `llama-server` as a child process on a random localhost port. If a build fails to start the next one is
  tried, and the one that works is remembered. The CUDA builds ship their runtime libraries, so users only need the
  NVIDIA driver, not the CUDA Toolkit.
- **LiteRT-LM**: `.litertlm` models run in-process through LiteRT-LM's C API, taken from Google's official
  `litert-lm-api` wheel on PyPI (Windows x64, Linux) or the Swift package (macOS). GPU first, then CPU. Models whose
  catalog entry has a drafter get an **MTP** (multi-token prediction) toggle in the composer, on by default.
  On Linux the first GPU load of each model runs in a child process, because a broken GPU driver (for example WSL's
  software Vulkan) crashes the process instead of returning an error.
- **Apple Intelligence** (macOS): the on-device Foundation model, listed like any installed model.

## Local API

Settings › Local API serves the installed models over OpenAI-style (`/v1/chat/completions`, `/v1/models`) and
Anthropic-style (`/v1/messages`, `/v1/messages/count_tokens`) endpoints, loopback only unless allowed, with an API key
by default. GGUF requests are proxied to `llama-server`; LiteRT-LM and Apple Intelligence are translated by the app.

## Data

| OS | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\LLMHub` |
| Linux | `~/.local/share/LLMHub` |
| macOS | `~/Library/Application Support/LLMHub` |

| Path | Contents |
|---|---|
| `models/<id>/` | Model files plus `manifest.json` |
| `engines/` | llama.cpp and LiteRT-LM builds |
| `llmhub.db` / `llmhub.sqlite` | Chats and messages (SQLite) |
| `settings.json` | Language, Hugging Face token, context length, MTP choices, local API |
| `logs/` | Daily log files |

## Localization

All strings live in `core/LLMHub.Core/strings.json` for en-US, es-ES, pt-BR, fr-FR, de-DE, it-IT and ru-RU, written
natively per language and without em or en dashes. Windows and Linux embed that file (`Loc.S("Key")`, in XAML
`{x:Bind services:Loc.S('Key')}`). After editing it, run `python3 desktop/tools/make_xcstrings.py` to update macOS.
