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
  tools/package/           Installer scripts (see Packaging)
  version.xcconfig         The app version for all three apps and every installer
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
(llama.cpp's Linux builds link OpenMP; without it `llama-server` exits with code 127). Studio › Search reads audio
and videos with `ffmpeg` or GStreamer's `gst-launch-1.0` (`gstreamer1.0-tools`), whichever is installed; GStreamer
needs `gstreamer1.0-libav` for H.264 and AAC.

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
# Studio › Search: index a folder (photos, audio or videos) and run queries; "=>" names the right answer
# (a file, or for videos file@from-to seconds), logged as SEARCHTEST PASS/FAIL. --playmoment then plays the best
# video moment, muted, and logs where the player is (macOS: --page studio).
llmhub --searchtest videos --folder <dir> --queries "fireworks at night=>clip.mp4@20-30" --playmoment --capture shot.png --page search --exit
# Linux only: run the built-in UI suite (optionally a comma-separated subset); exits with the failure count
llmhub --uitest [ChatReplyWithFooter,MtpToggle]
# macOS only: the chat mic on the live microphone for N seconds (play speech meanwhile), transcript to the log
LLMHub --dictatetest 12 [--noprompt]
```

On macOS `--capture` takes the window as the window server composites it (ScreenCaptureKit, own window, no screen
recording permission), so Liquid Glass shows. The XCUITest suite (`LLMHubUITests`) needs Automation Mode, which macOS
asks the signed-in user to approve; the hosted unit tests (`-only-testing:LLMHubTests`) do not.

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

## Packaging

The version lives in `desktop/version.xcconfig` (`MARKETING_VERSION`, plus `CURRENT_PROJECT_VERSION` as the build
number). Xcode, MSBuild (`desktop/Directory.Build.props`) and the scripts below all read it. Icons come from
`desktop/shared/icons`; `python3 desktop/tools/package/make_icons.py` regenerates the MSIX logos from
`llmhub-1024.png`. Output goes to `desktop/tools/package/out`, intermediate files to `desktop/tools/package/work`
(both ignored by git; delete `work` afterwards).

Installers carry the app and its runtimes only. llama.cpp, LiteRT-LM, whisper.cpp, ComfyUI with its Python, voices
and every model are downloaded by the app on first use, into the data folder (see Data), so they survive updates.

### Windows (x64 and Arm64)

Requirements: everything under Windows above, Visual Studio's C++ build tools for x64 and ARM64 (the installers ship
the Visual C++ runtime from its `VC\Redist` folder, which ONNX Runtime and the downloaded llama.cpp builds need) and
Inno Setup 6.3 or later (`winget install JRSoftware.InnoSetup`). Run on an x64 PC; Arm64 is cross-compiled.

```powershell
desktop\tools\package\windows\package-windows.ps1 [-Arch x64|arm64|all] [-Kind setup|msix|all] [-SdCli <sd-cli.exe>]
```

| File | What it is |
|---|---|
| `LLMHub-<version>-win-<arch>-setup.exe` | Per-user installer for direct download. No admin rights: installs to `%LOCALAPPDATA%\Programs\LLM Hub`, adds a Start menu shortcut (desktop icon optional) and an entry in Settings › Apps for uninstalling, which offers to delete the data folder too. Upgrades install over the old version. |
| `LLMHub-<version>-win-<arch>.msix`, `LLMHub-<version>-win.msixbundle` | Microsoft Store packages (`-p:WindowsPackageType=MSIX`, manifest in `windows/LLMHub/Package.appxmanifest`). Unsigned: Partner Center signs Store packages. |

Both are self-contained (.NET 10, Windows App SDK, Visual C++ runtime). The Arm64 build adds Qualcomm's QNN runtime and
`sd-cli.exe` for Studio's NPU video, built with `desktop\windows\sdcpp\build.ps1` (or passed with `-SdCli`); without it
Studio hides that workflow.

Signing: the installers work unsigned, but SmartScreen shows "Windows protected your PC" (More info › Run anyway) until
the publisher has a reputation. To sign `LLMHub.exe`, the installer and its uninstaller, set before running the script:
`LLMHUB_SIGN_CERT` (a .pfx) and `LLMHUB_SIGN_PASSWORD`, or `LLMHUB_SIGN_THUMBPRINT` (a certificate in the user's store,
for example on a token), or `LLMHUB_SIGN_COMMAND` (any command line with `{file}`, for example Azure Trusted Signing's
`signtool /dlib` command). `LLMHUB_SIGN_TIMESTAMP` changes the timestamp server.

Microsoft Store: reserve the name in Partner Center, then build with the values from Product management › Product
identity and upload the `.msixbundle`:

```powershell
desktop\tools\package\windows\package-windows.ps1 -Kind msix -StoreIdentityName <Package/Identity/Name> `
  -StorePublisher "<Package/Identity/Publisher>" -StorePublisherDisplayName "<Publisher display name>"
```

The package uses the `runFullTrust` restricted capability (a desktop app that starts engines as child processes), which
Partner Center asks to justify. Each upload needs a higher `MARKETING_VERSION`. The Windows App Certification Kit
(`appcert.exe test -appxpackagepath <msix> -reportoutputpath report.xml`) passes; its optional checks flag duplicate
WinRT GUIDs in Windows App SDK's own metadata and process launching, both expected for a full-trust app.

### macOS

Requirements: everything under macOS above. Apple silicon only (MLX and Apple Intelligence need it).

```bash
desktop/tools/package/macos/package-macos.sh [--identity "<signing identity>"] [--notarize <keychain profile>]
```

It builds Release, signs every nested library and the app with the hardened runtime and
`macos/LLMHub/LLMHub.entitlements`, and makes `LLMHub-<version>-macos-arm64.dmg` with the app and an Applications link
to drag it onto. Without `--identity` it uses the first "Developer ID Application" certificate in the keychain, else
signs ad hoc. An "Apple Development" identity works for testing on your own Macs only.

Entitlements: the app is not sandboxed (it runs llama.cpp, ComfyUI and Claude Code as child processes and reads folders
you pick), so network and file entitlements do not apply. Under the hardened runtime it needs only
`com.apple.security.device.audio-input` (dictation). No JIT or unsigned-memory entitlements: MLX compiles Metal
shaders through Metal, the bundled LiteRT-LM, ONNX Runtime and whisper.cpp libraries are signed with the app, and the
downloaded llama.cpp runs as its own process.

To ship outside the Mac App Store it must be signed with Developer ID and notarized:

1. Create the certificate (Account Holder role): Xcode › Settings › Accounts › Manage Certificates › + ›
   Developer ID Application. Check it with `security find-identity -v -p codesigning`.
2. Create an app-specific password at account.apple.com (Sign-In and Security › App-Specific Passwords), then save it
   for notarytool (it asks for the password): `xcrun notarytool store-credentials llmhub-notary --apple-id <Apple ID>
   --team-id 76K65T869A`.
3. Build, sign, notarize and staple: `desktop/tools/package/macos/package-macos.sh --identity "Developer ID Application:
   <Name> (76K65T869A)" --notarize llmhub-notary`. If Apple rejects it, `xcrun notarytool log <submission id>
   --keychain-profile llmhub-notary` says why.
4. Check: `spctl -a -t open --context context:primary-signature -vv <dmg>` says "accepted, source=Notarized Developer
   ID", and so does `spctl -a -t exec -vv /Applications/LLMHub.app` after installing it.

### Linux (x64)

Requirements: .NET 10 SDK and `dpkg-deb` (any Linux; WSL works).

```bash
desktop/tools/package/linux/package-linux.sh [--kind deb|appimage|all]
```

| File | What it is |
|---|---|
| `llmhub_<version>_amd64.deb` | Debian, Ubuntu 24.04 or later and derivatives. Installs to `/usr/lib/llmhub` with `llmhub` on PATH, a desktop entry, icons and AppStream data. Depends on GTK 4, libadwaita 1.4+, libgomp1, OpenSSL and ICU; recommends WebKitGTK 6.0 (Design preview), PipeWire or ALSA tools (dictation, read aloud), GStreamer or FFmpeg (Studio search) and git (Agent). |
| `LLMHub-<version>-x86_64.AppImage` | Every other distribution: one file, `chmod +x` and run. Uses the system's GTK 4 and libadwaita and says what to install when they are missing. |

Both are self-contained .NET publishes (no .NET install needed), with the desktop entry and icon under the GTK
application ID `app.llmhub.Desktop`. The script downloads a pinned appimagetool and AppImage runtime (checked by
SHA-256). AppImage rather than Flatpak: Flatpak's sandbox would cut the app off from what it is built around, namely
running the llama.cpp, ComfyUI and Python builds it downloads with the host's GPU drivers (CUDA, ROCm, Vulkan) and
driving Claude Code and git on the host for Agent; it would need `--talk-name=org.freedesktop.Flatpak` and device
access that Flathub does not accept for this. An AppImage runs unconfined, like the .deb.
