# LLM Hub for macOS 🖥️

Native macOS desktop edition of **LLM Hub** — bringing private, on-device AI inference, autonomous agent workflows, and creative studios to Apple Silicon Macs.

---

## 🌟 Key Features (1:1 iOS Feature Parity)

| Tool | macOS Implementation |
|---|---|
| **💬 AI Chat** | Multi-pane desktop layout with conversation sidebar, collapsible reasoning/thinking steps, full Markdown syntax highlighting with copy, attachment drag-and-drop, tok/s speed metrics, and runtime inference parameter tuning. |
| **📦 Model Manager** | Hugging Face model browser, categorized filtering, download speed & ETA tracking, active model selector, Reveal in Finder, and drag-and-drop local `.gguf` import. |
| **💻 Vibe Coder** | Desktop dual-pane code editor + Live interactive `WKWebView` preview with instant HTML/JS execution and one-click code export. |
| **🤖 AI Agent** | Autonomous multi-step ReAct agent with tool calling, MCP client integration, and **native macOS Terminal shell command execution** with interactive user approval. |
| **✍️ Writing Aid** | Side-by-side editor with Summarize, Expand, Rewrite, Grammar, and Tone adjustment modes. |
| **🌍 Translator** | Dual-pane offline neural translation supporting 50+ languages with instant language swap and copy. |
| **🎙️ Transcriber** | Drag-and-drop audio transcription with on-device Whisper models and `.txt` export. |
| **🛡️ Scam Detector** | Phishing and social engineering analyzer with threat risk gauges and security recommendations. |
| **🗣️ Vibe Voice** | Hands-free continuous voice conversation interface with fluid animated audio visualizer. |
| **🎨 Image Generator** | On-device Stable Diffusion studio with prompt tuning, negative prompts, aspect ratio selection, and image export. |
| **🔍 Image Upscaler** | 2× and 4× AI super-resolution using RealESRGAN and UltraSharp models with before/after comparison. |
| **🎥 Video Generator** | Text-to-video and image-to-video generation with motion controls and video preview. |
| **🎵 Music Generator** | On-device music composition with live audio generation and waveform player. |
| **🔎 Instant Media Search** | Semantic photo and audio search powered by on-device embeddings. |
| **🎞️ Video Moment Finder** | Jump directly to scenes in video files using audio-visual query matching. |
| **🎭 creAItor Designer** | Custom persona designer using the PCTF (Persona, Context, Task, Format) framework. |
| **⚙️ Desktop Settings** | Metal GPU layer offloading controls, CPU thread tuning, 17+ interface languages, storage manager, and cache clearing. |

---

## 🏗️ Architecture & Code Reuse

The macOS implementation lives in a dedicated folder (`macos/`) and **reuses shared iOS business logic, models, and runtimes**:

```
macos/
├── README.md               # Desktop documentation & build instructions
├── sync_shared.py          # Python script to manage shared iOS file links/copies
└── LLMHub/
    ├── Package.swift       # Swift Package Manager manifest for macOS executable
    ├── LLMHub-Info.plist   # macOS application bundle configuration
    ├── LLMHub.entitlements # Security entitlements (Network, Audio, Files)
    ├── LLMHub.xcodeproj/   # Xcode project for macOS
    ├── Resources/          # Symlinked/copied resources (models.json, configs.json, lproj)
    └── Sources/LLMHub/
        ├── App/            # macOS App entry point & NavigationSplitView
        │   ├── LLMHubMacApp.swift
        │   ├── ContentView.swift
        │   └── PlatformCompatibility.swift
        ├── Theme/          # Apollo macOS design system & liquid background
        │   └── ApolloThemeMac.swift
        ├── UI/             # Desktop-translated UI screens & components
        │   ├── HomeScreen.swift
        │   ├── ChatScreen.swift
        │   ├── ChatSettingsSheet.swift
        │   ├── ModelDownloadScreen.swift
        │   ├── SettingsScreen.swift
        │   ├── FeatureScreens.swift   (Writing Aid, Translator, Transcriber, Vibe Coder, etc.)
        │   ├── MediaScreens.swift     (Upscaler, Video Gen, Music Gen, Media Search)
        │   ├── AgentScreen.swift      (AI Agent + Terminal execution)
        │   ├── PremiumScreen.swift
        │   └── Components/
        │       ├── CodeBlockView.swift
        │       ├── MarkdownTextView.swift
        │       └── WebPreviewView.swift
        └── Shared/         # Reused business logic directly from ios/LLMHub
            ├── ModelData.swift
            ├── ModelDownloader.swift
            ├── ModelManager.swift
            ├── ChatModels.swift
            ├── ChatStore.swift
            ├── LLMBackend.swift
            ├── LocalizationManager.swift
            ├── PurchaseManager.swift
            └── ... (28 shared backend & service files)
```

---

## 🚀 How to Build & Run

### In Xcode (Recommended)
```bash
open macos/LLMHub/LLMHub.xcodeproj
```
Select **LLMHub** scheme and target **My Mac (Mac Catalyst or Native Apple Silicon)**, then press **Cmd+R** to build and run.

### Via Swift CLI
```bash
cd macos/LLMHub
swift build
```

### Syncing Shared iOS Files
If working across environments where symlinks need to be converted to local file copies:
```bash
python3 macos/sync_shared.py --copy
```
Or to restore symlinks:
```bash
python3 macos/sync_shared.py
```
