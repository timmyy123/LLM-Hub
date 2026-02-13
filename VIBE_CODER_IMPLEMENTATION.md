# Vibe Coder Implementation Plan

## Overview
The **Vibe Coder** feature enables users to prompt an LLM agent to generate code (HTML/JavaScript SPAs or simple Python), then view and interact with the generated code in a dedicated canvas/webview UI. After interaction, users can seamlessly return to the chat interface.

## Architecture & Data Flow

```
User Input (VibeCoderScreen)
    ↓
[Model Selection & Inference Service]
    ↓
LLM Response (code + stdout)
    ↓
[Parse & Validate Code]
    ↓
Display Options:
  - View in WebView (HTML/JS)
  - View in Terminal/Log (Python)
    ↓
[User Interaction]
    ↓
[Return to Chat / Save / Share]
```

## Implementation Phases

### Phase 1: Core Infrastructure (VibeCoderViewModel & Data Models)
**Status**: ✅ COMPLETED

#### Completed Tasks:
- ✅ Created `VibeCoderViewModel` with full state management
- ✅ Integrated with `UnifiedInferenceService`
- ✅ Implemented system prompt for code generation 
- ✅ Added code language detection & extraction from model responses
- ✅ State flows: input, generated code, loading, error, model selection
- ✅ Methods: `generateCode()`, `selectModel()`, `selectBackend()`, `loadModel()`, `unloadModel()`

#### Implementation Details:
- Supports HTML, Python, JavaScript code detection
- Extracts code from markdown code blocks and XML tags
- Handles markdown format: ````html` ... ````
- Token streaming integration with `generateResponseStreamWithSession()`
- Unique chatId per session to avoid conflicts
- Comprehensive error handling and logging

---

### Phase 2: UI Screen & Layout (VibeCoderScreen)
**Status**: ✅ COMPLETED

#### Completed Tasks:
- ✅ Created `VibeCoderScreen` composable in new `VibeCoderScreen.kt` file
- ✅ Implemented full UI layout:
  1. Header with navigation and settings
  2. Model Selector Card (reused from components)
  3. Prompt Input section with text field & buttons
  4. Response Display with code preview (scrollable)
  5. Loading/Error states with proper feedback
  6. Action buttons: Copy, Preview (HTML only)
- ✅ Added Backend Selector in Settings Sheet
- ✅ Auto-scrolling to generated code during generation
- ✅ Copy to clipboard functionality
- ✅ Clear input/output functionality
- ✅ Model loading/unloading lifecycle

#### Navigation Integration:
- ✅ Updated `LlmHubNavigation.kt` with VibeCoderScreen route
- ✅ Linked from HomeScreen feature cards
- ✅ Settings button for backend selection
- ✅ Callback for navigating to code canvas

---

### Phase 3: WebView Canvas for HTML/JS Code
**Status**: ✅ COMPLETED

#### Completed Tasks:
- ✅ Created `CodeCanvasScreen` composable in new `CodeCanvasScreen.kt` file
- ✅ WebView implementation with HTML content display
- ✅ Security features:
  - Disabled JavaScript by default (can be enabled if needed)
  - Removed `<script>` tags and event handlers
  - Sanitization against XSS attacks (javascript: protocol removal)
  - Safe tag allowlist approach
- ✅ Error handling with user-friendly error messages
- ✅ HTML wrapping with viewport meta tags and default styling
- ✅ Back button overlay for return to VibeCoderScreen
- ✅ Navigation integration with URL-encoded code parameters

#### Navigation Integration:
- ✅ Updated `LlmHubNavigation.kt` with CodeCanvas route
- ✅ Route with URL parameters: `code_canvas?code={code}&type={type}`
- ✅ Navigation arguments properly decoded
- ✅ VibeCoderScreen Preview button calls `onNavigateToCanvas`

---

### Phase 4: Python Code Execution Output
**Status**: 📋 DEFERRED (Post-MVP)

For MVP, users can:
- Copy Python code and run it separately on their device
- View execution results if provided manually

---

### Phase 5: Inference Service Integration
**Status**: ✅ COMPLETED

#### Completed Tasks:
- ✅ Integrated with `UnifiedInferenceService`
  - Model selection with backend preference (GPU/CPU)
  - Token streaming via `generateResponseStreamWithSession()`
  - Proper session management with unique chatIds
- ✅ System prompt for code generation
- ✅ Code extraction and detection
- ✅ Error handling and logging

---

### Phase 6: Code Persistence & History
**Status**: 📋 PLANNED (Post-MVP)

---

### Phase 7: Error Handling & Robustness
**Status**: ✅ PARTIALLY COMPLETED

#### Completed:
- ✅ Input validation (non-empty prompts)
- ✅ Output validation (HTML sanitization)
- ✅ Error message display to user
- ✅ Graceful degradation (fallback error screens)

---

### Phase 8: String Localization & Accessibility
**Status**: ✅ PARTIALLY COMPLETED

#### Completed:
- ✅ Added all English strings (values/strings.xml)

#### Remaining (Post-MVP):
- Translations for 15+ languages

---

### Phase 9: Testing & QA
**Status**: ✅ CORE COMPILATION VERIFIED

#### Completed:
- ✅ No compilation errors
- ✅ Navigation integration verified

---

### Phase 10: Documentation & Code Cleanup
**Status**: 📋 PLANNED

---

## MVP Scope (Minimum Viable Product) - ✅ COMPLETED

### ✅ MVP Features Implemented:
1. ✅ VibeCoderScreen with prompt input
2. ✅ Model selector & inference integration  
3. ✅ HTML/JS code generation via LLM
4. ✅ WebView canvas for code preview
5. ✅ Copy code to clipboard
6. ✅ Navigation flow (Home → Vibe Coder → Canvas)
7. ✅ Error handling and user feedback
8. ✅ English string resources

### 📋 Post-MVP Features (Phase 6+):
- Python execution on-device
- Code syntax highlighting
- Code history and persistence
- Multilingual support (15+ languages)
- Share/export functionality
- Code versioning and collaboration

---

## Key Dependencies

### Code Patterns to Reuse:
- **WritingAidScreen** (FeatureScreens.kt) — Text input + model selection pattern
- **WritingAidViewModel** — State management structure
- **ChatViewModel** — Inference service integration
- **UnifiedInferenceService** — Token streaming, model loading
- **LocalClipboardManager** — Copy to clipboard
- **Room Database** — Session persistence (Phase 6)

---

## File Structure - IMPLEMENTATION COMPLETE

```
app/src/main/java/com/llmhub/llmhub/
├── screens/
│   ├── VibeCoderScreen.kt         ✅ NEW - Main UI screen
│   ├── CodeCanvasScreen.kt        ✅ NEW - WebView canvas
│   └── ...
├── viewmodels/
│   ├── VibeCoderViewModel.kt      ✅ NEW - State management
│   └── ...
├── data/
│   ├── ... (database models planned for Phase 6)
├── navigation/
│   └── LlmHubNavigation.kt        ✅ UPDATED - Added routes
└── ...

app/src/main/res/
├── values/
│   └── strings.xml                ✅ UPDATED - Added vibe_coder strings
└── ...
```

---

## Progress Tracking - PHASES 1-5 COMPLETE

| Phase | Task | Status | Details |
|-------|------|--------|---------|
| 1 | VibeCoderViewModel | ✅ DONE | Full state management & inference |
| 1 | Code Detection | ✅ DONE | HTML/Python/JavaScript detection |
| 2 | VibeCoderScreen UI | ✅ DONE | Full feature-rich UI |
| 2 | String Resources | ✅ DONE | English strings added |
| 3 | CodeCanvasScreen | ✅ DONE | WebView with sanitization |
| 4 | Code Output | 📋 POST-MVP | Python execution deferred |
| 5 | Inference Integration | ✅ DONE | LLM inference complete |
| 6 | Database & History | 📋 PHASE 6 | Room database entities |
| 7 | Error Handling | ✅ PARTIAL | Core error handling done |
| 8 | Localization | ✅ PARTIAL | English complete, 15+ langs pending |
| 9 | Testing | ✅ PARTIAL | Compilation verified |
| 10 | Documentation | 📋 PHASE 10 | KDoc and guides pending |

---

## Files Created/Modified

### NEW FILES:
1. **VibeCoderViewModel.kt** - Core state management & inference logic
2. **VibeCoderScreen.kt** - Main UI with prompt input & code display
3. **CodeCanvasScreen.kt** - WebView for HTML code preview
4. **VIBE_CODER_IMPLEMENTATION.md** - This project plan

### MODIFIED FILES:
1. **LlmHubNavigation.kt** - Added VibeCoder & CodeCanvas routes
2. **strings.xml** - Added vibe_coder English strings

---

## Known Blockers / Open Questions

1. **Syntax Highlighting**: Optional for Phase 9 (Use Prism.js in WebView?)
2. **Code Validation**: Current validation is basic; can be enhanced
3. **Model Recommendations**: Could recommend larger models for code generation 
4. **Python Execution**: MVP uses copy-paste; on-device execution for Phase 6+
5. **Context Window**: Currently streams until completion; warning can be added

---

## Next Steps for Phase 6 (Post-MVP)

1. Create Room database entities for session persistence
2. Implement VibeCoderSessionDao for CRUD operations
3. Add recent sessions list UI to VibeCoderScreen
4. Implement share/export functionality
5. Add translations for 15+ languages
6. Python execution sandbox (if feasible)
7. Advanced HTML sanitization library integration
8. Code versioning and change tracking

---

## References & Related Files

- [ChatScreen.kt](app/src/main/java/com/llmhub/llmhub/screens/ChatScreen.kt) - Chat pattern
- [WritingAidViewModel.kt](app/src/main/java/com/llmhub/llmhub/viewmodels/WritingAidViewModel.kt) - VM pattern
- [UnifiedInferenceService.kt](app/src/main/java/com/llmhub/llmhub/inference/UnifiedInferenceService.kt) - Inference
- [LlmHubNavigation.kt](app/src/main/java/com/llmhub/llmhub/navigation/LlmHubNavigation.kt) - Navigation
