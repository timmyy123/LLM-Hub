#!/usr/bin/env python3
"""
Gemma-3 Model Summary for MediaPipe Android LLM App
Complete overview of available models and how to get them
"""
import os
from pathlib import Path

def print_section(title, char="="):
    """Print a formatted section header"""
    print(f"\n{title}")
    print(char * len(title))

def print_model_info(name, status, size, description, action):
    """Print formatted model information"""
    status_icon = "✅" if status == "available" else "🔧" if status == "convertible" else "❌"
    print(f"\n{status_icon} **{name}**")
    print(f"   Size: {size}")
    print(f"   Description: {description}")
    print(f"   Action: {action}")

def check_existing_models():
    """Check what models already exist in the models directory"""
    models_dir = Path("models")
    existing = []
    
    if models_dir.exists():
        for file in models_dir.glob("*.task"):
            existing.append(file.name)
        for file in models_dir.glob("*.bin"):
            existing.append(file.name)
    
    return existing

def main():
    """Display comprehensive Gemma-3 model summary"""
    print("🎯 GEMMA-3 MODELS FOR MEDIAPIPE ANDROID LLM")
    print("=" * 50)
    
    print("\n📱 **Your Requirements:** Gemma-3 models in 1B, 4B, and 12B sizes")
    print("🔧 **Target Platform:** Android with MediaPipe LLM Inference API")
    print("📁 **Required Format:** .task files")
    
    print_section("📊 MODEL AVAILABILITY STATUS")
    
    print_model_info(
        "Gemma-3 1B (INT4 Quantized)",
        "available",
        "~529MB",
        "Fastest, optimized for mobile devices",
        "Download directly from HuggingFace"
    )
    
    print_model_info(
        "Gemma-3 1B (INT8 Quantized)",
        "available", 
        "~1005MB",
        "Higher quality than INT4, still mobile-friendly",
        "Download directly from HuggingFace"
    )
    
    print_model_info(
        "Gemma-3n E2B (Multimodal)",
        "available",
        "~2.9GB",
        "Effective 2B parameters, supports text + images",
        "Download directly from HuggingFace"
    )
    
    print_model_info(
        "Gemma-3 4B",
        "convertible",
        "~4GB",
        "Medium size, good balance of quality and performance",
        "Convert from HuggingFace safetensors to .task"
    )
    
    print_model_info(
        "Gemma-3 12B",
        "convertible", 
        "~12GB",
        "Largest model, highest quality but resource intensive",
        "Convert from HuggingFace safetensors to .task"
    )
    
    print_section("🚀 QUICK START GUIDE")
    
    print("\n**Step 1: Download Available Models (1B variants)**")
    print("```")
    print("python public/download_gemma3_mediapipe.py list")
    print("python public/download_gemma3_mediapipe.py download gemma-3-1b-int4")
    print("python public/download_gemma3_mediapipe.py download gemma-3-1b-int8")
    print("```")
    
    print("\n**Step 2: Convert Missing Models (4B, 12B)**")
    print("```")
    print("python public/convert_gemma3_to_mediapipe.py list")
    print("python public/convert_gemma3_to_mediapipe.py convert gemma-3-4b")
    print("python public/convert_gemma3_to_mediapipe.py convert gemma-3-12b")
    print("```")
    
    print("\n**Step 3: Get Everything at Once**")
    print("```")
    print("# Download all available models")
    print("python public/download_gemma3_mediapipe.py download-all")
    print("")
    print("# Convert all missing models")
    print("python public/convert_gemma3_to_mediapipe.py convert-all")
    print("```")
    
    print_section("📁 EXPECTED OUTPUT STRUCTURE")
    
    print("""
models/
├── gemma3-1b-it-int4.task           (529MB)  ✅ Ready for Android
├── gemma3-1b-it-int8.task           (1GB)    ✅ Ready for Android  
├── gemma3n-e2b-multimodal.task      (2.9GB)  ✅ Ready for Android
├── gemma3-4b-it-converted.task      (4GB)    🔧 After conversion
└── gemma3-12b-it-converted.task     (12GB)   🔧 After conversion
""")
    
    print_section("⚙️ ANDROID INTEGRATION")
    
    print("""
**1. Copy .task files to Android assets:**
```
app/src/main/assets/models/
├── gemma3-1b-it-int4.task
├── gemma3-4b-it-converted.task
└── gemma3-12b-it-converted.task
```

**2. Update your Android model list:**
```kotlin
val models = listOf(
    ModelInfo("Gemma-3 1B", "gemma3-1b-it-int4.task", "529MB"),
    ModelInfo("Gemma-3 4B", "gemma3-4b-it-converted.task", "4GB"),
    ModelInfo("Gemma-3 12B", "gemma3-12b-it-converted.task", "12GB")
)
```

**3. Use in MediaPipe LLM Inference:**
```kotlin
val llmInference = LlmInference.createFromFile(context, modelPath)
val response = llmInference.generateResponse(prompt)
```
""")
    
    print_section("🔧 TROUBLESHOOTING")
    
    print("""
**If downloads fail:**
- Check internet connection
- Verify HuggingFace access (may need login for some models)
- Try downloading individual models instead of all at once

**If conversion fails:**
- Ensure you have enough disk space (models are large)
- Check Python dependencies: mediapipe, torch, transformers
- Try manual conversion using MediaPipe tools

**For manual conversion help:**
```
python public/convert_gemma3_to_mediapipe.py help
```

**Performance recommendations:**
- Start with Gemma-3 1B for mobile devices
- Use 4B for better quality if you have sufficient RAM
- Reserve 12B for high-end devices or specific use cases
""")
    
    print_section("📊 EXISTING FILES CHECK")
    
    existing_models = check_existing_models()
    if existing_models:
        print(f"\n✅ Found {len(existing_models)} existing model files:")
        for model in existing_models:
            print(f"   - {model}")
    else:
        print("\n📁 No existing model files found in 'models/' directory")
        print("   Run the download/conversion scripts to get started!")
    
    print_section("🎯 RECOMMENDED WORKFLOW")
    
    print("""
**For Mobile Development (Start Here):**
1. Download Gemma-3 1B INT4 (~529MB) - fastest for mobile
2. Test your Android app with this model first
3. If you need better quality, try Gemma-3 1B INT8 (~1GB)

**For Enhanced Performance:**
4. Convert Gemma-3 4B (~4GB) - best balance for most use cases
5. Test on your target devices to ensure performance is acceptable

**For Maximum Quality (Advanced):**
6. Convert Gemma-3 12B (~12GB) - only for high-end devices
7. Benchmark carefully due to large memory requirements

**Multimodal Features:**
- Download Gemma-3n E2B if you need image understanding capabilities
""")
    
    print(f"\n🔗 **Next Steps:**")
    print(f"   1. Choose your models based on device capabilities")
    print(f"   2. Run the appropriate download/conversion scripts")
    print(f"   3. Test models in your Android app")
    print(f"   4. Optimize for your specific use case")
    
    print(f"\n📚 **Documentation:**")
    print(f"   - MediaPipe LLM: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference")
    print(f"   - Gemma Models: https://ai.google.dev/gemma")
    print(f"   - Android Integration: https://developers.google.com/mediapipe")

if __name__ == "__main__":
    main() 