#!/usr/bin/env python3
"""
Download Gemma-3 models in MediaPipe .task format
Ready to use with Android MediaPipe LLM Inference API
"""
import os
import requests
from pathlib import Path
import sys

# Available Gemma-3 models in MediaPipe format
GEMMA3_MODELS = {
    "gemma-3-1b": {
        "name": "Gemma-3 1B (Instruction Tuned)",
        "description": "Smallest Gemma-3 model, optimized for mobile",
        "url": "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task",
        "filename": "gemma3-1b-it-int4.task",
        "size": "~529MB",
        "quantization": "dynamic_int4 QAT",
        "performance": "GPU: 2585 tk/s prefill, 56 tk/s decode (Samsung S24 Ultra)"
    },
    "gemma-3-1b-web": {
        "name": "Gemma-3 1B (Web Optimized)",
        "description": "Web-optimized version of Gemma-3 1B",
        "url": "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4-web.task",
        "filename": "gemma3-1b-it-int4-web.task", 
        "size": "~700MB",
        "quantization": "dynamic_int4",
        "performance": "GPU: 4339 tk/s prefill, 133 tk/s decode (MacBook M4 Max)"
    },
    "gemma-3n-2b": {
        "name": "Gemma-3n E2B (Vision + Text)",
        "description": "Multimodal model supporting text and images",
        "url": "https://huggingface.co/google/gemma-3n-E2B-it-litert-preview/resolve/main/gemma3n-e2b-it-int4-fp16.task",
        "filename": "gemma3n-e2b-it-int4-fp16.task",
        "size": "~2.9GB", 
        "quantization": "dynamic_int4",
        "performance": "GPU: 620 tk/s prefill, 23 tk/s decode + Vision support (Samsung S25 Ultra)"
    }
}

def download_file(url, filename, description):
    """Download a file with progress bar"""
    print(f"\n📥 Downloading {description}...")
    print(f"URL: {url}")
    print(f"File: {filename}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r💾 Progress: {percent:.1f}% ({downloaded // (1024*1024)}MB/{total_size // (1024*1024)}MB)", end='', flush=True)
        
        print(f"\n✅ Successfully downloaded: {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error downloading {filename}: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    print("🔥 Gemma-3 MediaPipe Model Downloader")
    print("=" * 50)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Download directory: {models_dir.absolute()}")
    
    # Show available models
    print("\n📋 Available Models:")
    for i, (key, model) in enumerate(GEMMA3_MODELS.items(), 1):
        print(f"{i}. {model['name']}")
        print(f"   📊 Size: {model['size']}")
        print(f"   ⚡ Performance: {model['performance']}")
        print(f"   📝 Description: {model['description']}")
        print()
    
    # Get user choice
    try:
        choice = input("Enter model numbers to download (e.g., '1,2,3' or 'all'): ").strip()
        
        if choice.lower() == 'all':
            selected_models = list(GEMMA3_MODELS.keys())
        else:
            indices = [int(x.strip()) for x in choice.split(',')]
            model_keys = list(GEMMA3_MODELS.keys())
            selected_models = [model_keys[i-1] for i in indices if 1 <= i <= len(model_keys)]
        
        if not selected_models:
            print("❌ No valid models selected!")
            return
            
    except (ValueError, IndexError):
        print("❌ Invalid input! Please enter valid numbers.")
        return
    
    # Download selected models
    successful_downloads = []
    failed_downloads = []
    
    for model_key in selected_models:
        model = GEMMA3_MODELS[model_key]
        filepath = models_dir / model['filename']
        
        if filepath.exists():
            print(f"\n⚠️ File {model['filename']} already exists. Skipping...")
            continue
            
        success = download_file(model['url'], filepath, model['name'])
        
        if success:
            successful_downloads.append(model['name'])
        else:
            failed_downloads.append(model['name'])
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Download Summary")
    print("=" * 50)
    
    if successful_downloads:
        print(f"✅ Successfully downloaded ({len(successful_downloads)}):")
        for model in successful_downloads:
            print(f"   • {model}")
    
    if failed_downloads:
        print(f"\n❌ Failed downloads ({len(failed_downloads)}):")
        for model in failed_downloads:
            print(f"   • {model}")
    
    print(f"\n📁 Models saved to: {models_dir.absolute()}")
    
    # Android integration instructions
    if successful_downloads:
        print("\n🔧 Next Steps for Android Integration:")
        print("1. Copy .task files to your Android app's assets folder")
        print("2. Update ModelData.kt with the downloaded model paths")
        print("3. Test with your MediaPipe implementation")
        print("\n💡 Performance Tips:")
        print("• Use Gemma-3 1B for fastest inference on mobile")
        print("• Use Gemma-3n E2B for multimodal (text + image) capabilities")
        print("• Enable GPU acceleration for better performance")

if __name__ == "__main__":
    main() 