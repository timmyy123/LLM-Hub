#!/usr/bin/env python3
"""
Download Gemma-3 models in MediaPipe .task format
Ready to use with Android MediaPipe LLM Inference API
"""
import os
import requests
from pathlib import Path
import sys
from urllib.parse import urlparse

# Available Gemma-3 models in MediaPipe format
GEMMA3_MODELS = {
    "gemma-3-1b-int4": {
        "name": "Gemma-3 1B (INT4 Quantized)",
        "description": "Smallest Gemma-3 model, optimized for mobile with INT4 quantization",
        "url": "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task",
        "filename": "gemma3-1b-it-int4.task",
        "size": "~529MB",
        "performance": "GPU: 2585 tk/s prefill, 56 tk/s decode (Samsung S24 Ultra)"
    },
    "gemma-3-1b-int8": {
        "name": "Gemma-3 1B (INT8 Quantized)",
        "description": "Gemma-3 1B with INT8 quantization, higher quality than INT4",
        "url": "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv1280.task",
        "filename": "gemma3-1b-it-int8.task",
        "size": "~1005MB",
        "performance": "GPU: 1191 tk/s prefill, 24 tk/s decode"
    },
    "gemma-3-1b-web": {
        "name": "Gemma-3 1B (Web Optimized)",
        "description": "Gemma-3 1B optimized for web deployment",
        "url": "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4-web.task",
        "filename": "gemma3-1b-it-web.task", 
        "size": "~700MB",
        "performance": "Web: 4339 tk/s prefill, 133 tk/s decode (MacBook M4 Max)"
    },
    "gemma-3n-e2b": {
        "name": "Gemma-3n E2B (Multimodal)",
        "description": "Effective 2B parameters, supports text + images",
        "url": "https://huggingface.co/google/gemma-3n-E2B-it-litert-preview/resolve/main/model.task",
        "filename": "gemma3n-e2b-multimodal.task",
        "size": "~2.9GB",
        "performance": "Multimodal: Text + Image processing"
    }
}

def download_file(url, filename, output_dir="models"):
    """Download a file with progress bar"""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"✅ {filename} already exists, skipping download")
        return str(output_path)
    
    print(f"🔽 Downloading {filename}...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)", end="")
        
        print(f"\n✅ Downloaded {filename} successfully!")
        return str(output_path)
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {filename}: {e}")
        if output_path.exists():
            output_path.unlink()  # Remove partial file
        return None

def list_available_models():
    """List all available models"""
    print("🎯 Available Gemma-3 Models for MediaPipe:")
    print("=" * 60)
    
    for model_id, info in GEMMA3_MODELS.items():
        print(f"\n📦 {model_id}")
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Size: {info['size']}")
        print(f"   Performance: {info['performance']}")

def download_model(model_id, output_dir="models"):
    """Download a specific model"""
    if model_id not in GEMMA3_MODELS:
        print(f"❌ Model '{model_id}' not found!")
        print("Available models:", list(GEMMA3_MODELS.keys()))
        return None
    
    model_info = GEMMA3_MODELS[model_id]
    print(f"\n🚀 Downloading {model_info['name']}...")
    
    return download_file(
        model_info['url'], 
        model_info['filename'], 
        output_dir
    )

def download_all_models(output_dir="models"):
    """Download all available models"""
    print("🚀 Downloading ALL Gemma-3 models...")
    
    downloaded = []
    failed = []
    
    for model_id in GEMMA3_MODELS:
        result = download_model(model_id, output_dir)
        if result:
            downloaded.append(model_id)
        else:
            failed.append(model_id)
    
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY:")
    print(f"✅ Successfully downloaded: {len(downloaded)} models")
    for model_id in downloaded:
        print(f"   - {model_id}")
    
    if failed:
        print(f"❌ Failed downloads: {len(failed)} models")
        for model_id in failed:
            print(f"   - {model_id}")
    
    return downloaded, failed

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("📱 Gemma-3 MediaPipe Model Downloader")
        print("=" * 40)
        print("Usage:")
        print("  python download_gemma3_mediapipe.py list")
        print("  python download_gemma3_mediapipe.py download <model_id>")
        print("  python download_gemma3_mediapipe.py download-all")
        print("\nExamples:")
        print("  python download_gemma3_mediapipe.py download gemma-3-1b-int4")
        print("  python download_gemma3_mediapipe.py download-all")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_available_models()
    
    elif command == "download" and len(sys.argv) >= 3:
        model_id = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else "models"
        download_model(model_id, output_dir)
    
    elif command == "download-all":
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else "models"
        download_all_models(output_dir)
    
    else:
        print("❌ Invalid command! Use 'list', 'download <model_id>', or 'download-all'")

if __name__ == "__main__":
    main() 