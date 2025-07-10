#!/usr/bin/env python3
"""
HuggingFace Authentication Setup for Gemma-3 Models
Helps you authenticate and provides alternative download methods
"""
import os
import sys
import subprocess
from pathlib import Path

def check_hf_auth():
    """Check if HuggingFace authentication is set up"""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Already logged in as: {user_info['name']}")
        return True
    except Exception:
        print("❌ Not logged in to HuggingFace")
        return False

def login_to_hf():
    """Help user log in to HuggingFace"""
    print("\n🔐 HuggingFace Login Required")
    print("=" * 40)
    print("Gemma models require accepting a license agreement.")
    print("\n📋 Steps to get access:")
    print("1. Go to: https://huggingface.co/google/gemma-3-1b-it")
    print("2. Click 'Agree and access repository'")
    print("3. Get your HuggingFace token from: https://huggingface.co/settings/tokens")
    print("4. Run the login command below")
    
    print("\n🔑 Login options:")
    print("Option 1 - Login with token:")
    print("   & \"C:\\ProgramData\\anaconda3\\python.exe\" -c \"from huggingface_hub import login; login()\"")
    print("\nOption 2 - Set environment variable:")
    print("   $env:HF_TOKEN = 'your_token_here'")
    
    return False

def get_direct_download_links():
    """Provide direct download links as fallback"""
    print("\n🔗 DIRECT DOWNLOAD LINKS (Alternative Method)")
    print("=" * 60)
    
    links = [
        {
            "name": "Gemma-3 1B INT4",
            "url": "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task",
            "size": "529MB",
            "filename": "gemma3-1b-it-int4.task"
        },
        {
            "name": "Gemma-3 1B INT8", 
            "url": "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv1280.task",
            "size": "1005MB",
            "filename": "gemma3-1b-it-int8.task"
        },
        {
            "name": "Gemma-3n E2B Multimodal",
            "url": "https://huggingface.co/google/gemma-3n-E2B-it-litert-preview/resolve/main/model.task",
            "size": "2.9GB", 
            "filename": "gemma3n-e2b-multimodal.task"
        }
    ]
    
    print("📋 Manual Download Instructions:")
    print("1. First, accept licenses at:")
    print("   - https://huggingface.co/litert-community/Gemma3-1B-IT")
    print("   - https://huggingface.co/google/gemma-3n-E2B-it-litert-preview")
    print("\n2. Then download files directly:")
    
    for link in links:
        print(f"\n📦 {link['name']} ({link['size']})")
        print(f"   URL: {link['url']}")
        print(f"   Save as: models/{link['filename']}")
    
    print("\n💡 You can also use wget or curl:")
    for link in links:
        print(f"\n# {link['name']}")
        print(f"curl -L -o \"models/{link['filename']}\" \"{link['url']}\"")

def create_manual_download_batch():
    """Create a batch file for manual downloads"""
    batch_content = '''@echo off
REM Manual download script for Gemma-3 models
REM Run this after accepting HuggingFace licenses

echo Creating models directory...
if not exist "models" mkdir "models"

echo.
echo Downloading Gemma-3 1B INT4 (529MB)...
curl -L -o "models/gemma3-1b-it-int4.task" "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task"

echo.
echo Downloading Gemma-3 1B INT8 (1005MB)...
curl -L -o "models/gemma3-1b-it-int8.task" "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv1280.task"

echo.
echo Downloading Gemma-3n E2B Multimodal (2.9GB)...
curl -L -o "models/gemma3n-e2b-multimodal.task" "https://huggingface.co/google/gemma-3n-E2B-it-litert-preview/resolve/main/model.task"

echo.
echo Download completed! Check the models/ directory.
pause
'''
    
    with open("public/manual_download.bat", "w") as f:
        f.write(batch_content)
    
    print(f"\n📄 Created manual download script: public/manual_download.bat")
    print("   Run this after accepting HuggingFace licenses")

def try_alternative_download():
    """Try downloading with authentication"""
    print("\n🔄 Trying authenticated download...")
    
    if not check_hf_auth():
        print("❌ Please authenticate first using the instructions above")
        return False
    
    try:
        from huggingface_hub import hf_hub_download
        
        models_to_try = [
            {
                "repo": "litert-community/Gemma3-1B-IT",
                "filename": "gemma3-1b-it-int4.task",
                "local_name": "gemma3-1b-it-int4.task"
            }
        ]
        
        os.makedirs("models", exist_ok=True)
        
        for model in models_to_try:
            print(f"📥 Downloading {model['local_name']}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=model["repo"],
                    filename=model["filename"],
                    local_dir="models",
                    local_dir_use_symlinks=False
                )
                print(f"✅ Successfully downloaded: {downloaded_path}")
                return True
            except Exception as e:
                print(f"❌ Failed to download {model['local_name']}: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    """Main function"""
    print("🔐 HuggingFace Authentication & Download Helper")
    print("=" * 50)
    
    # Check current auth status
    is_authenticated = check_hf_auth()
    
    if not is_authenticated:
        login_to_hf()
    else:
        print("✅ You're already authenticated!")
        
        # Try to download one model as a test
        print("\n🧪 Testing download access...")
        if try_alternative_download():
            print("✅ Downloads working! You can use the download scripts now.")
        else:
            print("❌ Downloads still failing. Using manual method...")
    
    # Always provide alternative methods
    get_direct_download_links()
    create_manual_download_batch()
    
    print("\n🎯 NEXT STEPS:")
    print("1. If not authenticated: Accept licenses and get HF token")
    print("2. If authenticated: Try run_scripts.bat download-all")
    print("3. If downloads fail: Use public/manual_download.bat")
    print("4. Once you have models: run_scripts.bat convert-all")

if __name__ == "__main__":
    main() 