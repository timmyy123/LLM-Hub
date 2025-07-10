#!/usr/bin/env python3
"""
Setup Android Models for MediaPipe LLM Hub
Copies downloaded .task models to Android app assets folder
"""
import os
import shutil
from pathlib import Path

def setup_android_models():
    """Copy models to Android assets folder for easy integration"""
    
    # Paths
    models_dir = Path("models")
    android_assets = Path("app/src/main/assets/models")
    
    print("🔧 Setting up Android Models")
    print("=" * 40)
    
    # Create assets/models directory if it doesn't exist
    android_assets.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created assets directory: {android_assets}")
    
    # Find all .task files in models directory
    task_files = list(models_dir.glob("*.task"))
    
    if not task_files:
        print("❌ No .task files found in models/ directory")
        print("   Run download or conversion scripts first!")
        return
    
    print(f"\n📁 Found {len(task_files)} model(s) to copy:")
    
    for task_file in task_files:
        dest_file = android_assets / task_file.name
        
        print(f"\n📋 Copying: {task_file.name}")
        print(f"   Size: {task_file.stat().st_size / (1024*1024):.1f} MB")
        print(f"   From: {task_file}")
        print(f"   To: {dest_file}")
        
        try:
            shutil.copy2(task_file, dest_file)
            print(f"   ✅ Success!")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🚀 Android Integration Guide:")
    print("=" * 40)
    print("""
    1. Your models are now in: app/src/main/assets/models/
    
    2. Update your Android code to load models from assets:
    
    ```kotlin
    // Load model from assets
    val modelPath = "models/gemma3-1b-it-int4.task"
    val llmInference = LlmInference.createFromAsset(context, modelPath)
    ```
    
    3. Alternative: Load from file path
    
    ```kotlin
    val modelFile = File(context.getExternalFilesDir(null), "models/gemma3-1b-it-int4.task")
    val llmInference = LlmInference.createFromFile(context, modelFile.absolutePath)
    ```
    
    4. Test your app!
    
    📱 The models should now work with your MediaPipe Android app.
    """)

def check_models():
    """Check what models are available"""
    models_dir = Path("models")
    assets_dir = Path("app/src/main/assets/models")
    
    print("📊 Model Status Check")
    print("=" * 30)
    
    print(f"\n📁 Models Directory ({models_dir}):")
    if models_dir.exists():
        task_files = list(models_dir.glob("*.task"))
        if task_files:
            for f in task_files:
                size_mb = f.stat().st_size / (1024*1024)
                print(f"   ✅ {f.name} ({size_mb:.1f} MB)")
        else:
            print("   ❌ No .task files found")
    else:
        print("   ❌ Directory doesn't exist")
    
    print(f"\n📱 Android Assets ({assets_dir}):")
    if assets_dir.exists():
        task_files = list(assets_dir.glob("*.task"))
        if task_files:
            for f in task_files:
                size_mb = f.stat().st_size / (1024*1024)
                print(f"   ✅ {f.name} ({size_mb:.1f} MB)")
        else:
            print("   ❌ No .task files found")
    else:
        print("   ❌ Directory doesn't exist")

def show_help():
    """Show usage instructions"""
    print("""
🔧 Android Model Setup Tool
==========================

Usage:
  python public/setup_android_models.py [command]

Commands:
  setup    Copy models from models/ to app/src/main/assets/models/
  check    Check status of models in both directories
  help     Show this help message

Examples:
  python public/setup_android_models.py setup
  python public/setup_android_models.py check

Note: Make sure you have downloaded .task models first!
    """)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        command = "check"
    else:
        command = sys.argv[1].lower()
    
    if command == "setup":
        setup_android_models()
    elif command == "check":
        check_models()
    elif command == "help":
        show_help()
    else:
        print(f"Unknown command: {command}")
        show_help() 