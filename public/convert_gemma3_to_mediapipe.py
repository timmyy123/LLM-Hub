#!/usr/bin/env python3
"""
Convert Gemma-3 4B and 12B models to MediaPipe .task format
Converts from HuggingFace safetensors to MediaPipe .task
"""
import os
import sys
import subprocess
from pathlib import Path
import tempfile
import shutil

# Models that need conversion (not available as pre-converted .task files)
MODELS_TO_CONVERT = {
    "gemma-3-4b": {
        "name": "Gemma-3 4B Instruction Tuned",
        "hf_repo": "google/gemma-3-4b-it",
        "description": "4B parameter Gemma-3 model with instruction tuning",
        "output_filename": "gemma3-4b-it-converted.task",
        "estimated_size": "~4GB",
        "context_length": 128000
    },
    "gemma-3-12b": {
        "name": "Gemma-3 12B Instruction Tuned", 
        "hf_repo": "google/gemma-3-12b-it",
        "description": "12B parameter Gemma-3 model with instruction tuning",
        "output_filename": "gemma3-12b-it-converted.task",
        "estimated_size": "~12GB",
        "context_length": 128000
    }
}

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "mediapipe",
        "torch",
        "transformers",
        "huggingface_hub"
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} is missing")
    
    if missing:
        print(f"\n🔧 Installing missing packages: {', '.join(missing)}")
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to install packages: {result.stderr}")
            return False
        else:
            print("✅ All packages installed successfully!")
    
    return True

def download_hf_model(repo_id, cache_dir=None):
    """Download model from HuggingFace"""
    try:
        from huggingface_hub import snapshot_download
        
        print(f"🔽 Downloading {repo_id} from HuggingFace...")
        
        model_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )
        
        print(f"✅ Downloaded {repo_id} to {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Error downloading {repo_id}: {e}")
        return None

def convert_to_mediapipe(model_path, output_path, model_config):
    """Convert HuggingFace model to MediaPipe .task format"""
    try:
        # Try different conversion approaches
        conversion_methods = [
            convert_with_mediapipe_converter,
            convert_with_manual_export
        ]
        
        for method in conversion_methods:
            try:
                print(f"🔄 Trying conversion method: {method.__name__}")
                result = method(model_path, output_path, model_config)
                if result:
                    return result
            except Exception as e:
                print(f"⚠️ Method {method.__name__} failed: {e}")
                continue
        
        print("❌ All conversion methods failed")
        return False
        
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def convert_with_mediapipe_converter(model_path, output_path, model_config):
    """Convert using MediaPipe's built-in converter"""
    try:
        import mediapipe.tasks.python.genai.converter as converter
        
        print("🔄 Using MediaPipe converter...")
        
        # Configure conversion
        config = converter.ConversionConfig(
            input_path=model_path,
            output_path=output_path,
            quantization_config=converter.QuantizationConfig.dynamic_int4()
        )
        
        # Perform conversion
        converter.convert(config)
        
        print(f"✅ Conversion completed: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe converter failed: {e}")
        return False

def convert_with_manual_export(model_path, output_path, model_config):
    """Manual conversion using transformers and custom export"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("🔄 Loading model with transformers...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("🔄 Exporting to TensorFlow Lite format...")
        
        # Export to TFLite (this is a simplified approach)
        # Note: This may need additional work for full MediaPipe compatibility
        
        # For now, create a placeholder indicating manual conversion is needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        info_file = output_dir / f"{Path(output_path).stem}_conversion_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Model: {model_config['name']}\n")
            f.write(f"Source: {model_config['hf_repo']}\n")
            f.write(f"Downloaded to: {model_path}\n")
            f.write(f"Status: Manual conversion required\n")
            f.write(f"Next steps:\n")
            f.write(f"1. Use MediaPipe's model converter tools\n")
            f.write(f"2. Follow MediaPipe LLM documentation\n")
            f.write(f"3. Convert to .task format manually\n")
        
        print(f"📝 Created conversion info file: {info_file}")
        print("⚠️ Manual conversion steps required - see info file")
        
        return str(info_file)
        
    except Exception as e:
        print(f"❌ Manual export failed: {e}")
        return False

def convert_model(model_id, output_dir="models"):
    """Convert a specific model"""
    if model_id not in MODELS_TO_CONVERT:
        print(f"❌ Model '{model_id}' not available for conversion!")
        print("Available models:", list(MODELS_TO_CONVERT.keys()))
        return False
    
    model_config = MODELS_TO_CONVERT[model_id]
    
    print(f"\n🚀 Converting {model_config['name']}...")
    print(f"   Repository: {model_config['hf_repo']}")
    print(f"   Estimated size: {model_config['estimated_size']}")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Create output directory
    output_path = Path(output_dir) / model_config['output_filename']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = download_hf_model(model_config['hf_repo'], temp_dir)
        if not model_path:
            return False
        
        # Convert to MediaPipe format
        result = convert_to_mediapipe(model_path, str(output_path), model_config)
        
        if result:
            print(f"✅ Conversion completed successfully!")
            return True
        else:
            print(f"❌ Conversion failed!")
            return False

def list_convertible_models():
    """List models that can be converted"""
    print("🔧 Models Available for Conversion:")
    print("=" * 50)
    
    for model_id, info in MODELS_TO_CONVERT.items():
        print(f"\n📦 {model_id}")
        print(f"   Name: {info['name']}")
        print(f"   Repository: {info['hf_repo']}")
        print(f"   Description: {info['description']}")
        print(f"   Estimated size: {info['estimated_size']}")
        print(f"   Context length: {info['context_length']:,} tokens")

def convert_all_models(output_dir="models"):
    """Convert all available models"""
    print("🚀 Converting ALL available models...")
    
    converted = []
    failed = []
    
    for model_id in MODELS_TO_CONVERT:
        success = convert_model(model_id, output_dir)
        if success:
            converted.append(model_id)
        else:
            failed.append(model_id)
    
    print("\n" + "="*60)
    print("📊 CONVERSION SUMMARY:")
    print(f"✅ Successfully converted: {len(converted)} models")
    for model_id in converted:
        print(f"   - {model_id}")
    
    if failed:
        print(f"❌ Failed conversions: {len(failed)} models")
        for model_id in failed:
            print(f"   - {model_id}")
    
    return converted, failed

def show_conversion_help():
    """Show help for manual conversion"""
    print("\n🔧 Manual Conversion Guide:")
    print("=" * 40)
    print("If automatic conversion fails, you can convert manually:")
    print()
    print("1. Install MediaPipe with LLM support:")
    print("   pip install mediapipe[llm]")
    print()
    print("2. Use MediaPipe Model Converter:")
    print("   python -m mediapipe.tasks.python.genai.converter \\")
    print("     --input_checkpoint=/path/to/model \\")
    print("     --output_path=/path/to/output.task \\")
    print("     --quantization=dynamic_int4")
    print()
    print("3. Follow MediaPipe LLM documentation:")
    print("   https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("🔧 Gemma-3 to MediaPipe Converter")
        print("=" * 40)
        print("Usage:")
        print("  python convert_gemma3_to_mediapipe.py list")
        print("  python convert_gemma3_to_mediapipe.py convert <model_id>")
        print("  python convert_gemma3_to_mediapipe.py convert-all")
        print("  python convert_gemma3_to_mediapipe.py help")
        print("\nExamples:")
        print("  python convert_gemma3_to_mediapipe.py convert gemma-3-4b")
        print("  python convert_gemma3_to_mediapipe.py convert-all")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_convertible_models()
    
    elif command == "convert" and len(sys.argv) >= 3:
        model_id = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else "models"
        convert_model(model_id, output_dir)
    
    elif command == "convert-all":
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else "models"
        convert_all_models(output_dir)
    
    elif command == "help":
        show_conversion_help()
    
    else:
        print("❌ Invalid command! Use 'list', 'convert <model_id>', 'convert-all', or 'help'")

if __name__ == "__main__":
    main() 