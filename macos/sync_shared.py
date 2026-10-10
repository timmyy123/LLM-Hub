#!/usr/bin/env python3
"""
Sync shared code and resources from ios/LLMHub to macos/LLMHub.
Supports symbolic links (default) or hard copies (--copy).
"""

import os
import sys
import shutil
import argparse

SHARED_FILES = [
    'ModelData.swift',
    'ModelDownloader.swift',
    'ModelManager.swift',
    'ChatModels.swift',
    'ChatStore.swift',
    'LLMBackend.swift',
    'LocalizationManager.swift',
    'PurchaseManager.swift',
    'MemoryStore.swift',
    'SimplifiedFileManager.swift',
    'PreviewServer.swift',
    'MCPClient.swift',
    'AgentViewModel.swift',
    'DocumentTextExtractor.swift',
    'AudioTools.swift',
    'AppleFoundationModelSupport.swift',
    'ThinkingComponents.swift',
    'WhisperBackend.swift',
    'LiteRTLMBackend.swift',
    'ChatAgentSkillsTools.swift',
    'AgentTools.swift',
    'MediaSearchCore.swift',
    'EmbeddingService.swift',
    'RagService.swift',
    'RagServiceManager.swift',
    'StableDiffusionBackend.swift',
    'ImageUpscalerBackend.swift',
    'VideoGeneratorBackend.swift',
]

SHARED_RESOURCES = [
    'models.json',
    'configs.json',
    'Icon.png'
]

def main():
    parser = argparse.ArgumentParser(description="Sync iOS shared files to macOS")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ios_src = os.path.join(repo_root, "ios", "LLMHub", "Sources", "LLMHub")
    macos_shared = os.path.join(repo_root, "macos", "LLMHub", "Sources", "LLMHub", "Shared")
    macos_res = os.path.join(repo_root, "macos", "LLMHub", "Resources")

    os.makedirs(macos_shared, exist_ok=True)
    os.makedirs(macos_res, exist_ok=True)

    print(f"Syncing shared files (copy={args.copy})...")
    for fname in SHARED_FILES:
        src = os.path.join(ios_src, fname)
        dst = os.path.join(macos_shared, fname)
        if not os.path.exists(src):
            print(f"Warning: source file missing: {src}")
            continue
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        if args.copy:
            shutil.copy2(src, dst)
        else:
            rel = os.path.relpath(src, macos_shared)
            os.symlink(rel, dst)
        print(f"  Synced: {fname}")

    print("Syncing resources...")
    for rname in SHARED_RESOURCES:
        src = os.path.join(ios_src, rname)
        dst = os.path.join(macos_res, rname)
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        if args.copy:
            shutil.copy2(src, dst)
        else:
            rel = os.path.relpath(src, macos_res)
            os.symlink(rel, dst)
        print(f"  Synced resource: {rname}")

    # Localization folders
    for item in os.listdir(ios_src):
        if item.endswith(".lproj"):
            src = os.path.join(ios_src, item)
            dst = os.path.join(macos_res, item)
            if os.path.exists(dst) or os.path.islink(dst):
                if os.path.islink(dst):
                    os.remove(dst)
                else:
                    shutil.rmtree(dst)
            if args.copy:
                shutil.copytree(src, dst)
            else:
                rel = os.path.relpath(src, macos_res)
                os.symlink(rel, dst)
            print(f"  Synced lproj: {item}")

    print("Done sync!")

if __name__ == "__main__":
    main()
