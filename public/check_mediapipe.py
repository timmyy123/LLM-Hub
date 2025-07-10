#!/usr/bin/env python3
import mediapipe as mp
import inspect

print("MediaPipe Module Contents:")
print("=" * 50)

# First check what's in mediapipe
print("Top level modules:")
for name, obj in inspect.getmembers(mp):
    if not name.startswith('_') and inspect.ismodule(obj):
        print(f"- {name}")

print("\nTasks module:")
if hasattr(mp, 'tasks'):
    for name, obj in inspect.getmembers(mp.tasks):
        if not name.startswith('_'):
            print(f"- {name}")

print("\nTrying to find LLM/GenAI conversion:")
try:
    # Check different possible paths
    paths_to_try = [
        'mp.tasks.python.genai',
        'mp.tasks.python.llm', 
        'mp.tasks.genai',
        'mp.tasks.llm',
        'mp.python.genai',
        'mp.genai'
    ]
    
    for path in paths_to_try:
        try:
            module = eval(path)
            print(f"✓ Found: {path}")
            print(f"  Contents: {[name for name, obj in inspect.getmembers(module) if not name.startswith('_')]}")
        except:
            print(f"✗ Not found: {path}")
            
except Exception as e:
    print(f"Error exploring: {e}")

# Try to find converter functionality
print("\nSearching for converter or conversion functionality:")
def find_converter(module, path="mp"):
    try:
        for name, obj in inspect.getmembers(module):
            if not name.startswith('_'):
                if 'convert' in name.lower() or 'llm' in name.lower() or 'genai' in name.lower():
                    print(f"Found in {path}: {name} ({type(obj)})")
                if inspect.ismodule(obj) and not name.startswith('_'):
                    find_converter(obj, f"{path}.{name}")
    except Exception as e:
        pass

find_converter(mp) 