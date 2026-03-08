import re
import glob

def check_file(path):
    print(f"Checking {path}")
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        orig_line = line
        line = line.strip()
        if not line: continue
        
        # Check standard format "key" = "value";
        # The key should be in quotes, then =, then value in quotes, then ;
        # The regex below checks if it matches the general format
        # and doesn't contain unescaped quotes in the middle.
        
        if not line.endswith(';'):
            print(f"[{path}:{i+1}] Missing semicolon: {line}")
        
        # Split by the first '='
        parts = line.split('=', 1)
        if len(parts) != 2:
            print(f"[{path}:{i+1}] Missing equals sign: {line}")
            continue
            
        key_part, val_part = parts[0].strip(), parts[1].strip()
        
        if not key_part.startswith('"') or not key_part.endswith('"'):
            print(f"[{path}:{i+1}] Key not properly quoted: {key_part}")
            
        # The value part should be `"value";`
        if not val_part.startswith('"'):
            print(f"[{path}:{i+1}] Value not properly quoted: {val_part}")
            
        val_content = val_part[1:-2] if val_part.endswith(';') else val_part[1:-1]
        
        # Remove literal `\"` 
        val_content_stripped = val_content.replace('\\"', '')
        if '"' in val_content_stripped:
             print(f"[{path}:{i+1}] Unescaped double quote inside value: {line}")

for p in glob.glob('*.lproj/Localizable.strings'):
    check_file(p)
