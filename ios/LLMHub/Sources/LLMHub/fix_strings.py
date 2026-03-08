import os
import glob
import re

paths = glob.glob('*.lproj/Localizable.strings')
for path in paths:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    changed = False
    new_lines = []
    
    for i, line in enumerate(lines):
        # A simple check: if a line matches '... = \"...\";'
        m = re.match(r'^([^=]+)=\s*\"(.*)\"\s*;\s*$', line)
        if m:
            key = m.group(1)
            val = m.group(2)
            # Replace unescaped quotes with escaped quotes
            # Wait, the user has \" inside string, so let's escape any quote that is not escaped
            new_val = re.sub(r'(?<!\\)\"', r'\"', val)
            
            # They also might have \' instead of ' or other things, but let's just do quotes
            # Actually, `?<!\\` requires not preceded by backslash. So we match `"` and replace with `\"`
            
            # Wait, re.sub replacement should be r'\"' which is \", but we need an actual backslash, so r'\\"'
            new_val = re.sub(r'(?<!\\)"', r'\\"', val)
            
            if new_val != val:
                new_line = f'{key}= "{new_val}";\n'
                new_lines.append(new_line)
                changed = True
                print(f'Fixed in {path}: {key.strip()} -> {new_val}')
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
