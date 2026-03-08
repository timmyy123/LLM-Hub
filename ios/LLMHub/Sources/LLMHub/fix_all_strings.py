import os, glob, re

paths = glob.glob('*.lproj/Localizable.strings')
for path in paths:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.splitlines()
    changed = False
    new_lines = []
    
    for i, line in enumerate(lines):
        m = re.match(r'^([^=]+)=\s*\"(.*)\"\s*;\s*$', line)
        if m:
            key = m.group(1)
            val = m.group(2)
            
            # Simple fix for unescaped quotes inside strings.
            # Convert already escaped quotes to a placeholder.
            val_placeholder = val.replace('\\"', '<<<<BACKSLASH_QUOTE>>>>')
            # Now any remaining double quotes are unescaped.
            val_fixed = val_placeholder.replace('"', '\\"')
            # Restore the placeholder back to escaped quotes.
            val_final = val_fixed.replace('<<<<BACKSLASH_QUOTE>>>>', '\\"')
            
            if val_final != val:
                new_line = f'{key}= "{val_final}";'
                new_lines.append(new_line)
                changed = True
                continue
        new_lines.append(line)
        
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines) + '\n')
        print(f'Fixed {path}')
