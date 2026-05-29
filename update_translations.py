import os

# Official translations organized by folder
translations = {
    'values':    ('    <string name="image_generator_width">Width</string>\n    <string name="image_generator_height">Height</string>'),
    'values-ar': ('    <string name="image_generator_width">العرض</string>\n    <string name="image_generator_height">الارتفاع</string>'),
    'values-de': ('    <string name="image_generator_width">Breite</string>\n    <string name="image_generator_height">Höhe</string>'),
    'values-es': ('    <string name="image_generator_width">Ancho</string>\n    <string name="image_generator_height">Alto</string>'),
    'values-fr': ('    <string name="image_generator_width">Largeur</string>\n    <string name="image_generator_height">Hauteur</string>'),
    'values-fa': ('    <string name="image_generator_width">عرض</string>\n    <string name="image_generator_height">ارتفاع</string>'),
    'values-he': ('    <string name="image_generator_width">רוחב</string>\n    <string name="image_generator_height">גובה</string>'),
    'values-id': ('    <string name="image_generator_width">Lebar</string>\n    <string name="image_generator_height">Tinggi</string>'),
    'values-it': ('    <string name="image_generator_width">Larghezza</string>\n    <string name="image_generator_height">Altezza</string>'),
    'values-in': ('    <string name="image_generator_width">Lebar</string>\n    <string name="image_generator_height">Tinggi</string>'),
    'values-iw': ('    <string name="image_generator_width">רוחב</string>\n    <string name="image_generator_height">גובה</string>'),
    'values-ja': ('    <string name="image_generator_width">幅</string>\n    <string name="image_generator_height">高さ</string>'),
    'values-ko': ('    <string name="image_generator_width">너비</string>\n    <string name="image_generator_height">높이</string>'),
    'values-pl': ('    <string name="image_generator_width">Szerokość</string>\n    <string name="image_generator_height">Wysokość</string>'),
    'values-pt': ('    <string name="image_generator_width">Largura</string>\n    <string name="image_generator_height">Altura</string>'),
    'values-ru': ('    <string name="image_generator_width">Ширина</string>\n    <string name="image_generator_height">Высота</string>'),
    'values-tr': ('    <string name="image_generator_width">Genişlik</string>\n    <string name="image_generator_height">Yükseklik</string>'),
    'values-uk': ('    <string name="image_generator_width">Ширина</string>\n    <string name="image_generator_height">Висота</string>'),
    'values-zh': ('    <string name="image_generator_width">宽度</string>\n    <string name="image_generator_height">高度</string>')
}

# Android resources base directory path
base_dir = '/home/angel-m/StudioProjects/LLM-Hub/android/app/src/main/res'

# The anchor tag used to locate the correct section across all languages
anchor_tag = 'name="image_generator_use_gpu_desc"'

for folder, xml_to_add in translations.items():
    file_path = os.path.join(base_dir, folder, 'strings.xml')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check if the resolution strings already exist to prevent duplication
        content = "".join(lines)
        if 'name="image_generator_width"' in content:
            print(f"Skipped (already exists): {folder}/strings.xml")
            continue
            
        updated = False
        new_lines = []
        
        # Process line by line to find the exact line matching the anchor tag
        for line in lines:
            new_lines.append(line)
            if anchor_tag in line:
                # Target found! Inject the new translation right after this line
                new_lines.append(f"{xml_to_add}\n")
                updated = True
        
        if updated:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"Successfully updated: {folder}/strings.xml")
        else:
            print(f"Warning: Anchor tag not found in {folder}/strings.xml (Falling back to end of file)")
            # Fallback to appending before </resources> if the anchor tag is missing in a specific file
            if '</resources>' in content:
                new_content = content.replace('</resources>', f'{xml_to_add}\n</resources>')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Successfully appended to end: {folder}/strings.xml")
    else:
        print(f"File not found: {file_path}")