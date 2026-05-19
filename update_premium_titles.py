import os
import re

translations = {
    'values': 'LLM Hub Premium Lifetime',
    'values-ar': 'LLM Hub Premium مدى الحياة',
    'values-de': 'LLM Hub Premium Lifetime',
    'values-es': 'LLM Hub Premium Lifetime',
    'values-fr': 'LLM Hub Premium Lifetime',
    'values-fa': 'LLM Hub Premium مادام‌العمر',
    'values-he': 'LLM Hub Premium לכל החיים',
    'values-id': 'LLM Hub Premium Seumur Hidup',
    'values-it': 'LLM Hub Premium Lifetime',
    'values-in': 'LLM Hub Premium Seumur Hidup',
    'values-iw': 'LLM Hub Premium לכל החיים',
    'values-ja': 'LLM Hub プレミアム ライフタイム',
    'values-ko': 'LLM Hub 프리미엄 평생',
    'values-pl': 'LLM Hub Premium na całe życie',
    'values-pt': 'LLM Hub Premium Vitalício',
    'values-ru': 'LLM Hub Premium (Навсегда)',
    'values-tr': 'LLM Hub Ömür Boyu Premium',
    'values-uk': 'LLM Hub Premium (Назавжди)',
    'values-zh': 'LLM Hub 终身高级版'
}

base_dir = '/Users/timmybrown/Documents/GitHub/LLM-Hub/android/app/src/main/res'

for folder, translated_text in translations.items():
    file_path = os.path.join(base_dir, folder, 'strings.xml')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the contents of the premium_title tag
        pattern = r'(<string name="premium_title">)(.*?)(</string>)'
        replacement = r'\g<1>' + translated_text + r'\g<3>'
        new_content = re.sub(pattern, replacement, content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated {folder}/strings.xml")
        else:
            print(f"No change needed for {folder}/strings.xml")
    else:
        print(f"File not found: {folder}/strings.xml")

