#!/usr/bin/env python3
import os
import re

translations = {
    "en": {
        "hf_token_title": "Hugging Face Token",
        "hf_token_subtitle": "Personal access token for model downloads",
        "hf_token_dialog_title": "Hugging Face Access Token",
        "hf_token_explanation": "If the app's default token fails, hits rate limits, or when downloading gated models that require an accepted license, enter your personal read token from huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Enter personal token (hf_...)",
        "hf_token_using_default": "Using default token",
        "hf_token_custom_active": "Custom token active",
        "hf_token_clear": "Reset to default",
    },
    "es": {
        "hf_token_title": "Token de Hugging Face",
        "hf_token_subtitle": "Token de acceso personal para descargar modelos",
        "hf_token_dialog_title": "Token de acceso de Hugging Face",
        "hf_token_explanation": "Si el token predeterminado falla, alcanza límites de velocidad o al descargar modelos restringidos con licencia aceptada, ingresa tu token de lectura personal de huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Ingresa token personal (hf_...)",
        "hf_token_using_default": "Usando token predeterminado",
        "hf_token_custom_active": "Token personalizado activo",
        "hf_token_clear": "Restablecer a predeterminado",
    },
    "pt": {
        "hf_token_title": "Token do Hugging Face",
        "hf_token_subtitle": "Token de acesso pessoal para descarregar modelos",
        "hf_token_dialog_title": "Token de acesso do Hugging Face",
        "hf_token_explanation": "Se o token predefinido falhar, atingir limites de taxa ou ao descarregar modelos restritos com licença aceite, introduza o seu token de leitura pessoal de huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Introduzir token pessoal (hf_...)",
        "hf_token_using_default": "A usar token predefinido",
        "hf_token_custom_active": "Token personalizado ativo",
        "hf_token_clear": "Repor predefinição",
    },
    "de": {
        "hf_token_title": "Hugging Face-Token",
        "hf_token_subtitle": "Persönliches Zugriffs-Token für Modell-Downloads",
        "hf_token_dialog_title": "Hugging Face-Zugriffstoken",
        "hf_token_explanation": "Falls das Standard-Token fehlschlägt, Ratenbegrenzungen erreicht oder Sie geschützte Modelle mit akzeptierter Lizenz herunterladen möchten, geben Sie Ihr persönliches Lese-Token von huggingface.co/settings/tokens ein.",
        "hf_token_placeholder": "Persönliches Token eingeben (hf_...)",
        "hf_token_using_default": "Standard-Token aktiv",
        "hf_token_custom_active": "Benutzerdefiniertes Token aktiv",
        "hf_token_clear": "Auf Standard zurücksetzen",
    },
    "fr": {
        "hf_token_title": "Jeton Hugging Face",
        "hf_token_subtitle": "Jeton d'accès personnel pour le téléchargement de modèles",
        "hf_token_dialog_title": "Jeton d'accès Hugging Face",
        "hf_token_explanation": "Si le jeton par défaut échoue, atteint des limites de débit ou pour télécharger des modèles sous licence acceptée, saisissez votre jeton de lecture personnel depuis huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Saisir un jeton personnel (hf_...)",
        "hf_token_using_default": "Jeton par défaut utilisé",
        "hf_token_custom_active": "Jeton personnalisé actif",
        "hf_token_clear": "Rétablir par défaut",
    },
    "it": {
        "hf_token_title": "Token Hugging Face",
        "hf_token_subtitle": "Token di accesso personale per scaricare modelli",
        "hf_token_dialog_title": "Token di accesso Hugging Face",
        "hf_token_explanation": "Se il token predefinito non funziona, supera i limiti di frequenza o per scaricare modelli con licenza accettata, inserisci il tuo token di lettura personale da huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Inserisci token personale (hf_...)",
        "hf_token_using_default": "Uso del token predefinito",
        "hf_token_custom_active": "Token personalizzato attivo",
        "hf_token_clear": "Ripristina predefinito",
    },
    "tr": {
        "hf_token_title": "Hugging Face Belirteci",
        "hf_token_subtitle": "Model indirmeleri için kişisel erişim belirteci",
        "hf_token_dialog_title": "Hugging Face Erişim Belirteci",
        "hf_token_explanation": "Varsayılan belirteç başarısız olursa, hız sınırına ulaşırsa veya lisans onaylı korumalı modelleri indirirken huggingface.co/settings/tokens adresinden alacağınız kişisel okuma belirtecinizi girin.",
        "hf_token_placeholder": "Kişisel belirteç girin (hf_...)",
        "hf_token_using_default": "Varsayılan belirteç kullanılıyor",
        "hf_token_custom_active": "Özel belirteç etkin",
        "hf_token_clear": "Varsayılana sıfırla",
    },
    "pl": {
        "hf_token_title": "Token Hugging Face",
        "hf_token_subtitle": "Osobisty token dostępu do pobierania modeli",
        "hf_token_dialog_title": "Token dostępu Hugging Face",
        "hf_token_explanation": "Jeśli domyślny token zawiedzie, osiągnie limity lub do pobierania modeli wymagających zaakceptowanej licencji, wprowadź osobisty token odczytu z huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Wpisz osobisty token (hf_...)",
        "hf_token_using_default": "Używany token domyślny",
        "hf_token_custom_active": "Aktywny token własny",
        "hf_token_clear": "Przywróć domyślny",
    },
    "ru": {
        "hf_token_title": "Токен Hugging Face",
        "hf_token_subtitle": "Личный токен доступа для загрузки моделей",
        "hf_token_dialog_title": "Токен доступа Hugging Face",
        "hf_token_explanation": "Если стандартный токен не работает, исчерпал лимит запросов или для загрузки лицензированных моделей введите свой личный токен чтения с huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Введите личный токен (hf_...)",
        "hf_token_using_default": "Используется стандартный токен",
        "hf_token_custom_active": "Пользовательский токен активен",
        "hf_token_clear": "Сбросить на стандартный",
    },
    "uk": {
        "hf_token_title": "Токен Hugging Face",
        "hf_token_subtitle": "Особистий токен доступу для завантаження моделей",
        "hf_token_dialog_title": "Токен доступу Hugging Face",
        "hf_token_explanation": "Якщо стандартний токен не працює, вичерпав ліміти або для завантаження ліцензованих моделей введіть свій особистий токен читання з huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Введіть особистий токен (hf_...)",
        "hf_token_using_default": "Використовується стандартний токен",
        "hf_token_custom_active": "Користувацький токен активний",
        "hf_token_clear": "Скинути до стандартного",
    },
    "ar": {
        "hf_token_title": "رمز Hugging Face",
        "hf_token_subtitle": "رمز وصول شخصي لتنزيل النماذج",
        "hf_token_dialog_title": "رمز وصول Hugging Face",
        "hf_token_explanation": "إذا تعذر استخدام الرمز الافتراضي أو تجاوز حد الاستخدام، أو لتنزيل النماذج التي تتطلب موافقة على الترخيص، أدخل رمز القراءة الشخصي الخاص بك من huggingface.co/settings/tokens.",
        "hf_token_placeholder": "أدخل الرمز الشخصي (hf_...)",
        "hf_token_using_default": "استخدام الرمز الافتراضي",
        "hf_token_custom_active": "الرمز المخصص مفعّل",
        "hf_token_clear": "إعادة الضبط للافتراضي",
    },
    "fa": {
        "hf_token_title": "توکن Hugging Face",
        "hf_token_subtitle": "توکن دسترسی شخصی برای دانلود مدل‌ها",
        "hf_token_dialog_title": "توکن دسترسی Hugging Face",
        "hf_token_explanation": "اگر توکن پیش‌فرض کار نکرد، به محدودیت نرخ رسید یا هنگام دانلود مدل‌های دارای مجوز تایید شده، توکن خواندن شخصی خود را از huggingface.co/settings/tokens وارد کنید.",
        "hf_token_placeholder": "وارد کردن توکن شخصی (hf_...)",
        "hf_token_using_default": "در حال استفاده از توکن پیش‌فرض",
        "hf_token_custom_active": "توکن سفارشی فعال است",
        "hf_token_clear": "بازنشانی به پیش‌فرض",
    },
    "he": {
        "hf_token_title": "אסימון Hugging Face",
        "hf_token_subtitle": "אסימון גישה אישי להורדת מודלים",
        "hf_token_dialog_title": "אסימון גישה של Hugging Face",
        "hf_token_explanation": "אם אסימון ברירת המחדל נכשל, מגיע למגבלת קצב או להורדת מודלים הדורשים רישיון מאושר, הזן את אסימון הקריאה האישי שלך מ-huggingface.co/settings/tokens.",
        "hf_token_placeholder": "הזן אסימון אישי (hf_...)",
        "hf_token_using_default": "בשימוש אסימון ברירת מחדל",
        "hf_token_custom_active": "אסימון מותאם אישית פעיל",
        "hf_token_clear": "איפוס לברירת מחדל",
    },
    "hi": {
        "hf_token_title": "Hugging Face टोकन",
        "hf_token_subtitle": "मॉडल डाउनलोड के लिए व्यक्तिगत एक्सेस टोकन",
        "hf_token_dialog_title": "Hugging Face एक्सेस टोकन",
        "hf_token_explanation": "यदि डिफ़ॉल्ट टोकन विफल होता है, सीमा समाप्त होती है, या लाइसेंस-स्वीकृत मॉडल डाउनलोड करने के लिए, huggingface.co/settings/tokens से अपना व्यक्तिगत रीड टोकन दर्ज करें।",
        "hf_token_placeholder": "व्यक्तिगत टोकन दर्ज करें (hf_...)",
        "hf_token_using_default": "डिफ़ॉल्ट टोकन का उपयोग हो रहा है",
        "hf_token_custom_active": "कस्टम टोकन सक्रिय है",
        "hf_token_clear": "डिफ़ॉल्ट पर रीसेट करें",
    },
    "id": {
        "hf_token_title": "Token Hugging Face",
        "hf_token_subtitle": "Token akses pribadi untuk mengunduh model",
        "hf_token_dialog_title": "Token Akses Hugging Face",
        "hf_token_explanation": "Jika token default gagal, mencapai batas penggunaan, atau saat mengunduh model berlisensi yang disetujui, masukkan token baca pribadi Anda dari huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Masukkan token pribadi (hf_...)",
        "hf_token_using_default": "Menggunakan token default",
        "hf_token_custom_active": "Token khusus aktif",
        "hf_token_clear": "Atur ulang ke default",
    },
    "ja": {
        "hf_token_title": "Hugging Face トークン",
        "hf_token_subtitle": "モデルダウンロード用の個人用アクセストークン",
        "hf_token_dialog_title": "Hugging Face アクセストークン",
        "hf_token_explanation": "デフォルトのトークンで失敗した場合や利用制限に達した際、またライセンス同意が必要な保護されたモデルをダウンロードするには、huggingface.co/settings/tokens で取得した個人用リードトークンを入力してください。",
        "hf_token_placeholder": "個人用トークンを入力 (hf_...)",
        "hf_token_using_default": "デフォルトトークンを使用中",
        "hf_token_custom_active": "カスタムトークン有効",
        "hf_token_clear": "デフォルトに戻す",
    },
    "ko": {
        "hf_token_title": "Hugging Face 토큰",
        "hf_token_subtitle": "모델 다운로드를 위한 개인 액세스 토큰",
        "hf_token_dialog_title": "Hugging Face 액세스 토큰",
        "hf_token_explanation": "기본 토큰이 실패하거나 요청 한도에 도달한 경우, 또는 라이선스 승인이 필요한 모델을 다운로드하려면 huggingface.co/settings/tokens에서 발급받은 개인 읽기 토큰을 입력하세요.",
        "hf_token_placeholder": "개인 토큰 입력 (hf_...)",
        "hf_token_using_default": "기본 토큰 사용 중",
        "hf_token_custom_active": "사용자 지정 토큰 활성",
        "hf_token_clear": "기본값으로 재설정",
    },
    "zh-TW": {
        "hf_token_title": "Hugging Face 權杖",
        "hf_token_subtitle": "下載模型的個人存取權杖",
        "hf_token_dialog_title": "Hugging Face 存取權杖",
        "hf_token_explanation": "若預設權杖失效、達到請求頻率上限，或下載需同意授權協議的模型，請輸入您在 huggingface.co/settings/tokens 取得的個人讀取權杖。",
        "hf_token_placeholder": "輸入個人權杖 (hf_...)",
        "hf_token_using_default": "使用預設權杖",
        "hf_token_custom_active": "自訂權杖已啟用",
        "hf_token_clear": "重設為預設值",
    },
    "nl": {
        "hf_token_title": "Hugging Face-token",
        "hf_token_subtitle": "Persoonlijk toegangstoken voor modeldownloads",
        "hf_token_dialog_title": "Hugging Face-toegangstoken",
        "hf_token_explanation": "Als het standaardtoken mislukt, de limiet bereikt of voor modellen met een geaccepteerde licentie, voer dan uw persoonlijke leestoken in via huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Persoonlijk token invoeren (hf_...)",
        "hf_token_using_default": "Standaardtoken in gebruik",
        "hf_token_custom_active": "Aangepast token actief",
        "hf_token_clear": "Herstellen naar standaard",
    },
    "da": {
        "hf_token_title": "Hugging Face-token",
        "hf_token_subtitle": "Personligt adgangstoken til download af modeller",
        "hf_token_dialog_title": "Hugging Face-adgangstoken",
        "hf_token_explanation": "Hvis standardtokenet mislykkes, rammer grænser eller ved download af modeller med godkendt licens, skal du indtaste dit personlige læsetoken fra huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Indtast personligt token (hf_...)",
        "hf_token_using_default": "Bruger standardtoken",
        "hf_token_custom_active": "Brugerdefineret token aktivt",
        "hf_token_clear": "Nulstil til standard",
    },
    "th": {
        "hf_token_title": "โทเค็น Hugging Face",
        "hf_token_subtitle": "โทเค็นการเข้าถึงส่วนตัวสำหรับการดาวน์โหลดโมเดล",
        "hf_token_dialog_title": "โทเค็นการเข้าถึง Hugging Face",
        "hf_token_explanation": "หากโทเค็นเริ่มต้นล้มเหลว ถึงขีดจำกัดการใช้งาน หรือเมื่อดาวน์โหลดโมเดลที่ต้องยอมรับสิทธิ์การใช้งาน ให้ป้อนโทเค็นการอ่านส่วนตัวจาก huggingface.co/settings/tokens",
        "hf_token_placeholder": "ป้อนโทเค็นส่วนตัว (hf_...)",
        "hf_token_using_default": "กำลังใช้โทเค็นเริ่มต้น",
        "hf_token_custom_active": "โทเค็นที่กำหนดเองใช้งานอยู่",
        "hf_token_clear": "รีเซ็ตเป็นค่าเริ่มต้น",
    },
    "vi": {
        "hf_token_title": "Mã Hugging Face",
        "hf_token_subtitle": "Mã truy cập cá nhân để tải xuống mô hình",
        "hf_token_dialog_title": "Mã truy cập Hugging Face",
        "hf_token_explanation": "Nếu mã mặc định bị lỗi, đạt giới hạn tốc độ hoặc khi tải mô hình cần chấp nhận giấy phép, hãy nhập mã đọc cá nhân từ huggingface.co/settings/tokens.",
        "hf_token_placeholder": "Nhập mã cá nhân (hf_...)",
        "hf_token_using_default": "Đang dùng mã mặc định",
        "hf_token_custom_active": "Mã tùy chỉnh đang hoạt động",
        "hf_token_clear": "Đặt lại về mặc định",
    }
}

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Update iOS Localizable.strings
ios_dir = os.path.join(root, "ios/LLMHub/Sources/LLMHub")
for fname in os.listdir(ios_dir):
    if fname.endswith(".lproj"):
        locale = fname[:-6]
        strings_path = os.path.join(ios_dir, fname, "Localizable.strings")
        if not os.path.exists(strings_path):
            continue
        trans = translations.get(locale, translations["en"])
        with open(strings_path, "r", encoding="utf-8") as f:
            content = f.read()

        new_entries = []
        for k, v in trans.items():
            if f'"{k}"' not in content:
                escaped = v.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                new_entries.append(f'"{k}" = "{escaped}";')

        if new_entries:
            with open(strings_path, "a", encoding="utf-8") as f:
                f.write("\n\n/* Hugging Face Token */\n" + "\n".join(new_entries) + "\n")
            print(f"Updated iOS {locale} with {len(new_entries)} entries")

# 2. Update Android strings.xml
android_map = {
    "values": "en",
    "values-in": "id",
    "values-iw": "he",
    "values-zh": "zh-TW"
}

res_dir = os.path.join(root, "android/app/src/main/res")
for folder in sorted(os.listdir(res_dir)):
    if folder.startswith("values"):
        strings_xml = os.path.join(res_dir, folder, "strings.xml")
        if not os.path.exists(strings_xml):
            continue
        locale = android_map.get(folder, folder.replace("values-", ""))
        trans = translations.get(locale, translations["en"])

        with open(strings_xml, "r", encoding="utf-8") as f:
            content = f.read()

        new_tags = []
        for k, v in trans.items():
            if f'name="{k}"' not in content:
                escaped = (v.replace('&', '&amp;')
                            .replace('<', '&lt;')
                            .replace('>', '&gt;')
                            .replace("'", "\\'")
                            .replace('"', '\\"'))
                new_tags.append(f'    <string name="{k}">{escaped}</string>')

        if new_tags:
            idx = content.rfind("</resources>")
            if idx != -1:
                updated = content[:idx] + "    <!-- Hugging Face Token -->\n" + "\n".join(new_tags) + "\n" + content[idx:]
                with open(strings_xml, "w", encoding="utf-8") as f:
                    f.write(updated)
                print(f"Updated Android {folder} ({locale}) with {len(new_tags)} tags")
