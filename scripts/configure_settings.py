#!/usr/bin/env python3
"""
Donanım profiline göre config/settings.json içindeki cihaz ayarlarını günceller.
Kullanım: python scripts/configure_settings.py --profile cpu|cuda
"""
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"

# Her profil için hangi anahtara hangi değer yazılacağı.
# "index_device" / "retrieval_device" colpali bölümünde tek geçiyor, güvenli.
# "device" yalnızca embedding bölümünde geçiyor.
# "chat_device" yalnızca embedding bölümünde geçiyor.
PROFILES: dict[str, dict[str, str]] = {
    "cpu": {
        "device":           "cpu",
        "chat_device":      "cpu",
        "index_device":     "cpu",
        "retrieval_device": "cpu",
    },
    "cuda": {
        "device":           "auto",    # embedding indexleme: GPU varsa otomatik
        "chat_device":      "cpu",     # LLM için GPU'yu serbest bırak
        "index_device":     "balanced", # ColPali indexleme: GPU + RAM taşması
        "retrieval_device": "cpu",     # ColPali retrieval: LLM ile çakışmasın
    },
}


def _set_value(content: str, key: str, value: str) -> tuple[str, int]:
    """JSON içindeki `"key": "..."` değerini değiştirir; yorum satırlarına dokunmaz."""
    pattern = rf'(?m)^(\s*"{re.escape(key)}"\s*:\s*)"[^"]*"'
    replacement = rf'\1"{value}"'
    new_content, count = re.subn(pattern, replacement, content)
    return new_content, count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="config/settings.json içindeki cihaz ayarlarını güncelle"
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        required=True,
        help="Donanım profili: cpu veya cuda",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Değişiklikleri dosyaya yazmadan göster",
    )
    args = parser.parse_args()

    if not SETTINGS_PATH.exists():
        print(f"❌ Settings dosyası bulunamadı: {SETTINGS_PATH}", file=sys.stderr)
        sys.exit(1)

    profile = PROFILES[args.profile]
    content = SETTINGS_PATH.read_text(encoding="utf-8")
    changed = False

    print(f"📋 Profil: {args.profile.upper()}")
    for key, value in profile.items():
        new_content, count = _set_value(content, key, value)
        if count == 0:
            print(f"  ⚠️  '{key}' ayarı bulunamadı, atlandı", file=sys.stderr)
            continue
        if new_content != content:
            print(f"  ✏️  {key} → \"{value}\"")
            content = new_content
            changed = True
        else:
            print(f"  ✓  {key} zaten \"{value}\"")

    if args.dry_run:
        print("ℹ️  --dry-run: dosya değiştirilmedi")
        return

    if changed:
        SETTINGS_PATH.write_text(content, encoding="utf-8")
        print(f"✅ {SETTINGS_PATH.relative_to(PROJECT_ROOT)} güncellendi")
    else:
        print("✅ Değişiklik gerekmedi")


if __name__ == "__main__":
    main()
