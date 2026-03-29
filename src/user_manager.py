"""
Kullanıcı yönetimi: kayıt, doğrulama, rol ve paylaşımlı erişim.
Veriler PROJECT_ROOT/data/users.json dosyasında saklanır.
"""

import hashlib
import json
import os
import secrets
import uuid
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USERS_FILE = PROJECT_ROOT / "data" / "users.json"

ROLE_ADMIN = "admin"
ROLE_USER = "user"


# ─── Şifreleme ────────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _make_salt() -> str:
    return secrets.token_hex(16)


# ─── Dosya okuma/yazma ────────────────────────────────────────────────────────

def _load() -> dict:
    if not USERS_FILE.exists():
        return {"users": []}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Kullanıcı işlemleri ──────────────────────────────────────────────────────

def get_user(username: str) -> Optional[dict]:
    """Kullanıcı kaydını döndür, yoksa None."""
    data = _load()
    for user in data["users"]:
        if user["username"] == username:
            return user
    return None


def verify_password(username: str, password: str) -> Optional[dict]:
    """
    Kimlik bilgilerini doğrula.
    Geçerliyse kullanıcı kaydını döndür, geçersizse None.

    Önce users.json kontrolü; başarısız olursa env değişkenlerine
    (CHAINLIT_APP_USERNAME / CHAINLIT_APP_PASSWORD) geri düşer.
    """
    user = get_user(username)
    if user is not None:
        expected = _hash_password(password, user["salt"])
        if secrets.compare_digest(expected, user["password_hash"]):
            return user
        return None

    # Env değişkeni fallback (ilk kurulum / eski yapılandırma)
    env_user = os.getenv("CHAINLIT_APP_USERNAME", "admin")
    env_pass = os.getenv("CHAINLIT_APP_PASSWORD", "admin123")
    if username == env_user and password == env_pass:
        return {
            "id": "env-admin",
            "username": username,
            "role": ROLE_ADMIN,
            "can_access_shared": True,
        }
    return None


def add_user(
    username: str,
    password: str,
    role: str = ROLE_USER,
    can_access_shared: bool = False,
) -> dict:
    """
    Yeni kullanıcı ekle.
    Aynı kullanıcı adı zaten varsa ValueError fırlatır.
    """
    if get_user(username) is not None:
        raise ValueError(f"Kullanıcı zaten mevcut: {username}")
    salt = _make_salt()
    user = {
        "id": str(uuid.uuid4()),
        "username": username,
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "role": role,
        "can_access_shared": can_access_shared,
    }
    data = _load()
    data["users"].append(user)
    _save(data)
    return user


def remove_user(username: str) -> bool:
    """Kullanıcıyı sil. Bulunursa True, yoksa False döner."""
    data = _load()
    before = len(data["users"])
    data["users"] = [u for u in data["users"] if u["username"] != username]
    if len(data["users"]) == before:
        return False
    _save(data)
    return True


def set_shared_access(username: str, can_access: bool) -> bool:
    """
    Kullanıcının paylaşımlı belgelere erişimini aç/kapat.
    Kullanıcı bulunursa True, yoksa False döner.
    """
    data = _load()
    for user in data["users"]:
        if user["username"] == username:
            user["can_access_shared"] = can_access
            _save(data)
            return True
    return False


def change_password(username: str, new_password: str) -> bool:
    """Kullanıcı şifresini değiştir."""
    data = _load()
    for user in data["users"]:
        if user["username"] == username:
            salt = _make_salt()
            user["salt"] = salt
            user["password_hash"] = _hash_password(new_password, salt)
            _save(data)
            return True
    return False


def list_users() -> list[dict]:
    """Tüm kullanıcıları döndür (şifre hash ve salt olmadan)."""
    data = _load()
    return [
        {
            "username": u["username"],
            "role": u["role"],
            "can_access_shared": u["can_access_shared"],
        }
        for u in data["users"]
    ]


# ─── Dizin yardımcıları ───────────────────────────────────────────────────────

def user_docs_dir(username: str) -> Path:
    return PROJECT_ROOT / "docs" / "users" / username


def shared_docs_dir() -> Path:
    return PROJECT_ROOT / "docs" / "shared"


def user_index_path(username: str) -> Path:
    return PROJECT_ROOT / "indexes" / "users" / username / "doc_index"


def shared_index_path() -> Path:
    return PROJECT_ROOT / "indexes" / "shared" / "doc_index"


def ensure_user_dirs(username: str) -> None:
    """Kullanıcıya ait docs ve indexes dizinlerini oluştur."""
    user_docs_dir(username).mkdir(parents=True, exist_ok=True)
    user_index_path(username).parent.mkdir(parents=True, exist_ok=True)


def ensure_shared_dirs() -> None:
    shared_docs_dir().mkdir(parents=True, exist_ok=True)
    shared_index_path().parent.mkdir(parents=True, exist_ok=True)
