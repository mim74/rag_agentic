import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import chainlit as cl
import requests
from chainlit.context import context
from chainlit.data import get_data_layer
from chainlit.types import ThreadDict
from chainlit.user import User

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from chainlit_local_data_layer import LocalDataLayer
from chat import _load_colpali_model_and_index, initialize_system, load_settings
from embedding import EmbeddingModel
from hf_load_hacks import resolve_hf_local_files_only
from lm_studio_client import LMStudioClient
from rag_agentic import agentic_rag
from rag_colpali import colpali_rag
from rag_simple import simple_rag
from user_manager import (
    ROLE_ADMIN,
    add_user,
    change_password,
    ensure_shared_dirs,
    ensure_user_dirs,
    get_user,
    list_users,
    remove_user,
    set_shared_access,
    shared_docs_dir,
    shared_index_path,
    user_docs_dir,
    user_index_path,
    verify_password,
)

LOCAL_DATA_DIR = PROJECT_ROOT / "exports" / "chainlit_datalayer"
SUPPORTED_EXTENSIONS = {".pdf", ".odt", ".docx"}


@cl.data_layer
def data_layer():
    return LocalDataLayer(LOCAL_DATA_DIR)


# ─── Ağır bileşenler (tüm kullanıcılar paylaşır, bir kez yüklenir) ────────────

@cl.cache
def _get_heavy_components():
    """Embedding model ve LM client — global cache, her kullanıcı oturumu paylaşır."""
    config = load_settings()
    emb_config = config["embedding"]
    emb_local = resolve_hf_local_files_only(emb_config.get("local_files_only"))
    embedding_model = EmbeddingModel(
        model_name=emb_config["model_name"],
        device=emb_config["device"],
        prefer_gpu_type=emb_config.get("prefer_gpu_type"),
        gpu_index=emb_config.get("gpu_index"),
        local_files_only=emb_local,
    )
    lm_client = LMStudioClient(
        base_url=config["lm_studio"]["base_url"],
        timeout=config["lm_studio"]["timeout"],
    )
    return embedding_model, lm_client, config


# ─── Kimlik doğrulama ─────────────────────────────────────────────────────────

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[User]:
    user_record = verify_password(username, password)
    if user_record is None:
        return None
    return User(
        identifier=username,
        metadata={
            "role": user_record.get("role", "user"),
            "can_access_shared": user_record.get("can_access_shared", False),
            "provider": "credentials",
        },
    )


# ─── Kullanıcıya özel sistem yükleme ─────────────────────────────────────────

def _user_record_from_session() -> dict:
    """Oturumdan ya da users.json'dan kullanıcı kaydını döndür."""
    username = cl.user_session.get("username") or ""
    record = get_user(username)
    if record:
        return record
    cl_user = context.session.user
    meta = getattr(cl_user, "metadata", {}) or {}
    return {
        "role": meta.get("role", "user"),
        "can_access_shared": meta.get("can_access_shared", False),
    }


async def _load_user_system(
    username: str,
    user_record: dict,
    status_msg: Optional[cl.Message] = None,
) -> None:
    """Kullanıcıya özel index yükle/oluştur ve oturuma kaydet."""
    ensure_user_dirs(username)
    ensure_shared_dirs()

    doc_dir = user_docs_dir(username)
    idx_path = user_index_path(username)
    can_shared = user_record.get("can_access_shared", False)

    shared_doc = shared_docs_dir() if can_shared else None
    shared_idx = shared_index_path() if can_shared else None

    embedding_model, lm_client, config = await cl.make_async(_get_heavy_components)()

    _, index, metadata, _, _ = await cl.make_async(initialize_system)(
        config,
        document_dir=doc_dir,
        index_path=idx_path,
        shared_document_dir=shared_doc,
        shared_index_path=shared_idx,
        embedding_model=embedding_model,
    )

    colpali_state = None
    use_colpali = config.get("agentic_rag", {}).get("use_colpali_vision", False)
    if use_colpali:
        c_model, c_processor, c_embs, c_meta = await cl.make_async(
            _load_colpali_model_and_index
        )(config)
        colpali_state = {
            "model": c_model,
            "processor": c_processor,
            "page_embeddings": c_embs,
            "metadata": c_meta,
            "top_k": config.get("colpali", {}).get("top_k_pages", 3),
            "score_device": "cuda"
            if str(config.get("colpali", {}).get("retrieval_device", "auto")).startswith("cuda")
            else None,
        }

    cl.user_session.set("embedding_model", embedding_model)
    cl.user_session.set("lm_client", lm_client)
    cl.user_session.set("index", index)
    cl.user_session.set("metadata", metadata)
    cl.user_session.set("config", config)
    cl.user_session.set("colpali_state", colpali_state)
    cl.user_session.set("username", username)
    cl.user_session.set("user_role", user_record.get("role", "user"))
    cl.user_session.set("can_access_shared", can_shared)

    if status_msg is not None:
        status_msg.content = "**Sistem hazır!** Belgelerinizle ilgili sorularınızı sorabilirsiniz."
        await status_msg.update()


async def ensure_system_loaded(status_message: Optional[cl.Message] = None) -> None:
    if cl.user_session.get("embedding_model"):
        return
    if status_message is not None:
        status_message.content = "Sistem başlatılıyor, lütfen bekleyin..."
        await status_message.send()
    cl_user = context.session.user
    username = getattr(cl_user, "identifier", None) or "anonymous"
    user_record = get_user(username) or {
        "role": getattr(cl_user, "metadata", {}).get("role", "user"),
        "can_access_shared": getattr(cl_user, "metadata", {}).get("can_access_shared", False),
    }
    await _load_user_system(username, user_record, status_message)


def sync_progress_callback(raw_response, action):
    async def _send_step():
        step_name = {
            "SEARCH": "Metin Araması",
            "VISUAL_SEARCH": "Görsel Arama",
            "ANSWER": "Cevap Oluşturuluyor",
        }.get(action.type, "Düşünme")
        step = cl.Step(name=step_name, type="tool")
        step.output = raw_response
        await step.send()

    cl.run_sync(_send_step())


def build_conversation_history(thread: ThreadDict):
    history = []
    pending_question = None
    steps = sorted(
        thread.get("steps", []),
        key=lambda step: step.get("createdAt") or step.get("start") or "",
    )
    for step in steps:
        step_type = step.get("type")
        content = step.get("output") or step.get("input") or ""
        if step_type == "user_message":
            pending_question = content
        elif step_type == "assistant_message" and pending_question is not None:
            history.append({"question": pending_question, "answer": content})
            pending_question = None
    return history


def build_thread_name(question: str):
    prefix = cl.user_session.get("thread_time_prefix")
    if not prefix:
        from datetime import datetime

        prefix = datetime.now().strftime("%d.%m %H:%M")
        cl.user_session.set("thread_time_prefix", prefix)
    short_question = question.strip().replace("\n", " ")
    if len(short_question) > 42:
        short_question = short_question[:39] + "..."
    return f"{prefix} · {short_question}"


async def update_thread_title_if_needed(question: str):
    if cl.user_session.get("thread_title_initialized"):
        return
    data_layer_instance = get_data_layer()
    if data_layer_instance:
        await data_layer_instance.update_thread(
            thread_id=context.session.thread_id,
            name=build_thread_name(question),
        )
    cl.user_session.set("thread_title_initialized", True)


def format_sources(chunks):
    if not chunks:
        return ""

    sources_dict = {}
    for chunk_meta, _score in chunks:
        source_name = chunk_meta.get("source", "Bilinmeyen")
        page_num = chunk_meta.get("page", "?")
        try:
            page_value = int(page_num)
        except (TypeError, ValueError):
            page_value = page_num
        sources_dict.setdefault(source_name, set()).add(page_value)

    lines = ["", "", "**Yararlanılan Kaynaklar:**"]
    for source_name, pages in sources_dict.items():
        sorted_pages = sorted(list(pages), key=lambda value: (isinstance(value, str), value))
        pages_str = ", ".join(str(page) for page in sorted_pages)
        lines.append(f"- {source_name} (Sayfa: {pages_str})")
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("thread_title_initialized", False)
    cl.user_session.set("thread_time_prefix", None)

    # Kullanıcı bilgilerini direkt oturuma yaz (context.session.user her zaman geçerli)
    cl_user = context.session.user
    username = getattr(cl_user, "identifier", None) or "anonymous"
    meta = getattr(cl_user, "metadata", {}) or {}
    cl.user_session.set("username", username)
    cl.user_session.set("user_role", meta.get("role", "user"))
    cl.user_session.set("can_access_shared", meta.get("can_access_shared", False))

    status_message = cl.Message(content="")
    await ensure_system_loaded(status_message)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("conversation_history", build_conversation_history(thread))
    cl.user_session.set("thread_title_initialized", bool(thread.get("name")))
    prefix = None
    thread_name = thread.get("name") or ""
    if " · " in thread_name:
        prefix = thread_name.split(" · ", 1)[0]
    cl.user_session.set("thread_time_prefix", prefix)

    cl_user = context.session.user
    username = getattr(cl_user, "identifier", None) or "anonymous"
    meta = getattr(cl_user, "metadata", {}) or {}
    cl.user_session.set("username", username)
    cl.user_session.set("user_role", meta.get("role", "user"))
    cl.user_session.set("can_access_shared", meta.get("can_access_shared", False))

    await ensure_system_loaded()


# ─── Dosya yükleme ────────────────────────────────────────────────────────────

async def _handle_file_upload(message: cl.Message, to_shared: bool = False) -> bool:
    """
    Mesaja eklenmiş desteklenen dosyaları ilgili dizine kaydeder ve
    kullanıcının index'ini artımlı olarak yeniden yükler.
    Dosya işlenirse True döndürür.
    """
    uploaded = [
        el for el in (message.elements or [])
        if hasattr(el, "path") and el.path
        and Path(el.path).suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not uploaded:
        return False

    username = cl.user_session.get("username") or "anonymous"
    role = cl.user_session.get("user_role", "user")

    if to_shared and role != ROLE_ADMIN:
        await cl.Message("⛔ Paylaşımlı alana yükleme için admin yetkisi gerekli.").send()
        return True

    target_dir = shared_docs_dir() if to_shared else user_docs_dir(username)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_names = []
    for el in uploaded:
        dest = target_dir / el.name
        shutil.copy2(Path(el.path), dest)
        saved_names.append(el.name)

    area = "paylaşımlı alan" if to_shared else "kişisel belge klasörünüz"
    info_msg = await cl.Message(
        content=f"📁 {len(saved_names)} dosya {area}na kaydedildi: "
                + ", ".join(f"`{n}`" for n in saved_names)
                + "\n⏳ Index güncelleniyor..."
    ).send()

    # initialize_system artımlı güncelleme yapar; sadece yeni dosyalar işlenir.
    user_record = _user_record_from_session()
    await _load_user_system(username, user_record)

    info_msg.content = f"✅ {len(saved_names)} dosya yüklendi ve index güncellendi."
    await info_msg.update()
    return True


# ─── Admin komutları ──────────────────────────────────────────────────────────

_ADMIN_HELP = """\
**Admin Komutları:**
- `/adduser <kullanıcı> <şifre>` — yeni kullanıcı ekle
- `/adduser <kullanıcı> <şifre> --shared` — paylaşımlı erişimle ekle
- `/removeuser <kullanıcı>` — kullanıcıyı sil
- `/listusers` — kullanıcıları listele
- `/setshared <kullanıcı> on|off` — paylaşımlı belge erişimini aç/kapat
- `/changepassword <kullanıcı> <yeni_şifre>` — şifre değiştir
- `/help` — bu yardımı göster

**Dosya yükleme:**
- Mesaja dosya ekle → kişisel belge klasörünüze kaydedilir
- `/shared` yazıp dosya ekle → paylaşımlı alana kaydedilir
"""


async def _handle_admin_command(text: str) -> bool:
    """
    Admin komutunu işle. Komut tanınırsa True döndürür.
    """
    role = cl.user_session.get("user_role", "user")
    parts = text.strip().split()
    cmd = parts[0].lower() if parts else ""

    # /help herkes kullanabilir, ama admin komutları admin'e özel
    if cmd == "/help":
        await cl.Message(_ADMIN_HELP).send()
        return True

    if role != ROLE_ADMIN:
        return False  # admin komutu değil, normal mesaj olarak işle

    if cmd == "/listusers":
        users = list_users()
        if not users:
            await cl.Message("Henüz kayıtlı kullanıcı yok.").send()
        else:
            lines = ["**Kullanıcı Listesi:**", ""]
            for u in users:
                shared_icon = "🌐" if u["can_access_shared"] else "🔒"
                role_icon = "👑" if u["role"] == ROLE_ADMIN else "👤"
                lines.append(f"{role_icon} {shared_icon} `{u['username']}` ({u['role']})")
            await cl.Message("\n".join(lines)).send()
        return True

    if cmd == "/adduser":
        if len(parts) < 3:
            await cl.Message("Kullanım: `/adduser <kullanıcı> <şifre> [--shared]`").send()
            return True
        uname, pwd = parts[1], parts[2]
        can_shared = "--shared" in parts
        try:
            add_user(uname, pwd, can_access_shared=can_shared)
            ensure_user_dirs(uname)
            shared_note = " (paylaşımlı erişim açık)" if can_shared else ""
            await cl.Message(f"✅ Kullanıcı eklendi: `{uname}`{shared_note}").send()
        except ValueError as exc:
            await cl.Message(f"❌ {exc}").send()
        return True

    if cmd == "/removeuser":
        if len(parts) < 2:
            await cl.Message("Kullanım: `/removeuser <kullanıcı>`").send()
            return True
        uname = parts[1]
        if remove_user(uname):
            await cl.Message(f"✅ Kullanıcı silindi: `{uname}`").send()
        else:
            await cl.Message(f"❌ Kullanıcı bulunamadı: `{uname}`").send()
        return True

    if cmd == "/setshared":
        if len(parts) < 3:
            await cl.Message("Kullanım: `/setshared <kullanıcı> on|off`").send()
            return True
        uname, flag = parts[1], parts[2].lower()
        if flag not in ("on", "off"):
            await cl.Message("❌ Değer `on` veya `off` olmalı.").send()
            return True
        val = flag == "on"
        if set_shared_access(uname, val):
            state = "açıldı ✅" if val else "kapatıldı 🔒"
            await cl.Message(f"`{uname}` için paylaşımlı erişim {state}").send()
        else:
            await cl.Message(f"❌ Kullanıcı bulunamadı: `{uname}`").send()
        return True

    if cmd == "/changepassword":
        if len(parts) < 3:
            await cl.Message("Kullanım: `/changepassword <kullanıcı> <yeni_şifre>`").send()
            return True
        uname, new_pwd = parts[1], parts[2]
        if change_password(uname, new_pwd):
            await cl.Message(f"✅ `{uname}` şifresi güncellendi.").send()
        else:
            await cl.Message(f"❌ Kullanıcı bulunamadı: `{uname}`").send()
        return True

    return False  # tanınmayan komut


# ─── Ana mesaj işleyici ───────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    await ensure_system_loaded()

    text = (message.content or "").strip()

    # Dosya yükleme: /shared komutuyla paylaşımlı alana, aksi hâlde kişisel dizine
    if message.elements:
        to_shared = text.lower().startswith("/shared")
        handled = await _handle_file_upload(message, to_shared=to_shared)
        if handled and (not text or to_shared or not text.replace("/shared", "").strip()):
            return
        # Hem dosya hem soru varsa: önce dosyayı işle, sonra soruya devam et
        if handled:
            text = text.replace("/shared", "").strip()
            if not text:
                return

    # Komut kontrolü: / ile başlayan hiçbir şey RAG'a düşmez
    if text.startswith("/"):
        if not await _handle_admin_command(text):
            await cl.Message("❓ Bilinmeyen komut veya yetki eksik. `/help` yazın.").send()
        return

    if not text:
        return

    question = text
    embedding_model = cl.user_session.get("embedding_model")
    index = cl.user_session.get("index")
    metadata = cl.user_session.get("metadata")
    lm_client = cl.user_session.get("lm_client")
    config = cl.user_session.get("config")
    colpali_state = cl.user_session.get("colpali_state")
    history = cl.user_session.get("conversation_history") or []

    await update_thread_title_if_needed(question)

    rag_mode = config.get("agentic_rag", {}).get("mode", "agentic")
    res_msg = cl.Message(content="")
    final_answer = ""
    chunks = None

    try:
        if rag_mode == "agentic":
            max_iterations = config.get("agentic_rag", {}).get("max_iterations", 5)

            def run_agentic():
                return agentic_rag(
                    question=question,
                    embedding_model=embedding_model,
                    index=index,
                    metadata=metadata,
                    lm_client=lm_client,
                    config=config,
                    progress_callback=sync_progress_callback,
                    max_iterations=max_iterations,
                    colpali_state=colpali_state,
                    conversation_history=history,
                )

            final_answer, chunks, _, _, _ = await cl.make_async(run_agentic)()

        elif rag_mode == "simple":

            def run_simple():
                return simple_rag(
                    question=question,
                    embedding_model=embedding_model,
                    index=index,
                    metadata=metadata,
                    lm_client=lm_client,
                    config=config,
                    stream_callback=None,
                )

            final_answer, chunks, _, _ = await cl.make_async(run_simple)()

        elif rag_mode == "colpali":
            if not colpali_state:
                await cl.Message("ColPali yüklenmemiş.").send()
                return

            def run_colpali():
                return colpali_rag(
                    question=question,
                    colpali_model=colpali_state["model"],
                    colpali_processor=colpali_state["processor"],
                    page_embeddings=colpali_state["page_embeddings"],
                    metadata=colpali_state["metadata"],
                    lm_client=lm_client,
                    config=config,
                )

            final_answer, _, _, _ = await cl.make_async(run_colpali)()

    except requests.exceptions.ReadTimeout:
        def run_simple_fallback():
            return simple_rag(
                question=question,
                embedding_model=embedding_model,
                index=index,
                metadata=metadata,
                lm_client=lm_client,
                config=config,
                stream_callback=None,
            )

        try:
            final_answer, chunks, _, _ = await cl.make_async(run_simple_fallback)()
        except Exception:
            timeout_sec = config.get("lm_studio", {}).get("timeout", 120)
            await cl.Message(
                content=(
                    "LM Studio zaman aşımına uğradı ve fallback cevap üretilemedi.\n"
                    "Lütfen daha kısa bir soru deneyin veya timeout değerini artırın "
                    f"(`config/settings.json`, mevcut: {timeout_sec}s)."
                )
            ).send()
            return

    except Exception as exc:
        await cl.Message(content=f"Yanıt üretilirken hata oluştu: {exc}").send()
        return

    res_msg.content = final_answer + format_sources(chunks)
    await res_msg.send()

    history.append({"question": question, "answer": final_answer, "mode": rag_mode})
    cl.user_session.set("conversation_history", history)
