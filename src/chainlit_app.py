import os
import sys
from pathlib import Path

import chainlit as cl
from chainlit.context import context
from chainlit.data import get_data_layer
from chainlit.types import ThreadDict
from chainlit.user import User

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from chainlit_local_data_layer import LocalDataLayer
from chat import _load_colpali_model_and_index, initialize_system, load_settings
from rag_agentic import agentic_rag
from rag_colpali import colpali_rag
from rag_simple import simple_rag

LOCAL_DATA_DIR = PROJECT_ROOT / "exports" / "chainlit_datalayer"


@cl.data_layer
def data_layer():
    return LocalDataLayer(LOCAL_DATA_DIR)


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    expected_username = os.getenv("CHAINLIT_APP_USERNAME", "admin")
    expected_password = os.getenv("CHAINLIT_APP_PASSWORD", "admin123")
    if username == expected_username and password == expected_password:
        return User(
            identifier=username,
            metadata={"role": "user", "provider": "credentials"},
        )
    return None


@cl.cache
def get_system_components():
    config = load_settings()
    embedding_model, index, metadata, lm_client, config = initialize_system(config)

    colpali_state = None
    use_colpali = config.get("agentic_rag", {}).get("use_colpali_vision", False)
    if use_colpali:
        c_model, c_processor, c_embs, c_meta = _load_colpali_model_and_index(config)
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

    return embedding_model, index, metadata, lm_client, config, colpali_state


async def ensure_system_loaded(status_message: cl.Message | None = None):
    if cl.user_session.get("embedding_model"):
        return

    if status_message is not None:
        status_message.content = "Sistem başlatılıyor, lütfen bekleyin..."
        await status_message.send()

    embedding_model, index, metadata, lm_client, config, colpali_state = await cl.make_async(
        get_system_components
    )()

    cl.user_session.set("embedding_model", embedding_model)
    cl.user_session.set("index", index)
    cl.user_session.set("metadata", metadata)
    cl.user_session.set("lm_client", lm_client)
    cl.user_session.set("config", config)
    cl.user_session.set("colpali_state", colpali_state)

    if status_message is not None:
        status_message.content = "**Sistem hazır!** Belgelerinizle ilgili sorularınızı sorabilirsiniz."
        await status_message.update()


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
    await ensure_system_loaded()


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content

    await ensure_system_loaded()

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

    res_msg.content = final_answer + format_sources(chunks)
    await res_msg.send()

    history.append({"question": question, "answer": final_answer, "mode": rag_mode})
    cl.user_session.set("conversation_history", history)
