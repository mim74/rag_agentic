"""Transformers model yükleme sırasında dar VRAM uyumu ve HF hub erişimi."""

import os
import socket
from contextlib import contextmanager
from typing import Any, Optional


def check_and_set_offline_mode(timeout: float = 3.0) -> None:
    """
    huggingface.co'ya TCP bağlantısı dener; başarısızsa ``HF_HUB_OFFLINE=1`` ve
    ``TRANSFORMERS_OFFLINE=1`` ortam değişkenlerini ayarlar.

    ``huggingface_hub.constants.HF_HUB_OFFLINE`` modül import edilirken
    ``os.environ`` üzerinden okunur; bu yüzden bu fonksiyon **SentenceTransformer**
    veya **transformers** import edilmeden ÖNCE çağrılmalıdır.

    Kullanıcı ``HF_HUB_OFFLINE`` veya ``TRANSFORMERS_OFFLINE``'ı önceden
    ayarlamışsa dokunulmaz.
    """
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on"):
            return

    try:
        conn = socket.create_connection(("huggingface.co", 443), timeout=timeout)
        conn.close()
    except (OSError, socket.timeout):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_hf_local_files_only(config_value: Optional[Any] = None) -> bool:
    """
    True ise ilgili ``from_pretrained`` çağrısında yalnızca yerel önbellek kullanılır.

    - ``True`` / ``False`` açıkça verilmişse bire bir kullanılır.
    - ``None`` (anahtar yok) ise ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` kontrol edilir.
    """
    if config_value is not None:
        return bool(config_value)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        v = os.environ.get(key, "").strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
    return False


def patch_safetensors_auto_conversion() -> None:
    """
    ``transformers.safetensors_conversion.auto_conversion`` her zaman
    ``ignore_errors_during_conversion=True`` ile çalışacak şekilde patch'ler.

    Arka plan thread'inde çalışır; hata olması model yüklemeyi etkilemez.
    Patchsiz hata varsa Python thread exception olarak stderr'e yazar.
    """
    try:
        import transformers.safetensors_conversion as _sc
    except ImportError:
        return
    if getattr(_sc, "_auto_conv_patched", False):
        return

    _orig = _sc.auto_conversion

    def _patched(*args, ignore_errors_during_conversion=False, **kwargs):
        return _orig(*args, ignore_errors_during_conversion=True, **kwargs)

    _sc.auto_conversion = _patched
    _sc._auto_conv_patched = True


def patch_find_adapter_for_offline() -> None:
    """
    ``sentence_transformers`` içindeki ``find_adapter_config_file`` referansını patch'ler.

    DNS başarısız olduğunda ``httpx`` istemcisi kapanır ve ``RuntimeError``
    fırlatır; bu hata ``_raise_exceptions_for_connection_errors=False``
    koruyucusunu atlar. Patch yalnızca bu hatayı yakalar; ``None`` döndürür.
    BGE-M3 ve ColQwen PEFT adapter kullanmaz, davranış değişmez.
    """
    try:
        import importlib
        import sentence_transformers.models.Transformer  # noqa: F401
        st_mod = importlib.import_module("sentence_transformers.models.Transformer")
    except ImportError:
        return
    if getattr(st_mod, "_find_adapter_offline_patched", False):
        return
    orig = getattr(st_mod, "find_adapter_config_file", None)
    if orig is None:
        return

    def _patched(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "client has been closed" in msg or "cannot send" in msg:
                return None
            raise

    st_mod.find_adapter_config_file = _patched
    st_mod._find_adapter_offline_patched = True


@contextmanager
def no_cuda_allocator_warmup():
    """
    transformers `caching_allocator_warmup` büyük bir boş tensör ayırarak yükleme hızını artırır;
    ColQwen boyutunda modellerde bu ek allocation tek başına OOM tetikleyebilir (15GB kartlar).
    """
    import transformers.modeling_utils as modeling_utils

    orig = getattr(modeling_utils, "caching_allocator_warmup", None)
    if orig is None:
        yield
        return

    def _noop(*_a, **_k):
        return None

    modeling_utils.caching_allocator_warmup = _noop
    try:
        yield
    finally:
        modeling_utils.caching_allocator_warmup = orig
