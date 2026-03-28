"""
ColPali MaxSim retrieval modülü.

Query token embedding'leri (M, D) ile sayfa patch embedding'leri (N, D)
arasında MaxSim skoru hesaplayarak en ilgili sayfaları bulur.

MaxSim = Σ_i max_j (q_i · p_j)
  q_i : i'nci query token embedding'i
  p_j : j'nci sayfa patch embedding'i
"""

import logging
from dataclasses import dataclass

from hf_load_hacks import no_cuda_allocator_warmup, patch_safetensors_auto_conversion
from pathlib import Path
from typing import Dict, List, Any, Optional, Type

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    """Retrieval sonucu: bir PDF sayfası"""
    page_id: int
    source: str
    page: int
    score: float
    image_path: str

    def __str__(self) -> str:
        return f"📄 {self.source} Sayfa {self.page} (skor: {self.score:.4f})"


# ─── Model yükleme ─────────────────────────────────────────────────────────────

def load_colpali_processor(
    processor_cls: Type,
    model_name: str,
    local_files_only: bool = False,
    processor_cache_dir: Optional[str] = None,
):
    """
    ColQwen2.5 processor'ı yükle; varsa yerel önbellek dizininden.

    LoRA adapter repoları (ör. vidore/colqwen2.5-v0.2) config.json içermez.
    transformers çevrimdışı modda bu dosyayı bulamayınca hata fırlatır.
    İlk başarılı yükleme yerel dizine kaydedilir; sonraki çalışmalarda
    oradan yüklenerek bu sorun atlanır.
    """
    _dir = Path(processor_cache_dir) if processor_cache_dir else None
    if _dir and _dir.is_dir() and (_dir / "tokenizer_config.json").exists():
        return processor_cls.from_pretrained(str(_dir))

    proc = processor_cls.from_pretrained(model_name, local_files_only=local_files_only)
    if _dir:
        _dir.mkdir(parents=True, exist_ok=True)
        proc.save_pretrained(str(_dir))
    return proc


def load_colpali_model(
    model_name: str,
    retrieval_device: str = "auto",
    quantization: str = None,
    local_files_only: bool = False,
    processor_cache_dir: Optional[str] = None,
):
    """
    ColQwen2.5 model ve processor'ı yükle.

    Args:
        model_name: HuggingFace model adı veya yerel dizin yolu
        retrieval_device: "cuda" | "cpu" | "auto"
            "cpu"  → sıfır VRAM, LM Studio ile paylaşım için önerilir
            "cuda" → hızlı GPU inference
            "auto" → CUDA varsa GPU, yoksa CPU
        quantization: None | "4bit" | "8bit"
            GPU üzerinde VRAM kullanımını azaltır (bitsandbytes gerekli).
            retrieval_device="cpu" ise göz ardı edilir.
        local_files_only: True ise yalnızca yerel/HF önbelleği (~/.cache/huggingface) kullanılır.
            False (varsayılan) eksik dosyada hub'dan indirebilir.
        processor_cache_dir: Processor'ın yerel kopyasının saklanacağı dizin.
            İlk başarılı yüklemede save_pretrained ile kaydedilir;
            sonraki çalışmalarda buradan yüklenir (çevrimdışı güvenilir).

    Returns:
        (model, processor) çifti
    """
    try:
        from colpali_engine.models import ColQwen2_5
        try:
            from colpali_engine.models import ColQwen2_5Processor  # type: ignore
        except ImportError:
            from colpali_engine.models import ColQwen2_5_Processor as ColQwen2_5Processor  # type: ignore
    except ImportError as e:
        raise ImportError(
            "colpali_engine import edilemedi veya API sürümü uyumsuz.\n"
            "Kurulum:\n"
            "  pip install -U colpali_engine\n"
            "Hata detayı:\n"
            f"  {e}"
        ) from e

    # Cihaz belirleme
    if retrieval_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        d = str(retrieval_device).strip()
        device = "cpu" if d.lower() == "cpu" else d

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    logger.debug(
        "ColQwen2.5 yükleniyor: %s (device=%s, quantization=%s)",
        model_name, device, quantization,
    )
    quant_label = f" [{quantization}]" if quantization and device.startswith("cuda") else ""
    print(f"🤖 ColQwen2.5 yükleniyor: {model_name} — device: {device}{quant_label}")

    # from_pretrained kwargs
    # low_cpu_mem_usage: ağırlıkları doğrudan GPU'ya yükler, CPU RAM'de tampon tutmaz
    load_kwargs: dict = {
        "torch_dtype": dtype,
        "device_map": device,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }

    if quantization and device.startswith("cuda"):
        try:
            from transformers import BitsAndBytesConfig
            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs.pop("torch_dtype", None)  # quantization ile çakışır
            elif quantization == "8bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                load_kwargs.pop("torch_dtype", None)
            else:
                logger.warning("Bilinmeyen quantization değeri '%s', göz ardı edildi.", quantization)
        except ImportError:
            logger.warning(
                "bitsandbytes yüklü değil, quantization atlandı. "
                "Kurmak için: pip install bitsandbytes"
            )

    patch_safetensors_auto_conversion()
    with no_cuda_allocator_warmup():
        model = ColQwen2_5.from_pretrained(model_name, **load_kwargs)

    processor = load_colpali_processor(
        ColQwen2_5Processor, model_name,
        local_files_only=local_files_only,
        processor_cache_dir=processor_cache_dir,
    )

    model.eval()

    print(f"   ✅ ColQwen2.5 hazır (device: {device})\n")
    return model, processor


# ─── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(query: str, model, processor) -> np.ndarray:
    """
    Sorguyu ColQwen2.5 ile embed et.

    Returns:
        (M, D) float32 numpy array — M: query token sayısı, D: embedding boyutu
    """
    batch = processor.process_queries([query]).to(model.device)

    with torch.no_grad():
        output = model(**batch)

    raw = output.embeddings if hasattr(output, "embeddings") else output
    # (1, M, D) → (M, D)
    return raw[0].float().cpu().numpy()


# ─── MaxSim skoru ──────────────────────────────────────────────────────────────

def maxsim_score(query_emb: np.ndarray, page_emb: np.ndarray) -> float:
    """
    İki embedding arasında MaxSim skoru hesapla (CPU / NumPy).

    query_emb : (M, D) — sorgu token embedding'leri
    page_emb  : (N, D) — sayfa patch embedding'leri

    Skor = Σ_i max_j (q_i · p_j)
    """
    sim_matrix = query_emb @ page_emb.T
    return float(sim_matrix.max(axis=1).sum())


def maxsim_score_torch(
    query_emb: np.ndarray, page_emb: np.ndarray, device: str
) -> float:
    """MaxSim — PyTorch ile belirtilen GPU/CPU üzerinde (retrieval için)."""
    dev = torch.device(device)
    q = torch.as_tensor(query_emb, dtype=torch.float32, device=dev)
    p = torch.as_tensor(page_emb, dtype=torch.float32, device=dev)
    sim_matrix = q @ p.T
    return float(sim_matrix.max(dim=1).values.sum().item())


# ─── Arama ────────────────────────────────────────────────────────────────────

def search_colpali(
    query_embedding: np.ndarray,
    page_embeddings: Dict[int, np.ndarray],
    metadata: List[Dict[str, Any]],
    top_k: int = 3,
    score_device: Optional[str] = None,
    query_text: Optional[str] = None,
) -> List[PageResult]:
    """
    MaxSim ile en ilgili sayfaları bul (ve varsa dosya adı eşleştirmesi yap).

    Args:
        query_embedding: (M, D) query token embedding'leri
        page_embeddings: {page_id: (N, D) ndarray} tüm sayfa patch embedding'leri
        metadata: Sayfa kayıt listesi [{page_id, source, page, image_path, ...}]
        top_k: Döndürülecek maksimum sayfa sayısı
        score_device: "cuda", "cuda:2" vb. ise MaxSim bu cihazda PyTorch ile hesaplanır.
            "cpu" veya None ise NumPy (CPU) kullanılır.
        query_text: Sorgu metni (dosya adı eşleştirmesi için)

    Returns:
        Skora göre azalan sırada PageResult listesi
    """
    if not page_embeddings:
        logger.warning("page_embeddings boş — retrieval yapılamıyor")
        return []

    meta_by_id = {rec["page_id"]: rec for rec in metadata}
    
    # 1. Dosya adı eşleştirmesi (eğer query_text verilmişse)
    mentioned_sources = set()
    if query_text:
        query_lower = query_text.lower()
        all_sources = set(rec.get("source", "") for rec in metadata if rec.get("source"))
        for src in all_sources:
            src_lower = src.lower()
            if src_lower in query_lower:
                mentioned_sources.add(src)
            else:
                src_no_ext = src_lower.rsplit('.', 1)[0]
                if len(src_no_ext) > 3 and src_no_ext in query_lower:
                    mentioned_sources.add(src)

    use_gpu = bool(score_device and score_device.startswith("cuda"))
    scores = []
    
    for page_id, page_emb in page_embeddings.items():
        rec = meta_by_id.get(page_id, {})
        source = rec.get("source", "")
        
        # Eğer bu dosya sorguda geçiyorsa, skoruna yapay bir bonus ekle (örn: +20.0)
        # MaxSim skorları genelde 10-30 arasında değişir
        bonus = 30.0 if source in mentioned_sources else 0.0
        
        if use_gpu:
            s = maxsim_score_torch(query_embedding, page_emb, score_device)
        else:
            s = maxsim_score(query_embedding, page_emb)
            
        scores.append((page_id, s + bonus))
        
    scores.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        "ColPali search tamamlandı: %d sayfa tarandı, en yüksek skor: %.4f",
        len(scores),
        scores[0][1] if scores else 0,
    )

    results: List[PageResult] = []
    for page_id, score in scores[:top_k]:
        rec = meta_by_id.get(page_id, {})
        results.append(PageResult(
            page_id=page_id,
            source=rec.get("source", "bilinmeyen"),
            page=rec.get("page", 0),
            score=score,
            image_path=rec.get("image_path", ""),
        ))

    return results
