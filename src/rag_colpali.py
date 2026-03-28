"""
ColPali tabanlı RAG modülü.

Akış:
  1. Sorguyu ColQwen2.5 ile embed et
  2. MaxSim retrieval ile top-K PDF sayfasını bul
  3. Bulunan sayfaları (PNG) base64 ile LM Studio vision LLM'e gönder
  4. Cevabı döndür
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch

from colpali_retrieval import embed_query, search_colpali, PageResult

logger = logging.getLogger(__name__)


COLPALI_SYSTEM_PROMPT = """Sen bir PDF araştırma asistanısın. Sana verilen PDF sayfası görsellerini analiz ederek soruları cevaplarsın.

KURALLAR:
1. SADECE verilen görsellerdeki bilgileri kullan, spekülasyon yapma.
2. Hangi sayfadan bilgi aldığını açıkça belirt: örn. "ED-73F.pdf Sayfa 5'te göre..."
3. Tablolar, diyagramlar ve şemalar için görsel içeriği ayrıntılı açıkla.
4. Teknik terimleri doğru kullan.
5. Cevabını Türkçe ver.
6. Eğer görsellerde soruya cevap yoksa bunu açıkça belirt."""


def colpali_rag(
    question: str,
    colpali_model,
    colpali_processor,
    page_embeddings: Dict[int, Any],
    metadata: List[Dict[str, Any]],
    lm_client,
    config: Dict[str, Any],
) -> Tuple[str, List[PageResult], float, Dict[str, int]]:
    """
    ColPali tabanlı tek seferlik RAG.

    Args:
        question: Kullanıcı sorusu
        colpali_model: Yüklü ColQwen2.5 modeli
        colpali_processor: ColQwen2.5 processor
        page_embeddings: {page_id: (N, D) ndarray}
        metadata: Sayfa metadata listesi
        lm_client: LMStudioClient (chat_completion_with_images desteği gerekli)
        config: settings.json içeriği

    Returns:
        (cevap_metni, bulunan_sayfalar, süre_saniye, token_kullanımı)
    """
    start_time = time.time()
    colpali_cfg = config.get("colpali", {})
    gen_cfg = config.get("generation", {})
    top_k = colpali_cfg.get("top_k_pages", 3)
    retrieval_device = colpali_cfg.get("retrieval_device", "auto")
    score_device = None
    if str(retrieval_device).startswith("cuda"):
        score_device = retrieval_device
    elif retrieval_device == "auto" and torch.cuda.is_available():
        score_device = "cuda"

    # 1. Sorguyu embed et
    logger.debug("Sorgu embed ediliyor...")
    query_emb = embed_query(question, colpali_model, colpali_processor)

    # 2. MaxSim retrieval (GPU: score_device ile matmul GPU'da; önceden tümü CPU NumPy idi)
    logger.debug("MaxSim retrieval (top_k=%d, score_device=%s)...", top_k, score_device)
    results = search_colpali(
        query_embedding=query_emb,
        page_embeddings=page_embeddings,
        metadata=metadata,
        top_k=top_k,
        score_device=score_device,
        query_text=question,
    )

    if not results:
        return (
            "İlgili sayfa bulunamadı. Lütfen sorunuzu farklı şekilde ifade edin.",
            [],
            time.time() - start_time,
            {},
        )

    # 3. Mevcut görsel dosyalarını filtrele
    image_paths = [
        Path(r.image_path)
        for r in results
        if r.image_path and Path(r.image_path).exists()
    ]

    if not image_paths:
        return (
            "Sayfa görselleri bulunamadı. Index'i yeniden oluşturun.",
            results,
            time.time() - start_time,
            {},
        )

    # 4. LLM için bağlam bilgisi
    source_lines = "\n".join(
        f"  [{i+1}] {r.source} — Sayfa {r.page} (MaxSim: {r.score:.3f})"
        for i, r in enumerate(results)
    )
    user_prompt = (
        f"Aşağıdaki PDF sayfaları soruyla ilgili olarak seçildi:\n"
        f"{source_lines}\n\n"
        f"SORU: {question}"
    )

    # 5. Vision LLM çağrısı
    logger.debug("Vision LLM çağrılıyor (%d görsel)...", len(image_paths))
    response = lm_client.chat_completion_with_images(
        text_prompt=user_prompt,
        image_paths=image_paths,
        system_prompt=COLPALI_SYSTEM_PROMPT,
        temperature=gen_cfg.get("temperature", 0.2),
        max_tokens=gen_cfg.get("max_tokens", 4096),
    )

    generation_time = time.time() - start_time
    token_usage = {
        "input_tokens": response.get("input_tokens", 0),
        "output_tokens": response.get("output_tokens", 0),
        "total_tokens": response.get("total_tokens", 0),
    }

    logger.debug(
        "ColPali RAG tamamlandı: %.2fs, %d token",
        generation_time,
        token_usage["total_tokens"],
    )

    return response["answer"], results, generation_time, token_usage
