"""
Basit (tek-atışlık) RAG implementasyonu.
Hızlı cevaplar için optimize edilmiş.
"""

from typing import List, Dict, Any
import requests
from indexer import search


def format_context(results: List[tuple[dict, float]]) -> str:
    """Retrieval sonuçlarını context string'e çevir"""
    context_parts = []
    for i, (meta, score) in enumerate(results, 1):
        context_parts.append(
            f"[{i}] Kaynak: {meta['source']} (Sayfa {meta['page']})\n"
            f"{meta['text']}\n"
        )
    return "\n".join(context_parts)


def simple_rag(
    question: str,
    embedding_model,
    index,
    metadata,
    lm_client,
    config: Dict[str, Any],
    stream_callback=None
) -> tuple[str, List[tuple], float, Dict[str, int]]:
    """
    Basit RAG: Tek seferde retrieval + generation
    
    Args:
        question: Kullanıcı sorusu
        embedding_model: Embedding modeli
        index: FAISS index
        metadata: Chunk metadata
        lm_client: LM Studio client
        config: Konfigürasyon
        stream_callback: Token streaming için callback (opsiyonel)
        
    Returns:
        (cevap, kullanılan_kaynaklar, süre, token_kullanımı)
    """
    import time
    
    start_time = time.time()
    
    retrieval_config = config["retrieval"]
    gen_config = config["generation"]
    
    # 1. Retrieval: En ilgili chunk'ları bul
    results = search(
        query=question,
        index=index,
        metadata=metadata,
        embedding_model=embedding_model,
        top_k=retrieval_config["top_k"]
    )
    
    # 2. Context oluştur
    context = format_context(results)
    
    # 3. Prompt oluştur
    messages = [
        {"role": "system", "content": gen_config["system_prompt"]},
        {"role": "user", "content": f"BAĞLAM:\n{context}\n\nSORU: {question}"}
    ]
    
    # 4. Generation parameters hazırla
    generation_params = {
        "temperature": gen_config["temperature"],
        "max_tokens": gen_config["max_tokens"]
    }
    
    # Opsiyonel parametreleri ekle
    for param in ["top_p", "top_k", "frequency_penalty", "presence_penalty", "repeat_penalty", "stop"]:
        if param in gen_config:
            generation_params[param] = gen_config[param]
    
    # Token kullanımını takip et
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    # 5. LLM'den cevap al - timeout hata yönetimi ile
    try:
        if stream_callback:
            # Streaming mode (token tracking yok)
            answer_text = ""
            for token in lm_client.chat_completion_stream(messages=messages, **generation_params):
                answer_text += token
                stream_callback(token)
            # Streaming modda token sayısı bilinemez
        else:
            # Non-streaming mode
            response = lm_client.chat_completion(messages=messages, **generation_params)
            answer_text = response["answer"]
            # Token istatistiklerini al
            token_usage["input_tokens"] = response.get("input_tokens", 0)
            token_usage["output_tokens"] = response.get("output_tokens", 0)
            token_usage["total_tokens"] = response.get("total_tokens", 0)
            
    except requests.exceptions.ReadTimeout as e:
        # Timeout hatası - alternatif strateji: daha küçük max_tokens ile tekrar dene
        print(f"[WARNING] LM Studio timeout: {e}. Daha küçük max_tokens ile tekrar deneniyor...")
        
        # Max tokens'ı yarıya indir (config'i değiştirme, yerel kopya kullan)
        original_max_tokens = generation_params.get("max_tokens", gen_config["max_tokens"])
        reduced_max_tokens = max(512, original_max_tokens // 2)
        generation_params = {**generation_params, "max_tokens": reduced_max_tokens}
        
        # Top-k değerini azaltarak context'i küçült (config mutasyonu yapma)
        original_top_k = retrieval_config["top_k"]
        reduced_top_k = max(5, original_top_k // 2)
        
        # Daha küçük context ile tekrar retrieval yap
        reduced_results = search(
            query=question,
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
            top_k=reduced_top_k
        )
        
        # Yeni context oluştur
        reduced_context = format_context(reduced_results)
        reduced_messages = [
            {"role": "system", "content": gen_config["system_prompt"]},
            {"role": "user", "content": f"BAĞLAM:\n{reduced_context}\n\nSORU: {question}\n\nNOT: Timeout nedeniyle cevap kısaltılmıştır."}
        ]
        
        try:
            # Daha küçük parametrelerle tekrar dene
            response = lm_client.chat_completion(messages=reduced_messages, **generation_params)
            answer_text = response["answer"]
            token_usage["input_tokens"] = response.get("input_tokens", 0)
            token_usage["output_tokens"] = response.get("output_tokens", 0)
            token_usage["total_tokens"] = response.get("total_tokens", 0)
            
            # Sonuçları güncelle
            results = reduced_results
            print(f"[INFO] Timeout sonrası başarılı: max_tokens={reduced_max_tokens}, top_k={reduced_top_k}")
            
        except requests.exceptions.ReadTimeout:
            # Hala timeout - basit bir hata mesajı döndür
            timeout_sec = getattr(lm_client, "timeout", 120)
            answer_text = (
                "Üzgünüm, LM Studio sunucusu zaman aşımına uğradı. "
                "Bu, modelin çok uzun süredir yanıt vermediği anlamına geliyor. "
                "Lütfen:\n"
                "1. LM Studio sunucusunun çalıştığından emin olun\n"
                "2. Daha küçük bir soru deneyin\n"
                f"3. config/settings.json'daki 'timeout' değerini artırın (şu anda {timeout_sec} saniye)\n"
                "4. Modelinizin yeterli kaynaklara sahip olduğundan emin olun"
            )
            # Boş token kullanımı
            token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    generation_time = time.time() - start_time
    
    return answer_text, results, generation_time, token_usage
