"""
ReAct Pattern ile Agentic RAG implementasyonu.
LLM düşünür, araştırır ve adım adım cevabı oluşturur.
"""

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import logging
import time
import requests
import torch
from rich.console import Console
from indexer import search as search_index
from rag_simple import simple_rag
from colpali_retrieval import embed_query, search_colpali
from agent_tools import (
    AgentAction,
    AgentMemory,
    SearchResult,
    VisualSearchResult,
    parse_agent_response,
    format_chunks_for_context,
)
from rag_colpali import COLPALI_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


REACT_SYSTEM_PROMPT = """Sen bir PDF araştırma asistanısın. Soruları cevaplamak için adım adım düşün ve sistematik bir şekilde çalış.

DÜŞÜNME ŞEKLİN:
- Adım adım mantık yürüt, acele etme
- Mevcut bilgileri analiz et, eksikleri belirle
- Bir sonraki adımı planla
- Karmaşık soruları parçalara ayır

KULLANILABILIR EYLEMLER:
1. SEARCH(query): Vektör veritabanından "query" ile metin/chunk araması yap
2. THINK(reasoning): Mevcut durumu değerlendir, derinlemesine düşün, sonraki adımı planla
3. ANSWER(text): Final cevabı ver ve araştırmayı bitir

ÇALIŞMA FORMATI:
Her adımda şu formatı kullan:

Thought: [Şu ana kadar ne öğrendim? Hangi bilgiler var?]
Reasoning: [Adım adım mantık yürütme: "Önce ...", "Sonra ...", "Bu bana şunu gösteriyor ..."]
Action: [SEARCH veya THINK veya ANSWER]
Action Input: [SEARCH için arama sorgusu, THINK için düşünce devamı, ANSWER için final cevap]

ÖNEMLİ KURALLAR:
1. DÜŞÜNME KURALI: THINK eylemi kullanırken "Reasoning:" bölümünde detaylı mantık yürüt.
2. KAYNAK BELİRTME: Cevap verirken kaynakları açıkça belirt: "ED-73F.pdf Sayfa 42'deki bilgi" şeklinde.
3. TÜRKÇE: Cevabını Türkçe ve net bir şekilde ver.
4. SPEKÜLASYON YAPMA: Sadece bulduğun belgelerden öğrendiklerini kullan.

ÖRNEK ÇALIŞMA:

--- ÖRNEK 1: ---
Soru: Mode S nedir?

Thought: Kullanıcı Mode S'in tanımını soruyor. Önce temel kavramları anlamam gerekiyor.
Reasoning: Önce Mode S'in ne olduğunu bulmalıyım. Bu bir radar protokolü olabilir. "selective addressing" özelliği ile ilgili olabilir.
Action: SEARCH
Action Input: Mode S nedir tanım selective addressing secondary surveillance radar

--- ÖRNEK 2: ---
[Arama sonuçlarını aldıktan sonra]
Thought: Mode S'in havacılıkta kullanılan bir ikincil gözetleme radarı protokolü olduğunu öğrendim.
Reasoning: Mode S, Mode A/C'den daha gelişmiş özelliklere sahip. Şimdi temel özelliklerini toparlayıp cevap verebilirim.
Action: ANSWER
Action Input: Mode S (Mode Select), havacılık için geliştirilmiş ikincil gözetleme radarı protokolüdür...

--- ÖRNEK 3: ---
Soru: Mode S UF=11 formatı nasıl çalışır?

Thought: UF=11'in spesifik bir format olduğunu biliyorum ama detaylarını bilmiyorum.
Reasoning: Önce Mode S mesaj formatının genel yapısını anlamalıyım, sonra UF=11'e özgü detaylara bakmalıyım.
Action: SEARCH
Action Input: Mode S message format structure UF field meaning

[Sonuç aldıktan sonra]
Thought: UF field'in Mode S mesajlarında formatı belirlediğini öğrendim.
Reasoning: Şimdi UF=11'e özgü bilgileri araştırmalıyım çünkü UF=11 "all-call" sorgulama formatı olabilir.
Action: SEARCH
Action Input: Mode S UF=11 all-call interrogation format

[Sonuç aldıktan sonra]
Thought: UF=11'in "all-call" sorgulama formatı olduğunu ve tüm uçaklara yayın yaptığını öğrendim.
Reasoning: Yeterli bilgi topladım. Şimdi bunları birleştirip net bir cevap oluşturabilirim.
Action: ANSWER
Action Input: Mode S UF=11 formatı "all-call" interrogation formatıdır...
"""

REACT_SYSTEM_PROMPT_VISUAL = """Sen bir PDF araştırma asistanısın. Soruları cevaplamak için adım adım düşün ve sistematik bir şekilde çalış.

DÜŞÜNME ŞEKLİN:
- Adım adım mantık yürüt, acele etme
- Mevcut bilgileri analiz et, eksikleri belirle
- Bir sonraki adımı planla
- Karmaşık soruları parçalara ayır
- Önce metin araması (SEARCH) ile başlamak genelde iyidir; tablo/diyagram/düzen için VISUAL_SEARCH ekle

KULLANILABILIR EYLEMLER:
1. SEARCH(query): Vektör veritabanından "query" ile metin/chunk araması yap
2. VISUAL_SEARCH(query): PDF sayfalarını görüntü tabanlı indekste (ColPali MaxSim) ara; tablolar, şemalar, diyagramlar ve düzen için uygundur
3. THINK(reasoning): Mevcut durumu değerlendir, derinlemesine düşün, sonraki adımı planla
4. ANSWER(text): Final cevabı ver ve araştırmayı bitir (görsel sayfalar toplandıysa sistem görselleri de kullanarak sonuçları birleştirir)

ÇALIŞMA FORMATI:
Her adımda şu formatı kullan:

Thought: [Şu ana kadar ne öğrendim? Hangi bilgiler var?]
Reasoning: [Adım adım mantık yürütme: "Önce ...", "Sonra ...", "Bu bana şunu gösteriyor ..."]
Action: [SEARCH veya VISUAL_SEARCH veya THINK veya ANSWER]
Action Input: [SEARCH veya VISUAL_SEARCH için arama sorgusu, THINK için düşünce devamı, ANSWER için final cevap]

ÖNEMLİ KURALLAR:
1. DÜŞÜNME KURALI: THINK eylemi kullanırken "Reasoning:" bölümünde detaylı mantık yürüt.
2. KAYNAK BELİRTME: Cevap verirken kaynakları açıkça belirt: "ED-73F.pdf Sayfa 42'deki bilgi" şeklinde.
3. TÜRKÇE: Cevabını Türkçe ve net bir şekilde ver.
4. SPEKÜLASYON YAPMA: Sadece bulduğun belgelerden öğrendiklerini kullan.
5. VISUAL_SEARCH'i gereksiz yere sık kullanma; metin araması yeterliyse SEARCH yeterlidir.

ÖRNEK:
Soru: Şekil 3'teki blok diyagramı açıkla.
Thought: Diyagram metin chunk'larında tam çıkmayabilir.
Reasoning: Önce konuyu metinle bulup sonra görsel sayfa araması yapmalıyım.
Action: SEARCH
Action Input: Şekil 3 blok diyagram sistem mimarisi
[Sonra]
Action: VISUAL_SEARCH
Action Input: figure 3 block diagram schematic
"""


def react_system_prompt(colpali_enabled: bool) -> str:
    return REACT_SYSTEM_PROMPT_VISUAL if colpali_enabled else REACT_SYSTEM_PROMPT


def _colpali_score_device(colpali_cfg: Dict[str, Any]) -> Optional[str]:
    retrieval_device = colpali_cfg.get("retrieval_device", "auto")
    if str(retrieval_device).startswith("cuda"):
        return retrieval_device
    if retrieval_device == "auto" and torch.cuda.is_available():
        return "cuda"
    return None


def build_colpali_agentic_state(
    config: Dict[str, Any],
    colpali_model,
    colpali_processor,
    page_embeddings,
    colpali_metadata,
) -> Dict[str, Any]:
    """agentic_rag için ColPali sözlüğü (tek yerden top_k / score_device)."""
    c = config.get("colpali", {})
    return {
        "model": colpali_model,
        "processor": colpali_processor,
        "page_embeddings": page_embeddings,
        "metadata": colpali_metadata,
        "top_k": c.get("top_k_pages", 4),
        "score_device": _colpali_score_device(c),
    }


def _finalize_with_vision_if_applicable(
    *,
    memory: AgentMemory,
    question: str,
    draft_answer: str,
    lm_client,
    config: Dict[str, Any],
    token_usage: Dict[str, int],
    colpali_state: Optional[Dict[str, Any]],
) -> str:
    """En az bir VISUAL_SEARCH sonucu ve geçerli görsel dosyası varsa tek vision çağrısı ile cevabı zenginleştir."""
    if not colpali_state or not memory.visual_searches:
        return draft_answer
    top_k = colpali_state.get("top_k", 4)
    pages = memory.deduped_visual_pages(max_pages=top_k)
    image_paths = [
        Path(p.image_path)
        for p in pages
        if p.image_path and Path(p.image_path).exists()
    ]
    if not image_paths:
        return draft_answer
    gen_cfg = config["generation"]
    all_chunks = memory.get_all_chunks()
    context = (
        format_chunks_for_context(all_chunks, max_chunks=30)
        if all_chunks
        else "(Metin aramasından chunk yok.)"
    )
    user_prompt = (
        "Aşağıdaki metin parçaları ve PDF sayfa görselleri birlikte değerlendir.\n\n"
        f"METİN BAĞLAMI:\n{context}\n\n"
        f"TASLAK CEVAP (metin tabanlı agent):\n{draft_answer}\n\n"
        f"SORU: {question}\n\n"
        "Tablolar ve diyagramları görsellere göre yorumla. Metin taslağıyla birleştirerek nihai cevabı Türkçe ver. "
        "Çelişki olursa ilgili sayfa görseline öncelik ver. Kaynakları belirt."
    )
    resp = lm_client.chat_completion_with_images(
        text_prompt=user_prompt,
        image_paths=image_paths,
        system_prompt=COLPALI_SYSTEM_PROMPT,
        temperature=gen_cfg.get("temperature", 0.2),
        max_tokens=gen_cfg.get("max_tokens", 4096),
    )
    token_usage["total_input_tokens"] += resp.get("input_tokens", 0)
    token_usage["total_output_tokens"] += resp.get("output_tokens", 0)
    token_usage["total_tokens"] += resp.get("total_tokens", 0)
    return resp["answer"]


def create_agent_prompt(
    question: str,
    memory: AgentMemory,
    iteration: int,
    colpali_enabled: bool = False,
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    """Agent için prompt oluştur"""
    
    prompt_parts = []
    
    if conversation_history:
        prompt_parts.append("=== ÖNCEKİ KONUŞMALAR ===")
        max_history = 5
        for i, entry in enumerate(conversation_history[-max_history:], 1):
            prompt_parts.append(f"\n[Geçmiş {i}]:")
            prompt_parts.append(f"Soru: {entry.get('question', '')[:200]}")
            answer = entry.get('answer', '')[:300]
            prompt_parts.append(f"Cevap: {answer}...")
            if entry.get('sources'):
                sources = entry.get('sources', [])
                source_names = [f"{s.get('name', '')} s.{s.get('page', '')}" for s in sources[:3]]
                prompt_parts.append(f"Kaynaklar: {', '.join(source_names)}")
        prompt_parts.append("\n" + "=" * 50 + "\n")
    
    prompt_parts.extend([
        f"SORU: {question}",
        "",
        f"=== İTERASYON {iteration} ===",
        ""
    ])
    
    if memory.searches:
        prompt_parts.append("ŞİMDİYE KADAR YAPTIĞIM METİN ARAMALARI (SEARCH):")
        for i, sr in enumerate(memory.searches, 1):
            prompt_parts.append(f"\nArama {i}: '{sr.query}'")
            prompt_parts.append(f"Sonuç: {len(sr.chunks)} chunk bulundu")
            
            for j, (meta, score) in enumerate(sr.chunks[:3], 1):
                preview = meta['text'][:150] + "..." if len(meta['text']) > 150 else meta['text']
                prompt_parts.append(
                    f"  [{j}] {meta['source']} (s.{meta['page']}, skor:{score:.3f})\n"
                    f"      {preview}"
                )
        prompt_parts.append("")
    else:
        prompt_parts.append("Henüz metin (SEARCH) araması yapmadım. Çoğu soruda önce SEARCH ile başlamalıyım.")
        prompt_parts.append("")

    if colpali_enabled and memory.visual_searches:
        prompt_parts.append("GÖRSEL ARAMALAR (VISUAL_SEARCH — sayfa adayları, MaxSim):")
        for i, vr in enumerate(memory.visual_searches, 1):
            prompt_parts.append(f"\nGörsel arama {i}: '{vr.query}' → {vr.get_summary()}")
            for j, p in enumerate(vr.pages[:5], 1):
                prompt_parts.append(
                    f"  [{j}] {p.source} — Sayfa {p.page} (MaxSim: {p.score:.3f})"
                )
        prompt_parts.append("")
    
    prompt_parts.extend([
        "ŞİMDİ NE YAPMALIYIM?",
        "Yukarıdaki formatı kullanarak düşün ve bir eylem belirle:",
        ""
    ])
    
    return "\n".join(prompt_parts)


def agentic_rag(
    question: str,
    embedding_model,
    index,
    metadata,
    lm_client,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[str, AgentAction], None]] = None,
    max_iterations: int = 5,
    colpali_state: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict]] = None,
) -> tuple[str, List[tuple], float, AgentMemory, Dict[str, int]]:
    """
    ReAct pattern ile agentic RAG

    colpali_state: ColPali eklentisi açıksa
        model, processor, page_embeddings, metadata (sayfa listesi),
        top_k, score_device anahtarları.
    """
    start_time = time.time()

    agentic_config = config.get("agentic_rag", {})
    gen_config = config["generation"]
    search_top_k = agentic_config.get("search_top_k", 15)
    agent_temp = agentic_config.get("agent_temperature", 0.3)
    min_searches = agentic_config.get("min_searches", 1)
    colpali_enabled = colpali_state is not None
    
    # Erken çıkış parametreleri
    enable_early_exit = agentic_config.get("enable_early_exit", True)
    early_exit_threshold = agentic_config.get("early_exit_threshold", 0.7)
    max_consecutive_think = agentic_config.get("max_consecutive_think", 2)

    memory = AgentMemory()
    final_answer = None
    token_usage = {"total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0}
    consecutive_think_count = 0  # Ardışık THINK aksiyonlarını say

    logger.debug(
        "Agentic RAG başlatılıyor: max_iterations=%d, colpali=%s",
        max_iterations,
        colpali_enabled,
    )

    system_prompt = react_system_prompt(colpali_enabled)

    for iteration in range(1, max_iterations + 1):
        memory.iteration_count = iteration
        logger.debug("Iterasyon %d/%d başlıyor", iteration, max_iterations)

        agent_prompt = create_agent_prompt(
            question, memory, iteration, colpali_enabled=colpali_enabled,
            conversation_history=conversation_history
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": agent_prompt},
        ]
        
        # LLM'den yanıt al (non-streaming agent decision için)
        try:
            response = lm_client.chat_completion(
                messages=messages,
                temperature=agent_temp,
                max_tokens=1024,  # Agent decisions için daha kısa
                stop=["---", "==="]  # Sadece bir action
            )
            
            # Token istatistiklerini topla
            token_usage["total_input_tokens"] += response.get("input_tokens", 0)
            token_usage["total_output_tokens"] += response.get("output_tokens", 0)
            token_usage["total_tokens"] += response.get("total_tokens", 0)
            
            agent_response = response["answer"]
            
        except requests.exceptions.ReadTimeout:
            console = Console()
            console.print(f"[yellow]⚠️  Timeout hatası (agent iterasyon {iteration}). Daha küçük max_tokens ile tekrar deneniyor...[/yellow]")

            try:
                response = lm_client.chat_completion(
                    messages=messages,
                    temperature=agent_temp,
                    max_tokens=512,
                    stop=["---", "==="]
                )

                token_usage["total_input_tokens"] += response.get("input_tokens", 0)
                token_usage["total_output_tokens"] += response.get("output_tokens", 0)
                token_usage["total_tokens"] += response.get("total_tokens", 0)

                agent_response = response["answer"]

            except requests.exceptions.ReadTimeout:
                console.print(f"[red]❌ Timeout hatası devam ediyor. Agentic moddan simple RAG moduna geçiliyor...[/red]")
                raise
        
        logger.debug("Iterasyon %d - Agent yanıtı:\n%s", iteration, agent_response)

        # Yanıtı parse et
        action = parse_agent_response(agent_response)
        logger.debug("Parsed action: %s", action.type)
        
        # Progress callback varsa çağır
        if progress_callback:
            try:
                progress_callback(agent_response, action)
            except Exception as callback_exc:
                logger.warning(
                    "Progress callback hatası (iterasyon %d): %s",
                    iteration,
                    callback_exc,
                )
        
        # Action'a göre işlem yap
        
        # Erken çıkış kontrolü: Eğer agent ANSWER vermeye karar verdiyse ve yeterli bilgi varsa
        if action.type == "ANSWER" and len(memory.searches) >= min_searches:
            # Agent cevap vermeye hazır, doğrudan çıkış yap
            final_answer = action.content
            logger.debug("Agent ANSWER verdi, erken çıkış yapılıyor (iterasyon %d)", iteration)
            break
            
        # THINK aksiyonlarını say
        if action.type == "THINK":
            consecutive_think_count += 1
            
            # Akıllı erken çıkış: Agent yeterli bilgiye ulaştığını belirtiyor mu?
            if enable_early_exit and len(memory.searches) >= min_searches and iteration < max_iterations:
                think_content = action.content.lower()
                if any(phrase in think_content for phrase in [
                    "yeterli bilgi", "şimdi cevap verebilirim", "artık cevap",
                    "bu bilgilerle", "bu kadar yeterli", "cevap verebilirim",
                    "şimdi cevap", "cevabı oluşturabilirim", "toplayabilirim"
                ]):
                    logger.debug("Agent THINK'de yeterli bilgi olduğunu belirtti, zorla ANSWER")
                    memory.add_thought("Yeterli bilgi topladım, şimdi ANSWER vermeliyim.")
                    continue
            
            # Erken çıkış: Eğer THINK sayısı >= iterasyon sayısı ise ve min_searches karşılandıysa
            # Bu, agent'ın neredeyse sadece düşündüğü ve yeterli arama yapmadığı anlamına gelir
            if enable_early_exit and consecutive_think_count >= iteration and len(memory.searches) >= min_searches:
                logger.debug("THINK sayısı (%d) >= iterasyon (%d) — Agent sadece düşünüyor, zorla ANSWER", consecutive_think_count, iteration)
                memory.add_thought(f"Çok fazla düşündüm (THINK {consecutive_think_count}, iterasyon {iteration}). Şimdi yeterli bilgi var, ANSWER vermeliyim.")
                continue
        else:
            consecutive_think_count = 0
            
        if action.type == "SEARCH":
            # Vektör DB'den ara
            if not action.query:
                # Query parse edilemedi, düşünceyi query olarak kullan
                action.query = action.content
            
            results = search_index(
                query=action.query,
                index=index,
                metadata=metadata,
                embedding_model=embedding_model,
                top_k=search_top_k
            )
            logger.debug("SEARCH sonucu: %d chunk bulundu", len(results))

            # Boş index/metadata: SEARCH kaçınılmaz olarak 0 döner.
            # Bu durumda büyük bir final LLM çağrısı yapmak hem yavaş hem de yanıltıcı.
            try:
                idx_total = int(getattr(index, "ntotal", 0))
            except Exception:
                idx_total = 0
            if (not metadata) and idx_total <= 0:
                final_answer = (
                    "Şu anda arama yapılacak bir index yok (belge/chunk bulunamadı). "
                    "Lütfen önce belge yükleyin ve index'in oluştuğundan emin olun, sonra tekrar sorun."
                )
                break
            
            # Sonucu hafızaya ekle
            search_result = SearchResult(query=action.query, chunks=results)
            memory.add_search(search_result)

        elif action.type == "VISUAL_SEARCH":
            if not colpali_state:
                memory.add_thought(
                    "VISUAL_SEARCH şu oturumda yok; metin SEARCH veya THINK kullan."
                )
                continue
            if not colpali_state.get("page_embeddings"):
                memory.add_thought(
                    "ColPali index boş! Görsel arama yapılamıyor. Lütfen metin SEARCH veya THINK kullan."
                )
                continue
            if not action.query:
                action.query = action.content
            logger.debug("VISUAL_SEARCH sorgusu: %s", action.query)
            q_emb = embed_query(
                action.query,
                colpali_state["model"],
                colpali_state["processor"],
            )
            results = search_colpali(
                query_embedding=q_emb,
                page_embeddings=colpali_state["page_embeddings"],
                metadata=colpali_state["metadata"],
                top_k=colpali_state["top_k"],
                score_device=colpali_state.get("score_device"),
                query_text=action.query,
            )
            memory.add_visual_search(
                VisualSearchResult(query=action.query, pages=results)
            )

        elif action.type == "ANSWER":
            if len(memory.searches) < min_searches:
                missing = min_searches - len(memory.searches)
                forced_thought = (
                    f"Henüz yeterli metin araması yapmadım (SEARCH). "
                    f"Cevap vermeden önce en az {missing} SEARCH daha yapmalıyım."
                )
                action = AgentAction(type="THINK", content=forced_thought)
                memory.add_thought(action.content)
            else:
                final_answer = action.content
                break

        elif action.type == "THINK":
            logger.debug("THINK action: %s...", action.content[:100])
            memory.add_thought(action.content)

        if iteration == max_iterations - 1:
            logger.debug("Son iterasyona yaklaşıldı (%d/%d)", iteration, max_iterations)
    
    # Eğer hiç cevap verilmediyse, zorla oluştur
    if not final_answer:
        all_chunks = memory.get_all_chunks()
        context = format_chunks_for_context(all_chunks, max_chunks=30)

        user_content_parts = []
        if conversation_history:
            user_content_parts.append("=== ÖNCEKİ KONUŞMALAR ===")
            for i, entry in enumerate(conversation_history[-3:], 1):
                user_content_parts.append(f"[Geçmiş {i}]:")
                user_content_parts.append(f"Soru: {entry.get('question', '')[:200]}")
                answer = entry.get('answer', '')[:300]
                user_content_parts.append(f"Cevap: {answer}...")
            user_content_parts.append("\n" + "=" * 50 + "\n")

        user_content_parts.append(f"BAĞLAM:\n{context}")
        user_content_parts.append(f"\nSORU: {question}")
        user_content_parts.append("\nÖNEMLİ: Maksimum iterasyona ulaştık. Şimdiye kadar topladığın bilgilerle en iyi cevabı ver.")

        messages = [
            {"role": "system", "content": gen_config["system_prompt"]},
            {"role": "user", "content": "\n".join(user_content_parts)}
        ]

        response = lm_client.chat_completion(
            messages=messages,
            temperature=gen_config.get("temperature", 0.2),
            max_tokens=gen_config.get("max_tokens", 8192)
        )

        token_usage["total_input_tokens"] += response.get("input_tokens", 0)
        token_usage["total_output_tokens"] += response.get("output_tokens", 0)
        token_usage["total_tokens"] += response.get("total_tokens", 0)

        final_answer = response["answer"]

    final_answer = _finalize_with_vision_if_applicable(
        memory=memory,
        question=question,
        draft_answer=final_answer,
        lm_client=lm_client,
        config=config,
        token_usage=token_usage,
        colpali_state=colpali_state,
    )

    all_sources = memory.get_all_chunks()
    generation_time = time.time() - start_time

    return final_answer, all_sources, generation_time, memory, token_usage
