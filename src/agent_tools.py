"""
ReAct Agent için temel araçlar ve veri yapıları.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class AgentAction:
    """Agent'ın alabileceği eylemler"""
    type: str  # "SEARCH", "VISUAL_SEARCH", "THINK", "ANSWER"
    content: str  # Eylem içeriği
    query: Optional[str] = None  # SEARCH / VISUAL_SEARCH için arama sorgusu
    
    def __str__(self):
        if self.type == "SEARCH":
            return f"🔍 SEARCH: {self.query}"
        if self.type == "VISUAL_SEARCH":
            return f"🖼️ VISUAL_SEARCH: {self.query}"
        elif self.type == "THINK":
            return f"💭 THINK: {self.content}"
        elif self.type == "ANSWER":
            return f"✅ ANSWER"
        return f"{self.type}: {self.content}"


@dataclass
class SearchResult:
    """Bir arama sonucu. chunks: (metadata_dict, similarity_score) çiftleri."""
    query: str
    chunks: List[tuple]  # List[tuple[Dict[str, Any], float]]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    
    def get_summary(self) -> str:
        """Sonuç özeti"""
        sources = set()
        for chunk, score in self.chunks:
            sources.add(f"{chunk['source']} (s.{chunk['page']})")
        return f"Bulundu: {len(self.chunks)} chunk, {len(sources)} kaynak"


@dataclass
class VisualSearchResult:
    """ColPali MaxSim ile yapılan bir görsel sayfa araması."""
    query: str
    pages: List[Any]  # List[PageResult]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    def get_summary(self) -> str:
        if not self.pages:
            return "İlgili sayfa bulunamadı"
        return f"{len(self.pages)} sayfa (MaxSim)"


class AgentMemory:
    """Agent'ın öğrenme hafızası"""
    
    def __init__(self):
        self.thoughts: List[str] = []
        self.searches: List[SearchResult] = []
        self.visual_searches: List[VisualSearchResult] = []
        self.iteration_count: int = 0
        
    def add_thought(self, thought: str):
        """Düşünce ekle"""
        self.thoughts.append(thought)
        
    def add_search(self, search_result: SearchResult):
        """Arama sonucu ekle"""
        self.searches.append(search_result)

    def add_visual_search(self, visual_result: VisualSearchResult):
        """Görsel (ColPali) arama sonucu ekle"""
        self.visual_searches.append(visual_result)

    def deduped_visual_pages(self, max_pages: int = 16) -> List[Any]:
        """Tüm görsel aramalardan (source, page) tekil; en yüksek skoru tut, skora göre sırala."""
        from colpali_retrieval import PageResult as PR

        best: Dict[tuple, Any] = {}
        for vs in self.visual_searches:
            for p in vs.pages:
                if not isinstance(p, PR):
                    continue
                key = (p.source, p.page)
                prev = best.get(key)
                if prev is None or p.score > prev.score:
                    best[key] = p
        ordered = sorted(best.values(), key=lambda x: x.score, reverse=True)
        return ordered[:max_pages]
        
    def get_all_chunks(self) -> List[tuple]:
        """Tüm bulunan chunk'ları döndür"""
        all_chunks = []
        for search in self.searches:
            all_chunks.extend(search.chunks)
        return all_chunks
    
    def get_context_summary(self) -> str:
        """Şimdiye kadar öğrenilenlerin özeti"""
        summary_parts = []
        
        for i, search in enumerate(self.searches, 1):
            summary_parts.append(
                f"Arama {i}: '{search.query}'\n"
                f"  → {search.get_summary()}"
            )
            
        if not summary_parts:
            return "Henüz arama yapılmadı."
            
        return "\n".join(summary_parts)
    
    def get_detailed_context(self) -> str:
        """Tüm bulunan bilgilerin detaylı özeti"""
        if not self.searches:
            return "Henüz bilgi toplanmadı."
        
        context_parts = []
        chunk_id = 1
        
        for search in self.searches:
            context_parts.append(f"\n## Arama: '{search.query}'")
            for meta, score in search.chunks[:5]:  # Her aramadan en iyi 5'i
                context_parts.append(
                    f"\n[{chunk_id}] {meta['source']} (Sayfa {meta['page']}, Skor: {score:.3f})\n"
                    f"{meta['text'][:300]}..."
                )
                chunk_id += 1
        
        return "\n".join(context_parts)


def parse_agent_response(response: str) -> Optional[AgentAction]:
    """
    LLM'den gelen yanıtı parse et ve AgentAction'a çevir.

    Beklenen format (her alan tek veya çok satırlı olabilir):
        Thought: [şu ana kadar ne öğrendim]
        Reasoning: [adım adım mantık yürütme]
        Action: [SEARCH/THINK/ANSWER]
        Action Input: [parametre — birden fazla satır olabilir]

    Satır satır okuma yerine state-machine yaklaşımı kullanılır:
    tanınan bir prefix gelene kadar devam eden satırlar mevcut alana eklenir.
    Bu sayede uzun çok satırlı ANSWER veya SEARCH sorguları kaybolmaz.
    """
    lines = response.strip().split('\n')

    thought_lines: List[str] = []
    reasoning_lines: List[str] = []
    action_type: Optional[str] = None
    action_input_lines: List[str] = []
    current_field: Optional[str] = None

    for raw_line in lines:
        stripped = raw_line.strip()

        if stripped.startswith("Thought:"):
            current_field = "thought"
            first = stripped[len("Thought:"):].strip()
            if first:
                thought_lines.append(first)
        elif stripped.startswith("Reasoning:"):
            current_field = "reasoning"
            first = stripped[len("Reasoning:"):].strip()
            if first:
                reasoning_lines.append(first)
        elif stripped.startswith("Action:"):
            current_field = "action"
            raw_action = stripped[len("Action:"):].strip().upper()
            action_type = raw_action.replace(" ", "_")
        elif stripped.startswith("Action Input:"):
            current_field = "action_input"
            first = stripped[len("Action Input:"):].strip()
            if first:
                action_input_lines.append(first)
        elif current_field == "thought":
            thought_lines.append(stripped)
        elif current_field == "reasoning":
            reasoning_lines.append(stripped)
        elif current_field == "action_input":
            action_input_lines.append(stripped)

    thought = "\n".join(thought_lines).strip() or None
    reasoning = "\n".join(reasoning_lines).strip() or None
    action_input = "\n".join(action_input_lines).strip() or None

    if not action_type:
        return AgentAction(type="THINK", content=response)

    full_content = ""
    if thought:
        full_content += f"Thought: {thought}\n"
    if reasoning:
        full_content += f"Reasoning: {reasoning}"

    if not full_content:
        full_content = thought or reasoning or response

    if action_type == "SEARCH":
        return AgentAction(type="SEARCH", content=full_content, query=action_input)
    if action_type == "VISUAL_SEARCH":
        return AgentAction(type="VISUAL_SEARCH", content=full_content, query=action_input)
    elif action_type == "ANSWER":
        return AgentAction(type="ANSWER", content=action_input or full_content or response)
    else:  # THINK veya diğer
        return AgentAction(type="THINK", content=full_content or action_input or response)


def format_chunks_for_context(chunks: List[tuple], max_chunks: int = 20) -> str:
    """Chunk'ları LLM için formatlı string'e çevir - kaynak bilgisiyle"""
    context_parts = []
    
    for i, (meta, score) in enumerate(chunks[:max_chunks], 1):
        # Kaynak bilgisini doğrudan etikette göster
        source_label = f"{meta['source']} - Sayfa {meta['page']}"
        context_parts.append(
            f"[{source_label}]\n"
            f"{meta['text']}\n"
        )
    
    return "\n".join(context_parts)