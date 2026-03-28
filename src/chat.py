"""
Sohbet arayüzü: FAISS metin araması + ColPali görsel arama + ReAct agentic iterasyonları.
"""

import sys
import logging
from datetime import datetime
import json5
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from embedding import EmbeddingModel
from hf_load_hacks import resolve_hf_local_files_only
from lm_studio_client import LMStudioClient
from indexer import (
    build_index,
    save_index,
    load_index,
    update_index_incremental,
    sync_full_directory_manifest,
)
from odt_exporter import ODTExporter
from rag_agentic import agentic_rag, build_colpali_agentic_state
from rag_simple import simple_rag
from agent_tools import AgentAction


console = Console()

# Proje kökü (src/chat.py -> src -> üst). Config, index ve belge yolları buna göre çözülür.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_settings() -> Dict[str, Any]:
    """Ayarları yükle."""
    config_path = PROJECT_ROOT / "config" / "settings.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config bulunamadı: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json5.load(f)


def initialize_system(config: Dict[str, Any]) -> tuple:
    """Sistemi başlat - embedding model ve index yükle/oluştur"""
    
    # Embedding modeli (HF hub / ~/.cache/huggingface; çevrimdışı: local_files_only veya HF_HUB_OFFLINE=1)
    emb_config = config["embedding"]
    emb_local = resolve_hf_local_files_only(emb_config.get("local_files_only"))
    embedding_model = EmbeddingModel(
        model_name=emb_config["model_name"],
        device=emb_config["device"],
        prefer_gpu_type=emb_config.get("prefer_gpu_type"),
        gpu_index=emb_config.get("gpu_index"),
        local_files_only=emb_local,
    )
    
    # Index yolu (proje köküne göre)
    index_path = PROJECT_ROOT / config["index"]["output_path"]
    index_file = Path(str(index_path) + ".index")
    document_dir = PROJECT_ROOT / "pdfs"

    # Index var mı kontrol et
    if index_file.exists():
        console.print("[cyan]📂 Mevcut index yükleniyor...[/cyan]")
        index, metadata = load_index(index_path)
        # Yeni eklenen belgeleri tespit edip index'e ekle
        if document_dir.exists():
            index, metadata, added = update_index_incremental(
                document_dir=document_dir,
                embedding_model=embedding_model,
                index=index,
                metadata=metadata,
                chunk_size=emb_config["chunk_size"],
                chunk_overlap=emb_config["chunk_overlap"],
                batch_size=emb_config["batch_size"],
                index_output_path=index_path,
            )
            if added > 0:
                save_index(index, metadata, index_path)
                console.print(f"[green]✅ Index güncellendi: {added} yeni chunk eklendi.[/green]\n")
    else:
        console.print("[yellow]⚠️  Index bulunamadı, yeni index oluşturuluyor...[/yellow]\n")
        
        if not document_dir.exists():
            console.print(f"[red]❌ Belge klasörü bulunamadı: {document_dir}[/red]")
            sys.exit(1)
        
        index, metadata = build_index(
            document_dir=document_dir,
            embedding_model=embedding_model,
            chunk_size=emb_config["chunk_size"],
            chunk_overlap=emb_config["chunk_overlap"],
            batch_size=emb_config["batch_size"]
        )
        
        # Index'i kaydet
        save_index(index, metadata, index_path)
        if document_dir.exists():
            sync_full_directory_manifest(document_dir, index_path)
        console.print()

    # Chat device ayarına göre model'i taşı
    chat_device = emb_config.get("chat_device", "auto")
    if chat_device != "auto":
        console.print(f"[cyan]🔄 Model chat için '{chat_device}' cihazına taşınıyor...[/cyan]")
        if chat_device == "cpu":
            embedding_model.to_cpu()
        else:
            embedding_model.to_device(chat_device)
    else:
        console.print("[cyan]ℹ️  Model indeksleme cihazında kalıyor (chat_device: auto)[/cyan]\n")
    
    # LM Studio client
    lm_config = config["lm_studio"]
    lm_client = LMStudioClient(
        base_url=lm_config["base_url"],
        timeout=lm_config["timeout"]
    )
    
    # LM Studio kontrolü
    if not lm_client.is_server_running():
        console.print("[red]❌ LM Studio server'a bağlanılamadı![/red]")
        console.print(f"[yellow]Lütfen LM Studio'yu başlatın: {lm_config['base_url']}[/yellow]")
        sys.exit(1)
    
    console.print(f"[green]✅ LM Studio bağlantısı başarılı: {lm_config['base_url']}[/green]\n")
    
    return embedding_model, index, metadata, lm_client, config


def _load_colpali_model_and_index(config: Dict[str, Any]) -> tuple:
    """ColPali model + sayfa index'i (LM Studio olmadan; metin pipeline eklentisi için)."""
    from colpali_retrieval import load_colpali_model
    from colpali_indexer import build_colpali_index, load_colpali_index

    colpali_cfg = config.get("colpali", {})
    hf_model = colpali_cfg.get("model_name", "vidore/colqwen2.5-v0.2")
    index_dir = PROJECT_ROOT / colpali_cfg.get("index_path", "indexes/colpali")
    page_images_dir = PROJECT_ROOT / colpali_cfg.get("page_images_dir", "page_images")
    dpi = colpali_cfg.get("render_dpi", 150)
    render_thread_count = colpali_cfg.get("render_thread_count", 4)
    batch_size = colpali_cfg.get("batch_size", 2)
    index_device = colpali_cfg.get("index_device", "cuda")
    retrieval_device = colpali_cfg.get("retrieval_device", "auto")
    quantization = colpali_cfg.get("quantization", None)
    mmap_embeddings = colpali_cfg.get("mmap_embeddings", True)
    colpali_local = resolve_hf_local_files_only(colpali_cfg.get("local_files_only"))

    meta_file = index_dir / "pages_meta.json"
    needs_build = False

    if meta_file.exists():
        console.print("[cyan]📂 Mevcut ColPali index yükleniyor...[/cyan]")
        page_embeddings, metadata = load_colpali_index(
            index_dir,
            mmap_embeddings=mmap_embeddings,
            page_images_root=page_images_dir,
        )
        missing = len(metadata) - len(page_embeddings)
        if missing > 0:
            console.print(f"[yellow]⚠️  {missing} sayfa eksik embedding — tamamlanacak[/yellow]\n")
            needs_build = True
    else:
        console.print("[yellow]⚠️  ColPali index bulunamadı, oluşturuluyor...[/yellow]\n")
        needs_build = True

    if needs_build:
        document_dir = PROJECT_ROOT / "pdfs"
        if not document_dir.exists():
            console.print(f"[red]❌ Belge klasörü bulunamadı: {document_dir}[/red]")
            sys.exit(1)

        page_embeddings, metadata = build_colpali_index(
            pdf_dir=document_dir,
            output_dir=index_dir,
            model_name=hf_model,
            page_images_dir=page_images_dir,
            dpi=dpi,
            batch_size=batch_size,
            render_thread_count=render_thread_count,
            index_device=index_device,
            balanced_gpu_memory_fraction=colpali_cfg.get("balanced_gpu_memory_fraction"),
            hub_local_files_only=colpali_local,
        )

    colpali_model, colpali_processor = load_colpali_model(
        hf_model,
        retrieval_device=retrieval_device,
        quantization=quantization,
        local_files_only=colpali_local,
        processor_cache_dir=str(index_dir / "processor"),
    )

    return colpali_model, colpali_processor, page_embeddings, metadata


def display_sources(results: List[tuple[dict, float]]):
    """Kaynakları tablo olarak göster"""
    if not results:
        return
        
    table = Table(title="📚 Kullanılan Kaynaklar", box=box.ROUNDED, show_header=True)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Belge", style="yellow")
    table.add_column("Sayfa", style="green", width=8)
    table.add_column("Benzerlik", style="magenta", width=10)
    table.add_column("İçerik Önizleme", style="dim")
    
    # Unique sources (agentic modda tekrar eden source'lar olabilir)
    seen = set()
    unique_results = []
    for meta, score in results:
        key = (meta['source'], meta['page'])
        if key not in seen:
            seen.add(key)
            unique_results.append((meta, score))
    
    for i, (meta, score) in enumerate(unique_results[:20], 1):
        preview = meta["text"][:100] + "..." if len(meta["text"]) > 100 else meta["text"]
        table.add_row(
            str(i),
            meta["source"],
            str(meta["page"]),
            f"{score:.3f}",
            preview
        )
    
    console.print(table)
    console.print()


def _display_and_record(
    question: str,
    answer_text: str,
    all_sources: List,
    generation_time: float,
    token_usage: Dict[str, Any],
    mode: str,
    conversation_history: List[Dict],
    memory=None,
):
    """Cevabı ekranda göster ve konuşma geçmişine ekle."""
    console.print("[bold cyan]💬 Final Cevap:[/bold cyan]")
    console.print("─" * console.width)
    console.print(answer_text, style="cyan")
    console.print("─" * console.width)
    console.print()

    display_sources(all_sources)

    if memory and getattr(memory, "visual_searches", None):
        vpages = memory.deduped_visual_pages(max_pages=12)
        if vpages:
            vtable = Table(
                title="Görsel aramadan gelen sayfalar (MaxSim)",
                box=box.ROUNDED,
                show_header=True,
            )
            vtable.add_column("#", style="cyan", width=3)
            vtable.add_column("Belge", style="yellow")
            vtable.add_column("Sayfa", style="green", width=8)
            vtable.add_column("Skor", style="magenta", width=10)
            for i, p in enumerate(vpages, 1):
                vtable.add_row(str(i), p.source, str(p.page), f"{p.score:.4f}")
            console.print(vtable)
            console.print()

    stats_parts = [f"⏱️  Süre: {generation_time:.2f}s"]
    if memory:
        stats_parts.append(f"🔄 İterasyon: {memory.iteration_count}")
        stats_parts.append(f"🔍 Arama: {len(memory.searches)}")
        if getattr(memory, "visual_searches", None):
            stats_parts.append(f"🖼️ Görsel arama: {len(memory.visual_searches)}")
    stats_parts.append(f"📚 Kaynak: {len(all_sources)} chunk")

    in_key = "total_input_tokens" if "total_input_tokens" in token_usage else "input_tokens"
    out_key = "total_output_tokens" if "total_output_tokens" in token_usage else "output_tokens"
    stats_parts.append(
        f"🧮 Token: {token_usage.get(in_key, 0):,}/{token_usage.get(out_key, 0):,} (giriş/çıkış) | "
        f"Σ: {token_usage.get('total_tokens', 0):,}"
    )
    console.print(f"[dim]{' | '.join(stats_parts)}[/dim]\n")

    sources_for_export = [
        {"name": meta["source"], "page": meta["page"], "score": score}
        for meta, score in all_sources[:20]
    ]
    entry: Dict[str, Any] = {
        "question": question,
        "answer": answer_text,
        "sources": sources_for_export,
        "mode": mode,
        "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "generation_time": generation_time,
        "token_usage": token_usage,
    }
    if memory:
        entry["agentic_metrics"] = {
            "iteration_count": memory.iteration_count,
            "search_count": len(memory.searches),
            "thought_count": len(memory.thoughts) if hasattr(memory, "thoughts") else 0,
            "visual_search_count": len(getattr(memory, "visual_searches", []) or []),
        }
    conversation_history.append(entry)


def chat_loop(
    embedding_model,
    index,
    metadata,
    lm_client,
    config,
    colpali_stack: Optional[Tuple[Any, Any, Any, List]] = None,
):
    """ReAct agentic döngüsü. colpali_stack: (model, processor, page_embeddings, pages_meta) veya yükleme başarısızsa None."""

    exporter = ODTExporter(export_dir="exports")
    conversation_history = []

    try:
        model_info = lm_client.get_model_info()
        model_name = model_info.get("model_name", "Bilinmeyen Model")
        console.print(f"[dim]📊 Aktif Model: {model_name}[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠️  Model bilgisi alınamadı: {e}[/yellow]")
        model_info = None

    colpali_note = ""
    if colpali_stack is not None:
        _pe = colpali_stack[2]
        colpali_note = f"\n[dim]ColPali: {len(_pe)} sayfa — agent VISUAL_SEARCH + final vision[/dim]"

    console.print(Panel.fit(
        "[bold cyan]🤖 Agentic RAG[/bold cyan]\n\n"
        "Metin indeksi (SEARCH) + görsel indeks (VISUAL_SEARCH) + iterasyonlu agent.\n"
        f"[dim]LLM: {model_info.get('model_name', 'Bilinmeyen') if model_info else 'Bilinmeyen'}[/dim]"
        f"{colpali_note}\n"
        "[dim]Komutlar: exit/q, save, export[/dim]",
        border_style="cyan",
    ))
    console.print()

    while True:
        try:
            question = console.input("[bold green]❓ Soru:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]👋 Görüşmek üzere![/yellow]")
            break

        if not question:
            continue

        if question.lower() == "q":
            console.print("[yellow]👋 Görüşmek üzere![/yellow]")
            break

        # Özel komutlar
        if question.lower() in ["save", "kaydet"]:
            if conversation_history:
                try:
                    # Son QA'nın performans metriklerini al
                    last_qa = conversation_history[-1]
                    performance_metrics = {
                        "mode": last_qa.get("mode", "agentic"),
                        "generation_time": last_qa.get("generation_time", 0),
                        "token_usage": last_qa.get("token_usage", {})
                    }
                    
                    filepath = exporter.export_last_qa(
                        question=last_qa["question"],
                        answer=last_qa["answer"],
                        sources=last_qa["sources"],
                        model_info=model_info,
                        performance_metrics=performance_metrics,
                        mode=last_qa.get("mode", "agentic")
                    )
                    console.print(f"[green]✅ Son soru-cevap kaydedildi: {filepath}[/green]\n")
                except Exception as e:
                    console.print(f"[red]❌ Kayıt hatası: {e}[/red]\n")
            else:
                console.print("[yellow]⚠️  Kaydedilecek konuşma yok![/yellow]\n")
            continue
        
        if question.lower() in ["export", "dışa aktar", "exportall"]:
            if conversation_history:
                try:
                    # Tüm konuşma için performans metriklerini topla
                    filepath = exporter.export_conversation(
                        conversation_history,
                        model_info=model_info,
                        performance_metrics={
                            "mode": "agentic",
                            "total_qa_count": len(conversation_history),
                            "average_generation_time": sum(qa.get("generation_time", 0) for qa in conversation_history) / len(conversation_history) if conversation_history else 0
                        }
                    )
                    console.print(f"[green]✅ Tüm konuşma kaydedildi ({len(conversation_history)} soru-cevap): {filepath}[/green]\n")
                except Exception as e:
                    console.print(f"[red]❌ Kayıt hatası: {e}[/red]\n")
            else:
                console.print("[yellow]⚠️  Kaydedilecek konuşma yok![/yellow]\n")
            continue

        try:
            console.print("[cyan]🧠 ReAct agent (metin + isteğe bağlı görsel arama)[/cyan]\n")

            iteration_count = 0

            def progress_callback(raw_response: str, action: AgentAction):
                nonlocal iteration_count
                iteration_count += 1

                console.print(f"[bold yellow]═══ İterasyon {iteration_count} ═══[/bold yellow]")
                console.print(f"{action}")

                if action.type == "SEARCH":
                    console.print(f"[dim]Arama sorgusu: {action.query}[/dim]")
                elif action.type == "VISUAL_SEARCH":
                    console.print(f"[dim]Görsel arama sorgusu: {action.query}[/dim]")
                elif action.type == "THINK":
                    console.print(f"[dim]💭 {action.content[:100]}...[/dim]")

                console.print()

            colpali_state = None
            if colpali_stack is not None:
                colpali_state = build_colpali_agentic_state(
                    config,
                    colpali_stack[0],
                    colpali_stack[1],
                    colpali_stack[2],
                    colpali_stack[3],
                )

            answer_text, all_sources, generation_time, memory, token_usage = agentic_rag(
                question=question,
                embedding_model=embedding_model,
                index=index,
                metadata=metadata,
                lm_client=lm_client,
                config=config,
                progress_callback=progress_callback,
                max_iterations=config.get("agentic_rag", {}).get("max_iterations", 5),
                colpali_state=colpali_state,
                conversation_history=conversation_history,
            )

            _display_and_record(
                question, answer_text, all_sources, generation_time,
                token_usage, "agentic", conversation_history, memory=memory,
            )

        except requests.exceptions.ReadTimeout:
            console.print(f"[yellow]⚠️  LM Studio sunucusu timeout oldu ({config['lm_studio']['timeout']}s). Simple RAG moduna geçiliyor...[/yellow]")
            
            try:
                logging.debug("Simple RAG moduna geçiliyor...")
                answer_text, all_sources, generation_time, token_usage = simple_rag(
                    question=question,
                    embedding_model=embedding_model,
                    index=index,
                    metadata=metadata,
                    lm_client=lm_client,
                    config=config
                )
                
                _display_and_record(
                    question, answer_text, all_sources, generation_time,
                    token_usage, "simple_rag_fallback", conversation_history,
                )
                
            except Exception:
                console.print("[red]❌ Simple RAG da timeout oldu. Lütfen LM Studio'yu kontrol edin.[/red]")
                console.print(f"[yellow]Timeout süresi: {config['lm_studio']['timeout']} saniye[/yellow]")
                console.print("[yellow]Öneriler:[/yellow]")
                console.print("[yellow]  1. LM Studio sunucusunun çalıştığından emin olun[/yellow]")
                console.print("[yellow]  2. config/settings.json'daki 'timeout' değerini artırın[/yellow]")
                console.print("[yellow]  3. Modelin yeterli RAM/VRAM'e sahip olduğundan emin olun[/yellow]\n")
                continue
                
        except Exception as e:
            console.print(f"[red]❌ Hata: {e}[/red]\n")
            import traceback
            traceback.print_exc()
            continue


def main():
    """Ana fonksiyon"""
    try:
        # Ayarları yükle
        config = load_settings()

        # features.debug_mode = true ise DEBUG, aksi hâlde WARNING seviyesi
        debug_mode = config.get("features", {}).get("debug_mode", False)
        logging.basicConfig(
            level=logging.DEBUG if debug_mode else logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        console.print("[bold cyan]🚀 Agentic RAG başlatılıyor (metin + ColPali)...[/bold cyan]\n")
        embedding_model, index, metadata, lm_client, config = initialize_system(config)

        colpali_stack = None
        console.print("[cyan]ColPali (görsel indeks) yükleniyor...[/cyan]")
        try:
            cm, cp, pe, pm = _load_colpali_model_and_index(config)
            colpali_stack = (cm, cp, pe, pm)
            console.print(f"[green]✅ ColPali hazır ({len(pe)} sayfa)[/green]\n")
        except Exception as e:
            console.print(f"[red]❌ ColPali yüklenemedi: {e}[/red]")
            console.print(
                "[yellow]Yalnızca metin araması (SEARCH) kullanılacak; VISUAL_SEARCH devre dışı.[/yellow]\n"
            )
            logging.exception("ColPali yüklemesi başarısız")

        chat_loop(
            embedding_model,
            index,
            metadata,
            lm_client,
            config,
            colpali_stack=colpali_stack,
        )
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Görüşmek üzere![/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()