"""
Index oluşturma ve kaydetme modülü.
Belgeleri yükler, embedding'e çevirir ve FAISS index'i oluşturur.
"""

import json
from pathlib import Path
from typing import List, Optional, Set
import numpy as np
import faiss

from document_loader import DocumentChunk, load_documents
from embedding import EmbeddingModel


def _iter_supported_documents(document_dir: Path) -> List[Path]:
    return sorted([
        *document_dir.rglob("*.pdf"),
        *document_dir.rglob("*.odt"),
        *document_dir.rglob("*.docx"),
    ])


def _document_rel_key(document_dir: Path, document_path: Path) -> str:
    """document_dir altındaki göreli posix yolu (alt klasör + çakışan isimler için)."""
    try:
        return document_path.resolve().relative_to(document_dir.resolve()).as_posix()
    except ValueError:
        return document_path.name


def load_document_manifest(index_output_path: Path) -> Set[str]:
    """İşlenmiş belge anahtarları (chunk üretilmese bile taranmış sayılır)."""
    path = Path(str(index_output_path) + ".manifest.json")
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = data.get("document_keys") or data.get("pdf_keys") or data.get("indexed_pdf_keys") or []
    if not isinstance(keys, list):
        return set()
    return set(keys)


def save_document_manifest(index_output_path: Path, keys: Set[str]) -> None:
    path = Path(str(index_output_path) + ".manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"document_keys": sorted(keys)}, f, ensure_ascii=False, indent=2)
    print(f"💾 Belge manifest kaydedildi: {path}")


def sync_full_directory_manifest(document_dir: Path, index_output_path: Path) -> None:
    """Tam index oluşturma sonrası: klasördeki tüm belgeleri manifest'e yaz."""
    keys = {_document_rel_key(document_dir, p) for p in _iter_supported_documents(document_dir)}
    save_document_manifest(index_output_path, keys)


def build_index(
    document_dir: Path,
    embedding_model: EmbeddingModel,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 32
) -> tuple[faiss.Index, List[dict]]:
    """
    Belgelerden FAISS index'i oluştur.
    
    Args:
        document_dir: Belge klasörü
        embedding_model: Embedding modeli
        chunk_size: Chunk boyutu
        chunk_overlap: Chunk overlap
        batch_size: Embedding batch size
        
    Returns:
        (faiss_index, metadata_list)
    """
    # Belgeleri yükle ve chunk'la
    print("=" * 60)
    print("📚 BELGE YÜKLEME")
    print("=" * 60)
    chunks = load_documents(document_dir, chunk_size, chunk_overlap)
    
    # Embedding oluştur
    print("=" * 60)
    print("🔢 EMBEDDING OLUŞTURMA")
    print("=" * 60)
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_model.encode(texts, batch_size=batch_size)
    
    # FAISS index oluştur
    print("\n" + "=" * 60)
    print("🔍 FAISS INDEX OLUŞTURMA")
    print("=" * 60)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity için normalize gerekli)
    index.add(embeddings.astype('float32'))
    
    print(f"✅ Index oluşturuldu: {index.ntotal} vektör, {dimension} boyut\n")
    
    # Metadata oluştur
    metadata = []
    for chunk in chunks:
        metadata.append({
            "text": chunk.text,
            "source": chunk.source,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id
        })
    
    return index, metadata


def get_indexed_sources(metadata: List[dict]) -> set:
    """Metadata chunk'larındaki source (dosya adı) değerleri."""
    if not isinstance(metadata, list):
        return set()
    out = set()
    for m in metadata:
        if isinstance(m, dict):
            s = m.get("source")
            if s:
                out.add(s)
    return out


def update_index_incremental(
    document_dir: Path,
    embedding_model: EmbeddingModel,
    index: faiss.Index,
    metadata: List[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 32,
    index_output_path: Optional[Path] = None,
) -> tuple[faiss.Index, List[dict], int]:
    """
    Yeni eklenen belgeleri tespit edip mevcut index'e ekler.
    Tanınan belgeler: metadata'daki source'lar + .manifest.json (çıkarılabilir metin
    olmayan dosyalar için chunk oluşmasa bile tekrar taranmaz).

    Returns:
        (güncel_index, güncel_metadata, eklenen_chunk_sayısı)
    """
    all_documents = _iter_supported_documents(document_dir)
    if not all_documents:
        return index, metadata, 0

    indexed_sources = get_indexed_sources(metadata)
    manifest: Set[str] = set()
    if index_output_path is not None:
        manifest = load_document_manifest(index_output_path)

    # Manifest: metni çıkmayan belgeler bir kez taranıp işaretlenir.
    # indexed_sources: chunk'ı olan dosya adları (pdf_loader source = basename).
    new_documents = []
    for p in all_documents:
        rk = _document_rel_key(document_dir, p)
        if rk in manifest:
            continue
        if p.name in indexed_sources:
            continue
        new_documents.append(p)

    if not new_documents:
        return index, metadata, 0

    print("=" * 60)
    print(f"📄 YENİ BELGELER TESPİT EDİLDİ ({len(new_documents)} dosya)")
    print("=" * 60)
    for p in new_documents:
        print(f"   + {p.name}")

    keys_done = {_document_rel_key(document_dir, p) for p in new_documents}

    chunks = load_documents(
        document_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        start_chunk_id=len(metadata),
        document_files=new_documents,
    )

    if index_output_path is not None:
        save_document_manifest(index_output_path, manifest | keys_done)

    if not chunks:
        print(
            "[!] Bu belgelerde çıkarılabilir metin yok (taranmış/OCR'siz olabilir). "
            "Manifest güncellendi; bir sonraki çalıştırmada tekrar işlenmeyecek.\n"
        )
        return index, metadata, 0

    texts = [c.text for c in chunks]
    embeddings = embedding_model.encode(texts, batch_size=batch_size)

    index.add(embeddings.astype("float32"))
    for c in chunks:
        metadata.append({
            "text": c.text,
            "source": c.source,
            "page": c.page,
            "chunk_id": c.chunk_id,
        })

    print(f"✅ {len(chunks)} yeni chunk index'e eklendi (toplam: {index.ntotal})\n")
    return index, metadata, len(chunks)


def save_index(index: faiss.Index, metadata: List[dict], output_path: Path):
    """
    Index ve metadata'yı kaydet.
    
    Args:
        index: FAISS index
        metadata: Metadata listesi
        output_path: Çıktı yolu (uzantı olmadan)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # FAISS index'i kaydet
    index_file = str(output_path) + ".index"
    faiss.write_index(index, index_file)
    print(f"💾 Index kaydedildi: {index_file}")
    
    # Metadata'yı kaydet
    metadata_file = str(output_path) + ".json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Metadata kaydedildi: {metadata_file}")


def load_index(index_path: Path) -> tuple[faiss.Index, List[dict]]:
    """
    Index ve metadata'yı yükle.
    
    Args:
        index_path: Index yolu (uzantı olmadan)
        
    Returns:
        (faiss_index, metadata_list)
    """
    index_file = str(index_path) + ".index"
    metadata_file = str(index_path) + ".json"
    
    # Dosya varlığını kontrol et
    if not Path(index_file).exists():
        raise FileNotFoundError(f"Index dosyası bulunamadı: {index_file}")
    if not Path(metadata_file).exists():
        raise FileNotFoundError(f"Metadata dosyası bulunamadı: {metadata_file}")
    
    # FAISS index'i yükle
    index = faiss.read_index(index_file)
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, list):
        raise ValueError(
            f"Metadata dosyası liste formatında olmalı: {metadata_file}"
        )

    print(f"✅ Index yüklendi: {index.ntotal} vektör")
    print(f"✅ Metadata yüklendi: {len(metadata)} chunk\n")
    
    return index, metadata


def merge_indexes(
    indexes: List[faiss.Index],
    metadatas: List[List[dict]],
) -> tuple[faiss.Index, List[dict]]:
    """
    Birden fazla FAISS IndexFlatIP'i tek bir index'e birleştir.
    Her index'in boyutu aynı olmalı.
    """
    if not indexes:
        raise ValueError("Birleştirilecek en az bir index gerekli")

    dim = indexes[0].d
    combined = faiss.IndexFlatIP(dim)
    combined_meta: List[dict] = []
    offset = 0
    for idx, meta in zip(indexes, metadatas):
        vectors = faiss.rev_swig_ptr(idx.get_xb(), idx.ntotal * dim).reshape(idx.ntotal, dim).copy()
        combined.add(vectors.astype("float32"))
        for m in meta:
            entry = dict(m)
            entry["chunk_id"] = offset
            combined_meta.append(entry)
            offset += 1
    return combined, combined_meta


def search(
    query: str,
    index: faiss.Index,
    metadata: List[dict],
    embedding_model: EmbeddingModel,
    top_k: int = 5
) -> List[tuple[dict, float]]:
    """
    Query için en benzer chunk'ları bul (vektör araması + dosya adı eşleştirme).
    
    Args:
        query: Arama sorgusu
        index: FAISS index
        metadata: Metadata listesi
        embedding_model: Embedding modeli
        top_k: Kaç sonuç döndürülecek
        
    Returns:
        [(metadata_dict, similarity_score), ...]
    """
    # 1. Sorguda geçen dosya adlarını (source) bul
    query_lower = query.lower()
    all_sources = set(m.get("source", "") for m in metadata if m.get("source"))
    mentioned_sources = set()
    
    for src in all_sources:
        src_lower = src.lower()
        if src_lower in query_lower:
            mentioned_sources.add(src)
        else:
            # Uzantısız halini kontrol et (örn: dosya.pdf -> dosya)
            src_no_ext = src_lower.rsplit('.', 1)[0]
            if len(src_no_ext) > 3 and src_no_ext in query_lower:
                mentioned_sources.add(src)

    exact_matches = []
    seen_chunks = set()
    
    if mentioned_sources:
        for src in mentioned_sources:
            # İlgili kaynağa ait chunk'ları al
            chunks_for_src = [m for m in metadata if m.get("source") == src]
            # Sayfa numarasına göre sırala ki ilk sayfalar (özet/başlık) gelsin
            chunks_for_src.sort(key=lambda x: x.get("page", 999999))
            
            # İlk 3 chunk'ı yapay yüksek skorla (örn: 2.0) ekle
            for m in chunks_for_src[:3]:
                exact_matches.append((m, 2.0))
                # dict olduğu için benzersiz kimlik olarak chunk_id veya text'in hash'ini kullanalım
                chunk_id = m.get("chunk_id", str(m.get("text", ""))[:50])
                seen_chunks.add(chunk_id)

    # 2. Query'yi embedding'e çevir
    query_embedding = embedding_model.encode([query], show_progress=False)
    
    # 3. FAISS search
    scores, indices = index.search(query_embedding.astype('float32'), top_k)
    
    # 4. Sonuçları birleştir
    results = exact_matches.copy()
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            m = metadata[idx]
            chunk_id = m.get("chunk_id", str(m.get("text", ""))[:50])
            if chunk_id not in seen_chunks:
                results.append((m, float(score)))
                seen_chunks.add(chunk_id)
                
    # Skora göre büyükten küçüğe sırala ve top_k kadarını döndür
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
