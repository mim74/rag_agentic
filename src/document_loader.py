"""
Belge yükleme ve text extraction modülü.
Sadece metin içeriğini çıkarır, görsel işleme yoktur.
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from odf import teletype
from odf.opendocument import load as load_odt_document
from odf.text import H, P
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    """Belgeden çıkarılan text parçası"""
    text: str
    source: str
    page: int
    chunk_id: int


def clean_text(text: str) -> str:
    """Text'i temizle"""
    text = " ".join(text.split())
    return text.strip()


def _extract_odt_text(odt_path: Path) -> str:
    document = load_odt_document(str(odt_path))
    text_parts = []
    for element in document.getElementsByType(H):
        extracted = clean_text(teletype.extractText(element))
        if extracted:
            text_parts.append(extracted)
    for element in document.getElementsByType(P):
        extracted = clean_text(teletype.extractText(element))
        if extracted:
            text_parts.append(extracted)
    return "\n\n".join(text_parts)


def load_documents(
    document_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    start_chunk_id: int = 0,
    document_files: Optional[List[Path]] = None,
) -> List[DocumentChunk]:
    """
    Belge klasöründeki desteklenen dosyaları yükle ve chunk'lara böl.

    Args:
        document_dir: Belge klasörü yolu (document_files verilmezse buradan taranır)
        chunk_size: Her chunk'ın karakter sayısı
        chunk_overlap: Chunk'lar arası örtüşme
        start_chunk_id: Başlangıç chunk ID'si (artımlı indeksleme için)
        document_files: Sadece bu dosyaları işle (None ise document_dir'deki tüm desteklenen belgeler)

    Returns:
        DocumentChunk listesi
    """
    if document_files is None:
        document_files = sorted([*document_dir.rglob("*.pdf"), *document_dir.rglob("*.odt")])

    if not document_files:
        raise ValueError(f"Desteklenen belge bulunamadı: {document_dir}")

    print(f"📚 {len(document_files)} belge işleniyor")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []
    chunk_counter = start_chunk_id

    for document_path in document_files:
        try:
            source_name = document_path.name
            if document_path.suffix.lower() == ".pdf":
                reader = PdfReader(document_path)
                print(f"   📄 {source_name} ({len(reader.pages)} sayfa)")
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue
                    text = clean_text(text)
                    chunks = splitter.split_text(text)
                    for chunk_text in chunks:
                        all_chunks.append(DocumentChunk(
                            text=chunk_text,
                            source=source_name,
                            page=page_num,
                            chunk_id=chunk_counter
                        ))
                        chunk_counter += 1
            elif document_path.suffix.lower() == ".odt":
                text = _extract_odt_text(document_path)
                print(f"   📝 {source_name} (ODT)")
                if not text:
                    continue
                chunks = splitter.split_text(text)
                for chunk_text in chunks:
                    all_chunks.append(DocumentChunk(
                        text=chunk_text,
                        source=source_name,
                        page=1,
                        chunk_id=chunk_counter
                    ))
                    chunk_counter += 1

        except Exception as e:
            print(f"   ⚠️  {document_path.name} okunamadı: {e}")
            continue

    print(f"✅ Toplam {len(all_chunks)} chunk oluşturuldu\n")
    return all_chunks
