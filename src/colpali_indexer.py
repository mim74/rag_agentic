"""
ColQwen2.5 tabanlı görsel PDF index modülü.

PDF sayfalarını PNG olarak render eder, ColQwen2.5 ile her sayfanın
patch embedding matrisini (N_patches × D) üretir ve diske kaydeder.
Mevcut metin pipeline'ına tamamen bağımsızdır.
"""

import json
import logging
import threading

from hf_load_hacks import no_cuda_allocator_warmup
from colpali_retrieval import load_colpali_processor
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import resource
except ImportError:
    resource = None  # type: ignore

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _mmap_safe_for_count(mmap_requested: bool, num_embeddings: int) -> bool:
    """
    Her mmap ayrı bir dosya tanımlayıcısı tutar; ulimit -n (RLIMIT_NOFILE) yetersizse
    OSError: [Errno 24] Too many open files oluşur. Mümkünse soft limiti yükseltir.
    """
    if not mmap_requested:
        return False
    if num_embeddings <= 0:
        return False
    if resource is None:
        print(
            "   ⚠️  mmap_embeddings atlandı (platform resource modülü yok); tam RAM yüklemesi."
        )
        return False

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Her .npy için ~1 FD + güvenlik payı
        need = num_embeddings + 4096
        if soft < need:
            new_soft = min(need, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < num_embeddings + 256:
            print(
                f"   ⚠️  mmap_embeddings atlandı: açık dosya limiti yetersiz "
                f"(soft={soft}, gerekli≈{num_embeddings}). Tam RAM yüklemesi.\n"
                f"   💡 mmap için shell'de: ulimit -n {min(num_embeddings + 8192, 1048576)} "
                f"(veya limits.conf ile kalıcı artırın)."
            )
            return False
    except (ValueError, OSError) as e:
        print(
            f"   ⚠️  mmap_embeddings atlandı (FD limiti artırılamadı: {e}); tam RAM yüklemesi."
        )
        return False

    return True


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _resolve_colpali_embedding_file(index_dir: Path, stored: str) -> Optional[Path]:
    """
    pages_meta.json'daki embedding_path başka makinede mutlak yol olarak kayıtlı olabilir.
    Önce mevcut index_dir/page_embeddings/<dosya_adı> dene (proje kopyası / taşınma).
    """
    if not stored:
        return None
    name = Path(stored).name
    local = index_dir / "page_embeddings" / name
    if _safe_is_file(local):
        return local
    legacy = Path(stored)
    if _safe_is_file(legacy):
        return legacy
    return None


def _resolve_page_image_file(page_images_root: Optional[Path], stored: str) -> Optional[Path]:
    """image_path için aynı taşınabilirlik: önce kayıtlı yol, yoksa page_images/<basename>."""
    if not stored:
        return None
    p = Path(stored)
    if _safe_is_file(p):
        return p
    if page_images_root is None:
        return None
    alt = page_images_root / p.name
    if _safe_is_file(alt):
        return alt
    return None


# ─── Render ────────────────────────────────────────────────────────────────────

def render_pdf_pages(
    pdf_dir: Path,
    output_dir: Path,
    dpi: int = 150,
    thread_count: int = 4,
) -> List[Dict[str, Any]]:
    """
    PDF klasöründeki tüm sayfaları PNG olarak render et.
    Zaten var olan görseller tekrar render edilmez.

    Args:
        thread_count: PDF sayfaları işlenirken kullanılacak CPU thread sayısı.
            output_folder ile birlikte kullanıldığında paralel render hızlanır.
            Önerilen: 4-8 (12'den fazla genelde fayda sağlamaz).

    Returns:
        [{"source": str, "page": int, "image_path": str}, ...]
    """
    try:
        from pdf2image import convert_from_path, pdfinfo_from_path
        import tempfile
    except ImportError:
        raise ImportError(
            "pdf2image yüklü değil. Kurmak için:\n"
            "  pip install pdf2image\n"
            "  sudo apt install poppler-utils"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"PDF bulunamadı: {pdf_dir}")

    print(f"📚 {len(pdf_files)} PDF dosyası bulundu")
    if thread_count > 1:
        print(f"   ⚡ PDF render: {thread_count} thread kullanılıyor")

    page_records: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:
        try:
            page_count = pdfinfo_from_path(str(pdf_path))["Pages"]
            expected_paths = [
                output_dir / f"{pdf_path.stem}_page_{p:04d}.png"
                for p in range(1, page_count + 1)
            ]
            all_exist = all(p.exists() for p in expected_paths)

            if all_exist:
                print(f"   📄 {pdf_path.name} ({page_count} sayfa) — mevcut PNG'ler kullanılıyor (atlandı)")
                for page_num in range(1, page_count + 1):
                    img_path = output_dir / f"{pdf_path.stem}_page_{page_num:04d}.png"
                    page_records.append({
                        "source": pdf_path.name,
                        "page": page_num,
                        "image_path": str(img_path),
                    })
                continue

            # Eksik sayfalar var, render et
            if thread_count > 1:
                with tempfile.TemporaryDirectory() as tmpdir:
                    images = convert_from_path(
                        str(pdf_path),
                        dpi=dpi,
                        output_folder=tmpdir,
                        thread_count=thread_count,
                        fmt="png",
                        use_pdftocairo=True,
                    )
            else:
                images = convert_from_path(str(pdf_path), dpi=dpi)
            print(f"   📄 {pdf_path.name} ({len(images)} sayfa)")

            for page_num, image in enumerate(images, start=1):
                img_filename = f"{pdf_path.stem}_page_{page_num:04d}.png"
                img_path = output_dir / img_filename

                if not img_path.exists():
                    image.save(str(img_path), "PNG")

                page_records.append({
                    "source": pdf_path.name,
                    "page": page_num,
                    "image_path": str(img_path),
                })

        except Exception as e:
            print(f"   ⚠️  {pdf_path.name} render edilemedi: {e}")
            continue

    print(f"✅ Toplam {len(page_records)} sayfa (mevcut + yeni render)")
    return page_records


# ─── Embedding helpers ─────────────────────────────────────────────────────────

def _embed_single_gpu(
    missing_pages: List[Tuple[int, str]],
    model_name: str,
    ColQwen2_5,
    ColQwen2_5Processor,
    emb_dir: Path,
    page_records: List[Dict[str, Any]],
    page_embeddings: Dict[int, np.ndarray],
    batch_size: int,
    device: str,
    hf_local_files_only: bool = False,
):
    """Eksik sayfaları tek GPU üzerinde embed et."""
    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    print(f"   Cihaz: {device} | dtype: {dtype}")

    with no_cuda_allocator_warmup():
        model = ColQwen2_5.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device,
            local_files_only=hf_local_files_only, low_cpu_mem_usage=True,
        )
    processor = load_colpali_processor(
        ColQwen2_5Processor, model_name, local_files_only=hf_local_files_only,
    )
    model.eval()
    print("   ✅ Model yüklendi\n")

    print("=" * 60)
    print("🔢 SAYFA EMBEDDİNG ÜRETİLİYOR")
    print("=" * 60)

    try:
        for batch_start in range(0, len(missing_pages), batch_size):
            batch = missing_pages[batch_start: batch_start + batch_size]

            images = [Image.open(img_path).convert("RGB") for _, img_path in batch]
            batch_input = processor.process_images(images).to(device)

            with torch.no_grad():
                output = model(**batch_input)

            raw = output.embeddings if hasattr(output, "embeddings") else output

            for i, (page_idx, _) in enumerate(batch):
                emb_np = raw[i].float().cpu().numpy()
                emb_path = emb_dir / f"page_{page_idx:05d}.npy"
                np.save(str(emb_path), emb_np)
                page_records[page_idx]["embedding_path"] = str(emb_path)
                page_records[page_idx]["page_id"] = page_idx
                page_embeddings[page_idx] = emb_np

            done = batch_start + len(batch)
            if done % (batch_size * 5) == 0 or done == len(missing_pages):
                print(f"   [{done}/{len(missing_pages)}] yeni sayfa işlendi")
    except KeyboardInterrupt:
        saved = len(page_embeddings)
        print(f"\n⚠️  Ctrl+C — durduruldu. {saved} sayfa kaydedildi, kalan tekrar çalıştırınca devam eder.")
    finally:
        del model
        torch.cuda.empty_cache()


def _colpali_balanced_max_memory(fraction: float) -> Dict[str, Any]:
    """
    Yalnızca cuda:0 + CPU. ColQwen2.5 + custom_text_proj, birden fazla CUDA kartına
    bölündüğünde tensör cihazları uyumsuz kalıyor (HF device_map çoklu GPU uyumsuz).
    """
    props = torch.cuda.get_device_properties(0)
    gib = max(1, int(props.total_memory / (1024**3) * fraction))
    return {0: f"{gib}GiB", "cpu": "256GiB"}


def _align_colqwen_proj_device(model) -> None:
    """custom_text_proj dil modelinin son katmanıyla aynı cihazda olmalı (CPU offload senaryosu)."""
    proj = getattr(model, "custom_text_proj", None)
    lm = getattr(model, "language_model", None)
    if proj is None or lm is None:
        return
    layers = getattr(lm, "layers", None)
    if not layers:
        return
    ref = next(layers[-1].parameters(), None)
    if ref is None:
        return
    p0 = next(proj.parameters(), None)
    if p0 is not None and p0.device != ref.device:
        proj.to(device=ref.device, dtype=ref.dtype)


def _colqwen_vision_input_device(model) -> torch.device:
    """pixel_values genelde vision gövdesinin ilk parametresiyle aynı cihazda olmalı."""
    vis = getattr(model, "visual", None)
    if vis is not None:
        v0 = next(vis.parameters(), None)
        if v0 is not None:
            return v0.device
    for p in model.parameters():
        if p.device.type == "cuda":
            return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _embed_balanced(
    missing_pages: List[Tuple[int, str]],
    model_name: str,
    ColQwen2_5,
    ColQwen2_5Processor,
    emb_dir: Path,
    page_records: List[Dict[str, Any]],
    page_embeddings: Dict[int, np.ndarray],
    batch_size: int,
    n_gpus: int,
    gpu_memory_fraction: float = 0.72,
    hf_local_files_only: bool = False,
):
    """
    Tek model: ağırlıklar yalnızca cuda:0 + CPU'ya yayılır (RAM taşması).
    cuda:1 / cuda:2 burada kullanılmaz — ColQwen ColPali ileri geçişi çoklu CUDA shard ile uyumlu değil.
    """
    max_mem = _colpali_balanced_max_memory(gpu_memory_fraction)
    if n_gpus > 1:
        print(
            f"   Mod: balanced — cuda:0 + CPU taşması (sistemde {n_gpus} GPU var; "
            "ColQwen indekslemede ek CUDA kartları devre dışı, aksi halde tensör cihaz hatası oluşuyor).\n"
        )
    else:
        print("   Mod: balanced — cuda:0 + CPU taşması (VRAM tasarrufu)\n")
    print(f"   max_memory: {max_mem}\n")

    with no_cuda_allocator_warmup():
        model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_mem,
            local_files_only=hf_local_files_only,
            low_cpu_mem_usage=True,
        )
    _align_colqwen_proj_device(model)
    processor = load_colpali_processor(
        ColQwen2_5Processor, model_name, local_files_only=hf_local_files_only,
    )
    model.eval()
    inp_device = _colqwen_vision_input_device(model)
    print(f"   ✅ Model yüklendi (işlemci girdileri → {inp_device})\n")

    print("=" * 60)
    print("🔢 SAYFA EMBEDDİNG ÜRETİLİYOR (balanced: cuda:0 + CPU)")
    print("=" * 60)

    try:
        for batch_start in range(0, len(missing_pages), batch_size):
            batch = missing_pages[batch_start : batch_start + batch_size]

            images = [Image.open(img_path).convert("RGB") for _, img_path in batch]
            batch_input = processor.process_images(images).to(inp_device)

            with torch.no_grad():
                output = model(**batch_input)

            raw = output.embeddings if hasattr(output, "embeddings") else output

            for i, (page_idx, _) in enumerate(batch):
                emb_np = raw[i].float().cpu().numpy()
                emb_path = emb_dir / f"page_{page_idx:05d}.npy"
                np.save(str(emb_path), emb_np)
                page_records[page_idx]["embedding_path"] = str(emb_path)
                page_records[page_idx]["page_id"] = page_idx
                page_embeddings[page_idx] = emb_np

            done = batch_start + len(batch)
            if done % (batch_size * 5) == 0 or done == len(missing_pages):
                print(f"   [{done}/{len(missing_pages)}] yeni sayfa işlendi")
    except KeyboardInterrupt:
        saved = len(page_embeddings)
        print(f"\n⚠️  Ctrl+C — durduruldu. {saved} sayfa kaydedildi, kalan tekrar çalıştırınca devam eder.")
    finally:
        del model
        torch.cuda.empty_cache()


def _embed_multi_gpu(
    missing_pages: List[Tuple[int, str]],
    model_name: str,
    ColQwen2_5,
    ColQwen2_5Processor,
    emb_dir: Path,
    page_records: List[Dict[str, Any]],
    page_embeddings: Dict[int, np.ndarray],
    batch_size: int,
    n_gpus: int,
    hf_local_files_only: bool = False,
):
    """Eksik sayfaları birden fazla GPU'ya dağıtarak paralel embed et."""
    dtype = torch.bfloat16
    cancel = threading.Event()

    models = {}
    processors = {}
    for gpu_id in range(n_gpus):
        device = f"cuda:{gpu_id}"
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"   GPU {gpu_id}: {gpu_name} ({device}) — model yükleniyor...")
        with no_cuda_allocator_warmup():
            models[gpu_id] = ColQwen2_5.from_pretrained(
                model_name, torch_dtype=dtype, device_map=device,
                local_files_only=hf_local_files_only, low_cpu_mem_usage=True,
            )
        models[gpu_id].eval()
        processors[gpu_id] = load_colpali_processor(
            ColQwen2_5Processor, model_name, local_files_only=hf_local_files_only,
        )
    print(f"   ✅ {n_gpus} GPU'da model yüklendi\n")

    print("=" * 60)
    print(f"🔢 SAYFA EMBEDDİNG ÜRETİLİYOR ({n_gpus} GPU paralel)")
    print("=" * 60)

    gpu_chunks: List[List[Tuple[int, str]]] = [[] for _ in range(n_gpus)]
    for i, item in enumerate(missing_pages):
        gpu_chunks[i % n_gpus].append(item)

    lock = threading.Lock()
    processed = [0]
    total = len(missing_pages)

    def gpu_worker(gpu_id: int, pages: List[Tuple[int, str]]):
        model = models[gpu_id]
        processor = processors[gpu_id]
        device = f"cuda:{gpu_id}"

        for batch_start in range(0, len(pages), batch_size):
            if cancel.is_set():
                return

            batch = pages[batch_start: batch_start + batch_size]

            images = [Image.open(img_path).convert("RGB") for _, img_path in batch]
            batch_input = processor.process_images(images).to(device)

            with torch.no_grad():
                output = model(**batch_input)

            if cancel.is_set():
                return

            raw = output.embeddings if hasattr(output, "embeddings") else output

            with lock:
                for i, (page_idx, _) in enumerate(batch):
                    emb_np = raw[i].float().cpu().numpy()
                    emb_path = emb_dir / f"page_{page_idx:05d}.npy"
                    np.save(str(emb_path), emb_np)
                    page_records[page_idx]["embedding_path"] = str(emb_path)
                    page_records[page_idx]["page_id"] = page_idx
                    page_embeddings[page_idx] = emb_np
                    processed[0] += 1

                if processed[0] % (batch_size * 5) == 0 or processed[0] == total:
                    print(f"   [{processed[0]}/{total}] yeni sayfa işlendi")

    try:
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = []
            for gpu_id in range(n_gpus):
                if gpu_chunks[gpu_id]:
                    futures.append(executor.submit(gpu_worker, gpu_id, gpu_chunks[gpu_id]))
            for f in as_completed(futures):
                f.result()
    except KeyboardInterrupt:
        cancel.set()
        saved = len(page_embeddings)
        print(f"\n⚠️  Ctrl+C — durduruldu. {saved} sayfa kaydedildi, kalan tekrar çalıştırınca devam eder.")
    finally:
        for gpu_id in list(models.keys()):
            del models[gpu_id]
            del processors[gpu_id]
        torch.cuda.empty_cache()


# ─── Index oluşturma ───────────────────────────────────────────────────────────

def build_colpali_index(
    pdf_dir: Path,
    output_dir: Path,
    model_name: str,
    page_images_dir: Path,
    dpi: int = 150,
    batch_size: int = 2,
    render_thread_count: int = 4,
    index_device: str = "cuda",
    balanced_gpu_memory_fraction: Optional[float] = None,
    hub_local_files_only: bool = False,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]]]:
    """
    PDF sayfalarını ColQwen2.5 ile vektörleştirip ColPali index'i oluştur.

    Args:
        pdf_dir: PDF kaynak klasörü
        output_dir: Index çıktı dizini (embedding .npy + pages_meta.json)
        model_name: HuggingFace model adı veya yerel yol
        page_images_dir: PNG sayfaların kaydedileceği dizin
        dpi: Render çözünürlüğü
        batch_size: Aynı anda işlenecek sayfa sayısı
        index_device: cpu | cuda / cuda:N | balanced | multi
            balanced = cuda:0 + CPU taşması; çoklu CUDA shard desteklenmez (ColQwen/ColPali)
            multi = her GPU'ya tam kopya (çok VRAM, veri paralelliği)
        balanced_gpu_memory_fraction: index_device balanced iken cuda:0 kotası (0.35–0.95), None = 0.72
        hub_local_files_only: True ise ColQwen yüklemesinde HF hub'a istek yok (çevrimdışı önbellek)

    Returns:
        (page_embeddings, metadata)
        page_embeddings: {page_id: np.ndarray şeklinde (N_patches, D)}
        metadata: Sayfa kayıt listesi
    """
    try:
        from colpali_engine.models import ColQwen2_5
        try:
            # Bazı colpali_engine sürümlerinde adlandırma bu şekilde
            from colpali_engine.models import ColQwen2_5Processor  # type: ignore
        except ImportError:
            # Bazı sürümlerde Processor adı underscore içeriyor
            from colpali_engine.models import ColQwen2_5_Processor as ColQwen2_5Processor  # type: ignore
    except ImportError as e:
        raise ImportError(
            "colpali_engine import edilemedi veya API sürümü uyumsuz.\n"
            "Kurulum:\n"
            "  pip install -U colpali_engine\n"
            "Hata detayı:\n"
            f"  {e}"
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = output_dir / "page_embeddings"
    emb_dir.mkdir(exist_ok=True)

    # 1. Sayfaları render et
    print("\n" + "=" * 60)
    print("🖼️  PDF SAYFASI RENDER")
    print("=" * 60)
    page_records = render_pdf_pages(
        pdf_dir, page_images_dir, dpi=dpi, thread_count=render_thread_count
    )

    # 2. Mevcut embedding'leri yükle, eksikleri tespit et
    page_embeddings: Dict[int, np.ndarray] = {}
    missing_pages: List[Tuple[int, str]] = []  # (page_idx, image_path)

    for page_idx, rec in enumerate(page_records):
        emb_path = emb_dir / f"page_{page_idx:05d}.npy"
        if emb_path.exists():
            page_embeddings[page_idx] = np.load(str(emb_path))
            page_records[page_idx]["embedding_path"] = str(emb_path)
            page_records[page_idx]["page_id"] = page_idx
        else:
            missing_pages.append((page_idx, rec["image_path"]))

    skipped = len(page_embeddings)
    if skipped:
        print(f"\n   📂 {skipped} sayfa mevcut embedding'den yüklendi")

    if not missing_pages:
        print(f"\n✅ {len(page_embeddings)} sayfa (tümü mevcut, yeni üretim yok)")
        _save_metadata(page_records, output_dir)
        return page_embeddings, page_records

    # 3. GPU yapılandırması
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    idx_dev = str(index_device).strip().lower()
    use_multi = idx_dev == "multi" and n_gpus >= 2
    use_balanced = idx_dev == "balanced" and n_gpus >= 1

    print("\n" + "=" * 60)
    print("🤖 COLQWEN2.5 MODELİ YÜKLENİYOR")
    print("=" * 60)

    interrupted = False
    try:
        if use_multi:
            print(
                "⚠️  index_device=multi: Her GPU'ya TAM model kopyası yüklenir — "
                "VRAM ≈ (GPU sayısı × tek model). Hız için; dar VRAM'da OOM olur.\n"
                "   Çok GPU'da tek ağırlık kümesi için: index_device: \"balanced\"\n"
            )
            _embed_multi_gpu(
                missing_pages, model_name, ColQwen2_5, ColQwen2_5Processor,
                emb_dir, page_records, page_embeddings, batch_size, n_gpus,
                hf_local_files_only=hub_local_files_only,
            )
        elif use_balanced:
            frac = balanced_gpu_memory_fraction
            if frac is None:
                bgf = 0.72
            else:
                bgf = max(0.35, min(0.95, float(frac)))
            _embed_balanced(
                missing_pages, model_name, ColQwen2_5, ColQwen2_5Processor,
                emb_dir, page_records, page_embeddings, batch_size, n_gpus,
                gpu_memory_fraction=bgf,
                hf_local_files_only=hub_local_files_only,
            )
        else:
            idx = idx_dev
            if idx == "cpu":
                device = "cpu"
            elif n_gpus and index_device and idx_dev not in ("multi", "balanced"):
                # "cuda" / "cuda:1" vb.; ":" yoksa ilk GPU
                device = index_device if ":" in str(index_device) else "cuda:0"
            else:
                device = "cuda:0" if n_gpus else "cpu"
            _embed_single_gpu(
                missing_pages, model_name, ColQwen2_5, ColQwen2_5Processor,
                emb_dir, page_records, page_embeddings, batch_size, device,
                hf_local_files_only=hub_local_files_only,
            )
    except KeyboardInterrupt:
        interrupted = True

    total_new = len(page_embeddings) - skipped
    print(f"\n✅ {len(page_embeddings)} sayfa ({skipped} mevcut atlandı, {total_new} yeni üretildi)")

    # Metadata'yı her durumda kaydet (kısmi ilerleme de korunur)
    _save_metadata(page_records, output_dir)

    if interrupted:
        raise KeyboardInterrupt

    return page_embeddings, page_records


# ─── Kaydet / Yükle ────────────────────────────────────────────────────────────

def _save_metadata(metadata: List[Dict[str, Any]], output_dir: Path) -> None:
    meta_file = output_dir / "pages_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Metadata kaydedildi: {meta_file}")


def save_colpali_index(
    metadata: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Metadata dosyasını güncelle (embedding .npy dosyaları zaten yerinde)."""
    _save_metadata(metadata, output_dir)
    logger.debug("ColPali index metadata güncellendi: %s", output_dir)


def load_colpali_index(
    index_dir: Path,
    mmap_embeddings: bool = False,
    page_images_root: Optional[Path] = None,
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, Any]]]:
    """
    Kaydedilmiş ColPali index'ini yükle.

    Args:
        index_dir: build_colpali_index() çıktı dizini
        mmap_embeddings: True ise .npy dosyaları disk üzerinden memory-map edilir;
            başlangıçta RAM kullanımı çok düşer; tarama sırasında disk/önbellek kullanılır.
        page_images_root: Varsa, metadata'daki eski mutlak image_path'ler bu dizindeki
            aynı dosya adıyla eşlenir (proje kopyası).

    Returns:
        (page_embeddings, metadata)
    """
    meta_file = index_dir / "pages_meta.json"

    if not meta_file.exists():
        raise FileNotFoundError(
            f"ColPali metadata bulunamadı: {meta_file}\n"
            "İlk kullanımda index oluşturmak için chat uygulamasını bir kez çalıştırın (otomatik oluşturur)."
        )

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata: List[Dict[str, Any]] = json.load(f)

    for rec in metadata:
        ep = rec.get("embedding_path")
        if ep:
            resolved = _resolve_colpali_embedding_file(index_dir, ep)
            if resolved is not None:
                rec["embedding_path"] = str(resolved)
        ip = rec.get("image_path")
        if ip:
            resolved_img = _resolve_page_image_file(page_images_root, ip)
            if resolved_img is not None:
                rec["image_path"] = str(resolved_img)

    n_existing = sum(
        1
        for rec in metadata
        if rec.get("embedding_path") and _safe_is_file(Path(rec["embedding_path"]))
    )
    # mmap = her .npy için ayrı FD; gerekli limit ≈ diskteki embedding dosya sayısı
    use_mmap = _mmap_safe_for_count(mmap_embeddings, n_existing)

    if use_mmap:
        print(f"📂 ColPali index yükleniyor: {len(metadata)} sayfa (mmap — düşük RAM başlangıç)...")
    else:
        print(f"📂 ColPali index yükleniyor: {len(metadata)} sayfa...")

    page_embeddings: Dict[int, np.ndarray] = {}
    missing = 0

    for rec in metadata:
        page_id = rec.get("page_id")
        emb_path = rec.get("embedding_path")

        if page_id is None or not emb_path:
            missing += 1
            continue

        emb_file = _resolve_colpali_embedding_file(index_dir, emb_path)
        if emb_file is None:
            logger.warning("Embedding dosyası bulunamadı (page_id=%d): %s", page_id, emb_path)
            missing += 1
            continue
        try:
            if use_mmap:
                page_embeddings[page_id] = np.load(str(emb_file), mmap_mode="r")
            else:
                page_embeddings[page_id] = np.load(str(emb_file))
        except OSError as e:
            logger.warning(
                "Embedding okunamadı (page_id=%d) %s: %s", page_id, emb_file, e
            )
            missing += 1

    if missing:
        print(f"   ⚠️  {missing} sayfa embedding'i yüklenemedi")

    print(f"✅ ColPali index yüklendi: {len(page_embeddings)} sayfa\n")
    return page_embeddings, metadata
