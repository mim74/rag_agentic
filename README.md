# RAG Agentic — PDF ile sohbet (metin + görsel)

PDF arşiviniz üzerinde **FAISS metin araması**, **ColPali (ColQwen) ile görsel sayfa araması** ve **ReAct tarzı agentic iterasyonlar** kullanan yerel RAG sistemi. LLM çıktısı **LM Studio** üzerinden (OpenAI uyumlu API) alınır.

## Özellikler

- **Metin RAG**: `pypdf` ile metin çıkarma, chunk’lama (`langchain-text-splitters`), `sentence-transformers` embedding, **FAISS** ile benzerlik araması.
- **Agentic akış**: `SEARCH` (metin), `VISUAL_SEARCH` (ColPali MaxSim), `THINK`, `ANSWER`; ayarlardan `max_iterations` ve `min_searches` ile sınırlanır.
- **Görsel tamamlama**: Agent en az bir görsel arama yaptıysa ve sayfa PNG’leri mevcutsa, nihai cevap için **tek seferlik vision** çağrısı (LM Studio’da çok modlu model gerekir).
- **Artımlı indeks**: Yeni PDF’ler tespit edilip metin indeksine eklenir; çıkarılabilir metin olmayan dosyalar `.manifest.json` ile işaretlenir (her seferinde yeniden taranmaz).
- **ColPali indeksi**: Sayfalar PNG’ye render edilir, ColQwen2.5 ile sayfa embedding’leri üretilir (ilk çalıştırmada veya eksik sayfada otomatik oluşturma).
- **ODT dışa aktarma**: Konuşmayı `exports/` altına kaydetme (`odfpy`).

## Gereksinimler

- Python 3.10+ önerilir.
- **LM Studio**: Sunucu açık olmalı; görsel cevaplar için **vision destekli** bir sohbet modeli yükleyin.
- **Poppler** (pdf2image): `sudo apt install poppler-utils` (Debian/Ubuntu benzeri).
- **GPU**: ColPali indeks üretiminde `index_device: "balanced"` yalnızca **cuda:0 + CPU taşması** kullanır (ColQwen2.5 + ColPali ileri geçişi çoklu CUDA kartına güvenle bölünemiyor). Tüm kartlarda tam hız için `multi` (her GPU’ya tam model, çok VRAM) veya tek kartta `cuda:0`.

## Kurulum

```bash
git clone <repo-url> rag_agentic
cd rag_agentic
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch’u kendi donanımınıza göre [pytorch.org](https://pytorch.org) üzerinden kurmak genelde daha doğrudur; ardından `pip install -r requirements.txt` içindeki diğer paketler yeterlidir.

`config/settings.json` içinde `embedding.model_name` ve `colpali.model_name` Hugging Face repo kimliğidir; ağırlıklar `~/.cache/huggingface` (veya `HF_HOME`) altında saklanır. **Çevrimdışı:** `embedding` / `colpali` içinde `"local_files_only": true` ekleyin veya (anahtar yokken) `export HF_HUB_OFFLINE=1`. JSON’da açıkça `false` yazarsanız ortam değişkeni göz ardı edilir. Modellerin tamamı önbellekte olmalı; yoksa önce çevrimiçi çalıştırın.

## Kullanım

1. PDF’leri `pdfs/` klasörüne koyun.
2. LM Studio’da modeli yükleyip API sunucusunu başlatın (`settings.json` → `lm_studio.base_url`).
3. Uygulamayı çalıştırın:

```bash
source .venv/bin/activate
python src/chat.py
# veya: ./run.sh
```

İlk açılışta metin indeksi (`indexes/pdf_index.*`) ve ColPali çıktıları (`indexes/colpali/`, `page_images/`) gerektiğinde oluşturulur; süre ve disk kullanımı PDF sayısına bağlıdır.

Sohbet komutları: `exit` / `q`, `save`, `export`. Her soru **agentic** pipeline ile işlenir. LM Studio zaman aşımında yedek olarak tek adımlı metin RAG (`rag_simple`) denenebilir.

## Yapılandırma

Dosya: `config/settings.json` (JSON5: yorum ve sondaki virgül desteklenir).

| Bölüm | Anlamı |
|--------|--------|
| `embedding` | Model, `device` / `chat_device`, `chunk_size`, `chunk_overlap`, `batch_size` |
| `lm_studio` | `base_url`, `timeout` |
| `index.output_path` | FAISS + metadata + `.manifest.json` öneki (uzantısız yol) |
| `generation` | `system_prompt`, `temperature`, `max_tokens` |
| `agentic_rag` | `max_iterations`, `search_top_k`, `min_searches`, `agent_temperature` |
| `colpali` | ColQwen HF `model_name`, `index_path`, `page_images_dir`, `top_k_pages`, `retrieval_device`, `mmap_embeddings`, vb. |

## Proje yapısı

```
rag_agentic/
├── config/settings.json
├── src/
│   ├── chat.py              # Giriş: metin + ColPali yükleme, sohbet döngüsü
│   ├── rag_agentic.py       # ReAct döngüsü, görsel final birleştirme
│   ├── rag_simple.py        # Tek atımlı metin RAG (yedek)
│   ├── rag_colpali.py       # ColPali tek tur (vision); agent finalinde prompt paylaşımı
│   ├── colpali_retrieval.py # MaxSim, model yükleme
│   ├── colpali_indexer.py   # PNG render, embedding üretimi
│   ├── indexer.py           # FAISS, artımlı güncelleme, PDF manifest
│   ├── pdf_loader.py
│   ├── embedding.py
│   ├── lm_studio_client.py
│   ├── agent_tools.py
│   └── odt_exporter.py
├── pdfs/                    # Kaynak PDF’ler
├── indexes/                 # pdf_index + colpali (git’e genelde eklenmez)
├── page_images/             # Render edilen sayfa görselleri
├── exports/                 # ODT çıktıları
├── requirements.txt
├── run.sh
└── README.md
```

## İndeksi sıfırlama

Metin indeksini baştan oluşturmak için:

```bash
rm -f indexes/pdf_index.index indexes/pdf_index.json indexes/pdf_index.manifest.json
python src/chat.py
```

ColPali’yi sıfırlamak için `indexes/colpali/` ve isteğe bağlı `page_images/` silinebilir (yeniden oluşturma maliyetlidir).

## Sorun giderme

| Sorun | Öneri |
|--------|--------|
| LM Studio bağlanmıyor | Sunucu ve port; güvenlik duvarı |
| `pdf2image` hatası | Poppler kurulumu |
| ColPali import hatası | `pip install -U colpali-engine` |
| “Yeni PDF” her seferinde | Metin çıkmıyorsa manifest güncellenir; `.manifest.json` oluşumunu kontrol edin |
| Vision cevap boş / hatalı | LM Studio’da çok modlu model; `lm_studio_client` görsel boyutu sınırlı (JPEG 1024px kenar) |
| ColPali `CUDA out of memory` (özellikle yükleme sırasında) | Kod transformers’ın ekstra VRAM warmup’ını kapatır; yine OOM ise `balanced_gpu_memory_fraction` düşürün, `batch_size` azaltın veya `index_device`/`retrieval_device`: `cpu` |

## Lisans ve teşekkür

Kullanılan başlıca projeler: **sentence-transformers**, **FAISS**, **transformers**, **colpali-engine** / **ColQwen**, **LM Studio**, **Rich**, **pypdf**, **odfpy**.
