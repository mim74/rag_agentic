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
git clone https://github.com/mim74/rag_agentic.git rag_agentic
cd rag_agentic
./install.sh
```

`install.sh` ne yapar:
- `.venv` oluşturur (yoksa)
- Makinede NVIDIA GPU ve CUDA sürümünü algılar
- **GPU varsa** CUDA'ya uygun PyTorch wheel'ini (cu121/cu124) kurar ve `settings.json` içindeki cihaz ayarlarını GPU profiline getirir
- **GPU yoksa** CPU-only PyTorch kurar, `settings.json` tüm cihazları `"cpu"` olarak ayarlar
- `requirements.txt` içindeki diğer bağımlılıkları kurar

Ek seçenekler:
```bash
./install.sh --no-config   # settings.json'a dokunmadan kur
./install.sh --dry-run     # ne yapacağını göster, dosya değiştirme
```

Ayarları elle yeniden uygulamak için:
```bash
.venv/bin/python scripts/configure_settings.py --profile cpu    # ya da cuda
```

`config/settings.json` içinde `embedding.model_name` ve `colpali.model_name` Hugging Face repo kimliğidir; ağırlıklar `~/.cache/huggingface` (veya `HF_HOME`) altında saklanır. **Çevrimdışı:** `embedding` / `colpali` içinde `"local_files_only": true` ekleyin veya (anahtar yokken) `export HF_HUB_OFFLINE=1`. JSON'da açıkça `false` yazarsanız ortam değişkeni göz ardı edilir. Modellerin tamamı önbellekte olmalı; yoksa önce çevrimiçi çalıştırın.

## Kullanım

1. LM Studio'da modeli yükleyip API sunucusunu başlatın (`settings.json` → `lm_studio.base_url`).
2. Uygulamayı Chainlit arayüzüyle başlatın:

```bash
source .venv/bin/activate
python src/chat.py
# veya: ./run.sh
```

İlk açılışta metin indeksleri ve ColPali çıktıları gerektiğinde oluşturulur.

Sohbet komutları: `exit` / `q`, `save`, `export`. Her soru **agentic** pipeline ile işlenir.

## Çok Kullanıcılı Yapı

Chainlit arayüzü (`chainlit_app.py`) tam çok kullanıcılı modu destekler.

### İlk Giriş (Admin)

İlk kurulumda kullanıcı henüz `data/users.json`'da yok. Giriş için env değişkenleri kullanılır:

```bash
export CHAINLIT_APP_USERNAME=admin
export CHAINLIT_APP_PASSWORD=admin123
```

Admin giriş yaptıktan sonra komutlarla kalıcı kullanıcılar eklenebilir.

### Admin Komutları (sohbet kutusunda)

| Komut | Açıklama |
|---|---|
| `/adduser ali gizli123` | Yeni kullanıcı ekle |
| `/adduser ali gizli123 --shared` | Paylaşımlı belgelere de erişimli ekle |
| `/removeuser ali` | Kullanıcıyı sil |
| `/listusers` | Tüm kullanıcıları listele |
| `/setshared ali on` | Ali'ye paylaşımlı erişim ver |
| `/setshared ali off` | Ali'nin paylaşımlı erişimini kapat |
| `/changepassword ali yeni123` | Şifre değiştir |
| `/help` | Yardım |

### Belge Yükleme (sohbet kutusunda)

- Dosyayı mesaja ekle → **kişisel** belge klasörüne kaydedilir, index anında güncellenir
- `/shared` yazıp dosya ekle → **paylaşımlı** alana kaydedilir (yalnızca admin)

### Dizin Yapısı

```
docs/
├── shared/          # Admin tarafından yüklenen ortak belgeler
└── users/
    ├── admin/       # Admin'in kişisel belgeleri
    └── <kullanıcı>/ # Her kullanıcının kişisel belgeleri

indexes/
├── shared/          # Paylaşımlı FAISS indeksi
└── users/
    └── <kullanıcı>/ # Kullanıcıya özel FAISS indeksi

data/
└── users.json       # Kullanıcı kayıtları (şifreler hash+salt ile)
```

Kullanıcı giriş yaptığında: kendi indeksi + (yetkisi varsa) paylaşımlı indeks birleştirilerek sorgu için tek indeks oluşturulur. Paylaşımlı erişim değişikliği bir sonraki oturumda geçerli olur.

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
│   ├── chainlit_app.py      # Chainlit arayüzü (çok kullanıcılı, dosya yükleme)
│   ├── chat.py              # Giriş: metin + ColPali yükleme, sohbet döngüsü
│   ├── user_manager.py      # Kullanıcı CRUD, şifre hash, dizin yardımcıları
│   ├── rag_agentic.py       # ReAct döngüsü, görsel final birleştirme
│   ├── rag_simple.py        # Tek atımlı metin RAG (yedek)
│   ├── rag_colpali.py       # ColPali tek tur (vision)
│   ├── colpali_retrieval.py # MaxSim, model yükleme
│   ├── colpali_indexer.py   # PNG render, embedding üretimi
│   ├── indexer.py           # FAISS, artımlı güncelleme, manifest
│   ├── document_loader.py   # PDF / ODT / DOCX metin çıkarma
│   ├── embedding.py
│   ├── lm_studio_client.py
│   ├── agent_tools.py
│   └── odt_exporter.py
├── docs/
│   ├── shared/              # Admin'in yüklediği ortak belgeler
│   └── users/<kullanıcı>/  # Kişisel belgeler
├── indexes/
│   ├── shared/              # Paylaşımlı FAISS indeksi
│   └── users/<kullanıcı>/  # Kişisel FAISS indeksleri
├── data/users.json          # Kullanıcı kayıtları (git'e eklenmez)
├── page_images/             # Render edilen sayfa görselleri
├── exports/                 # ODT çıktıları
├── scripts/configure_settings.py
├── install.sh
├── run.sh
└── README.md
```

## İndeksi sıfırlama

Belirli bir kullanıcının indeksini sıfırlamak:
```bash
rm -rf indexes/users/<kullanıcı>/
```

Paylaşımlı indeksi sıfırlamak:
```bash
rm -rf indexes/shared/
```

ColPali'yi sıfırlamak için `indexes/colpali/` ve isteğe bağlı `page_images/` silinebilir (yeniden oluşturma maliyetlidir).

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
