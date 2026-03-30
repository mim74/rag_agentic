# TODO - Radeon 7900 XTX Gecis Plani

Bu dosya, projeyi yeni makineye tasidiktan sonra LM Studio / vLLM / PyTorch seceneklerini temiz sekilde dogrulamak icin hazirlandi.

## 1) Yeni Makinede Ortam Dogrulama

- [ ] `uname -a` ile kernel ve sistem bilgisini kaydet
- [ ] `lspci | grep -iE "vga|display"` ile GPU modelini dogrula
- [ ] `lsmod | grep -i amdgpu || true` ile AMD surucusunu kontrol et
- [ ] `rocminfo` ve `clinfo` ciktilarini al

Komutlar:

```bash
uname -a
lspci | grep -iE "vga|display"
lsmod | grep -i amdgpu || true
rocminfo | head -n 80
/opt/rocm/bin/rocminfo | head -n 80 || true
clinfo | head -n 80 || true
```

## 2) PyTorch ROCm Kontrolu

- [ ] Sanal ortamda `torch` surumu ve GPU gorunurlugunu test et
- [ ] `torch.version.hip` degerini kontrol et (ROCm icin onemli)

Komutlar:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('hip_version', getattr(torch.version, 'hip', None))"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_gpu')"
```

## 3) Backend Mimarisi (Secilebilir Yapi)

- [ ] Tek bir backend secimi tanimla: `LLM_BACKEND=lmstudio|vllm|pytorch`
- [ ] HTTP tabanli provider: LM Studio ve vLLM icin ortak istemci
- [ ] In-process provider: PyTorch/Transformers
- [ ] Ayarlari `config/settings.json` + env degiskenlerinden oku

Onerilen env anahtarlari:

```bash
LLM_BACKEND=lmstudio
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=<model-adi>
LLM_API_KEY=<opsiyonel>
```

vLLM icin ornek:

```bash
LLM_BACKEND=vllm
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=<model-adi>
```

## 4) vLLM Uygunluk Karari (AMD/ROCm)

- [ ] ROCm saglikli calisiyorsa vLLM kurulumu dene
- [ ] Kurulum ve baslatma loglarini kaydet
- [ ] Basit bir test istegi ile endpoint dogrula
- [ ] Sorun varsa fallback olarak LM Studio veya PyTorch kullan

Not:
- vLLM, yuksek throughput icin guclu bir secenektir.
- AMD ROCm destegi surum/kart kombinasyonuna gore degisebilir.

## 5) Fonksiyonel Test Plani (Tum Backendler)

- [ ] Kisa prompt testi (tek istek)
- [ ] Uzun baglam testi (RAG context)
- [ ] Eszamanli 5 istek testi
- [ ] Gecikme/kararlilik notlarini kaydet

## 6) Son Karar

- [ ] Gelistirme icin varsayilan backend belirle
- [ ] Uretim/agir yuk icin backend belirle
- [ ] README veya proje notlarina secimi yaz

---

## Hedef

Projede tek kod tabaniyla, makineye gore su seceneklerden biriyle calisabilmek:

1. LM Studio (OpenAI-compatible HTTP)
2. vLLM (OpenAI-compatible HTTP)
3. PyTorch/Transformers (in-process)
