#!/bin/bash
# RAG Agentic GUI (Chainlit) başlatma script'i

cd "$(dirname "$0")"

# Virtual environment'ı aktifleştir
source .venv/bin/activate

export CHAINLIT_AUTH_SECRET="${CHAINLIT_AUTH_SECRET:-local-dev-chainlit-auth-secret}"
export CHAINLIT_APP_USERNAME="${CHAINLIT_APP_USERNAME:-admin}"
export CHAINLIT_APP_PASSWORD="${CHAINLIT_APP_PASSWORD:-admin123}"

echo "Chainlit arayüzü başlatılıyor..."
echo "Uygulama tarayıcınızda http://localhost:8000 adresinde açılacaktır."
echo "Giriş bilgileri: ${CHAINLIT_APP_USERNAME} / ${CHAINLIT_APP_PASSWORD}"

# Chainlit uygulamasını başlat
# Not: -w (watch) index oluşturma sırasında indexes/ altında dosyalar yazıldıkça
# restart tetikleyip yeniden indexlemeye sebep olabiliyor.
chainlit run src/chainlit_app.py --port 8000
