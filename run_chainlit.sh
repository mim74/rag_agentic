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

# Chainlit uygulamasını başlat (-w ile dosya değişikliklerini anında algılar)
chainlit run src/chainlit_app.py -w --port 8000
