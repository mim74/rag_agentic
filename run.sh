#!/bin/bash
# RAG Agentic başlatma script'i

cd "$(dirname "$0")"

# Otomatik: başlangıçta huggingface.co erişimi yoksa HF_HUB_OFFLINE=1 ayarlanır.
# Zorla çevrimdışı (opsiyonel):
# export HF_HUB_OFFLINE=1

# Virtual environment'ı aktifleştir
source .venv/bin/activate

# Chat'i başlat
python src/chat.py
