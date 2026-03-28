"""
Embedding oluşturma modülü.
CPU, CUDA (NVIDIA), ve ROCm (AMD) destekler.
"""

import os
import warnings
from typing import List
import numpy as np
import torch

# HuggingFace ve model yükleme uyarılarını kapat
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

# Logging seviyesini ayarla
import logging
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

from hf_load_hacks import (
    check_and_set_offline_mode,
    patch_find_adapter_for_offline,
    patch_safetensors_auto_conversion,
)

check_and_set_offline_mode()

from sentence_transformers import SentenceTransformer

patch_find_adapter_for_offline()
patch_safetensors_auto_conversion()


def list_available_gpus():
    """Sistemdeki tüm GPU'ları listele."""
    if not torch.cuda.is_available():
        return []
    
    gpus = []
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_type = "AMD/ROCm" if ("AMD" in gpu_name or "Radeon" in gpu_name) else "NVIDIA"
        gpus.append({
            'index': i,
            'name': gpu_name,
            'type': gpu_type
        })
    return gpus


def detect_device(prefer_gpu_type: str = None, gpu_index: int = None) -> str:
    """
    En iyi kullanılabilir cihazı otomatik tespit et.
    
    Args:
        prefer_gpu_type: "amd" veya "nvidia" - Belirli GPU tipini tercih et
        gpu_index: Kullanılacak GPU'nun index'i (0, 1, 2, ...)
    
    Returns:
        Device string: "cuda:0", "cuda:1", "mps", veya "cpu"
    """
    if torch.cuda.is_available():
        # Tüm GPU'ları listele
        available_gpus = list_available_gpus()
        
        if len(available_gpus) > 1:
            print(f"🔍 {len(available_gpus)} GPU tespit edildi:")
            for gpu in available_gpus:
                print(f"   [{gpu['index']}] {gpu['name']} ({gpu['type']})")
        
        # GPU index belirtildiyse onu kullan
        if gpu_index is not None:
            if gpu_index < len(available_gpus):
                selected_gpu = available_gpus[gpu_index]
                device = f"cuda:{gpu_index}"
                print(f"\n✅ Seçilen GPU [{gpu_index}]: {selected_gpu['name']} ({selected_gpu['type']})")
                
                # GPU bellek bilgisi
                props = torch.cuda.get_device_properties(gpu_index)
                print(f"   GPU Bellek: {props.total_memory / 1024**3:.1f} GB")
                
                return device
            else:
                print(f"⚠️  GPU index {gpu_index} bulunamadı, varsayılan GPU kullanılacak")
        
        # GPU tipi tercihi belirtildiyse
        if prefer_gpu_type:
            prefer_gpu_type = prefer_gpu_type.lower()
            for gpu in available_gpus:
                if prefer_gpu_type == "amd" and gpu['type'] == "AMD/ROCm":
                    device = f"cuda:{gpu['index']}"
                    print(f"\n✅ AMD GPU seçildi [{gpu['index']}]: {gpu['name']}")
                    props = torch.cuda.get_device_properties(gpu['index'])
                    print(f"   GPU Bellek: {props.total_memory / 1024**3:.1f} GB")
                    return device
                elif prefer_gpu_type == "nvidia" and gpu['type'] == "NVIDIA":
                    device = f"cuda:{gpu['index']}"
                    print(f"\n✅ NVIDIA GPU seçildi [{gpu['index']}]: {gpu['name']}")
                    props = torch.cuda.get_device_properties(gpu['index'])
                    print(f"   GPU Bellek: {props.total_memory / 1024**3:.1f} GB")
                    return device
            
            print(f"⚠️  {prefer_gpu_type.upper()} GPU bulunamadı, varsayılan GPU kullanılacak")
        
        # Varsayılan: İlk GPU'yu kullan
        device_name = torch.cuda.get_device_name(0)
        device = "cuda:0"
        
        gpu_type = "AMD/ROCm" if ("AMD" in device_name or "Radeon" in device_name) else "NVIDIA"
        print(f"\n✅ Varsayılan GPU [0]: {device_name} ({gpu_type})")
        
        if hasattr(torch.version, 'hip'):
            print(f"   ROCm version: {torch.version.hip}")
        elif hasattr(torch.version, 'cuda'):
            print(f"   CUDA version: {torch.version.cuda}")
        
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU Bellek: {props.total_memory / 1024**3:.1f} GB")
        
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ Apple Silicon (MPS) tespit edildi")
        return "mps"
    else:
        print("⚠️  GPU bulunamadı, CPU kullanılıyor")
        return "cpu"


class EmbeddingModel:
    """Sentence-transformers embedding modeli wrapper'ı"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: str = "auto",
                 prefer_gpu_type: str = None,
                 gpu_index: int = None,
                 local_files_only: bool = False):
        """
        Args:
            model_name: HuggingFace model adı
            device: Cihaz seçimi
                    "auto" = Otomatik en iyi cihazı seç
                    "cuda" veya "cuda:0" = Belirli GPU kullan
                    "cpu" = CPU kullan
            prefer_gpu_type: "amd" veya "nvidia" - Birden fazla GPU varsa tercih edilen tip
            gpu_index: Kullanılacak GPU'nun index'i (0, 1, 2, ...)
            local_files_only: True = hub'a istek yok (çevrimdışı; model önbellekte olmalı)
        """
        # Otomatik cihaz tespiti
        if device == "auto":
            device = detect_device(prefer_gpu_type=prefer_gpu_type, gpu_index=gpu_index)
        elif device.startswith("cuda") and ":" not in device and gpu_index is not None:
            # "cuda" yazılmış ama index belirtilmiş
            device = f"cuda:{gpu_index}"
        
        print(f"🔄 Embedding modeli yükleniyor: {model_name}")
        print(f"   Cihaz: {device}")
        
        # GPU kullanılıyorsa bilgi ver
        if device.startswith("cuda"):
            # cuda:0, cuda:1 gibi formatlardan index'i çıkar
            gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
            if torch.cuda.is_available() and gpu_idx < torch.cuda.device_count():
                gpu_props = torch.cuda.get_device_properties(gpu_idx)
                print(f"   GPU Bellek: {gpu_props.total_memory / 1024**3:.1f} GB")
        
        self.device = device
        self.model = SentenceTransformer(
            model_name, device=device, local_files_only=local_files_only
        )
        print(f"✅ Model yüklendi\n")
    
    def _clear_gpu_cache(self):
        """GPU cache'ini temizle (yardımcı metod)."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Text'leri embedding'e çevir.
        
        Args:
            texts: Text listesi
            batch_size: Batch boyutu
            show_progress: İlerleme göster
            
        Returns:
            Embedding matrisi (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Cosine similarity için normalize
        )
        return embeddings
    
    def to_cpu(self):
        """Model'i CPU'ya taşı ve GPU belleğini temizle."""
        if self.device.startswith("cuda") or self.device == "mps":
            print(f"🔄 Model GPU'dan CPU'ya taşınıyor...")
            self.model = self.model.to('cpu')
            self.device = 'cpu'
            self._clear_gpu_cache()
            print(f"✅ Model CPU'ya taşındı, GPU belleği temizlendi\n")
    
    def to_device(self, device: str = None):
        """Model'i belirtilen cihaza taşı."""
        target_device = device or self.device
        
        if target_device != self.device:
            print(f"🔄 Model {self.device} -> {target_device} taşınıyor...")
            self.model = self.model.to(target_device)
            self.device = target_device
            print(f"✅ Model {target_device} cihazına taşındı\n")
    
    def free_memory(self):
        """
        Model'i bellekten kaldır ve GPU belleğini tamamen temizle.
        Not: Bu işlemden sonra model tekrar yüklenmeli.
        """
        print(f"🗑️  Model bellekten kaldırılıyor...")
        
        # Model'i sil
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        
        # GPU belleğini temizle
        self._clear_gpu_cache()
        print(f"✅ Model bellekten kaldırıldı, bellek temizlendi\n")
    
    def __del__(self):
        """Destructor - nesne silinirken GPU belleğini temizle."""
        if hasattr(self, 'device') and (self.device.startswith("cuda") or self.device == "mps"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
