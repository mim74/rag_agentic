"""
LM Studio REST API client.
Sadeleştirilmiş versiyon - sadece chat completion.
"""

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Union
import requests


class LMStudioClient:
    """LM Studio OpenAI-compatible API client"""
    
    def __init__(self, base_url: str = "http://localhost:1234", timeout: int = 120):
        """
        Args:
            base_url: LM Studio server URL
            timeout: Request timeout (saniye)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def is_server_running(self) -> bool:
        """LM Studio server'ın çalışıp çalışmadığını kontrol et"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        LM Studio'dan model bilgilerini al.
        
        Returns:
            {
                "model_id": str,      # Aktif model ID
                "model_name": str,    # Model adı (varsa)
                "loaded_models": List[Dict]  # Tüm yüklü modeller
            }
            
        Raises:
            requests.exceptions.RequestException: Connection/timeout errors
        """
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # OpenAI-compatible models endpoint yanıtı
            models = data.get("data", [])
            
            if not models:
                return {
                    "model_id": "unknown",
                    "model_name": "Bilinmeyen Model",
                    "loaded_models": []
                }
            
            # İlk modeli al (genellikle aktif model)
            first_model = models[0]
            model_id = first_model.get("id", "unknown")
            model_name = first_model.get("name", model_id)
            
            return {
                "model_id": model_id,
                "model_name": model_name,
                "loaded_models": models
            }
            
        except requests.exceptions.RequestException as e:
            print(f"[WARNING] Model bilgisi alınamadı: {e}")
            return {
                "model_id": "unknown",
                "model_name": "Bilinmeyen Model (LM Studio'dan alınamadı)",
                "loaded_models": []
            }
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completion.
        
        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            {
                "answer": str,
                "generation_time": float,
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int
            }
            
        Raises:
            requests.exceptions.RequestException: Connection/timeout errors
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            if not response.ok:
                # Hata detayını göster (LM Studio genellikle JSON body döner)
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise requests.exceptions.HTTPError(
                    f"LM Studio {response.status_code} hatası: {detail}",
                    response=response,
                )

        except requests.exceptions.ReadTimeout as e:
            # Timeout durumunda özel hata mesajı
            raise requests.exceptions.ReadTimeout(
                f"LM Studio sunucusu yanıt vermedi. Timeout: {self.timeout} saniye. "
                f"Sunucunun çalıştığından emin olun veya timeout süresini artırın."
            ) from e
        
        generation_time = time.time() - start_time
        data = response.json()
        
        # OpenAI format response parse
        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        return {
            "answer": answer,
            "generation_time": generation_time,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
    
    @staticmethod
    def _image_to_base64(img_path: Path, max_size: int = 1024, quality: int = 85) -> str:
        """
        Görseli JPEG'e sıkıştır, gerekirse yeniden boyutlandır ve base64'e çevir.
        max_size: uzun kenarın maksimum piksel değeri.
        """
        from PIL import Image as PILImage
        import io

        with PILImage.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def chat_completion_with_images(
        self,
        text_prompt: str,
        image_paths: List[Path],
        system_prompt: str = "",
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Görsel içerikli vision chat completion (OpenAI vision formatı).
        LM Studio'da Qwen3.5 gibi multimodal bir model yüklü olmalıdır.

        Args:
            text_prompt: Kullanıcı metin sorusu
            image_paths: Gönderilecek görsel dosyalarının yolları (PNG/JPG)
            system_prompt: Sistem mesajı (boş bırakılabilir)
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maksimum üretilecek token

        Returns:
            chat_completion() ile aynı format:
            {"answer": str, "generation_time": float, "input_tokens": int, ...}
        """
        # Her görseli yeniden boyutlandır (gerekirse) ve base64'e çevir
        # LM Studio büyük payload'ları reddedebilir; max 1024px kısa kenar.
        image_parts = []
        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.exists():
                continue
            b64 = self._image_to_base64(img_path, max_size=1024)
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        # Kullanıcı mesajı: önce görseller, sonra metin
        user_content: List[Dict] = image_parts + [{"type": "text", "text": text_prompt}]

        messages: List[Dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        return self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        OpenAI-compatible streaming chat completion.
        Token'ları tek tek yield eder.

        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Yields:
            Her token string olarak
        """
        kwargs.setdefault("stream", True)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

        except requests.exceptions.ReadTimeout as e:
            raise requests.exceptions.ReadTimeout(
                f"LM Studio sunucusu yanıt vermedi. Timeout: {self.timeout} saniye. "
                f"Sunucunun çalıştığından emin olun veya timeout süresini artırın."
            ) from e

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')

            if line.startswith('data: '):
                data_str = line[6:]

                if data_str == '[DONE]':
                    break

                try:
                    data = json.loads(data_str)
                    delta = data['choices'][0].get('delta', {})

                    if 'content' in delta:
                        yield delta['content']

                except json.JSONDecodeError:
                    continue
