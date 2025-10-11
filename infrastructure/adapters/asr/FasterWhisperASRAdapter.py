from pathlib import Path
from typing import Optional

import torch

# 导入领域层
from domain.entities import (
    # Entities
    TextSegment,
    # Value Objects
    TimeRange, LanguageCode, )


class FasterWhisperASRAdapter:
    """Faster-Whisper ASR 适配器（更快的实现）"""

    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _load_model(self):
        """懒加载模型"""
        if self._model is None:
            from faster_whisper import WhisperModel
            print(f"🔄 加载 Faster-Whisper 模型: {self.model_size}")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if torch.cuda.is_available() else "int8"
            )
        return self._model

    def transcribe(
            self,
            audio_path: Path,
            language: Optional[LanguageCode] = None
    ) -> tuple[tuple[TextSegment, ...], LanguageCode]:
        """实现 ASRProvider 接口"""
        model = self._load_model()

        lang_code = language.value if language and language != LanguageCode.AUTO else None

        segments_gen, info = model.transcribe(
            str(audio_path),
            language=lang_code,
            beam_size=1,
            best_of=1,
            vad_filter=True
        )

        # 转换为领域对象
        segments = tuple(
            TextSegment(
                text=seg.text.strip(),
                time_range=TimeRange(seg.start, seg.end),
                language=LanguageCode(info.language)
            )
            for seg in segments_gen
        )

        detected_language = LanguageCode(info.language)

        return segments, detected_language

    def unload(self):
        """卸载模型"""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 Faster-Whisper 模型已卸载")