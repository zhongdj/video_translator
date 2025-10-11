from pathlib import Path
from typing import Optional

import torch
import whisper

# 导入领域层
from domain.entities import (
    # Entities
    TextSegment,
    # Value Objects
    TimeRange, LanguageCode, )


class WhisperASRAdapter:
    """Whisper ASR 适配器"""

    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):  # 改为可选
        self.model_size = model_size
        # 自动检测设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model = None
        print(f"🎯 ASR 设备: {self.device}")

    def _load_model(self):
        """带错误处理的模型加载"""
        if self._model is None:
            try:
                print(f"🔄 加载 Whisper 模型: {self.model_size} -> {self.device}")
                self._model = whisper.load_model(self.model_size, device=self.device)
                print("✅ 模型加载成功")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                # 降级到 CPU
                self.device = "cpu"
                print(f"🔄 尝试使用 CPU 加载模型...")
                self._model = whisper.load_model(self.model_size, device="cpu")
        return self._model

    def transcribe(
            self,
            audio_path: Path,
            language: Optional[LanguageCode] = None
    ) -> tuple[tuple[TextSegment, ...], LanguageCode]:
        """实现 ASRProvider 接口"""

        # 每次重新加载模型（模仿原项目）
        print(f"🔄 加载 Whisper 模型: {self.model_size}")
        model = whisper.load_model(self.model_size, device=self.device)

        # 转录参数
        options = {
            "fp16": torch.cuda.is_available(),
            "beam_size": 1,
            "best_of": 1
        }

        if language and language != LanguageCode.AUTO:
            options["language"] = language.value

        # 执行转录
        result = model.transcribe(str(audio_path), **options)

        # 转换为领域对象
        segments = tuple(
            TextSegment(
                text=seg["text"].strip(),
                time_range=TimeRange(seg["start"], seg["end"]),
                language=LanguageCode(result["language"])
            )
            for seg in result["segments"]
        )

        detected_language = LanguageCode(result["language"])

        # 模仿原项目的清理逻辑
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return segments, detected_language

    def unload(self):
        """卸载模型释放资源"""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 Whisper 模型已卸载")