from pathlib import Path

from domain.ports import TranslationProvider, TTSProvider
from infrastructure.adapters.asr.WhisperASRAdapter import WhisperASRAdapter
from infrastructure.adapters.storage.FileCacheRepositoryAdapter import FileCacheRepositoryAdapter
from infrastructure.adapters.storage.TranslationContextRepository import TranslationContextRepositoryAdapter
from infrastructure.adapters.subtitle.PySRTSubtitleWriterAdapter import PySRTSubtitleWriterAdapter
from infrastructure.adapters.translation.enhanced_translation_adapter import create_enhanced_translation_provider
from infrastructure.adapters.tts.indextts_adapter import IndexTTSAdapter
from infrastructure.adapters.video.FFmpegVideoProcessorAdapter import FFmpegVideoProcessorAdapter


class DependencyContainer:
    """依赖注入容器"""

    def __init__(self):
        self.cache_repo = FileCacheRepositoryAdapter()
        self.video_processor = FFmpegVideoProcessorAdapter()
        self.subtitle_writer = PySRTSubtitleWriterAdapter()
        self.translator_context_repo = TranslationContextRepositoryAdapter(Path("./translation_contexts"))

        # 懒加载的模型
        self._asr = None
        self._translator = None
        self._tts = None

    def get_asr(self, model_size: str = "large-v3", device: str = "cuda") -> WhisperASRAdapter:
        """获取 ASR 提供者（懒加载）"""
        if self._asr is None or getattr(self._asr, 'model_size', None) != model_size:
            if self._asr is not None:
                self._asr.unload()
            self._asr = WhisperASRAdapter(model_size=model_size,device=device)
        return self._asr

    # def get_translator(self, model_name: str = "Qwen/Qwen2.5-7B") -> QwenTranslationAdapter:
    #     """获取翻译提供者（懒加载）"""
    #     if self._translator is None:
    #         self._translator = QwenTranslationAdapter(model_name=model_name)
    #     return self._translator

    def get_translator(self) -> TranslationProvider:
        """获取增强的翻译提供者"""
        if self._translator is None:
            self._translator = create_enhanced_translation_provider()
        return self._translator

    def get_tts(self) -> TTSProvider:
        """获取 TTS 提供者（懒加载）"""
        if self._tts is None:
            self._tts = IndexTTSAdapter()
        return self._tts

    def cleanup(self):
        """清理所有资源"""
        if self._asr:
            self._asr.unload()
        if self._translator:
            self._translator.unload()
        if self._tts:
            self._tts.unload()

# 全局容器
container = DependencyContainer()