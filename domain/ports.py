# ============== Ports (接口定义) ============== #
from abc import ABC, abstractmethod
from typing import Protocol, Optional

from domain.entities import *


class ASRProvider(Protocol):
    """自动语音识别提供者接口"""

    def transcribe(
            self,
            audio_path: Path,
            language: Optional[LanguageCode] = None
    ) -> tuple[tuple[TextSegment, ...], LanguageCode]:
        """
        转录音频为文本
        Returns: (segments, detected_language)
        """
        ...

    def unload(self):
        """
        释放内存
        :return:
        """

class TranslationProvider(Protocol):
    """翻译提供者接口"""

    def translate(
            self,
            segments: tuple[TextSegment, ...],
            source_lang: LanguageCode,
            target_lang: LanguageCode
    ) -> tuple[TextSegment, ...]:
        """翻译文本片段"""
        ...


class TTSProvider(ABC):
    """TTS 提供者接口"""

    @abstractmethod
    def synthesize(
            self,
            text: str,
            voice_profile: VoiceProfile,
            target_duration: Optional[float] = None
    ) -> AudioSample:
        """
        单句合成

        Args:
            text: 待合成文本
            voice_profile: 声音配置
            target_duration: 目标时长

        Returns:
            合成的音频
        """
        pass

    @abstractmethod
    def batch_synthesize(
            self,
            texts: list[str],
            reference_audio_path: Path,
            language: LanguageCode
    ) -> tuple[AudioSample, ...]:
        """
        批量合成（同一说话人）

        关键优化：
        1. 说话人条件只提取一次
        2. 所有文本批量推理
        3. 减少 GPU 上下文切换

        Args:
            texts: 待合成文本列表
            reference_audio_path: 参考音频路径
            language: 目标语言

        Returns:
            合成的音频列表（顺序与输入一致）
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """加载模型"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """卸载模型"""
        pass


class VideoProcessor(Protocol):
    """视频处理器接口"""

    def extract_audio(self, video: Video) -> Path:
        """从视频提取音频"""
        ...

    def extract_reference_audio(
            self,
            video: Video,
            duration: float
    ) -> Path:
        """提取参考音频片段"""
        ...

    def merge_audio_video(
            self,
            video: Video,
            audio_track: AudioTrack,
            output_path: Path
    ) -> Path:
        """合并音频和视频"""
        ...

    def burn_subtitles(
            self,
            video: Video,
            subtitle: Subtitle,
            output_path: Path
    ) -> Path:
        """烧录字幕到视频"""
        ...


class SubtitleWriter(Protocol):
    """字幕写入器接口"""

    def write_srt(self, subtitle: Subtitle, output_path: Path) -> Path:
        """写入 SRT 格式字幕"""
        ...

    def write_ass(self, subtitle: Subtitle, output_path: Path) -> Path:
        """写入 ASS 格式字幕"""
        ...


class CacheRepository(Protocol):
    """缓存仓储接口"""

    def get(self, key: str) -> Optional[dict]:
        """获取缓存"""
        ...

    def set(self, key: str, value: dict) -> None:
        """设置缓存"""
        ...

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        ...


class FileStorage(Protocol):
    """文件存储接口"""

    def save(self, data: bytes, path: Path) -> Path:
        """保存文件"""
        ...

    def load(self, path: Path) -> bytes:
        """加载文件"""
        ...

    def exists(self, path: Path) -> bool:
        """检查文件是否存在"""
        ...