"""
Application Layer - 应用层用例
纯函数编排，无副作用，依赖注入接口
"""
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path

# 导入领域层
from domain.entities import (
    # Entities
    Video, Subtitle, AudioTrack, VoiceProfile, ProcessedVideo, TextSegment,
    # Value Objects
    TimeRange, LanguageCode, AudioSample,
)

from domain.ports import (
# Ports
    ASRProvider, TranslationProvider, TTSProvider,
    VideoProcessor, SubtitleWriter, CacheRepository,

)

from domain.services import (
    # Domain Services
    merge_bilingual_subtitles, calculate_speed_adjustment,
    calculate_cache_key, split_audio_by_segments,
)

# ============== Use Case Results (用例结果) ============== #

@dataclass(frozen=True)
class SubtitleGenerationResult:
    """字幕生成结果"""
    original_subtitle: Subtitle
    translated_subtitle: Subtitle
    detected_language: LanguageCode
    cache_hit: bool


@dataclass(frozen=True)
class VoiceCloningResult:
    """语音克隆结果"""
    audio_track: AudioTrack
    voice_profile: VoiceProfile
    total_segments: int
    cache_hit: bool


@dataclass(frozen=True)
class VideoSynthesisResult:
    """视频合成结果"""
    output_paths: tuple[Path, ...]
    processing_time: float

# ============== Progress Callback Type ============== #

ProgressCallback = Optional[Callable[[float, str], None]]