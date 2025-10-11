from pathlib import Path

from domain.value_objects import *


@dataclass(frozen=True)
class Subtitle:
    """字幕实体（不可变）"""
    segments: tuple[TextSegment, ...]  # 使用 tuple 保证不可变
    language: LanguageCode

    def __post_init__(self):
        if not self.segments:
            raise ValueError("segments cannot be empty")


@dataclass(frozen=True)
class AudioTrack:
    """音轨实体（不可变）"""
    audio: AudioSample
    language: LanguageCode


@dataclass(frozen=True)
class VoiceProfile:
    """声音配置文件实体（不可变）"""
    reference_audio_path: Path
    language: LanguageCode
    duration: float

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError("duration must be positive")


@dataclass(frozen=True)
class Video:
    """视频实体（不可变）"""
    path: Path
    duration: float
    has_audio: bool


@dataclass(frozen=True)
class ProcessedVideo:
    """处理后的视频实体（不可变）"""
    original_video: Video
    subtitles: tuple[Subtitle, ...]
    audio_tracks: tuple[AudioTrack, ...]
    output_paths: tuple[Path, ...]