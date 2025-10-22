from pathlib import Path
from typing import Optional

from domain.value_objects import *


@dataclass(frozen=True)
class Subtitle:
    """字幕实体（不可变）"""
    segments: tuple[TextSegment, ...]  # 使用 tuple 保证不可变
    language: LanguageCode
    path: Optional[Path] = None

    def __post_init__(self):
        if not self.segments:
            raise ValueError("segments cannot be empty")

    def with_path(self, path: Path) -> 'Subtitle':
        """返回带有新路径的 Subtitle 副本"""
        return Subtitle(
            segments=self.segments,
            language=self.language,
            path=path
        )

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


"""
Domain Layer - 扩展实体定义
支持分段语音克隆和增量合成
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from domain.value_objects import *


@dataclass(frozen=True)
class AudioSegment:
    """音频片段实体（不可变）"""
    segment_index: int
    audio: AudioSample
    text_segment: TextSegment
    cache_key: str
    file_path: Optional[Path] = None  # 缓存文件路径

    def with_file_path(self, path: Path) -> 'AudioSegment':
        """返回带文件路径的新实例"""
        return AudioSegment(
            segment_index=self.segment_index,
            audio=self.audio,
            text_segment=self.text_segment,
            cache_key=self.cache_key,
            file_path=path
        )


@dataclass(frozen=True)
class SegmentReviewStatus:
    """片段审核状态（不可变）"""
    segment_index: int
    subtitle_approved: bool  # 字幕是否已审核通过
    audio_approved: bool  # 音频是否已审核通过
    subtitle_modified: bool  # 字幕是否被修改过
    needs_regeneration: bool  # 是否需要重新生成音频

    def mark_subtitle_modified(self) -> 'SegmentReviewStatus':
        """标记字幕已修改"""
        return SegmentReviewStatus(
            segment_index=self.segment_index,
            subtitle_approved=False,
            audio_approved=False,
            subtitle_modified=True,
            needs_regeneration=True
        )

    def mark_approved(self) -> 'SegmentReviewStatus':
        """标记已审核通过"""
        return SegmentReviewStatus(
            segment_index=self.segment_index,
            subtitle_approved=True,
            audio_approved=True,
            subtitle_modified=self.subtitle_modified,
            needs_regeneration=False
        )


@dataclass(frozen=True)
class IncrementalSynthesisResult:
    """增量合成结果（不可变）"""
    total_segments: int
    cached_segments: int  # 使用缓存的片段数
    regenerated_segments: int  # 重新生成的片段数
    audio_segments: tuple[AudioSegment, ...]
    synthesis_time: float