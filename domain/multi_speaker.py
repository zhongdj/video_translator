"""
Domain Layer - 多说话人支持
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from domain.value_objects import LanguageCode


@dataclass(frozen=True)
class SpeakerId:
    """说话人标识（值对象）"""
    id: str  # 唯一标识，如 "speaker_1", "speaker_2"
    name: str  # 显示名称，如 "主讲人", "旁白"

    def __post_init__(self):
        if not self.id or not self.id.strip():
            raise ValueError("speaker id cannot be empty")


@dataclass(frozen=True)
class MultiSpeakerVoiceProfile:
    """多说话人声音配置（不可变）"""
    speaker_id: SpeakerId
    reference_audio_path: Path
    language: LanguageCode
    duration: float

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if not self.reference_audio_path.exists():
            raise ValueError(f"reference audio not found: {self.reference_audio_path}")


@dataclass(frozen=True)
class SegmentSpeakerAssignment:
    """片段-说话人分配（值对象）"""
    segment_index: int
    speaker_id: SpeakerId

    def __post_init__(self):
        if self.segment_index < 0:
            raise ValueError("segment_index must be >= 0")


@dataclass(frozen=True)
class MultiSpeakerConfig:
    """多说话人配置（不可变）"""
    voice_profiles: tuple[MultiSpeakerVoiceProfile, ...]
    segment_assignments: tuple[SegmentSpeakerAssignment, ...]
    default_speaker_id: SpeakerId  # 未分配片段的默认说话人

    def __post_init__(self):
        if not self.voice_profiles:
            raise ValueError("at least one voice profile required")

        # 验证 default_speaker_id 存在于 voice_profiles 中
        speaker_ids = {vp.speaker_id.id for vp in self.voice_profiles}
        if self.default_speaker_id.id not in speaker_ids:
            raise ValueError(f"default speaker {self.default_speaker_id.id} not in voice profiles")

        # 验证所有分配的 speaker_id 都有对应的 voice_profile
        for assignment in self.segment_assignments:
            if assignment.speaker_id.id not in speaker_ids:
                raise ValueError(
                    f"segment {assignment.segment_index} assigned to unknown speaker {assignment.speaker_id.id}"
                )

    def get_speaker_for_segment(self, segment_index: int) -> SpeakerId:
        """获取指定片段的说话人"""
        for assignment in self.segment_assignments:
            if assignment.segment_index == segment_index:
                return assignment.speaker_id
        return self.default_speaker_id

    def get_voice_profile(self, speaker_id: SpeakerId) -> Optional[MultiSpeakerVoiceProfile]:
        """获取指定说话人的声音配置"""
        for vp in self.voice_profiles:
            if vp.speaker_id.id == speaker_id.id:
                return vp
        return None