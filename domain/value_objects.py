"""
Domain Layer - 核心领域模型
纯业务逻辑，无副作用，不依赖外层
"""
from dataclasses import dataclass
from enum import Enum


# ============== Value Objects (值对象) ============== #

@dataclass(frozen=True)
class TimeRange:
    """时间范围值对象（不可变）"""
    start_seconds: float
    end_seconds: float

    def __post_init__(self):
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be >= 0")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be > start_seconds")

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


class LanguageCode(str, Enum):
    """语言代码枚举"""
    CHINESE = "zh"
    ENGLISH = "en"
    PORTUGUESE = "pt"
    JAPANESE = "ja"
    KOREAN = "ko"
    AUTO = "auto"


@dataclass(frozen=True)
class TextSegment:
    """文本片段值对象"""
    text: str
    time_range: TimeRange
    language: LanguageCode

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("text cannot be empty")


@dataclass(frozen=True)
class AudioSample:
    """音频样本值对象（不可变）"""
    samples: tuple  # 使用 tuple 保证不可变
    sample_rate: int

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not self.samples:
            raise ValueError("samples cannot be empty")

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate