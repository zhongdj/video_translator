from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json


@dataclass(frozen=True)
class TranslationContext:
    """翻译上下文配置（不可变）"""
    domain: str  # 领域名称，如 "inline_skating"
    system_prompt: str  # 系统提示词
    terminology: dict[str, str]  # 术语表 {source: target}

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "domain": self.domain,
            "system_prompt": self.system_prompt,
            "terminology": self.terminology
        }

    @staticmethod
    def from_dict(data: dict) -> 'TranslationContext':
        """从字典创建"""
        return TranslationContext(
            domain=data["domain"],
            system_prompt=data["system_prompt"],
            terminology=data.get("terminology", {})
        )


@dataclass(frozen=True)
class TranslationQualityIssue:
    """翻译质量问题（不可变）"""
    segment_index: int
    issue_type: str  # "terminology", "fluency", "accuracy", "context"
    severity: str  # "low", "medium", "high"
    description: str
    original_text: str
    translated_text: str
    suggestion: Optional[str] = None