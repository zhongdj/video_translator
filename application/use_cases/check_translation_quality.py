"""
Application Layer - 翻译质量检查用例
完整实现版本
"""
from dataclasses import dataclass
from typing import Optional, Callable

from domain.entities import Subtitle, TextSegment, LanguageCode
from domain.ports import TranslationProvider


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


@dataclass(frozen=True)
class TranslationQualityReport:
    """翻译质量报告"""
    total_segments: int
    issues_found: int
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    issues: tuple[TranslationQualityIssue, ...]
    overall_quality: str  # "excellent", "good", "fair", "poor"
    requires_review: bool


def _assess_overall_quality(
        total_segments: int,
        high_severity: int,
        medium_severity: int
) -> tuple[str, bool]:
    """
    评估整体质量

    Returns:
        (quality_level, requires_review)
    """
    if high_severity > 0:
        return "poor", True

    issue_ratio = medium_severity / total_segments if total_segments > 0 else 0

    if issue_ratio > 0.3:
        return "fair", True
    elif issue_ratio > 0.1:
        return "good", True
    else:
        return "excellent", False


def _check_terminology(
        segment: TextSegment,
        original_segment: TextSegment,
        index: int,
        terminology: dict[str, str]
) -> list[TranslationQualityIssue]:
    """检查术语使用"""
    issues = []

    for source_term, target_term in terminology.items():
        if source_term.lower() in original_segment.text.lower():
            if target_term not in segment.text:
                issues.append(TranslationQualityIssue(
                    segment_index=index,
                    issue_type="terminology",
                    severity="high",
                    description=f"专业术语 '{source_term}' 应翻译为 '{target_term}'",
                    original_text=original_segment.text,
                    translated_text=segment.text,
                    suggestion=f"建议使用术语 '{target_term}'"
                ))

    return issues


def _check_length_anomaly(
        segment: TextSegment,
        original_segment: TextSegment,
        index: int
) -> list[TranslationQualityIssue]:
    """检查长度异常"""
    issues = []

    orig_len = len(original_segment.text)
    trans_len = len(segment.text)

    if orig_len == 0:
        return issues

    length_ratio = trans_len / orig_len

    # 译文过长（可能冗余）
    if length_ratio > 2.5:
        issues.append(TranslationQualityIssue(
            segment_index=index,
            issue_type="fluency",
            severity="medium",
            description="译文过长，可能存在冗余表达",
            original_text=original_segment.text,
            translated_text=segment.text,
            suggestion="建议简化表达，使用更简洁的译文"
        ))

    # 译文过短（可能遗漏信息）
    elif length_ratio < 0.3 and orig_len > 10:
        issues.append(TranslationQualityIssue(
            segment_index=index,
            issue_type="accuracy",
            severity="high",
            description="译文过短，可能遗漏重要信息",
            original_text=original_segment.text,
            translated_text=segment.text,
            suggestion="请检查是否完整翻译了原文内容"
        ))

    return issues


def _check_empty_translation(
        segment: TextSegment,
        original_segment: TextSegment,
        index: int
) -> list[TranslationQualityIssue]:
    """检查空白翻译"""
    issues = []

    if not segment.text.strip() and original_segment.text.strip():
        issues.append(TranslationQualityIssue(
            segment_index=index,
            issue_type="accuracy",
            severity="high",
            description="翻译结果为空",
            original_text=original_segment.text,
            translated_text=segment.text,
            suggestion="需要重新翻译此片段"
        ))

    return issues


def _check_segment_quality(
        segment: TextSegment,
        original_segment: TextSegment,
        index: int,
        terminology: dict[str, str]
) -> list[TranslationQualityIssue]:
    """
    检查单个片段的翻译质量

    综合多个检查维度
    """
    all_issues = []

    # 1. 术语使用检查
    all_issues.extend(_check_terminology(
        segment, original_segment, index, terminology
    ))

    # 2. 长度异常检查
    all_issues.extend(_check_length_anomaly(
        segment, original_segment, index
    ))

    # 3. 空白翻译检查
    all_issues.extend(_check_empty_translation(
        segment, original_segment, index
    ))

    return all_issues


def check_translation_quality_use_case(
        original_subtitle: Subtitle,
        translated_subtitle: Subtitle,
        translation_provider: Optional[TranslationProvider] = None,
        terminology: dict[str, str] = None,
        progress: Optional[Callable[[float, str], None]] = None
) -> TranslationQualityReport:
    """
    检查翻译质量用例（纯函数）

    Args:
        original_subtitle: 原始字幕
        translated_subtitle: 翻译后的字幕
        translation_provider: 翻译提供者（保留用于未来扩展）
        terminology: 术语表
        progress: 进度回调

    Returns:
        TranslationQualityReport: 质量报告

    流程：
    1. 逐句对比原文和译文
    2. 检查术语使用
    3. 检查流畅度
    4. 检查准确性
    5. 生成质量报告
    """
    if progress:
        progress(0.0, "开始质量检查")

    terminology = terminology or {}
    all_issues = []

    total = len(original_subtitle.segments)

    # 检查片段数量是否匹配
    if len(translated_subtitle.segments) != total:
        all_issues.append(TranslationQualityIssue(
            segment_index=-1,
            issue_type="accuracy",
            severity="high",
            description=f"字幕片段数量不匹配：原文{total}个，译文{len(translated_subtitle.segments)}个",
            original_text="",
            translated_text="",
            suggestion="请检查翻译过程是否完整"
        ))

        # 如果数量不匹配，提前返回
        return TranslationQualityReport(
            total_segments=total,
            issues_found=1,
            high_severity_count=1,
            medium_severity_count=0,
            low_severity_count=0,
            issues=tuple(all_issues),
            overall_quality="poor",
            requires_review=True
        )

    # 逐个检查片段
    for idx, (orig_seg, trans_seg) in enumerate(
            zip(original_subtitle.segments, translated_subtitle.segments)
    ):
        if progress:
            progress(idx / total, f"检查片段 {idx + 1}/{total}")

        segment_issues = _check_segment_quality(
            trans_seg, orig_seg, idx, terminology
        )
        all_issues.extend(segment_issues)

    # 统计问题严重度
    high_severity = sum(1 for i in all_issues if i.severity == "high")
    medium_severity = sum(1 for i in all_issues if i.severity == "medium")
    low_severity = sum(1 for i in all_issues if i.severity == "low")

    # 评估整体质量
    overall_quality, requires_review = _assess_overall_quality(
        total, high_severity, medium_severity
    )

    if progress:
        progress(1.0, "质量检查完成")

    return TranslationQualityReport(
        total_segments=total,
        issues_found=len(all_issues),
        high_severity_count=high_severity,
        medium_severity_count=medium_severity,
        low_severity_count=low_severity,
        issues=tuple(all_issues),
        overall_quality=overall_quality,
        requires_review=requires_review
    )