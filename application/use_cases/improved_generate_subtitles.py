"""
Application Layer - 改进的字幕生成用例（带上下文和质量检查）
完整实现版本
"""
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path

from domain.entities import Video, Subtitle, TextSegment, TimeRange, LanguageCode
from domain.ports import ASRProvider, TranslationProvider, VideoProcessor, CacheRepository
from domain.services import calculate_cache_key


@dataclass(frozen=True)
class ImprovedSubtitleGenerationResult:
    """改进的字幕生成结果"""
    original_subtitle: Subtitle
    translated_subtitle: Subtitle
    detected_language: LanguageCode
    cache_hit: bool
    quality_report: Optional[object] = None  # TranslationQualityReport


def _apply_translation_context(
        translation_provider: TranslationProvider,
        context
):
    """
    应用翻译上下文到提供者

    如果翻译提供者支持，设置系统提示词和术语表
    """
    if context is None:
        return

    # 设置系统提示词
    if hasattr(translation_provider, 'set_system_prompt'):
        translation_provider.set_system_prompt(context.system_prompt)

    # 设置术语表
    if hasattr(translation_provider, 'set_terminology'):
        translation_provider.set_terminology(context.terminology)


def _translate_with_context(
        segments: tuple[TextSegment, ...],
        source_lang: LanguageCode,
        target_lang: LanguageCode,
        translation_provider: TranslationProvider,
        context: Optional[object]
) -> tuple[TextSegment, ...]:
    """
    使用上下文进行翻译

    在翻译前应用上下文配置
    """
    if context:
        _apply_translation_context(translation_provider, context)

    result = translation_provider.translate(segments, source_lang, target_lang)
    return result


def _reconstruct_segments_from_cache(
        cached_segments: list,
        language: LanguageCode
) -> tuple[TextSegment, ...]:
    """从缓存重建文本片段"""
    return tuple(
        TextSegment(
            text=seg["text"],
            time_range=TimeRange(seg["start"], seg["end"]),
            language=language
        )
        for seg in cached_segments
    )


def _serialize_segments(segments: tuple[TextSegment, ...]) -> list:
    """序列化文本片段为字典列表"""
    return [
        {
            "text": seg.text,
            "start": seg.time_range.start_seconds,
            "end": seg.time_range.end_seconds
        }
        for seg in segments
    ]


def _load_subtitles_from_cache(
        cache_repo: CacheRepository,
        cache_key: str
) -> Optional[ImprovedSubtitleGenerationResult]:
    """从缓存加载字幕"""
    if not cache_repo.exists(cache_key):
        return None

    try:
        cached = cache_repo.get(cache_key)
        detected_lang = LanguageCode(cached["detected_language"])

        # 加载原始语言字幕
        original_segments = _reconstruct_segments_from_cache(
            cached.get(f"{detected_lang.value}_segments", []),
            detected_lang
        )

        # 加载中文字幕
        zh_segments = _reconstruct_segments_from_cache(
            cached.get("zh_segments", []),
            LanguageCode.CHINESE
        )

        # 如果没有原始语言字幕，使用中文字幕
        if not original_segments:
            original_segments = zh_segments

        original_subtitle = Subtitle(original_segments, detected_lang)
        translated_subtitle = Subtitle(zh_segments, LanguageCode.CHINESE)

        return ImprovedSubtitleGenerationResult(
            original_subtitle=original_subtitle,
            translated_subtitle=translated_subtitle,
            detected_language=detected_lang,
            cache_hit=True,
            quality_report=None
        )

    except (KeyError, ValueError) as e:
        print(f"⚠️  缓存数据损坏: {e}")
        return None


def _translate_english_to_chinese(
        original_segments: tuple[TextSegment, ...],
        translation_provider: TranslationProvider,
        context: Optional[object]
) -> tuple[tuple[TextSegment, ...], tuple[TextSegment, ...]]:
    """英文翻译策略：en -> zh"""
    en_segments = original_segments
    zh_segments = _translate_with_context(
        original_segments,
        LanguageCode.ENGLISH,
        LanguageCode.CHINESE,
        translation_provider,
        context
    )
    translation_provider.unload()
    return en_segments, zh_segments


def _translate_chinese_to_english(
        original_segments: tuple[TextSegment, ...],
        translation_provider: TranslationProvider,
        context: Optional[object]
) -> tuple[tuple[TextSegment, ...], tuple[TextSegment, ...]]:
    """中文翻译策略：zh -> en"""
    zh_segments = original_segments
    en_segments = _translate_with_context(
        original_segments,
        LanguageCode.CHINESE,
        LanguageCode.ENGLISH,
        translation_provider,
        context
    )
    translation_provider.unload()
    return en_segments, zh_segments


def _translate_other_to_bilingual(
        original_segments: tuple[TextSegment, ...],
        detected_language: LanguageCode,
        translation_provider: TranslationProvider,
        context: Optional[object]
) -> tuple[tuple[TextSegment, ...], tuple[TextSegment, ...]]:
    """其他语言翻译策略：other -> en -> zh"""
    # 第一步：original -> en
    en_segments = _translate_with_context(
        original_segments,
        detected_language,
        LanguageCode.ENGLISH,
        translation_provider,
        context
    )

    if not en_segments:
        raise ValueError(f"第一步翻译失败！{detected_language.value} -> en 返回空结果")

    # 第二步：en -> zh
    zh_segments = _translate_with_context(
        en_segments,
        LanguageCode.ENGLISH,
        LanguageCode.CHINESE,
        translation_provider,
        context
    )
    translation_provider.unload()
    if not zh_segments:
        raise ValueError(f"第二步翻译失败！en -> zh 返回空结果")

    return en_segments, zh_segments


def _execute_translation_strategy(
        original_segments: tuple[TextSegment, ...],
        detected_language: LanguageCode,
        translation_provider: TranslationProvider,
        context: Optional[object],
        progress: Optional[Callable[[float, str], None]]
) -> tuple[tuple[TextSegment, ...], tuple[TextSegment, ...]]:
    """
    执行翻译策略（策略模式）

    根据检测的语言选择合适的翻译路径
    """
    if detected_language == LanguageCode.ENGLISH:
        if progress:
            progress(0.7, "翻译 英文 -> 中文")
        return _translate_english_to_chinese(
            original_segments, translation_provider, context
        )

    elif detected_language == LanguageCode.CHINESE:
        if progress:
            progress(0.7, "翻译 中文 -> 英文")
        return _translate_chinese_to_english(
            original_segments, translation_provider, context
        )

    else:
        if progress:
            progress(0.65, f"翻译 {detected_language.value} -> 英文 -> 中文")
        return _translate_other_to_bilingual(
            original_segments, detected_language, translation_provider, context
        )


def _validate_translation_results(
        en_segments: tuple,
        zh_segments: tuple,
        detected_language: LanguageCode
):
    """验证翻译结果"""
    if not en_segments:
        raise ValueError(f"en_segments 为空！检测语言: {detected_language.value}")
    if not zh_segments:
        raise ValueError(f"zh_segments 为空！检测语言: {detected_language.value}")


def _build_cache_data(
        source_language: Optional[LanguageCode],
        detected_language: LanguageCode,
        original_segments: tuple[TextSegment, ...],
        en_segments: tuple[TextSegment, ...],
        zh_segments: tuple[TextSegment, ...]
) -> dict:
    """构建缓存数据"""
    cache_data = {
        "source_language": source_language,
        "detected_language": detected_language.value,
        "zh_segments": _serialize_segments(zh_segments),
        "en_segments": _serialize_segments(en_segments)
    }

    # 如果原始语言不是中英文，也保存原始语言片段
    if detected_language not in [LanguageCode.CHINESE, LanguageCode.ENGLISH]:
        cache_data[f"{detected_language.value}_segments"] = _serialize_segments(original_segments)

    return cache_data


def improved_generate_subtitles_use_case(
        video: Video,
        asr_provider: ASRProvider,
        translation_provider: TranslationProvider,
        video_processor: VideoProcessor,
        cache_repo: CacheRepository,
        translation_context: Optional[object] = None,
        target_language: LanguageCode = LanguageCode.CHINESE,
        source_language: Optional[LanguageCode] = None,
        enable_quality_check: bool = True,
        progress: Optional[Callable[[float, str], None]] = None
) -> ImprovedSubtitleGenerationResult:
    """
    改进的字幕生成用例（纯函数）

    新特性：
    1. 支持翻译上下文（系统提示词 + 术语表）
    2. 可选的质量检查
    3. 更清晰的缓存键（包含上下文）

    Args:
        video: 视频对象
        asr_provider: ASR 提供者
        translation_provider: 翻译提供者
        video_processor: 视频处理器
        cache_repo: 缓存仓储
        translation_context: 翻译上下文（可选）
        target_language: 目标语言
        source_language: 源语言（可选，自动检测）
        enable_quality_check: 是否启用质量检查
        progress: 进度回调

    Returns:
        ImprovedSubtitleGenerationResult: 包含字幕和质量报告
    """
    if progress:
        progress(0.0, "开始生成字幕")

    # 1. 计算缓存键（包含上下文信息）
    cache_params = {
        "target_language": target_language.value,
        "source_language": source_language
    }

    if translation_context:
        cache_params["context_domain"] = translation_context.domain

    cache_key = calculate_cache_key(video.path, "subtitles_v2", cache_params)
    # 2. 尝试从缓存加载
    cached_result = _load_subtitles_from_cache(cache_repo, cache_key)
    if cached_result:
        if progress:
            progress(1.0, "字幕缓存命中")
        return cached_result

    # 3. 提取音频
    if progress:
        progress(0.1, "提取音频")
    audio_path = video_processor.extract_audio(video)

    # 4. ASR 识别
    if progress:
        progress(0.3, "语音识别中")
    original_segments, detected_language = asr_provider.transcribe(
        audio_path, source_language
    )
    asr_provider.unload()

    # 5. 执行翻译策略
    if progress:
        context_name = translation_context.domain if translation_context else "general"
        progress(0.6, f"翻译中（使用 {context_name} 上下文）")

    en_segments, zh_segments = _execute_translation_strategy(
        original_segments,
        detected_language,
        translation_provider,
        translation_context,
        progress
    )

    # 6. 验证翻译结果
    _validate_translation_results(en_segments, zh_segments, detected_language)

    # 7. 质量检查（可选）
    quality_report = None
    if enable_quality_check:
        if progress:
            progress(0.9, "检查翻译质量")

        from application.use_cases.check_translation_quality import (
            check_translation_quality_use_case
        )

        original_for_check = Subtitle(original_segments, detected_language)
        translated_for_check = Subtitle(zh_segments, LanguageCode.CHINESE)

        terminology = translation_context.terminology if translation_context else {}

        quality_report = check_translation_quality_use_case(
            original_for_check,
            translated_for_check,
            translation_provider,
            terminology=terminology,
            progress=None
        )

    # 8. 保存缓存
    cache_data = _build_cache_data(source_language,
                                   detected_language, original_segments, en_segments, zh_segments
                                   )
    cache_repo.set(cache_key, cache_data)

    if progress:
        progress(1.0, "字幕生成完成")

    return ImprovedSubtitleGenerationResult(
        original_subtitle=Subtitle(original_segments, detected_language),
        translated_subtitle=Subtitle(zh_segments, LanguageCode.CHINESE),
        detected_language=detected_language,
        cache_hit=False,
        quality_report=quality_report
    )
