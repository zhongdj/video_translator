"""
生成字幕用例 - 重构版

重构要点：
1. 提取缓存加载逻辑
2. 提取翻译策略
3. 简化主流程
4. 移除冗余日志
"""

from application import *
from domain.entities import (
    Video, Subtitle, TextSegment,
    TimeRange, LanguageCode,
)
from domain.ports import (
    ASRProvider, TranslationProvider, VideoProcessor, CacheRepository,
)
from domain.services import calculate_cache_key


# ============== 缓存辅助函数 ============== #

def _reconstruct_segments_from_cache(cached_segments: list, language: LanguageCode) -> tuple[TextSegment, ...]:
    """从缓存重建文本片段"""
    return tuple(
        TextSegment(
            text=seg["text"],
            time_range=TimeRange(seg["start"], seg["end"]),
            language=language
        )
        for seg in cached_segments
    )


def _load_subtitles_from_cache(cache_repo: CacheRepository, cache_key: str) -> Optional[SubtitleGenerationResult]:
    """从缓存加载字幕"""
    if not cache_repo.exists(cache_key):
        return None

    cached = cache_repo.get(cache_key)
    detected_lang = LanguageCode(cached["detected_language"])

    original_segments = _reconstruct_segments_from_cache(
        cached.get(f"{detected_lang.value}_segments", []),
        detected_lang
    )

    zh_segments = _reconstruct_segments_from_cache(
        cached.get("zh_segments", []),
        LanguageCode.CHINESE
    )

    original_subtitle = Subtitle(original_segments, detected_lang) if original_segments else Subtitle(
        zh_segments, detected_lang
    )
    translated_subtitle = Subtitle(zh_segments, LanguageCode.CHINESE)

    return SubtitleGenerationResult(
        original_subtitle=original_subtitle,
        translated_subtitle=translated_subtitle,
        detected_language=detected_lang,
        cache_hit=True
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


def _build_cache_data(
        detected_language: LanguageCode,
        original_segments: tuple[TextSegment, ...],
        en_segments: tuple[TextSegment, ...],
        zh_segments: tuple[TextSegment, ...]
) -> dict:
    """构建缓存数据"""
    cache_data = {
        "detected_language": detected_language.value,
        "zh_segments": _serialize_segments(zh_segments),
        "en_segments": _serialize_segments(en_segments)
    }

    # 如果原始语言不是中英文，也保存原始语言片段
    if detected_language not in [LanguageCode.CHINESE, LanguageCode.ENGLISH]:
        cache_data[f"{detected_language.value}_segments"] = _serialize_segments(original_segments)

    return cache_data


def _validate_cache_data(cache_data: dict):
    """验证缓存数据完整性"""
    if not cache_data.get('en_segments'):
        raise ValueError("缓存数据验证失败：en_segments 为空")
    if not cache_data.get('zh_segments'):
        raise ValueError("缓存数据验证失败：zh_segments 为空")


# ============== 翻译策略函数 ============== #

def _translate_english_to_chinese(
        original_segments: tuple[TextSegment, ...],
        translation_provider: TranslationProvider
) -> tuple[tuple[TextSegment, ...], tuple[TextSegment, ...]]:
    """英文翻译策略：en -> zh"""
    en_segments = original_segments
    zh_segments = translation_provider.translate(
        original_segments,
        LanguageCode.ENGLISH,
        LanguageCode.CHINESE
    )
    return en_segments, zh_segments


def _translate_chinese_to_english(
        original_segments: tuple[TextSegment, ...],
        translation_provider: TranslationProvider
) -> tuple[tuple[TextSegment, ...], tuple[TextSegment, ...]]:
    """中文翻译策略：zh -> en"""
    zh_segments = original_segments
    en_segments = translation_provider.translate(
        original_segments,
        LanguageCode.CHINESE,
        LanguageCode.ENGLISH
    )
    return en_segments, zh_segments


def _translate_other_to_bilingual(
        original_segments: tuple[TextSegment, ...],
        detected_language: LanguageCode,
        translation_provider: TranslationProvider
) -> tuple[tuple[TextSegment, ...], tuple[TextSegment, ...]]:
    """其他语言翻译策略：other -> en -> zh"""
    # 第一步：original -> en
    en_segments = translation_provider.translate(
        original_segments,
        detected_language,
        LanguageCode.ENGLISH
    )

    if not en_segments:
        raise ValueError(f"第一步翻译失败！{detected_language.value} -> en 返回空结果")

    # 第二步：en -> zh
    zh_segments = translation_provider.translate(
        en_segments,
        LanguageCode.ENGLISH,
        LanguageCode.CHINESE
    )

    if not zh_segments:
        raise ValueError(f"第二步翻译失败！en -> zh 返回空结果")

    return en_segments, zh_segments


def _execute_translation_strategy(
        original_segments: tuple[TextSegment, ...],
        detected_language: LanguageCode,
        translation_provider: TranslationProvider,
        progress: ProgressCallback
) -> tuple[tuple[TextSegment, ...], tuple[TextSegment, ...]]:
    """执行翻译策略（策略模式）"""
    if detected_language == LanguageCode.ENGLISH:
        if progress:
            progress(0.7, "翻译 英文 -> 中文")
        return _translate_english_to_chinese(original_segments, translation_provider)

    elif detected_language == LanguageCode.CHINESE:
        if progress:
            progress(0.7, "翻译 中文 -> 英文")
        return _translate_chinese_to_english(original_segments, translation_provider)

    else:
        if progress:
            progress(0.65, f"翻译 {detected_language.value} -> 英文")
        return _translate_other_to_bilingual(original_segments, detected_language, translation_provider)


def _validate_translation_results(en_segments: tuple, zh_segments: tuple, detected_language: LanguageCode):
    """验证翻译结果"""
    if not en_segments:
        raise ValueError(f"en_segments 为空！检测语言: {detected_language.value}")
    if not zh_segments:
        raise ValueError(f"zh_segments 为空！检测语言: {detected_language.value}")


# ============== 主用例函数 ============== #

def generate_subtitles_use_case(
        video: Video,
        asr_provider: ASRProvider,
        translation_provider: TranslationProvider,
        video_processor: VideoProcessor,
        cache_repo: CacheRepository,
        target_language: LanguageCode = LanguageCode.CHINESE,
        source_language: Optional[LanguageCode] = None,
        progress: ProgressCallback = None
) -> SubtitleGenerationResult:
    """
    生成字幕用例（重构版）

    流程简化：
    1. 检查缓存
    2. ASR 识别
    3. 智能翻译（策略模式）
    4. 保存缓存
    """
    if progress:
        progress(0.0, "开始生成字幕")

    # 1. 计算缓存键
    cache_key = calculate_cache_key(
        video.path,
        "subtitles",
        {
            "target_language": target_language.value,
            "source_language": source_language.value if source_language else "auto"
        }
    )

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
    original_segments, detected_language = asr_provider.transcribe(audio_path, source_language)

    # 5. 执行翻译策略
    if progress:
        progress(0.6, "翻译中")

    en_segments, zh_segments = _execute_translation_strategy(
        original_segments,
        detected_language,
        translation_provider,
        progress
    )

    # 6. 验证翻译结果
    _validate_translation_results(en_segments, zh_segments, detected_language)

    # 7. 保存缓存
    cache_data = _build_cache_data(detected_language, original_segments, en_segments, zh_segments)
    _validate_cache_data(cache_data)
    cache_repo.set(cache_key, cache_data)

    if progress:
        progress(1.0, "字幕生成完成")

    return SubtitleGenerationResult(
        original_subtitle=Subtitle(original_segments, detected_language),
        translated_subtitle=Subtitle(zh_segments, LanguageCode.CHINESE),
        detected_language=detected_language,
        cache_hit=False
    )