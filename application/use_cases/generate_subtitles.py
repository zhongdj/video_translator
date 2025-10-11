from application import *
from domain.entities import (
    Video, Subtitle, TextSegment,
    TimeRange, LanguageCode,
)
from domain.ports import (
    ASRProvider, TranslationProvider, VideoProcessor, CacheRepository,
)
from domain.services import calculate_cache_key


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
    生成字幕用例（修复版）- 支持多级翻译

    流程:
    1. 检查缓存
    2. 提取音频
    3. ASR 识别
    4. 智能翻译（非中英文先翻译成英文，再翻译成中文）
    5. 返回结果
    """
    if progress:
        progress(0.0, "开始生成字幕")

    # 1. 检查缓存
    cache_key = calculate_cache_key(
        video.path,
        "subtitles",
        {
            "target_language": target_language.value,
            "source_language": source_language.value if source_language else "auto"
        }
    )

    if cache_repo.exists(cache_key):
        cached = cache_repo.get(cache_key)
        if progress:
            progress(1.0, "字幕缓存命中")

        # 从缓存重建对象
        original_segments = tuple(
            TextSegment(
                text=seg["text"],
                time_range=TimeRange(seg["start"], seg["end"]),
                language=LanguageCode(cached["detected_language"])
            )
            for seg in cached["original_segments"]
        )

        translated_segments = tuple(
            TextSegment(
                text=seg["text"],
                time_range=TimeRange(seg["start"], seg["end"]),
                language=target_language
            )
            for seg in cached["translated_segments"]
        )

        return SubtitleGenerationResult(
            original_subtitle=Subtitle(original_segments, LanguageCode(cached["detected_language"])),
            translated_subtitle=Subtitle(translated_segments, target_language),
            detected_language=LanguageCode(cached["detected_language"]),
            cache_hit=True
        )

    # 2. 提取音频
    if progress:
        progress(0.1, "提取音频")
    audio_path = video_processor.extract_audio(video)

    # 3. ASR 识别
    if progress:
        progress(0.3, "语音识别中")
    original_segments, detected_language = asr_provider.transcribe(
        audio_path,
        source_language
    )

    # 4. 智能翻译流程
    if progress:
        progress(0.6, "翻译中")

    def needs_two_step_translation(source_lang: LanguageCode, target_lang: LanguageCode) -> bool:
        """判断是否需要两步翻译"""
        # 如果源语言和目标语言都是中英文之一，直接翻译
        common_langs = {LanguageCode.ENGLISH, LanguageCode.CHINESE}
        if source_lang in common_langs and target_lang in common_langs:
            return False
        # 如果源语言不是中英文，目标语言是中文，需要两步翻译
        if source_lang not in common_langs and target_lang == LanguageCode.CHINESE:
            return True
        # 如果源语言不是中英文，目标语言是英文，直接翻译
        if source_lang not in common_langs and target_lang == LanguageCode.ENGLISH:
            return False
        # 其他情况直接翻译
        return False

    if needs_two_step_translation(detected_language, target_language):
        print(f"🔀 检测到 {detected_language.value} -> 中文，启用两步翻译流程")

        # 第一步：翻译成英文
        if progress:
            progress(0.7, f"翻译 {detected_language.value} -> 英文")

        english_segments = translation_provider.translate(
            original_segments,
            detected_language,
            LanguageCode.ENGLISH
        )

        # 第二步：从英文翻译成中文
        if progress:
            progress(0.8, "翻译 英文 -> 中文")

        translated_segments = translation_provider.translate(
            english_segments,
            LanguageCode.ENGLISH,
            target_language
        )

        print("✅ 两步翻译完成")
    else:
        # 直接翻译
        print(f"🔀 直接翻译: {detected_language.value} -> {target_language.value}")
        translated_segments = translation_provider.translate(
            original_segments,
            detected_language,
            target_language
        )

    # 5. 保存缓存
    cache_data = {
        "detected_language": detected_language.value,
        "original_segments": [
            {"text": seg.text, "start": seg.time_range.start_seconds, "end": seg.time_range.end_seconds}
            for seg in original_segments
        ],
        "translated_segments": [
            {"text": seg.text, "start": seg.time_range.start_seconds, "end": seg.time_range.end_seconds}
            for seg in translated_segments
        ]
    }
    cache_repo.set(cache_key, cache_data)

    if progress:
        progress(1.0, "字幕生成完成")

    return SubtitleGenerationResult(
        original_subtitle=Subtitle(original_segments, detected_language),
        translated_subtitle=Subtitle(translated_segments, target_language),
        detected_language=detected_language,
        cache_hit=False
    )