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
    生成字幕用例（最终调试版）- 始终生成中英文字幕
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

        detected_lang = LanguageCode(cached["detected_language"])

        print(f"💾 从缓存加载:")
        print(f"   检测语言: {detected_lang.value}")
        print(f"   缓存键: {list(cached.keys())}")

        # 从缓存重建
        original_segments = tuple(
            TextSegment(
                text=seg["text"],
                time_range=TimeRange(seg["start"], seg["end"]),
                language=detected_lang
            )
            for seg in cached.get(f"{detected_lang.value}_segments", [])
        )

        zh_segments = tuple(
            TextSegment(
                text=seg["text"],
                time_range=TimeRange(seg["start"], seg["end"]),
                language=LanguageCode.CHINESE
            )
            for seg in cached.get("zh_segments", [])
        )

        en_segments = tuple(
            TextSegment(
                text=seg["text"],
                time_range=TimeRange(seg["start"], seg["end"]),
                language=LanguageCode.ENGLISH
            )
            for seg in cached.get("en_segments", [])
        )

        print(f"   zh_segments: {len(zh_segments)}")
        print(f"   en_segments: {len(en_segments)}")

        return SubtitleGenerationResult(
            original_subtitle=Subtitle(original_segments, detected_lang) if original_segments else Subtitle(zh_segments,
                                                                                                            detected_lang),
            translated_subtitle=Subtitle(zh_segments, LanguageCode.CHINESE),
            detected_language=detected_lang,
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
    asr_provider.unload()
    print(f"\n🎤 ASR 识别完成:")
    print(f"   检测语言: {detected_language.value}")
    print(f"   片段数量: {len(original_segments)}")

    # 4. 智能翻译流程
    if progress:
        progress(0.6, "翻译中")

    # 初始化变量
    en_segments = None
    zh_segments = None

    print(f"\n{'=' * 60}")
    print(f"🔍 翻译流程调试")
    print(f"   检测语言: {detected_language.value}")
    print(f"   目标语言: {target_language.value}")
    print(f"{'=' * 60}")

    if detected_language == LanguageCode.ENGLISH:
        # 情况1: 原始是英文
        print(f"\n📍 情况1: 原始语言是英文")
        en_segments = original_segments
        print(f"   ✅ en_segments 已设置（使用 original_segments）")
        print(f"   📊 en_segments 长度: {len(en_segments)}")

        if progress:
            progress(0.7, "翻译 英文 -> 中文")
        zh_segments = translation_provider.translate(
            original_segments,
            LanguageCode.ENGLISH,
            LanguageCode.CHINESE
        )
        print(f"   ✅ zh_segments 已生成")
        print(f"   📊 zh_segments 长度: {len(zh_segments)}")

    elif detected_language == LanguageCode.CHINESE:
        # 情况2: 原始是中文
        print(f"\n📍 情况2: 原始语言是中文")
        zh_segments = original_segments
        print(f"   ✅ zh_segments 已设置（使用 original_segments）")
        print(f"   📊 zh_segments 长度: {len(zh_segments)}")

        if progress:
            progress(0.7, "翻译 中文 -> 英文")
        en_segments = translation_provider.translate(
            original_segments,
            LanguageCode.CHINESE,
            LanguageCode.ENGLISH
        )
        print(f"   ✅ en_segments 已生成")
        print(f"   📊 en_segments 长度: {len(en_segments)}")

    else:
        # 情况3: 其他语言
        print(f"\n📍 情况3: 原始语言是 {detected_language.value}")
        print(f"   需要两步翻译: {detected_language.value} -> en -> zh")

        # 第一步: original -> en
        print(f"\n   🔄 第一步: {detected_language.value} -> 英文")
        if progress:
            progress(0.65, f"翻译 {detected_language.value} -> 英文")

        en_segments = translation_provider.translate(
            original_segments,
            detected_language,
            LanguageCode.ENGLISH
        )

        print(f"   ✅ 第一步完成")
        print(f"   📊 en_segments 类型: {type(en_segments)}")
        print(f"   📊 en_segments 长度: {len(en_segments) if en_segments else 'None/Empty'}")

        if not en_segments:
            raise ValueError(f"❌ 第一步翻译失败！en_segments 为空")

        if len(en_segments) > 0:
            print(f"   📝 第一个英文片段: {en_segments[0].text[:50]}...")

        # 第二步: en -> zh
        print(f"\n   🔄 第二步: 英文 -> 中文")
        if progress:
            progress(0.8, "翻译 英文 -> 中文")

        zh_segments = translation_provider.translate(
            en_segments,
            LanguageCode.ENGLISH,
            LanguageCode.CHINESE
        )

        print(f"   ✅ 第二步完成")
        print(f"   📊 zh_segments 类型: {type(zh_segments)}")
        print(f"   📊 zh_segments 长度: {len(zh_segments) if zh_segments else 'None/Empty'}")

        if not zh_segments:
            raise ValueError(f"❌ 第二步翻译失败！zh_segments 为空")

        if len(zh_segments) > 0:
            print(f"   📝 第一个中文片段: {zh_segments[0].text[:50]}...")

    # 最终验证
    print(f"\n{'=' * 60}")
    print(f"🔍 最终验证")
    print(f"{'=' * 60}")

    if not en_segments:
        raise ValueError(f"❌ en_segments 为空！检测语言: {detected_language.value}")
    if not zh_segments:
        raise ValueError(f"❌ zh_segments 为空！检测语言: {detected_language.value}")

    print(f"✅ en_segments: {len(en_segments)} 片段")
    print(f"✅ zh_segments: {len(zh_segments)} 片段")

    # 5. 保存缓存
    print(f"\n💾 保存缓存:")
    cache_data = {
        "detected_language": detected_language.value,
        "zh_segments": [
            {"text": seg.text, "start": seg.time_range.start_seconds, "end": seg.time_range.end_seconds}
            for seg in zh_segments
        ],
        "en_segments": [
            {"text": seg.text, "start": seg.time_range.start_seconds, "end": seg.time_range.end_seconds}
            for seg in en_segments
        ]
    }

    # 如果原始语言不是中英文，也保存
    if detected_language not in [LanguageCode.CHINESE, LanguageCode.ENGLISH]:
        cache_data[f"{detected_language.value}_segments"] = [
            {"text": seg.text, "start": seg.time_range.start_seconds, "end": seg.time_range.end_seconds}
            for seg in original_segments
        ]
        print(f"   {detected_language.value}_segments: {len(original_segments)} 片段")

    print(f"   zh_segments: {len(cache_data['zh_segments'])} 条目")
    print(f"   en_segments: {len(cache_data['en_segments'])} 条目")

    # 验证缓存数据
    if not cache_data['en_segments']:
        raise ValueError("❌ 缓存数据验证失败：en_segments 为空")
    if not cache_data['zh_segments']:
        raise ValueError("❌ 缓存数据验证失败：zh_segments 为空")

    cache_repo.set(cache_key, cache_data)
    print(f"✅ 缓存保存成功")

    if progress:
        progress(1.0, "字幕生成完成")

    print(f"\n📊 字幕生成结果:")
    print(f"   检测语言: {detected_language.value}")
    print(f"   中文字幕: {len(zh_segments)} 片段")
    print(f"   英文字幕: {len(en_segments)} 片段")

    return SubtitleGenerationResult(
        original_subtitle=Subtitle(original_segments, detected_language),
        translated_subtitle=Subtitle(zh_segments, LanguageCode.CHINESE),
        detected_language=detected_language,
        cache_hit=False
    )