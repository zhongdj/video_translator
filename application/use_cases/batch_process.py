"""
优化的批量处理用例 - 按阶段执行，避免重复加载模型

设计理念：
1. 阶段化处理：ASR -> Translation -> TTS -> Synthesis
2. 模型复用：每个阶段的模型只加载一次
3. 函数式风格：纯函数 + 不可变数据结构
4. 管道模式：数据在阶段间流动

性能对比：
- 传统方式：N个视频 × 3个模型 = 3N次加载
- 优化方式：3个模型各加载1次 = 3次加载
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

from domain.entities import (
    Video, Subtitle, AudioTrack, ProcessedVideo,
    LanguageCode,
)
from domain.ports import (
    ASRProvider, TranslationProvider, TTSProvider,
    VideoProcessor, SubtitleWriter, CacheRepository,
)
from domain.services import (
    merge_bilingual_subtitles,
    calculate_cache_key,
)


# ============== 中间数据结构 ============== #

@dataclass(frozen=True)
class VideoWithSubtitles:
    """视频 + 字幕的中间结果"""
    video: Video
    original_subtitle: Subtitle
    translated_subtitle: Subtitle
    detected_language: LanguageCode
    cache_hit_subtitle: bool


@dataclass(frozen=True)
class VideoWithAudio:
    """视频 + 字幕 + 音频的中间结果"""
    video: Video
    original_subtitle: Subtitle
    translated_subtitle: Subtitle
    detected_language: LanguageCode
    audio_track: Optional[AudioTrack]
    cache_hit_audio: bool


# ============== 阶段性处理函数 ============== #

def stage1_batch_asr(
        videos: tuple[Video, ...],
        asr_provider: ASRProvider,
        translation_provider: TranslationProvider,
        video_processor: VideoProcessor,
        cache_repo: CacheRepository,
        target_language: LanguageCode = LanguageCode.CHINESE,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[VideoWithSubtitles, ...]:
    """
    阶段1: 批量 ASR + 翻译

    对所有视频执行语音识别和翻译，模型只加载一次
    """
    if progress:
        progress(0.0, "阶段1: 批量语音识别和翻译")

    results = []
    total = len(videos)

    for idx, video in enumerate(videos):
        video_progress = idx / total

        if progress:
            progress(video_progress, f"ASR: 处理视频 {idx + 1}/{total} - {video.path.name}")

        # 检查缓存
        cache_key = calculate_cache_key(
            video.path,
            "subtitles",
            {
                "target_language": target_language.value,
                "source_language": "auto"
            }
        )

        cache_hit = cache_repo.exists(cache_key)

        if cache_hit:
            # 从缓存加载
            cached = cache_repo.get(cache_key)
            detected_lang = LanguageCode(cached["detected_language"])

            from domain.entities import TextSegment, TimeRange

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

            original_subtitle = Subtitle(original_segments, detected_lang) if original_segments else Subtitle(
                zh_segments, detected_lang)
            translated_subtitle = Subtitle(zh_segments, LanguageCode.CHINESE)

            print(f"  💾 缓存命中: {video.path.name}")
        else:
            # 执行 ASR
            audio_path = video_processor.extract_audio(video)
            original_segments, detected_lang = asr_provider.transcribe(audio_path, None)

            # 执行翻译
            from domain.entities import TextSegment

            en_segments = None
            zh_segments = None

            if detected_lang == LanguageCode.ENGLISH:
                en_segments = original_segments
                zh_segments = translation_provider.translate(
                    original_segments,
                    LanguageCode.ENGLISH,
                    LanguageCode.CHINESE
                )
            elif detected_lang == LanguageCode.CHINESE:
                zh_segments = original_segments
                en_segments = translation_provider.translate(
                    original_segments,
                    LanguageCode.CHINESE,
                    LanguageCode.ENGLISH
                )
            else:
                # 其他语言：先翻译到英文，再翻译到中文
                en_segments = translation_provider.translate(
                    original_segments,
                    detected_lang,
                    LanguageCode.ENGLISH
                )
                zh_segments = translation_provider.translate(
                    en_segments,
                    LanguageCode.ENGLISH,
                    LanguageCode.CHINESE
                )

            # 保存缓存
            cache_data = {
                "detected_language": detected_lang.value,
                "zh_segments": [
                    {"text": seg.text, "start": seg.time_range.start_seconds, "end": seg.time_range.end_seconds}
                    for seg in zh_segments
                ],
                "en_segments": [
                    {"text": seg.text, "start": seg.time_range.start_seconds, "end": seg.time_range.end_seconds}
                    for seg in en_segments
                ]
            }

            if detected_lang not in [LanguageCode.CHINESE, LanguageCode.ENGLISH]:
                cache_data[f"{detected_lang.value}_segments"] = [
                    {"text": seg.text, "start": seg.time_range.start_seconds, "end": seg.time_range.end_seconds}
                    for seg in original_segments
                ]

            cache_repo.set(cache_key, cache_data)

            original_subtitle = Subtitle(original_segments, detected_lang)
            translated_subtitle = Subtitle(zh_segments, LanguageCode.CHINESE)

            print(f"  ✅ 完成: {video.path.name} ({detected_lang.value} -> zh)")

        # 构建中间结果
        result = VideoWithSubtitles(
            video=video,
            original_subtitle=original_subtitle,
            translated_subtitle=translated_subtitle,
            detected_language=detected_lang,
            cache_hit_subtitle=cache_hit
        )
        results.append(result)

    if progress:
        progress(1.0, f"阶段1完成: 处理了 {total} 个视频")

    return tuple(results)


def stage2_batch_tts(
        video_subtitles: tuple[VideoWithSubtitles, ...],
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        cache_repo: CacheRepository,
        enable_voice_cloning: bool = True,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[VideoWithAudio, ...]:
    """
    阶段2: 批量 TTS（语音克隆）

    对所有视频执行语音合成，TTS 模型只加载一次
    """
    if progress:
        progress(0.0, "阶段2: 批量语音克隆")

    if not enable_voice_cloning:
        # 跳过语音克隆，直接返回
        results = tuple(
            VideoWithAudio(
                video=vs.video,
                original_subtitle=vs.original_subtitle,
                translated_subtitle=vs.translated_subtitle,
                detected_language=vs.detected_language,
                audio_track=None,
                cache_hit_audio=False
            )
            for vs in video_subtitles
        )
        if progress:
            progress(1.0, "阶段2跳过: 未启用语音克隆")
        return results

    results = []
    total = len(video_subtitles)

    for idx, vs in enumerate(video_subtitles):
        video_progress = idx / total

        if progress:
            progress(video_progress, f"TTS: 处理视频 {idx + 1}/{total} - {vs.video.path.name}")

        # 检查缓存
        cache_key = calculate_cache_key(
            vs.video.path,
            "voice_cloning",
            {
                "language": vs.translated_subtitle.language.value,
                "reference": "auto",
                "num_segments": len(vs.translated_subtitle.segments)
            }
        )

        cache_hit = cache_repo.exists(cache_key)

        if cache_hit:
            # 从缓存加载
            cached = cache_repo.get(cache_key)

            from domain.entities import AudioSample, VoiceProfile

            audio_sample = AudioSample(
                samples=tuple(cached["audio_samples"]),
                sample_rate=cached["sample_rate"]
            )

            audio_track = AudioTrack(audio_sample, vs.translated_subtitle.language)

            print(f"  💾 缓存命中: {vs.video.path.name}")
        else:
            # 提取参考音频
            reference_audio_path = video_processor.extract_reference_audio(
                vs.video,
                10.0
            )

            # 创建声音配置
            from domain.entities import VoiceProfile

            voice_profile = VoiceProfile(
                reference_audio_path=reference_audio_path,
                language=vs.translated_subtitle.language,
                duration=10.0
            )

            # 逐句合成
            synthesized_segments = []
            for seg_idx, segment in enumerate(vs.translated_subtitle.segments):
                audio_sample = tts_provider.synthesize(
                    text=segment.text,
                    voice_profile=voice_profile,
                    target_duration=segment.time_range.duration
                )
                synthesized_segments.append((audio_sample, segment))

            # 拼接音频
            total_samples = int(vs.video.duration * synthesized_segments[0][0].sample_rate)
            full_audio_list = [0.0] * total_samples

            for audio_sample, segment in synthesized_segments:
                start_idx = int(segment.time_range.start_seconds * audio_sample.sample_rate)
                for i, sample in enumerate(audio_sample.samples):
                    if start_idx + i < total_samples:
                        full_audio_list[start_idx + i] = sample

            from domain.entities import AudioSample

            full_audio = AudioSample(
                samples=tuple(full_audio_list),
                sample_rate=synthesized_segments[0][0].sample_rate
            )

            # 保存缓存
            cache_data = {
                "audio_samples": list(full_audio.samples),
                "sample_rate": full_audio.sample_rate,
                "reference_audio": str(reference_audio_path),
                "reference_duration": 10.0
            }
            cache_repo.set(cache_key, cache_data)

            audio_track = AudioTrack(full_audio, vs.translated_subtitle.language)

            print(f"  ✅ 完成: {vs.video.path.name}")

        # 构建中间结果
        result = VideoWithAudio(
            video=vs.video,
            original_subtitle=vs.original_subtitle,
            translated_subtitle=vs.translated_subtitle,
            detected_language=vs.detected_language,
            audio_track=audio_track,
            cache_hit_audio=cache_hit
        )
        results.append(result)

    if progress:
        progress(1.0, f"阶段2完成: 处理了 {total} 个视频")

    return tuple(results)


def stage3_batch_synthesis(
        video_audios: tuple[VideoWithAudio, ...],
        video_processor: VideoProcessor,
        subtitle_writer: SubtitleWriter,
        output_dir: Path,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[ProcessedVideo, ...]:
    """
    阶段3: 批量视频合成

    为所有视频生成字幕文件和最终视频
    """
    if progress:
        progress(0.0, "阶段3: 批量视频合成")

    results = []
    total = len(video_audios)

    for idx, va in enumerate(video_audios):
        video_progress = idx / total

        if progress:
            progress(video_progress, f"合成: 处理视频 {idx + 1}/{total} - {va.video.path.name}")

        # 创建双语字幕
        bilingual = merge_bilingual_subtitles(
            va.translated_subtitle,
            va.original_subtitle
        )

        # 执行视频合成
        from application.use_cases.synthesize_video_use_case import synthesize_video_use_case

        synthesis_result = synthesize_video_use_case(
            video=va.video,
            subtitles=(
                va.translated_subtitle,
                va.original_subtitle,
                bilingual
            ),
            audio_track=va.audio_track,
            video_processor=video_processor,
            subtitle_writer=subtitle_writer,
            output_dir=output_dir,
            burn_subtitles=True,
            progress=None  # 不传递进度，避免过多输出
        )

        # 构建结果
        processed = ProcessedVideo(
            original_video=va.video,
            subtitles=(
                va.translated_subtitle,
                va.original_subtitle,
                bilingual
            ),
            audio_tracks=(va.audio_track,) if va.audio_track else tuple(),
            output_paths=synthesis_result.output_paths
        )

        results.append(processed)
        print(f"  ✅ 完成: {va.video.path.name}")

    if progress:
        progress(1.0, f"阶段3完成: 处理了 {total} 个视频")

    return tuple(results)


# ============== 主用例函数 ============== #

def batch_process_use_case(
        videos: tuple[Video, ...],
        asr_provider: ASRProvider,
        translation_provider: TranslationProvider,
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        subtitle_writer: SubtitleWriter,
        cache_repo: CacheRepository,
        output_dir: Path,
        enable_voice_cloning: bool = True,
        target_language: LanguageCode = LanguageCode.CHINESE,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[ProcessedVideo, ...]:
    """
    优化的批量处理用例（纯函数）

    按阶段执行，避免重复加载模型：
    1. 阶段1: 批量 ASR + 翻译（所有视频）
    2. 阶段2: 批量 TTS（所有视频）
    3. 阶段3: 批量视频合成（所有视频）

    性能提升：
    - 传统方式：3N 次模型加载（N = 视频数量）
    - 优化方式：3 次模型加载（每种模型加载1次）

    Args:
        videos: 待处理的视频列表
        asr_provider: ASR 提供者
        translation_provider: 翻译提供者
        tts_provider: TTS 提供者
        video_processor: 视频处理器
        subtitle_writer: 字幕写入器
        cache_repo: 缓存仓储
        output_dir: 输出目录
        enable_voice_cloning: 是否启用语音克隆
        target_language: 目标语言
        progress: 进度回调

    Returns:
        处理结果列表
    """
    if progress:
        progress(0.0, f"开始优化批量处理 {len(videos)} 个视频")

    print(f"\n{'=' * 60}")
    print(f"🚀 优化批量处理模式")
    print(f"   视频数量: {len(videos)}")
    print(f"   语音克隆: {'启用' if enable_voice_cloning else '禁用'}")
    print(f"{'=' * 60}\n")

    # 阶段1: 批量 ASR + 翻译
    print(f"📝 阶段1: 批量语音识别和翻译")
    video_subtitles = stage1_batch_asr(
        videos=videos,
        asr_provider=asr_provider,
        translation_provider=translation_provider,
        video_processor=video_processor,
        cache_repo=cache_repo,
        target_language=target_language,
        progress=lambda p, d: progress(p * 0.4, d) if progress else None
    )
    print(f"✅ 阶段1完成\n")

    # 阶段2: 批量 TTS
    print(f"🎤 阶段2: 批量语音克隆")
    video_audios = stage2_batch_tts(
        video_subtitles=video_subtitles,
        tts_provider=tts_provider,
        video_processor=video_processor,
        cache_repo=cache_repo,
        enable_voice_cloning=enable_voice_cloning,
        progress=lambda p, d: progress(0.4 + p * 0.4, d) if progress else None
    )
    print(f"✅ 阶段2完成\n")

    # 阶段3: 批量视频合成
    print(f"🎬 阶段3: 批量视频合成")
    results = stage3_batch_synthesis(
        video_audios=video_audios,
        video_processor=video_processor,
        subtitle_writer=subtitle_writer,
        output_dir=output_dir,
        progress=lambda p, d: progress(0.8 + p * 0.2, d) if progress else None
    )
    print(f"✅ 阶段3完成\n")

    if progress:
        progress(1.0, "优化批量处理完成")

    # 统计信息
    cache_hits_subtitle = sum(1 for vs in video_subtitles if vs.cache_hit_subtitle)
    cache_hits_audio = sum(1 for va in video_audios if va.cache_hit_audio)

    print(f"\n{'=' * 60}")
    print(f"✅ 批量处理完成")
    print(f"   总视频数: {len(videos)}")
    print(f"   字幕缓存命中: {cache_hits_subtitle}/{len(videos)}")
    if enable_voice_cloning:
        print(f"   音频缓存命中: {cache_hits_audio}/{len(videos)}")
    print(f"   输出文件数: {sum(len(r.output_paths) for r in results)}")
    print(f"{'=' * 60}\n")

    return results