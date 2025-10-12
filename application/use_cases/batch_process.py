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
from datetime import datetime
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

            original_subtitle = Subtitle(original_segments, detected_lang) if original_segments else Subtitle(zh_segments, detected_lang)
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

    asr_provider.unload()
    if progress:
        progress(1.0, f"阶段1完成: 处理了 {total} 个视频")

    return tuple(results)


"""
优化的 stage2_batch_tts - 使用 batch_infer_same_speaker 批量处理 segments
"""


def stage2_batch_tts(
        video_subtitles: tuple[VideoWithSubtitles, ...],
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        cache_repo: CacheRepository,
        enable_voice_cloning: bool = True,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[VideoWithAudio, ...]:
    """
    阶段2: 批量 TTS（语音克隆）- 优化版本

    关键优化：
    1. 使用 batch_infer_same_speaker 批量处理同一视频的所有 segments
    2. 一次性提取并缓存说话人条件，避免重复计算
    3. 减少 GPU 上下文切换，提高吞吐量

    Args:
        video_subtitles: 带字幕的视频中间结果
        tts_provider: TTS 提供者
        video_processor: 视频处理器
        cache_repo: 缓存仓储
        enable_voice_cloning: 是否启用语音克隆
        progress: 进度回调

    Returns:
        带音频的视频中间结果
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
            audio_track = _load_audio_from_cache(
                cache_repo,
                cache_key,
                vs.video.path.name,
                vs.translated_subtitle.language
            )

            if audio_track is None:
                # 缓存损坏，重新生成
                cache_hit = False

        if not cache_hit:
            # 批量合成音频
            audio_track = _batch_synthesize_segments(
                vs=vs,
                tts_provider=tts_provider,
                video_processor=video_processor,
                cache_repo=cache_repo,
                cache_key=cache_key
            )

            print(f"  ✅ 完成: {vs.video.path.name} (批量处理 {len(vs.translated_subtitle.segments)} 个片段)")

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


def _load_audio_from_cache(
        cache_repo: CacheRepository,
        cache_key: str,
        video_name: str,
        language: LanguageCode
) -> Optional[AudioTrack]:
    """从缓存加载音频轨道"""
    try:
        cached = cache_repo.get(cache_key)

        if cached is None or "audio_samples" not in cached or "sample_rate" not in cached:
            print(f"  ⚠️  缓存数据损坏，重新生成: {video_name}")
            return None

        from domain.entities import AudioSample

        audio_sample = AudioSample(
            samples=tuple(cached["audio_samples"]),
            sample_rate=cached["sample_rate"]
        )

        audio_track = AudioTrack(audio_sample, language)

        print(f"  💾 缓存命中: {video_name}")
        return audio_track

    except (KeyError, TypeError) as e:
        print(f"  ⚠️  缓存数据解析失败: {e}，重新生成")
        return None


def _batch_synthesize_segments(
        vs: VideoWithSubtitles,
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        cache_repo: CacheRepository,
        cache_key: str
) -> AudioTrack:
    """
    批量合成音频片段 - 核心优化逻辑

    关键步骤：
    1. 提取参考音频（一次性）
    2. 准备批量文本列表
    3. 调用 batch_infer_same_speaker（一次性处理所有 segments）
    4. 将生成的音频片段拼接到完整音频
    """
    # 1. 提取参考音频
    reference_audio_path = video_processor.extract_reference_audio(
        vs.video,
        duration=10.0
    )

    # 2. 准备批量文本
    segments = vs.translated_subtitle.segments
    texts = [segment.text for segment in segments]

    print(f"  🎤 批量合成: {len(texts)} 个片段")

    # 3. 调用批量推理（核心优化点）
    synthesized_audios = tts_provider.batch_synthesize(
        texts=texts,
        reference_audio_path=reference_audio_path,
        language=vs.translated_subtitle.language
    )

    # 4. 构建完整音频轨道
    audio_track = _assemble_full_audio(
        synthesized_audios=synthesized_audios,
        segments=segments,
        video_duration=vs.video.duration,
        language=vs.translated_subtitle.language
    )

    # 5. 保存缓存
    _save_audio_to_cache(
        cache_repo=cache_repo,
        cache_key=cache_key,
        audio_track=audio_track,
        reference_audio_path=reference_audio_path
    )

    return audio_track


def _assemble_full_audio(
        synthesized_audios: tuple,
        segments: tuple,
        video_duration: float,
        language: LanguageCode
) -> AudioTrack:
    """
    将批量合成的音频片段拼接成完整音频

    Args:
        synthesized_audios: batch_synthesize 返回的音频列表
        segments: 对应的字幕片段
        video_duration: 视频总时长
        language: 目标语言

    Returns:
        完整的音频轨道
    """
    from domain.entities import AudioSample

    # 获取采样率（假设所有片段采样率相同）
    sample_rate = synthesized_audios[0].sample_rate

    # 初始化完整音频数组
    total_samples = int(video_duration * sample_rate)
    full_audio_list = [0.0] * total_samples

    # 按时间轴放置每个音频片段
    for audio_sample, segment in zip(synthesized_audios, segments):
        start_idx = int(segment.time_range.start_seconds * sample_rate)

        # 复制音频数据到对应位置
        for i, sample in enumerate(audio_sample.samples):
            target_idx = start_idx + i
            if target_idx < total_samples:
                full_audio_list[target_idx] = sample

    # 构建完整音频
    full_audio = AudioSample(
        samples=tuple(full_audio_list),
        sample_rate=sample_rate
    )

    return AudioTrack(full_audio, language)


def _save_audio_to_cache(
        cache_repo: CacheRepository,
        cache_key: str,
        audio_track: AudioTrack,
        reference_audio_path: Path
) -> None:
    """保存音频到缓存"""
    cache_data = {
        "audio_samples": list(audio_track.audio.samples),
        "sample_rate": audio_track.audio.sample_rate,
        "reference_audio": str(reference_audio_path),
        "reference_duration": 10.0
    }
    cache_repo.set(cache_key, cache_data)


def stage3_batch_synthesis(
        video_audios: tuple[VideoWithAudio, ...],
        video_processor: VideoProcessor,
        subtitle_writer: SubtitleWriter,
        cache_repo: CacheRepository,  # 新增缓存仓储参数
        output_dir: Path,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[ProcessedVideo, ...]:
    """
    阶段3: 批量视频合成（带缓存和断点续传）

    为所有视频生成字幕文件和最终视频，支持断点续传
    """
    if progress:
        progress(0.0, "阶段3: 批量视频合成")

    results = []
    total = len(video_audios)

    for idx, va in enumerate(video_audios):
        video_progress = idx / total

        if progress:
            progress(video_progress, f"合成: 处理视频 {idx + 1}/{total} - {va.video.path.name}")

        # 检查视频合成缓存
        cache_key = calculate_cache_key(
            va.video.path,
            "video_synthesis",
            {
                "subtitles_hash": hash((va.original_subtitle, va.translated_subtitle)),
                "audio_track_hash": hash(va.audio_track) if va.audio_track else "no_audio",
                "output_dir": str(output_dir)
            }
        )

        cache_hit = cache_repo.exists(cache_key)
        processed_video = None

        if cache_hit:
            # 尝试从缓存加载已处理的视频信息
            try:
                cached_data = cache_repo.get(cache_key)
                if cached_data and _validate_cached_video(cached_data, output_dir):
                    processed_video = _load_processed_video_from_cache(cached_data, va)
                    print(f"  💾 视频合成缓存命中: {va.video.path.name}")
            except (KeyError, ValueError, FileNotFoundError) as e:
                print(f"  ⚠️  视频合成缓存损坏: {e}，重新生成")
                cache_hit = False

        if not cache_hit:
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
            processed_video = ProcessedVideo(
                original_video=va.video,
                subtitles=(
                    va.translated_subtitle,
                    va.original_subtitle,
                    bilingual
                ),
                audio_tracks=(va.audio_track,) if va.audio_track else tuple(),
                output_paths=synthesis_result.output_paths
            )

            # 保存视频合成缓存
            _save_video_synthesis_cache(
                cache_repo=cache_repo,
                cache_key=cache_key,
                processed_video=processed_video,
                output_dir=output_dir
            )

            print(f"  ✅ 完成: {va.video.path.name}")

        results.append(processed_video)

    if progress:
        progress(1.0, f"阶段3完成: 处理了 {total} 个视频")

    return tuple(results)


def _validate_cached_video(cached_data: dict, output_dir: Path) -> bool:
    """
    验证缓存的视频文件是否有效

    Args:
        cached_data: 缓存数据
        output_dir: 输出目录

    Returns:
        bool: 缓存是否有效
    """
    try:
        output_paths = cached_data.get("output_paths", [])
        if not output_paths:
            return False

        for path_str in output_paths:
            output_path = Path(path_str)
            # 检查文件是否存在且大小合理（至少1KB）
            if not output_path.exists() or output_path.stat().st_size < 1024:
                return False

        return True
    except (KeyError, OSError):
        return False


def _load_processed_video_from_cache(cached_data: dict, va: VideoWithAudio) -> ProcessedVideo:
    """
    从缓存加载已处理的视频信息

    Args:
        cached_data: 缓存数据
        va: 视频音频中间结果

    Returns:
        ProcessedVideo: 处理后的视频
    """
    output_paths = tuple(Path(path_str) for path_str in cached_data["output_paths"])

    return ProcessedVideo(
        original_video=va.video,
        subtitles=(
            va.translated_subtitle,
            va.original_subtitle,
            merge_bilingual_subtitles(va.translated_subtitle, va.original_subtitle)
        ),
        audio_tracks=(va.audio_track,) if va.audio_track else tuple(),
        output_paths=output_paths
    )


def _save_video_synthesis_cache(
        cache_repo: CacheRepository,
        cache_key: str,
        processed_video: ProcessedVideo,
        output_dir: Path
) -> None:
    """
    保存视频合成缓存

    Args:
        cache_repo: 缓存仓储
        cache_key: 缓存键
        processed_video: 处理后的视频
        output_dir: 输出目录
    """
    cache_data = {
        "output_paths": [str(path) for path in processed_video.output_paths],
        "original_video": str(processed_video.original_video.path),
        "timestamp": datetime.now().isoformat()
    }

    cache_repo.set(cache_key, cache_data)


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

    print(f"\n{'='*60}")
    print(f"🚀 优化批量处理模式")
    print(f"   视频数量: {len(videos)}")
    print(f"   语音克隆: {'启用' if enable_voice_cloning else '禁用'}")
    print(f"{'='*60}\n")

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
        cache_repo=cache_repo,  # 新增参数
        output_dir=output_dir,
        progress=lambda p, d: progress(0.8 + p * 0.2, d) if progress else None
    )
    print(f"✅ 阶段3完成\n")

    if progress:
        progress(1.0, "优化批量处理完成")

    # 统计信息
    cache_hits_subtitle = sum(1 for vs in video_subtitles if vs.cache_hit_subtitle)
    cache_hits_audio = sum(1 for va in video_audios if va.cache_hit_audio)

    print(f"\n{'='*60}")
    print(f"✅ 批量处理完成")
    print(f"   总视频数: {len(videos)}")
    print(f"   字幕缓存命中: {cache_hits_subtitle}/{len(videos)}")
    if enable_voice_cloning:
        print(f"   音频缓存命中: {cache_hits_audio}/{len(videos)}")
    print(f"   输出文件数: {sum(len(r.output_paths) for r in results)}")
    print(f"{'='*60}\n")

    return results