"""
优化的批量处理用例 - 修复字幕逻辑版本

关键修复：
1. ✅ 字幕命名清晰：original/target/secondary
2. ✅ TTS 使用目标语言（中文）字幕
3. ✅ 双语字幕顺序正确（中文在上，英文在下）
4. ✅ 完整的导入和辅助函数
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List, Tuple

from domain.entities import (
    Video, Subtitle, AudioTrack, ProcessedVideo,
    LanguageCode, AudioSample, TextSegment, TimeRange, VoiceProfile
)
from domain.ports import (
    ASRProvider, TranslationProvider, TTSProvider,
    VideoProcessor, SubtitleWriter, CacheRepository,
    AudioFileRepository,
)
from domain.services import (
    merge_bilingual_subtitles,
    calculate_cache_key,
)

# ============== 中间数据结构 ============== #
@dataclass(frozen=True)
class VideoWithSubtitles:
    """
    视频 + 字幕的中间结果

    字幕命名规范：
    - original_subtitle: ASR识别的原始语言字幕
    - target_subtitle: 目标语言字幕（中文）
    - secondary_subtitle: 次要语言字幕（英文）
    """
    video: Video
    original_subtitle: Subtitle  # 原始识别语言（zh/en/pt/ja等）
    target_subtitle: Subtitle  # 目标语言（中文）
    secondary_subtitle: Subtitle  # 次要语言（英文）
    detected_language: LanguageCode
    cache_hit_subtitle: bool = False


@dataclass(frozen=True)
class VideoWithAudio:
    """视频 + 字幕 + 音频的中间结果"""
    video: Video
    original_subtitle: Subtitle
    target_subtitle: Subtitle  # 中文字幕
    secondary_subtitle: Subtitle  # 英文字幕
    detected_language: LanguageCode
    audio_track: Optional[AudioTrack]
    cache_hit_audio: bool


# ============== 辅助函数（纯逻辑） ============== #
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


def _deserialize_segments(
    data: List[dict],
    language: LanguageCode
) -> Tuple[TextSegment, ...]:
    """反序列化文本片段"""
    return tuple(
        TextSegment(
            text=item["text"],
            time_range=TimeRange(
                start_seconds=item["start"],
                end_seconds=item["end"],
            ),
            language=language
        )
        for item in data
    )


# ============== 音频处理（重构版） ============== #
def _load_audio_from_cache(
        audio_repo: AudioFileRepository,  # ✅ 使用Port接口
        cache_key: str,
        language: LanguageCode
) -> Optional[AudioTrack]:
    """从缓存加载音频轨道（纯函数）"""
    audio_sample, metadata = audio_repo.load_audio(cache_key)

    if audio_sample is None:
        return None

    return AudioTrack(audio_sample, language)


def _save_audio_to_cache(
        audio_repo: AudioFileRepository,  # ✅ 使用Port接口
        cache_key: str,
        audio_track: AudioTrack,
        reference_audio_path: Path
) -> None:
    """保存音频到缓存（纯函数）"""
    metadata = {
        "language": audio_track.language.value,
        "sample_rate": audio_track.audio.sample_rate,
        "reference_audio": str(reference_audio_path),
    }

    audio_repo.save_audio(cache_key, audio_track.audio, metadata)


def _assemble_full_audio(
        synthesized_audios: tuple,
        segments: tuple[TextSegment, ...],
        video_duration: float,
        language: LanguageCode
) -> AudioTrack:
    """拼接音频片段（纯逻辑）"""
    if not synthesized_audios:
        raise ValueError("没有可拼接的音频")

    sample_rate = synthesized_audios[0].sample_rate
    total_samples = int(video_duration * sample_rate)
    full_audio_list = [0.0] * total_samples

    for audio_sample, segment in zip(synthesized_audios, segments):
        start_idx = int(segment.time_range.start_seconds * sample_rate)
        for i, sample in enumerate(audio_sample.samples):
            target_idx = start_idx + i
            if target_idx < total_samples:
                full_audio_list[target_idx] = sample

    full_audio = AudioSample(
        samples=tuple(full_audio_list),
        sample_rate=sample_rate
    )

    return AudioTrack(full_audio, language)


# ============== 字幕缓存辅助函数 ============== #
def _load_subtitle_segments(
        cached_data: dict,
        language: LanguageCode,
        segment_key: str
) -> tuple[TextSegment, ...]:
    """从缓存加载指定语言的字幕片段"""
    segments_data = cached_data.get(segment_key, [])
    if not segments_data:
        return tuple()

    return tuple(
        TextSegment(
            text=seg["text"],
            time_range=TimeRange(seg["start"], seg["end"]),
            language=language
        )
        for seg in segments_data
    )


def _reconstruct_subtitles_from_cache(
        cached: dict,
        detected_lang: LanguageCode
) -> tuple[Subtitle, Subtitle, Subtitle]:
    """
    从缓存重建所有字幕

    Returns:
        (original_subtitle, target_subtitle, secondary_subtitle)
    """
    # 1. 恢复中文字幕（目标语言）
    zh_segments = _load_subtitle_segments(cached, LanguageCode.CHINESE, "zh_segments")
    target_subtitle = Subtitle(zh_segments, LanguageCode.CHINESE)

    # 2. 恢复英文字幕（次要语言）
    en_segments = _load_subtitle_segments(cached, LanguageCode.ENGLISH, "en_segments")
    secondary_subtitle = Subtitle(en_segments, LanguageCode.ENGLISH)

    # 3. 恢复原始语言字幕
    if detected_lang == LanguageCode.CHINESE:
        original_subtitle = target_subtitle
    elif detected_lang == LanguageCode.ENGLISH:
        original_subtitle = secondary_subtitle
    else:
        original_segments = _load_subtitle_segments(
            cached,
            detected_lang,
            f"{detected_lang.value}_segments"
        )
        if original_segments:
            original_subtitle = Subtitle(original_segments, detected_lang)
        else:
            original_subtitle = target_subtitle

    return original_subtitle, target_subtitle, secondary_subtitle

# ============== 字幕翻译策略函数 ============== #

def _translate_subtitles(
        original_segments: tuple[TextSegment, ...],
        detected_lang: LanguageCode,
        translation_provider: TranslationProvider
) -> tuple[Subtitle, Subtitle]:
    """
    翻译字幕，始终返回中文字幕和英文字幕

    Returns:
        (target_subtitle: 中文, secondary_subtitle: 英文)
    """
    if detected_lang == LanguageCode.CHINESE:
        zh_segments = original_segments
        en_segments = translation_provider.translate(
            original_segments,
            LanguageCode.CHINESE,
            LanguageCode.ENGLISH
        )

    elif detected_lang == LanguageCode.ENGLISH:
        en_segments = original_segments
        zh_segments = translation_provider.translate(
            original_segments,
            LanguageCode.ENGLISH,
            LanguageCode.CHINESE
        )

    else:
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

    target_subtitle = Subtitle(zh_segments, LanguageCode.CHINESE)
    secondary_subtitle = Subtitle(en_segments, LanguageCode.ENGLISH)

    return target_subtitle, secondary_subtitle

# ============== 阶段性处理函数 ============== #
def phase1_extract_asr(
        videos: List[Video],
        cache_repo: CacheRepository,
        video_processor: VideoProcessor,
        asr_provider: ASRProvider,
        progress: Optional[Callable[[float, str], None]] = None,
) -> Tuple[VideoWithSubtitles, ...]:
    """
    音频提取 + ASR，结果落盘并返回 VideoWithSubtitles（原始字幕）。
    """
    total = len(videos)
    out: List[VideoWithSubtitles] = []

    for idx, video in enumerate(videos):
        if progress:
            progress(idx / total, f"Phase-1 ASR: {idx + 1}/{total}  {video.path.name}")

        cache_key = calculate_cache_key(video.path, "phase1_asr", {})
        if cache_repo.exists(cache_key):
            try:
                cached = cache_repo.get(cache_key)
                detected_lang = LanguageCode(cached["detected_language"])
                original_sub = Subtitle(
                    segments=_deserialize_segments(cached["segments"], detected_lang),
                    language=detected_lang,
                )
                out.append(VideoWithSubtitles(video, original_sub, None, None, detected_lang, True))
                print(f"  💾 Phase-1 缓存命中: {video.path.name}")
                continue
            except (KeyError, ValueError):
                print(f"  ⚠️  Phase-1 缓存损坏，重新生成: {video.path.name}")

        # 真正干活
        audio_path = video_processor.extract_audio(video)
        segments, detected_lang = asr_provider.transcribe(audio_path)
        cache_repo.set(cache_key, {
            "detected_language": detected_lang.value,
            "segments": _serialize_segments(segments),
        })
        original_sub = Subtitle(segments, detected_lang)
        out.append(VideoWithSubtitles(video, original_sub, None, None, detected_lang, False))
        print(f"  ✅ Phase-1 完成: {video.path.name}  ({detected_lang.value})")

    return tuple(out)

def phase2_translate(
        videos: List[Video],
        cache_repo: CacheRepository,
        translation_provider: TranslationProvider,
        target_language: LanguageCode = LanguageCode.CHINESE,
        progress: Optional[Callable[[float, str], None]] = None,
) -> Tuple[VideoWithSubtitles, ...]:
    """
    读取 Phase-1 缓存 → 翻译（或缓存命中）→ 返回 VideoWithSubtitles 元组
    """
    total = len(videos)
    out: List[VideoWithSubtitles] = []

    for idx, video in enumerate(videos):
        if progress:
            progress(idx / total, f"Phase-2 Trans: {idx + 1}/{total}  {video.path.name}")

        # 1. Phase-2 缓存 key
        trans_key = calculate_cache_key(
            video.path,
            "phase2_trans",
            {"target_language": target_language.value},
        )

        # 2. 读 Phase-1 原始字幕（必须存在）
        asr_key = calculate_cache_key(video.path, "phase1_asr", {})
        if not cache_repo.exists(asr_key):
            raise RuntimeError(f"Phase-1 缓存缺失，无法翻译: {video.path.name}")
        asr_cached = cache_repo.get(asr_key)
        detected_lang = LanguageCode(asr_cached["detected_language"])
        original_sub = Subtitle(
            segments=_deserialize_segments(asr_cached["segments"], detected_lang),
            language=detected_lang,
        )

        # 3. 如果 Phase-2 已存在，直接还原
        if cache_repo.exists(trans_key):
            try:
                trans_cached = cache_repo.get(trans_key)
                zh_sub = Subtitle(
                    segments=_deserialize_segments(trans_cached["zh_segments"], _deserialize_segments),
                    language=LanguageCode.CHINESE,
                )
                en_sub = Subtitle(
                    segments=_deserialize_segments(trans_cached["en_segments"], _deserialize_segments),
                    language=LanguageCode.ENGLISH,
                )
                out.append(VideoWithSubtitles(video, original_sub, zh_sub, en_sub, detected_lang, True))
                print(f"  💾 Phase-2 缓存命中: {video.path.name}")
                continue
            except (KeyError, ValueError):
                print(f"  ⚠️  Phase-2 缓存损坏，重新翻译: {video.path.name}")

        # 4. 真正翻译
        zh_sub, en_sub = _translate_subtitles(
            original_sub.segments,
            detected_lang,
            translation_provider,
        )

        # 5. 写 Phase-2 缓存
        cache_repo.set(
            trans_key,
            {
                "zh_segments": _serialize_segments(zh_sub.segments),
                "en_segments": _serialize_segments(en_sub.segments),
            },
        )
        out.append(out.append(VideoWithSubtitles(
            video=video,
            original_subtitle=original_sub,
            target_subtitle=zh_sub,  # 中文留空
            secondary_subtitle=en_sub,  # 英文留空
            detected_language=detected_lang,
            cache_hit_subtitle=False  # 或 False，视你逻辑而定
        )))
        print(f"  ✅ Phase-2 完成: {video.path.name}")

    return tuple(out)


def stage1_batch_asr(
        videos: tuple[Video, ...],
        asr_provider: ASRProvider,
        translation_provider: TranslationProvider,
        video_processor: VideoProcessor,
        cache_repo: CacheRepository,
        target_language: LanguageCode = LanguageCode.CHINESE,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[VideoWithSubtitles, ...]:
    """阶段1: 批量 ASR + 翻译"""

    if progress:
        progress(0.0, "阶段1: 批量语音识别和翻译")

    total = len(videos)

    videos_with_orig = phase1_extract_asr(list(videos), cache_repo, video_processor, asr_provider, progress)
    asr_provider.unload()

    videos_with_subs = phase2_translate(
        videos=[vws.video for vws in videos_with_orig],
        cache_repo=cache_repo,
        translation_provider=translation_provider,
        target_language=target_language,
        progress=lambda ratio, msg: print(f"{ratio:.1%} {msg}"),
    )
    translation_provider.unload()

    if progress:
        progress(1.0, f"阶段1完成: 处理了 {total} 个视频")

    del video_processor, asr_provider, translation_provider
    return tuple(videos_with_subs)


def stage2_batch_tts(
        video_subtitles: tuple,
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        cache_repo: CacheRepository,
        audio_repo: AudioFileRepository,  # ✅ 新增参数
        enable_voice_cloning: bool = True,
        reference_audio_file: Path = None,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple:
    """阶段2: 批量TTS（重构版）"""

    if not enable_voice_cloning:
        return _skip_voice_cloning(video_subtitles, progress)

    results = []
    total = len(video_subtitles)

    for idx, vs in enumerate(video_subtitles):
        if progress:
            progress(idx / total, f"TTS: {idx + 1}/{total} - {vs.video.path.name}")

        cache_key = calculate_cache_key(
            vs.video.path,
            "voice_cloning",
            {
                "language": vs.target_subtitle.language.value,
                "num_segments": len(vs.target_subtitle.segments)
            }
        )

        # ✅ 使用Port接口检查缓存
        cache_hit = audio_repo.exists(cache_key)
        audio_track = None

        if cache_hit:
            audio_track = _load_audio_from_cache(
                audio_repo,
                cache_key,
                vs.target_subtitle.language
            )
            if audio_track is None:
                cache_hit = False

        if not cache_hit:
            # 批量合成
            texts = [seg.text for seg in vs.target_subtitle.segments]
            synthesized_audios = tts_provider.batch_synthesize(
                texts=texts,
                reference_audio_path=reference_audio_file,
                language=vs.target_subtitle.language
            )

            # 拼接音频
            audio_track = _assemble_full_audio(
                synthesized_audios=synthesized_audios,
                segments=vs.target_subtitle.segments,
                video_duration=vs.video.duration,
                language=vs.target_subtitle.language
            )

            # ✅ 使用Port接口保存
            _save_audio_to_cache(
                audio_repo,
                cache_key,
                audio_track,
                reference_audio_file
            )

        # 构建结果
        result = VideoWithAudio(
            video=vs.video,
            original_subtitle=vs.original_subtitle,
            target_subtitle=vs.target_subtitle,
            secondary_subtitle=vs.secondary_subtitle,
            detected_language=vs.detected_language,
            audio_track=audio_track,
            cache_hit_audio=cache_hit
        )
        results.append(result)

    if progress:
        progress(1.0, f"阶段2完成: 处理了 {total} 个视频")

    tts_provider.unload()
    return tuple(results)


def _skip_voice_cloning(video_subtitles, progress):
    """跳过语音克隆的辅助函数"""

    results = tuple(
        VideoWithAudio(
            video=vs.video,
            original_subtitle=vs.original_subtitle,
            target_subtitle=vs.target_subtitle,
            secondary_subtitle=vs.secondary_subtitle,
            detected_language=vs.detected_language,
            audio_track=None,
            cache_hit_audio=False
        )
        for vs in video_subtitles
    )

    if progress:
        progress(1.0, "阶段2跳过: 未启用语音克隆")

    return results

def stage3_batch_synthesis(
        video_audios: tuple[VideoWithAudio, ...],
        video_processor: VideoProcessor,
        subtitle_writer: SubtitleWriter,
        output_dir: Path,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[ProcessedVideo, ...]:
    """阶段3: 批量视频合成"""

    if progress:
        progress(0.0, "阶段3: 批量视频合成")

    results = []
    total = len(video_audios)

    for idx, va in enumerate(video_audios):
        if progress:
            progress(idx / total, f"合成: 处理视频 {idx + 1}/{total} - {va.video.path.name}")

        # ✅ 双语字幕：中文在上，英文在下
        bilingual = merge_bilingual_subtitles(
            va.target_subtitle,
            va.secondary_subtitle
        )

        from application.use_cases.synthesize_video_use_case import synthesize_video_use_case

        synthesis_result = synthesize_video_use_case(
            video=va.video,
            subtitles=(
                va.target_subtitle,
                va.secondary_subtitle,
                bilingual
            ),
            audio_track=va.audio_track,
            video_processor=video_processor,
            subtitle_writer=subtitle_writer,
            output_dir=output_dir,
            burn_subtitles=True,
            progress=None
        )

        processed = ProcessedVideo(
            original_video=va.video,
            subtitles=(va.target_subtitle, va.secondary_subtitle, bilingual),
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
        audio_repo: AudioFileRepository,
        output_dir: Path,
        enable_voice_cloning: bool = True,
        reference_audio_file: Path = None,
        target_language: LanguageCode = LanguageCode.CHINESE,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[ProcessedVideo, ...]:
    """
    优化的批量处理用例（纯函数）

    按阶段执行，避免重复加载模型
    """
    if progress:
        progress(0.0, f"开始优化批量处理 {len(videos)} 个视频")

    print(f"\n{'=' * 60}")
    print(f"🚀 优化批量处理模式")
    print(f"   视频数量: {len(videos)}")
    print(f"   语音克隆: {'启用' if enable_voice_cloning else '禁用'}")
    print(f"{'=' * 60}\n")

    # 阶段1
    print(f"📝 阶段1: 批量语音识别和翻译")
    video_subtitles = stage1_batch_asr(
        videos, asr_provider, translation_provider,
        video_processor, cache_repo, target_language,
        lambda p, d: progress(p * 0.4, d) if progress else None
    )
    print(f"✅ 阶段1完成\n")

    # 阶段2
    print(f"🎤 阶段2: 批量语音克隆")
    video_audios = stage2_batch_tts(
        video_subtitles, tts_provider, video_processor,
        cache_repo, audio_repo, enable_voice_cloning, reference_audio_file,
        lambda p, d: progress(0.4 + p * 0.4, d) if progress else None
    )
    print(f"✅ 阶段2完成\n")

    # 阶段3
    print(f"🎬 阶段3: 批量视频合成")
    results = stage3_batch_synthesis(
        video_audios, video_processor, subtitle_writer, output_dir,
        lambda p, d: progress(0.8 + p * 0.2, d) if progress else None
    )
    print(f"✅ 阶段3完成\n")

    if progress:
        progress(1.0, "优化批量处理完成")

    # 统计
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
