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
from typing import Optional, Callable
from datetime import datetime

from domain.entities import (
    Video, Subtitle, AudioTrack, ProcessedVideo,
    LanguageCode, AudioSample, TextSegment, TimeRange, VoiceProfile
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
    """
    视频 + 字幕的中间结果

    字幕命名规范：
    - original_subtitle: ASR识别的原始语言字幕
    - target_subtitle: 目标语言字幕（中文）
    - secondary_subtitle: 次要语言字幕（英文）
    """
    video: Video
    original_subtitle: Subtitle      # 原始识别语言（zh/en/pt/ja等）
    target_subtitle: Subtitle        # 目标语言（中文）
    secondary_subtitle: Subtitle     # 次要语言（英文）
    detected_language: LanguageCode
    cache_hit_subtitle: bool


@dataclass(frozen=True)
class VideoWithAudio:
    """视频 + 字幕 + 音频的中间结果"""
    video: Video
    original_subtitle: Subtitle
    target_subtitle: Subtitle        # 中文字幕
    secondary_subtitle: Subtitle     # 英文字幕
    detected_language: LanguageCode
    audio_track: Optional[AudioTrack]
    cache_hit_audio: bool


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


# ============== 音频处理辅助函数 ============== #

def _load_audio_from_cache(
        cache_repo: CacheRepository,
        cache_key: str,
        video_name: str,
        language: LanguageCode
) -> Optional[AudioTrack]:
    """从缓存加载音频轨道"""
    try:
        cached = cache_repo.get(cache_key)
        if cached is None:
            return None

        # 优先从音频文件加载
        if "audio_file" in cached:
            audio_file = Path(cached["audio_file"])
            if audio_file.exists():
                try:
                    import numpy as np
                    import soundfile as sf

                    audio_data, sample_rate = sf.read(str(audio_file))
                    audio_sample = AudioSample(
                        samples=tuple(audio_data.tolist()),
                        sample_rate=sample_rate
                    )
                    audio_track = AudioTrack(audio_sample, language)
                    print(f"  💾 音频缓存命中（文件）: {video_name}")
                    return audio_track
                except Exception as e:
                    print(f"  ⚠️  音频文件加载失败: {e}")
                    return None

        # 兼容旧格式：从内存加载
        if "audio_samples" in cached and "sample_rate" in cached:
            audio_sample = AudioSample(
                samples=tuple(cached["audio_samples"]),
                sample_rate=cached["sample_rate"]
            )
            audio_track = AudioTrack(audio_sample, language)
            print(f"  💾 音频缓存命中（内存）: {video_name}")
            return audio_track

        return None

    except (KeyError, TypeError) as e:
        print(f"  ⚠️  音频缓存解析失败: {e}")
        return None


def _assemble_full_audio(
        synthesized_audios: tuple,
        segments: tuple[TextSegment, ...],
        video_duration: float,
        language: LanguageCode
) -> AudioTrack:
    """将批量合成的音频片段拼接成完整音频"""
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


def _save_audio_to_cache(
        cache_repo: CacheRepository,
        cache_key: str,
        audio_track: AudioTrack,
        reference_audio_path: Path,
        video_path: Path
) -> None:
    """保存音频到缓存（文件存储）"""
    cache_dir = video_path.parent / ".audio_cache"
    cache_dir.mkdir(exist_ok=True)

    audio_cache_path = cache_dir / f"audio_{cache_key[:16]}.wav"

    try:
        import numpy as np
        import soundfile as sf

        audio_data = np.array(audio_track.audio.samples, dtype=np.float32)
        sf.write(str(audio_cache_path), audio_data, audio_track.audio.sample_rate)

        cache_data = {
            "audio_file": str(audio_cache_path),
            "sample_rate": audio_track.audio.sample_rate,
            "language": audio_track.language.value,
            "reference_audio": str(reference_audio_path),
            "reference_duration": 10.0
        }

        cache_repo.set(cache_key, cache_data)
        print(f"  💾 音频已保存到文件: {audio_cache_path.name}")

    except Exception as e:
        print(f"  ⚠️  音频缓存保存失败: {e}")
        cache_data = {
            "audio_samples": list(audio_track.audio.samples),
            "sample_rate": audio_track.audio.sample_rate,
            "language": audio_track.language.value,
            "reference_audio": str(reference_audio_path),
            "reference_duration": 10.0
        }
        cache_repo.set(cache_key, cache_data)


def _batch_synthesize_segments(
        vs: VideoWithSubtitles,
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        reference_audio_file: Path,
        cache_repo: CacheRepository,
        cache_key: str
) -> AudioTrack:
    """批量合成音频片段（使用中文字幕）"""

    reference_audio_path = reference_audio_file
    #video_processor.extract_reference_audio(vs.video, duration=10.0)

    # ✅ 使用目标语言（中文）字幕
    segments = vs.target_subtitle.segments
    texts = [segment.text for segment in segments]

    print(f"  🎤 批量合成中文配音: {len(texts)} 个片段")

    # 批量合成
    synthesized_audios = tts_provider.batch_synthesize(texts=texts, reference_audio_path=reference_audio_path, language=vs.target_subtitle.language)

    # 拼接完整音频
    audio_track = _assemble_full_audio(
        synthesized_audios=synthesized_audios,
        segments=segments,
        video_duration=vs.video.duration,
        language=vs.target_subtitle.language
    )

    # 保存缓存
    _save_audio_to_cache(
        cache_repo=cache_repo,
        cache_key=cache_key,
        audio_track=audio_track,
        reference_audio_path=reference_audio_path,
        video_path=vs.video.path
    )

    return audio_track


def _skip_voice_cloning(
        video_subtitles: tuple[VideoWithSubtitles, ...],
        progress: Optional[Callable[[float, str], None]]
) -> tuple[VideoWithAudio, ...]:
    """跳过语音克隆"""
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
    """阶段1: 批量 ASR + 翻译"""

    if progress:
        progress(0.0, "阶段1: 批量语音识别和翻译")

    results = []
    total = len(videos)

    for idx, video in enumerate(videos):
        if progress:
            progress(idx / total, f"ASR: 处理视频 {idx + 1}/{total} - {video.path.name}")

        cache_key = calculate_cache_key(
            video.path,
            "subtitles",
            {"target_language": target_language.value, "source_language": "auto"}
        )

        cache_hit = cache_repo.exists(cache_key)

        if cache_hit:
            try:
                cached = cache_repo.get(cache_key)
                detected_lang = LanguageCode(cached["detected_language"])

                original_subtitle, target_subtitle, secondary_subtitle = \
                    _reconstruct_subtitles_from_cache(cached, detected_lang)

                print(f"  💾 字幕缓存命中: {video.path.name}")

            except (KeyError, ValueError) as e:
                print(f"  ⚠️  字幕缓存损坏: {e}，重新生成")
                cache_hit = False

        if not cache_hit:
            audio_path = video_processor.extract_audio(video)
            original_segments, detected_lang = asr_provider.transcribe(audio_path, None)

            original_subtitle = Subtitle(original_segments, detected_lang)
            target_subtitle, secondary_subtitle = _translate_subtitles(
                original_segments,
                detected_lang,
                translation_provider
            )

            cache_data = {
                "detected_language": detected_lang.value,
                "zh_segments": _serialize_segments(target_subtitle.segments),
                "en_segments": _serialize_segments(secondary_subtitle.segments)
            }

            if detected_lang not in [LanguageCode.CHINESE, LanguageCode.ENGLISH]:
                cache_data[f"{detected_lang.value}_segments"] = _serialize_segments(original_segments)

            cache_repo.set(cache_key, cache_data)
            print(f"  ✅ 完成: {video.path.name} ({detected_lang.value} -> zh, en)")

        result = VideoWithSubtitles(
            video=video,
            original_subtitle=original_subtitle,
            target_subtitle=target_subtitle,
            secondary_subtitle=secondary_subtitle,
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
        reference_audio_file: Path = None,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[VideoWithAudio, ...]:
    """阶段2: 批量 TTS（使用中文字幕）"""

    if progress:
        progress(0.0, "阶段2: 批量语音克隆")

    if not enable_voice_cloning:
        return _skip_voice_cloning(video_subtitles, progress)

    results = []
    total = len(video_subtitles)

    for idx, vs in enumerate(video_subtitles):
        if progress:
            progress(idx / total, f"TTS: 处理视频 {idx + 1}/{total} - {vs.video.path.name}")

        # ✅ 使用目标语言（中文）字幕
        cache_key = calculate_cache_key(
            vs.video.path,
            "voice_cloning",
            {
                "language": vs.target_subtitle.language.value,
                "reference": "auto",
                "num_segments": len(vs.target_subtitle.segments)
            }
        )

        cache_hit = cache_repo.exists(cache_key)
        audio_track = None

        if cache_hit:
            audio_track = _load_audio_from_cache(
                cache_repo,
                cache_key,
                vs.video.path.name,
                vs.target_subtitle.language
            )
            if audio_track is None:
                cache_hit = False

        if not cache_hit:
            audio_track = _batch_synthesize_segments(
                vs=vs,
                tts_provider=tts_provider,
                video_processor=video_processor,
                reference_audio_file=reference_audio_file,
                cache_repo=cache_repo,
                cache_key=cache_key
            )
            print(f"  ✅ 完成: {vs.video.path.name}")

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

    return tuple(results)


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
        cache_repo, enable_voice_cloning,reference_audio_file,
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