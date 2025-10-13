"""
语音克隆用例 - 重构版

重构要点：
1. 提取缓存处理逻辑
2. 提取音频合成逻辑
3. 简化主流程
"""

from application import *
from domain.entities import (
    Video, Subtitle, AudioTrack, VoiceProfile,
    AudioSample,
)
from domain.ports import (
    TTSProvider, VideoProcessor, CacheRepository,
)
from domain.services import calculate_cache_key


# ============== 缓存辅助函数 ============== #

def _load_voice_cloning_from_cache(
        cached: dict,
        language: LanguageCode
) -> tuple[AudioTrack, VoiceProfile]:
    """从缓存加载语音克隆结果"""
    audio_sample = AudioSample(
        samples=tuple(cached["audio_samples"]),
        sample_rate=cached["sample_rate"]
    )

    voice_profile = VoiceProfile(
        reference_audio_path=Path(cached["reference_audio"]),
        language=language,
        duration=cached["reference_duration"]
    )

    return AudioTrack(audio_sample, language), voice_profile


def _save_voice_cloning_to_cache(
        cache_repo: CacheRepository,
        cache_key: str,
        audio: AudioSample,
        reference_audio_path: Path,
        reference_duration: float
):
    """保存语音克隆结果到缓存"""
    cache_data = {
        "audio_samples": list(audio.samples),
        "sample_rate": audio.sample_rate,
        "reference_audio": str(reference_audio_path),
        "reference_duration": reference_duration
    }
    cache_repo.set(cache_key, cache_data)


# ============== 音频处理辅助函数 ============== #

def _prepare_reference_audio(
        video: Video,
        video_processor: VideoProcessor,
        reference_audio_path: Optional[Path],
        reference_duration: float
) -> Path:
    """准备参考音频"""
    if reference_audio_path is not None:
        return reference_audio_path

    return video_processor.extract_reference_audio(video, reference_duration)


def _synthesize_single_segment(
        segment: TextSegment,
        tts_provider: TTSProvider,
        voice_profile: VoiceProfile
) -> tuple[AudioSample, TextSegment]:
    """合成单个音频片段"""
    audio_sample = tts_provider.synthesize(
        text=segment.text,
        voice_profile=voice_profile,
        target_duration=segment.time_range.duration
    )
    return audio_sample, segment


def _synthesize_all_segments(
        segments: tuple[TextSegment, ...],
        tts_provider: TTSProvider,
        voice_profile: VoiceProfile,
        progress: ProgressCallback
) -> list[tuple[AudioSample, TextSegment]]:
    """逐句合成所有音频片段"""
    synthesized_segments = []
    total_segments = len(segments)

    for idx, segment in enumerate(segments):
        if progress:
            prog = 0.2 + (idx / total_segments) * 0.7
            progress(prog, f"合成语音 {idx + 1}/{total_segments}")

        synthesized_segment = _synthesize_single_segment(segment, tts_provider, voice_profile)
        synthesized_segments.append(synthesized_segment)

    return synthesized_segments


def _create_empty_audio_buffer(video_duration: float, sample_rate: int) -> list[float]:
    """创建空音频缓冲区"""
    total_samples = int(video_duration * sample_rate)
    return [0.0] * total_samples


def _fill_audio_buffer(
        buffer: list[float],
        synthesized_segments: list[tuple[AudioSample, TextSegment]]
):
    """将合成的音频片段填充到缓冲区"""
    for audio_sample, segment in synthesized_segments:
        start_idx = int(segment.time_range.start_seconds * audio_sample.sample_rate)
        for i, sample in enumerate(audio_sample.samples):
            if start_idx + i < len(buffer):
                buffer[start_idx + i] = sample


def _merge_audio_segments(
        synthesized_segments: list[tuple[AudioSample, TextSegment]],
        video_duration: float
) -> AudioSample:
    """拼接所有音频片段"""
    sample_rate = synthesized_segments[0][0].sample_rate
    buffer = _create_empty_audio_buffer(video_duration, sample_rate)
    _fill_audio_buffer(buffer, synthesized_segments)

    return AudioSample(
        samples=tuple(buffer),
        sample_rate=sample_rate
    )


# ============== 主用例函数 ============== #

def clone_voice_use_case(
        video: Video,
        subtitle: Subtitle,
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        cache_repo: CacheRepository,
        reference_audio_path: Optional[Path] = None,
        reference_duration: float = 10.0,
        progress: ProgressCallback = None
) -> VoiceCloningResult:
    """
    语音克隆用例（重构版）

    流程简化：
    1. 检查缓存
    2. 准备参考音频
    3. 合成语音片段
    4. 拼接音频
    5. 保存缓存
    """
    if progress:
        progress(0.0, "开始语音克隆")

    # 1. 计算缓存键并检查缓存
    cache_key = calculate_cache_key(
        video.path,
        "voice_cloning",
        {
            "language": subtitle.language.value,
            "reference": str(reference_audio_path) if reference_audio_path else "auto",
            "num_segments": len(subtitle.segments)
        }
    )

    if cache_repo.exists(cache_key):
        cached = cache_repo.get(cache_key)
        audio_track, voice_profile = _load_voice_cloning_from_cache(cached, subtitle.language)

        if progress:
            progress(1.0, "语音克隆缓存命中")

        return VoiceCloningResult(
            audio_track=audio_track,
            voice_profile=voice_profile,
            total_segments=len(subtitle.segments),
            cache_hit=True
        )

    # 2. 准备参考音频
    if progress:
        progress(0.1, "准备参考音频")

    reference_audio = _prepare_reference_audio(
        video, video_processor, reference_audio_path, reference_duration
    )

    # 3. 创建声音配置
    voice_profile = VoiceProfile(
        reference_audio_path=reference_audio,
        language=subtitle.language,
        duration=reference_duration
    )

    # 4. 合成所有音频片段
    if progress:
        progress(0.2, "合成语音")

    synthesized_segments = _synthesize_all_segments(
        subtitle.segments,
        tts_provider,
        voice_profile,
        progress
    )

    # 5. 拼接音频
    if progress:
        progress(0.9, "拼接音频")

    full_audio = _merge_audio_segments(synthesized_segments, video.duration)

    # 6. 保存缓存
    _save_voice_cloning_to_cache(
        cache_repo,
        cache_key,
        full_audio,
        reference_audio,
        reference_duration
    )

    if progress:
        progress(1.0, "语音克隆完成")

    return VoiceCloningResult(
        audio_track=AudioTrack(full_audio, subtitle.language),
        voice_profile=voice_profile,
        total_segments=len(subtitle.segments),
        cache_hit=False
    )