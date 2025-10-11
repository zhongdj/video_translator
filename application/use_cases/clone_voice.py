from application import *

# 导入领域层
from domain.entities import (
    # Entities
    Video, Subtitle, AudioTrack, VoiceProfile,  # Value Objects
    AudioSample,
)

from domain.ports import (
# Ports
    TTSProvider,
    VideoProcessor, CacheRepository,

)

from domain.services import (
    # Domain Services
    calculate_cache_key, )

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
    语音克隆用例（纯函数）

    流程:
    1. 检查缓存
    2. 提取/使用参考音频
    3. 创建声音配置
    4. 逐句合成
    5. 拼接音频
    """
    if progress:
        progress(0.0, "开始语音克隆")

    # 1. 检查缓存
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
        if progress:
            progress(1.0, "语音克隆缓存命中")

        # 从缓存重建
        audio_sample = AudioSample(
            samples=tuple(cached["audio_samples"]),
            sample_rate=cached["sample_rate"]
        )

        voice_profile = VoiceProfile(
            reference_audio_path=Path(cached["reference_audio"]),
            language=subtitle.language,
            duration=cached["reference_duration"]
        )

        return VoiceCloningResult(
            audio_track=AudioTrack(audio_sample, subtitle.language),
            voice_profile=voice_profile,
            total_segments=len(subtitle.segments),
            cache_hit=True
        )

    # 2. 提取或使用参考音频
    if progress:
        progress(0.1, "准备参考音频")

    if reference_audio_path is None:
        reference_audio_path = video_processor.extract_reference_audio(
            video,
            reference_duration
        )

    # 3. 创建声音配置
    voice_profile = VoiceProfile(
        reference_audio_path=reference_audio_path,
        language=subtitle.language,
        duration=reference_duration
    )

    # 4. 逐句合成
    if progress:
        progress(0.2, "合成语音")

    synthesized_segments = []
    total_segments = len(subtitle.segments)

    for idx, segment in enumerate(subtitle.segments):
        if progress:
            prog = 0.2 + (idx / total_segments) * 0.7
            progress(prog, f"合成语音 {idx + 1}/{total_segments}")

        # 合成单个片段
        audio_sample = tts_provider.synthesize(
            text=segment.text,
            voice_profile=voice_profile,
            target_duration=segment.time_range.duration
        )

        synthesized_segments.append((audio_sample, segment))

    # 5. 拼接音频（创建完整音轨）
    if progress:
        progress(0.9, "拼接音频")

    # 计算总样本数
    total_samples = int(video.duration * synthesized_segments[0][0].sample_rate)
    full_audio_list = [0.0] * total_samples

    # 填充各个片段
    for audio_sample, segment in synthesized_segments:
        start_idx = int(segment.time_range.start_seconds * audio_sample.sample_rate)
        for i, sample in enumerate(audio_sample.samples):
            if start_idx + i < total_samples:
                full_audio_list[start_idx + i] = sample

    full_audio = AudioSample(
        samples=tuple(full_audio_list),
        sample_rate=synthesized_segments[0][0].sample_rate
    )

    # 保存缓存
    cache_data = {
        "audio_samples": list(full_audio.samples),
        "sample_rate": full_audio.sample_rate,
        "reference_audio": str(reference_audio_path),
        "reference_duration": reference_duration
    }
    cache_repo.set(cache_key, cache_data)

    if progress:
        progress(1.0, "语音克隆完成")

    return VoiceCloningResult(
        audio_track=AudioTrack(full_audio, subtitle.language),
        voice_profile=voice_profile,
        total_segments=len(subtitle.segments),
        cache_hit=False
    )