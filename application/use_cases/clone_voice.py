"""
语音克隆用例 - 修复缓存验证问题
"""

from application import *

# 导入领域层
from domain.entities import (
    Video, Subtitle, AudioTrack, VoiceProfile,
    AudioSample,
)

from domain.ports import (
    TTSProvider,
    VideoProcessor, CacheRepository,
)

from domain.services import (
    calculate_cache_key,
)


def validate_voice_cache(cached):
    """验证语音缓存数据的完整性"""
    if cached is None:
        return False

    required_keys = ["audio_samples", "sample_rate", "reference_audio", "reference_duration"]

    for key in required_keys:
        if key not in cached:
            return False

    if not isinstance(cached["audio_samples"], (list, tuple)):
        return False

    if not isinstance(cached["sample_rate"], int):
        return False

    if len(cached["audio_samples"]) == 0:
        return False

    return True


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
    语音克隆用例（纯函数）- 修复版

    流程:
    1. 检查缓存（带验证）
    2. 提取/使用参考音频
    3. 创建声音配置
    4. 逐句合成
    5. 拼接音频
    """
    if progress:
        progress(0.0, "开始语音克隆")

    # 1. 检查缓存（带完整性验证）
    cache_key = calculate_cache_key(
        video.path,
        "voice_cloning",
        {
            "language": subtitle.language.value,
            "reference": str(reference_audio_path) if reference_audio_path else "auto",
            "num_segments": len(subtitle.segments)
        }
    )

    cache_hit = False

    if cache_repo.exists(cache_key):
        cached = cache_repo.get(cache_key)

        # 验证缓存数据完整性
        if validate_voice_cache(cached):
            try:
                audio_sample = AudioSample(
                    samples=tuple(cached["audio_samples"]),
                    sample_rate=cached["sample_rate"]
                )

                voice_profile = VoiceProfile(
                    reference_audio_path=Path(cached["reference_audio"]),
                    language=subtitle.language,
                    duration=cached["reference_duration"]
                )

                if progress:
                    progress(1.0, "语音克隆缓存命中")

                print(f"💾 语音克隆缓存命中: {video.path.name}")

                return VoiceCloningResult(
                    audio_track=AudioTrack(audio_sample, subtitle.language),
                    voice_profile=voice_profile,
                    total_segments=len(subtitle.segments),
                    cache_hit=True
                )
            except Exception as e:
                print(f"⚠️  缓存数据解析失败: {e}，将重新生成")
                cache_hit = False
        else:
            print(f"⚠️  缓存数据损坏，将重新生成")
            cache_hit = False

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
        try:
            audio_sample = tts_provider.synthesize(
                text=segment.text,
                voice_profile=voice_profile,
                target_duration=segment.time_range.duration
            )

            synthesized_segments.append((audio_sample, segment))
        except Exception as e:
            print(f"❌ 片段 {idx} 合成失败: {e}")
            # 使用静音代替
            silent_samples = int(22050 * segment.time_range.duration)
            silent_audio = AudioSample(
                samples=tuple([0.0] * silent_samples),
                sample_rate=22050
            )
            synthesized_segments.append((silent_audio, segment))

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

    # 保存缓存（带完整数据）
    try:
        cache_data = {
            "audio_samples": list(full_audio.samples),
            "sample_rate": full_audio.sample_rate,
            "reference_audio": str(reference_audio_path),
            "reference_duration": reference_duration
        }

        # 再次验证要保存的数据
        if validate_voice_cache(cache_data):
            cache_repo.set(cache_key, cache_data)
            print(f"✅ 语音缓存已保存")
        else:
            print(f"⚠️  缓存数据验证失败，跳过保存")
    except Exception as e:
        print(f"⚠️  保存缓存失败: {e}")

    if progress:
        progress(1.0, "语音克隆完成")

    return VoiceCloningResult(
        audio_track=AudioTrack(full_audio, subtitle.language),
        voice_profile=voice_profile,
        total_segments=len(subtitle.segments),
        cache_hit=False
    )