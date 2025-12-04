"""
Application Layer - 多说话人语音克隆用例
"""
import time
from pathlib import Path
from typing import Optional, Callable, Dict

from domain.entities import (
    Video, Subtitle, AudioSegment, IncrementalSynthesisResult,
    AudioSample, AudioTrack
)
from domain.multi_speaker import (
    MultiSpeakerConfig, SpeakerId, MultiSpeakerVoiceProfile
)
from domain.ports import (
    TTSProvider, VideoProcessor,
    CacheRepository, AudioSegmentRepository
)


def _get_segment_cache_key_multi(
        video_path: Path,
        segment_index: int,
        text: str,
        speaker_id: str
) -> str:
    """生成多说话人片段缓存键"""
    import hashlib
    content = f"{video_path.name}_{segment_index}_{text}_{speaker_id}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _synthesize_segment_with_speaker(
        text_seg,
        segment_index: int,
        voice_profile: MultiSpeakerVoiceProfile,
        tts_provider: TTSProvider,
        video: Video,
        audio_repo: AudioSegmentRepository
) -> AudioSegment:
    """使用指定说话人合成单个片段"""
    from domain.entities import VoiceProfile

    # 转换为单说话人 VoiceProfile（适配现有 TTS 接口）
    single_voice_profile = VoiceProfile(
        reference_audio_path=voice_profile.reference_audio_path,
        language=voice_profile.language,
        duration=voice_profile.duration
    )

    # 合成音频
    audio_sample = tts_provider.synthesize(
        text=text_seg.text,
        voice_profile=single_voice_profile,
        target_duration=text_seg.time_range.duration
    )

    # 创建缓存键
    cache_key = _get_segment_cache_key_multi(
        video.path,
        segment_index,
        text_seg.text,
        voice_profile.speaker_id.id
    )

    # 创建音频片段实体
    audio_seg = AudioSegment(
        segment_index=segment_index,
        audio=audio_sample,
        text_segment=text_seg,
        cache_key=cache_key
    )

    # 保存到仓储
    file_path = audio_repo.save_segment(segment_index, audio_seg, video.path)
    return audio_seg.with_file_path(file_path)


def multi_speaker_voice_cloning_use_case(
        video: Video,
        subtitle: Subtitle,
        multi_speaker_config: MultiSpeakerConfig,
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        audio_repo: AudioSegmentRepository,
        cache_repo: CacheRepository,
        progress: Optional[Callable[[float, str, int, Optional[AudioSegment]], None]] = None
) -> IncrementalSynthesisResult:
    """
    多说话人语音克隆用例

    Args:
        video: 视频对象
        subtitle: 字幕对象
        multi_speaker_config: 多说话人配置
        tts_provider: TTS 提供者
        video_processor: 视频处理器
        audio_repo: 音频片段仓储
        cache_repo: 缓存仓储
        progress: 进度回调

    Returns:
        IncrementalSynthesisResult: 合成结果
    """
    start_time = time.perf_counter()

    if progress:
        progress(0.0, "开始多说话人语音克隆", -1, None)

    # 统计各说话人的片段数
    speaker_stats = {}
    for idx in range(len(subtitle.segments)):
        speaker_id = multi_speaker_config.get_speaker_for_segment(idx)
        speaker_stats[speaker_id.id] = speaker_stats.get(speaker_id.id, 0) + 1

    print(f"\n📊 多说话人配置:")
    print(f"   总片段数: {len(subtitle.segments)}")
    print(f"   说话人数: {len(multi_speaker_config.voice_profiles)}")
    for speaker_id, count in speaker_stats.items():
        print(f"   - {speaker_id}: {count} 个片段")

    # 检查缓存
    cached_segments = {}
    missing_indices = []

    for idx, text_seg in enumerate(subtitle.segments):
        speaker_id = multi_speaker_config.get_speaker_for_segment(idx)

        # 尝试加载缓存
        audio_seg = audio_repo.load_segment(idx, video.path, text_seg)

        # 验证缓存的说话人是否匹配
        if audio_seg:
            expected_cache_key = _get_segment_cache_key_multi(
                video.path, idx, text_seg.text, speaker_id.id
            )
            if audio_seg.cache_key == expected_cache_key:
                cached_segments[idx] = audio_seg
            else:
                missing_indices.append(idx)
        else:
            missing_indices.append(idx)

    print(f"  💾 缓存命中: {len(cached_segments)}/{len(subtitle.segments)}")

    # 合成缺失片段
    all_segments = dict(cached_segments)

    if missing_indices:
        print(f"  🎤 需要合成 {len(missing_indices)} 个片段")

        for i, idx in enumerate(missing_indices):
            text_seg = subtitle.segments[idx]
            speaker_id = multi_speaker_config.get_speaker_for_segment(idx)
            voice_profile = multi_speaker_config.get_voice_profile(speaker_id)

            if progress:
                ratio = i / len(missing_indices)
                progress(
                    ratio,
                    f"合成片段 {idx + 1}/{len(subtitle.segments)} [说话人: {speaker_id.name}]",
                    idx,
                    None
                )

            # 合成
            audio_seg = _synthesize_segment_with_speaker(
                text_seg=text_seg,
                segment_index=idx,
                voice_profile=voice_profile,
                tts_provider=tts_provider,
                video=video,
                audio_repo=audio_repo
            )

            all_segments[idx] = audio_seg

            if progress:
                progress(
                    (i + 1) / len(missing_indices),
                    f"完成片段 {idx + 1} [说话人: {speaker_id.name}]",
                    idx,
                    audio_seg
                )

            print(f"  ✅ 片段 {idx} 已合成 [说话人: {speaker_id.name}]")

    synthesis_time = time.perf_counter() - start_time

    if progress:
        progress(1.0, "多说话人合成完成", -1, None)

    return IncrementalSynthesisResult(
        total_segments=len(subtitle.segments),
        cached_segments=len(cached_segments),
        regenerated_segments=len(missing_indices),
        audio_segments=tuple(
            all_segments[i] for i in sorted(all_segments.keys())
        ),
        synthesis_time=synthesis_time
    )