"""
Application Layer - 增量语音克隆用例
支持分段合成、缓存和增量更新
"""
import time
from pathlib import Path
from typing import Optional, Callable, Dict

from domain.entities import (
    Video, Subtitle, AudioSegment, IncrementalSynthesisResult,
    VoiceProfile, AudioSample, AudioTrack
)
from domain.ports import (
    TTSProvider, VideoProcessor,
    CacheRepository, AudioSegmentRepository
)


# ============== 辅助函数 ============== #

def _get_segment_cache_key(
        video_path: Path,
        segment_index: int,
        text: str
) -> str:
    """生成片段缓存键（基于内容哈希）"""
    import hashlib
    content = f"{video_path.name}_{segment_index}_{text}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _check_cached_segments(
        subtitle: Subtitle,
        video: Video,
        audio_repo: AudioSegmentRepository
) -> Dict[int, AudioSegment]:
    """检查哪些片段已有缓存"""
    cached = {}

    for idx, seg in enumerate(subtitle.segments):
        if audio_repo.exists(idx, video.path):
            audio_seg = audio_repo.load_segment(idx, video.path, seg)
            if audio_seg:
                cached[idx] = audio_seg

    return cached


def _synthesize_missing_segments(
        subtitle: Subtitle,
        cached_segments: Dict[int, AudioSegment],
        tts_provider: TTSProvider,
        voice_profile: VoiceProfile,
        video: Video,
        audio_repo: AudioSegmentRepository,
        progress: Optional[Callable[[float, str, int, AudioSegment], None]] = None
) -> Dict[int, AudioSegment]:
    """
    合成缺失的片段

    Args:
        progress: 进度回调 (ratio, message, segment_index, audio_segment)
    """
    all_segments = {}
    all_segments.update(cached_segments)

    total = len(subtitle.segments)
    missing_indices = [
        idx for idx in range(total)
        if idx not in cached_segments
    ]

    if not missing_indices:
        return all_segments

    print(f"  🎤 需要合成 {len(missing_indices)} 个新片段")

    for i, idx in enumerate(missing_indices):
        text_seg = subtitle.segments[idx]

        if progress:
            ratio = i / len(missing_indices)
            progress(ratio, f"合成片段 {idx + 1}/{total}", idx, None)

        # 单段合成
        audio_sample = tts_provider.synthesize(
            text=text_seg.text,
            voice_profile=voice_profile,
            target_duration=text_seg.time_range.duration
        )

        # 创建实体
        cache_key = _get_segment_cache_key(
            video.path, idx, text_seg.text
        )

        audio_seg = AudioSegment(
            segment_index=idx,
            audio=audio_sample,
            text_segment=text_seg,
            cache_key=cache_key
        )

        # 保存到仓储
        file_path = audio_repo.save_segment(idx, audio_seg, video.path)
        audio_seg = audio_seg.with_file_path(file_path)

        all_segments[idx] = audio_seg

        # 实时回调（携带音频片段）
        if progress:
            progress(
                (i + 1) / len(missing_indices),
                f"完成片段 {idx + 1}/{total}",
                idx,
                audio_seg
            )

        print(f"  ✅ 片段 {idx} 已合成并缓存")

    return all_segments


def _merge_segments_to_track(
        audio_segments: Dict[int, AudioSegment],
        video_duration: float,
        language
) -> AudioTrack:
    """将片段合并为完整音轨"""
    if not audio_segments:
        raise ValueError("没有音频片段可合并")

    # 按索引排序
    sorted_segments = sorted(audio_segments.items())

    sample_rate = sorted_segments[0][1].audio.sample_rate
    total_samples = int(video_duration * sample_rate)
    buffer = [0.0] * total_samples

    for idx, audio_seg in sorted_segments:
        text_seg = audio_seg.text_segment
        start_idx = int(text_seg.time_range.start_seconds * sample_rate)

        for i, sample in enumerate(audio_seg.audio.samples):
            target_idx = start_idx + i
            if target_idx < total_samples:
                buffer[target_idx] = sample

    full_audio = AudioSample(
        samples=tuple(buffer),
        sample_rate=sample_rate
    )

    return AudioTrack(full_audio, language)


# ============== 主用例函数 ============== #

def incremental_voice_cloning_use_case(
        video: Video,
        subtitle: Subtitle,
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        audio_repo: AudioSegmentRepository,
        cache_repo: CacheRepository,
        reference_audio_path: Optional[Path] = None,
        reference_duration: float = 10.0,
        progress: Optional[Callable[[float, str, int, Optional[AudioSegment]], None]] = None
) -> IncrementalSynthesisResult:
    """
    增量语音克隆用例

    特性:
    1. 逐片段合成
    2. 实时缓存
    3. 断点续传
    4. 进度回调携带音频片段

    Args:
        progress: 回调函数 (ratio, message, segment_index, audio_segment)
            - ratio: 进度比例 0.0-1.0
            - message: 进度描述
            - segment_index: 当前片段索引
            - audio_segment: 当前完成的音频片段（可选）
    """
    start_time = time.perf_counter()

    if progress:
        progress(0.0, "开始增量语音克隆", -1, None)

    # 1. 准备参考音频
    if progress:
        progress(0.05, "准备参考音频", -1, None)

    if reference_audio_path is None:
        reference_audio_path = video_processor.extract_reference_audio(
            video, reference_duration
        )

    voice_profile = VoiceProfile(
        reference_audio_path=reference_audio_path,
        language=subtitle.language,
        duration=reference_duration
    )

    # 2. 检查已缓存片段
    if progress:
        progress(0.1, "检查缓存", -1, None)

    cached_segments = _check_cached_segments(subtitle, video, audio_repo)
    print(f"  💾 缓存命中: {len(cached_segments)}/{len(subtitle.segments)} 片段")

    # 3. 合成缺失片段（带实时回调）
    def synthesis_progress(ratio, msg, idx, audio_seg):
        overall_ratio = 0.1 + ratio * 0.8  # 10%-90%
        if progress:
            progress(overall_ratio, msg, idx, audio_seg)

    all_segments = _synthesize_missing_segments(
        subtitle=subtitle,
        cached_segments=cached_segments,
        tts_provider=tts_provider,
        voice_profile=voice_profile,
        video=video,
        audio_repo=audio_repo,
        progress=synthesis_progress
    )

    # 4. 合并为完整音轨
    if progress:
        progress(0.9, "合并音频片段", -1, None)

    audio_track = _merge_segments_to_track(
        all_segments, video.duration, subtitle.language
    )

    # 5. 保存完整音轨缓存（可选）
    if progress:
        progress(0.95, "保存完整音轨", -1, None)

    synthesis_time = time.perf_counter() - start_time

    if progress:
        progress(1.0, "增量合成完成", -1, None)

    return IncrementalSynthesisResult(
        total_segments=len(subtitle.segments),
        cached_segments=len(cached_segments),
        regenerated_segments=len(subtitle.segments) - len(cached_segments),
        audio_segments=tuple(
            all_segments[i] for i in sorted(all_segments.keys())
        ),
        synthesis_time=synthesis_time
    )


def regenerate_modified_segments_use_case(
        video: Video,
        original_subtitle: Subtitle,
        modified_subtitle: Subtitle,
        modified_indices: set[int],
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        audio_repo: AudioSegmentRepository,
        reference_audio_path: Path,
        progress: Optional[Callable[[float, str, int, Optional[AudioSegment]], None]] = None
) -> IncrementalSynthesisResult:
    """
    重新生成修改过的片段

    Args:
        modified_indices: 被修改的片段索引集合
    """
    start_time = time.perf_counter()

    if not modified_indices:
        print("  ℹ️  没有修改的片段，跳过重新生成")
        # 加载所有现有片段
        all_segments = {}
        for idx, seg in enumerate(modified_subtitle.segments):
            audio_seg = audio_repo.load_segment(idx, video.path, seg)
            if audio_seg:
                all_segments[idx] = audio_seg

        return IncrementalSynthesisResult(
            total_segments=len(modified_subtitle.segments),
            cached_segments=len(all_segments),
            regenerated_segments=0,
            audio_segments=tuple(
                all_segments[i] for i in sorted(all_segments.keys())
            ),
            synthesis_time=0.0
        )

    print(f"  🔄 需要重新生成 {len(modified_indices)} 个片段")

    voice_profile = VoiceProfile(
        reference_audio_path=reference_audio_path,
        language=modified_subtitle.language,
        duration=10.0
    )

    # 重新生成修改的片段
    regenerated = {}
    total = len(modified_indices)

    for i, idx in enumerate(sorted(modified_indices)):
        text_seg = modified_subtitle.segments[idx]

        if progress:
            progress(
                i / total,
                f"重新生成片段 {idx + 1}",
                idx,
                None
            )

        # 删除旧缓存
        audio_repo.delete_segment(idx, video.path)

        # 合成新音频
        audio_sample = tts_provider.synthesize(
            text=text_seg.text,
            voice_profile=voice_profile,
            target_duration=text_seg.time_range.duration
        )

        cache_key = _get_segment_cache_key(
            video.path, idx, text_seg.text
        )

        audio_seg = AudioSegment(
            segment_index=idx,
            audio=audio_sample,
            text_segment=text_seg,
            cache_key=cache_key
        )

        # 保存
        file_path = audio_repo.save_segment(idx, audio_seg, video.path)
        audio_seg = audio_seg.with_file_path(file_path)

        regenerated[idx] = audio_seg

        if progress:
            progress(
                (i + 1) / total,
                f"完成片段 {idx + 1}",
                idx,
                audio_seg
            )

        print(f"  ✅ 片段 {idx} 已重新生成")

    # 加载未修改的片段
    all_segments = dict(regenerated)
    for idx in range(len(modified_subtitle.segments)):
        if idx not in modified_indices:
            audio_seg = audio_repo.load_segment(
                idx, video.path, modified_subtitle.segments[idx]
            )
            if audio_seg:
                all_segments[idx] = audio_seg

    synthesis_time = time.perf_counter() - start_time

    return IncrementalSynthesisResult(
        total_segments=len(modified_subtitle.segments),
        cached_segments=len(all_segments) - len(regenerated),
        regenerated_segments=len(regenerated),
        audio_segments=tuple(
            all_segments[i] for i in sorted(all_segments.keys())
        ),
        synthesis_time=synthesis_time
    )