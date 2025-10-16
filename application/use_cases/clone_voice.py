"""
语音克隆用例 - 持久化缓存版
确保GPU生成的音频数据能够持久化保存，支持断点续传
"""

from pathlib import Path
from typing import Optional
import pickle
import hashlib
import array

from domain.entities import (
    Video, Subtitle, AudioTrack, VoiceProfile, AudioSample, TextSegment, LanguageCode
)
from domain.ports import TTSProvider, VideoProcessor, CacheRepository
from domain.services import calculate_cache_key


# ============== 音频文件缓存管理 ============== #

class AudioFileCache:
    """音频文件缓存管理器"""

    def __init__(self, cache_dir: Path = Path(".cache/audio")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_audio_file_path(self, cache_key: str) -> Path:
        """获取音频文件路径"""
        return self.cache_dir / f"{cache_key}.wav"

    def get_metadata_file_path(self, cache_key: str) -> Path:
        """获取元数据文件路径"""
        return self.cache_dir / f"{cache_key}.meta"

    def save_audio_data(self, cache_key: str, audio_sample: AudioSample, metadata: dict) -> bool:
        """保存音频数据和元数据"""
        try:
            # 保存音频数据为WAV文件
            audio_path = self.get_audio_file_path(cache_key)
            self._save_as_wav(audio_path, audio_sample)

            # 保存元数据
            meta_path = self.get_metadata_file_path(cache_key)
            with open(meta_path, 'wb') as f:
                pickle.dump(metadata, f)

            print(f"✅ 音频数据已保存: {audio_path}")
            return True

        except Exception as e:
            print(f"❌ 保存音频数据失败: {e}")
            return False

    def load_audio_data(self, cache_key: str) -> tuple[Optional[AudioSample], Optional[dict]]:
        """加载音频数据和元数据"""
        try:
            audio_path = self.get_audio_file_path(cache_key)
            meta_path = self.get_metadata_file_path(cache_key)

            if not audio_path.exists() or not meta_path.exists():
                return None, None

            # 加载元数据
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)

            # 加载音频数据
            audio_sample = self._load_from_wav(audio_path)

            print(f"✅ 音频数据已加载: {audio_path}")
            return audio_sample, metadata

        except Exception as e:
            print(f"❌ 加载音频数据失败: {e}")
            return None, None

    def _save_as_wav(self, file_path: Path, audio_sample: AudioSample):
        """将音频数据保存为WAV文件"""
        import wave
        import struct

        # 将浮点数转换为16位PCM
        samples_int16 = [int(sample * 32767) for sample in audio_sample.samples]

        with wave.open(str(file_path), 'w') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(audio_sample.sample_rate)

            # 写入数据
            for sample in samples_int16:
                wav_file.writeframes(struct.pack('<h', sample))

    def _load_from_wav(self, file_path: Path) -> AudioSample:
        """从WAV文件加载音频数据"""
        import wave
        import struct

        with wave.open(str(file_path), 'r') as wav_file:
            # 获取参数
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            # 读取数据
            frames = wav_file.readframes(n_frames)

            # 转换为浮点数
            samples = []
            for i in range(0, len(frames), 2):
                sample = struct.unpack('<h', frames[i:i+2])[0]
                samples.append(sample / 32767.0)  # 转换为[-1, 1]范围的浮点数

            return AudioSample(
                samples=tuple(samples),
                sample_rate=sample_rate
            )


# ============== 分段合成状态管理 ============== #

class SegmentSynthesisState:
    """分段合成状态管理器"""

    def __init__(self, cache_dir: Path = Path(".cache/segments")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_segment_file_path(self, cache_key: str, segment_index: int) -> Path:
        """获取分段音频文件路径"""
        return self.cache_dir / f"{cache_key}_seg_{segment_index:04d}.wav"

    def save_segment(self, cache_key: str, segment_index: int, audio_sample: AudioSample) -> bool:
        """保存分段音频"""
        try:
            file_path = self.get_segment_file_path(cache_key, segment_index)

            import wave
            import struct

            # 将浮点数转换为16位PCM
            samples_int16 = [int(sample * 32767) for sample in audio_sample.samples]

            with wave.open(str(file_path), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(audio_sample.sample_rate)

                for sample in samples_int16:
                    wav_file.writeframes(struct.pack('<h', sample))

            return True

        except Exception as e:
            print(f"❌ 保存分段 {segment_index} 失败: {e}")
            return False

    def load_segment(self, cache_key: str, segment_index: int) -> Optional[AudioSample]:
        """加载分段音频"""
        try:
            file_path = self.get_segment_file_path(cache_key, segment_index)

            if not file_path.exists():
                return None

            import wave
            import struct

            with wave.open(str(file_path), 'r') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                frames = wav_file.readframes(n_frames)

                samples = []
                for i in range(0, len(frames), 2):
                    sample = struct.unpack('<h', frames[i:i+2])[0]
                    samples.append(sample / 32767.0)

                return AudioSample(
                    samples=tuple(samples),
                    sample_rate=sample_rate
                )

        except Exception as e:
            print(f"❌ 加载分段 {segment_index} 失败: {e}")
            return None

    def get_completed_segments(self, cache_key: str, total_segments: int) -> list[int]:
        """获取已完成的段落下标列表"""
        completed = []
        for i in range(total_segments):
            file_path = self.get_segment_file_path(cache_key, i)
            if file_path.exists():
                completed.append(i)
        return completed


# ============== 缓存辅助函数 ============== #

def _load_voice_cloning_from_cache(
        audio_cache: AudioFileCache,
        cache_key: str
) -> tuple[AudioTrack, VoiceProfile]:
    """从文件缓存加载语音克隆结果"""
    audio_sample, metadata = audio_cache.load_audio_data(cache_key)

    if audio_sample is None or metadata is None:
        raise ValueError("缓存数据加载失败")

    voice_profile = VoiceProfile(
        reference_audio_path=Path(metadata["reference_audio"]),
        language=LanguageCode(metadata["language"]),
        duration=metadata["reference_duration"]
    )

    return AudioTrack(audio_sample, voice_profile.language), voice_profile


def _save_voice_cloning_to_cache(
        audio_cache: AudioFileCache,
        cache_key: str,
        audio: AudioSample,
        voice_profile: VoiceProfile
):
    """保存语音克隆结果到文件缓存"""
    metadata = {
        "reference_audio": str(voice_profile.reference_audio_path),
        "language": voice_profile.language.value,
        "reference_duration": voice_profile.duration,
        "sample_rate": audio.sample_rate,
        "num_samples": len(audio.samples)
    }

    audio_cache.save_audio_data(cache_key, audio, metadata)


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


def _synthesize_segments_with_checkpoint(
        segments: tuple[TextSegment, ...],
        tts_provider: TTSProvider,
        voice_profile: VoiceProfile,
        segment_state: SegmentSynthesisState,
        cache_key: str,
        progress_callback: Optional[callable]
) -> list[tuple[AudioSample, TextSegment]]:
    """带检查点的分段合成 - 支持断点续传"""
    synthesized_segments = []
    total_segments = len(segments)

    # 检查已完成的片段
    completed_indices = segment_state.get_completed_segments(cache_key, total_segments)
    print(f"📊 断点续传: 已完成 {len(completed_indices)}/{total_segments} 个片段")

    for idx, segment in enumerate(segments):
        # 如果这个片段已经完成，直接加载
        if idx in completed_indices:
            if progress_callback:
                progress = 0.2 + (idx / total_segments) * 0.7
                progress_callback(progress, f"加载已合成片段 {idx + 1}/{total_segments}")

            audio_sample = segment_state.load_segment(cache_key, idx)
            if audio_sample:
                synthesized_segments.append((audio_sample, segment))
                continue

        # 合成新片段
        if progress_callback:
            progress = 0.2 + (idx / total_segments) * 0.7
            progress_callback(progress, f"合成语音 {idx + 1}/{total_segments}")

        audio_sample, segment = _synthesize_single_segment(segment, tts_provider, voice_profile)

        # 立即保存片段到磁盘
        if segment_state.save_segment(cache_key, idx, audio_sample):
            print(f"✅ 片段 {idx} 已保存到磁盘")

        synthesized_segments.append((audio_sample, segment))

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
    if not synthesized_segments:
        raise ValueError("没有可合成的音频片段")

    sample_rate = synthesized_segments[0][0].sample_rate
    buffer = _create_empty_audio_buffer(video_duration, sample_rate)
    _fill_audio_buffer(buffer, synthesized_segments)

    return AudioSample(
        samples=tuple(buffer),
        sample_rate=sample_rate
    )


# ============== 主用例函数 ============== #

class VoiceCloningResult:
    """语音克隆结果"""
    def __init__(self, audio_track: AudioTrack, voice_profile: VoiceProfile,
                 total_segments: int, cache_hit: bool):
        self.audio_track = audio_track
        self.voice_profile = voice_profile
        self.total_segments = total_segments
        self.cache_hit = cache_hit


def clone_voice_use_case(
        video: Video,
        subtitle: Subtitle,
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        cache_repo: CacheRepository,
        reference_audio_path: Optional[Path] = None,
        reference_duration: float = 10.0,
        progress: Optional[callable] = None
) -> VoiceCloningResult:
    """
    语音克隆用例（持久化缓存版）
    支持断点续传，确保GPU耗时任务的结果能够持久化保存
    """
    if progress:
        progress(0.0, "开始语音克隆")

    # 初始化缓存管理器
    audio_cache = AudioFileCache()
    segment_state = SegmentSynthesisState()

    # 1. 计算缓存键
    cache_key = calculate_cache_key(
        video.path,
        "clone_voice",
        {
            "target_language": subtitle.language.value,
            "source_language": "auto",
            "reference_audio_hash": str(reference_audio_path) if reference_audio_path else "default"
        }
    )

    # 2. 检查完整音频缓存
    try:
        audio_sample, metadata = audio_cache.load_audio_data(cache_key)
        if audio_sample is not None:
            voice_profile = VoiceProfile(
                reference_audio_path=Path(metadata["reference_audio"]),
                language=LanguageCode(metadata["language"]),
                duration=metadata["reference_duration"]
            )

            if progress:
                progress(1.0, "语音克隆完整缓存命中")

            print("✅ 使用完整音频缓存")
            return VoiceCloningResult(
                audio_track=AudioTrack(audio_sample, subtitle.language),
                voice_profile=voice_profile,
                total_segments=len(subtitle.segments),
                cache_hit=True
            )
    except Exception as e:
        print(f"⚠️ 完整缓存加载失败: {e}")

    # 3. 准备参考音频
    if progress:
        progress(0.1, "准备参考音频")

    reference_audio = _prepare_reference_audio(
        video, video_processor, reference_audio_path, reference_duration
    )

    # 4. 创建声音配置
    voice_profile = VoiceProfile(
        reference_audio_path=reference_audio,
        language=subtitle.language,
        duration=reference_duration
    )

    # 5. 带检查点的分段合成
    if progress:
        progress(0.2, "合成语音片段")

    synthesized_segments = _synthesize_segments_with_checkpoint(
        subtitle.segments,
        tts_provider,
        voice_profile,
        segment_state,
        cache_key,
        progress
    )

    tts_provider.unload()

    # 6. 拼接音频
    if progress:
        progress(0.9, "拼接音频")

    full_audio = _merge_audio_segments(synthesized_segments, video.duration)

    # 7. 保存完整音频到缓存
    try:
        _save_voice_cloning_to_cache(
            audio_cache,
            cache_key,
            full_audio,
            voice_profile
        )
        print("✅ 完整音频已保存到缓存")
    except Exception as e:
        print(f"⚠️ 完整音频缓存保存失败: {e}")

    # 8. 清理分段缓存（可选，为了节省空间）
    try:
        for i in range(len(subtitle.segments)):
            segment_file = segment_state.get_segment_file_path(cache_key, i)
            if segment_file.exists():
                segment_file.unlink()
        print("✅ 分段缓存已清理")
    except Exception as e:
        print(f"⚠️ 分段缓存清理失败: {e}")

    if progress:
        progress(1.0, "语音克隆完成")

    return VoiceCloningResult(
        audio_track=AudioTrack(full_audio, subtitle.language),
        voice_profile=voice_profile,
        total_segments=len(subtitle.segments),
        cache_hit=False
    )