"""
Infrastructure Layer - 音频片段仓储适配器
基于文件系统的实现
"""
import json
from pathlib import Path
from typing import Optional, List
import numpy as np
import soundfile as sf

from domain.entities import AudioSegment, TextSegment, AudioSample, TimeRange, LanguageCode
from domain.ports import AudioSegmentRepository
from domain.services import calculate_cache_key


class AudioSegmentRepositoryAdapter(AudioSegmentRepository):
    """音频片段仓储适配器（文件系统）"""

    def __init__(self, base_dir: Path = Path(".cache/audio_segments")):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_video_dir(self, video_path: Path) -> Path:
        """获取视频对应的缓存目录"""
        # 使用视频文件名作为目录名
        video_hash = calculate_cache_key(
            video_path,
            "audio_segments",
            {}
        )[:16]
        video_dir = self.base_dir / f"{video_path.stem}_{video_hash}"
        video_dir.mkdir(exist_ok=True)
        return video_dir

    def _get_segment_files(
            self,
            segment_index: int,
            video_path: Path
    ) -> tuple[Path, Path]:
        """获取片段的音频文件和元数据文件路径"""
        video_dir = self._get_video_dir(video_path)
        audio_file = video_dir / f"seg_{segment_index:04d}.wav"
        meta_file = video_dir / f"seg_{segment_index:04d}.json"
        return audio_file, meta_file

    def save_segment(
            self,
            segment_index: int,
            audio_segment: AudioSegment,
            video_path: Path
    ) -> Path:
        """保存音频片段"""
        audio_file, meta_file = self._get_segment_files(
            segment_index, video_path
        )

        # 保存音频
        audio_data = np.array(
            audio_segment.audio.samples,
            dtype=np.float32
        )
        sf.write(
            str(audio_file),
            audio_data,
            audio_segment.audio.sample_rate
        )

        # 保存元数据
        meta_data = {
            "segment_index": segment_index,
            "text": audio_segment.text_segment.text,
            "start": audio_segment.text_segment.time_range.start_seconds,
            "end": audio_segment.text_segment.time_range.end_seconds,
            "language": audio_segment.text_segment.language.value,
            "sample_rate": audio_segment.audio.sample_rate,
            "cache_key": audio_segment.cache_key
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        return audio_file

    def load_segment(
            self,
            segment_index: int,
            video_path: Path,
            text_segment: TextSegment
    ) -> Optional[AudioSegment]:
        """加载音频片段"""
        audio_file, meta_file = self._get_segment_files(
            segment_index, video_path
        )

        if not audio_file.exists() or not meta_file.exists():
            return None

        try:
            # 加载元数据
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)

            # 验证文本是否匹配（简单校验）
            if meta_data["text"] != text_segment.text:
                print(f"  ⚠️  片段 {segment_index} 文本不匹配，缓存失效")
                return None

            # 加载音频
            audio_data, sample_rate = sf.read(str(audio_file))

            audio_sample = AudioSample(
                samples=tuple(float(s) for s in audio_data),
                sample_rate=sample_rate
            )

            # 重建 TextSegment（使用缓存的数据）
            cached_text_segment = TextSegment(
                text=meta_data["text"],
                time_range=TimeRange(
                    meta_data["start"],
                    meta_data["end"]
                ),
                language=LanguageCode(meta_data["language"])
            )

            audio_segment = AudioSegment(
                segment_index=segment_index,
                audio=audio_sample,
                text_segment=cached_text_segment,
                cache_key=meta_data["cache_key"],
                file_path=audio_file
            )

            return audio_segment

        except Exception as e:
            print(f"  ⚠️  加载片段 {segment_index} 失败: {e}")
            return None

    def exists(
            self,
            segment_index: int,
            video_path: Path
    ) -> bool:
        """检查片段是否存在"""
        audio_file, meta_file = self._get_segment_files(
            segment_index, video_path
        )
        return audio_file.exists() and meta_file.exists()

    def delete_segment(
            self,
            segment_index: int,
            video_path: Path
    ) -> bool:
        """删除音频片段"""
        audio_file, meta_file = self._get_segment_files(
            segment_index, video_path
        )

        deleted = False
        if audio_file.exists():
            audio_file.unlink()
            deleted = True
        if meta_file.exists():
            meta_file.unlink()
            deleted = True

        return deleted

    def list_segments(
            self,
            video_path: Path
    ) -> List[int]:
        """列出所有已缓存的片段索引"""
        video_dir = self._get_video_dir(video_path)

        if not video_dir.exists():
            return []

        indices = []
        for audio_file in video_dir.glob("seg_*.wav"):
            try:
                # 从文件名提取索引
                index = int(audio_file.stem.split('_')[1])
                indices.append(index)
            except (ValueError, IndexError):
                continue

        return sorted(indices)

    def get_segment_path(
            self,
            segment_index: int,
            video_path: Path
    ) -> Path:
        """获取片段的存储路径（用于前端播放）"""
        audio_file, _ = self._get_segment_files(
            segment_index, video_path
        )
        return audio_file

    def clear_video_cache(self, video_path: Path) -> int:
        """清除视频的所有音频片段缓存"""
        video_dir = self._get_video_dir(video_path)

        if not video_dir.exists():
            return 0

        count = 0
        for file in video_dir.iterdir():
            file.unlink()
            count += 1

        video_dir.rmdir()
        return count