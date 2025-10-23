"""
Infrastructure Layer - 音频文件仓储适配器
封装所有音频文件I/O操作
"""
from pathlib import Path
from typing import Optional
import wave
import struct
import json

from domain.entities import AudioSample
from domain.ports import AudioFileRepository


class AudioFileRepositoryAdapter(AudioFileRepository):
    """音频文件仓储适配器（基于文件系统）"""

    def __init__(self, base_dir: Path = Path(".cache/audio")):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_audio(
            self,
            cache_key: str,
            audio: AudioSample,
            metadata: dict
    ) -> Path:
        """保存音频到持久化存储"""
        audio_path = self._get_audio_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)

        # 保存音频文件
        self._write_wav(audio_path, audio)

        # 保存元数据
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return audio_path

    def load_audio(
            self,
            cache_key: str
    ) -> tuple[Optional[AudioSample], Optional[dict]]:
        """从持久化存储加载音频"""
        audio_path = self._get_audio_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)

        if not audio_path.exists() or not meta_path.exists():
            return None, None

        try:
            # 加载音频
            audio_sample = self._read_wav(audio_path)

            # 加载元数据
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            return audio_sample, metadata

        except Exception as e:
            print(f"❌ 加载音频失败: {e}")
            return None, None

    def exists(self, cache_key: str) -> bool:
        """检查音频是否存在"""
        audio_path = self._get_audio_path(cache_key)
        meta_path = self._get_metadata_path(cache_key)
        return audio_path.exists() and meta_path.exists()

    # ============== 私有方法（技术实现） ============== #

    def _get_audio_path(self, cache_key: str) -> Path:
        """获取音频文件路径"""
        return self.base_dir / f"{cache_key}.wav"

    def _get_metadata_path(self, cache_key: str) -> Path:
        """获取元数据文件路径"""
        return self.base_dir / f"{cache_key}.meta"

    def _write_wav(self, file_path: Path, audio_sample: AudioSample):
        """写入WAV文件（技术细节封装）"""
        # 转换为16位PCM
        samples_int16 = [
            int(max(-1.0, min(1.0, sample)) * 32767)
            for sample in audio_sample.samples
        ]

        with wave.open(str(file_path), 'w') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(audio_sample.sample_rate)

            for sample in samples_int16:
                wav_file.writeframes(struct.pack('<h', sample))

    def _read_wav(self, file_path: Path) -> AudioSample:
        """读取WAV文件（技术细节封装）"""
        with wave.open(str(file_path), 'r') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            frames = wav_file.readframes(n_frames)

            # 转换为浮点数
            samples = []
            for i in range(0, len(frames), 2):
                sample = struct.unpack('<h', frames[i:i + 2])[0]
                samples.append(sample / 32767.0)

            return AudioSample(
                samples=tuple(samples),
                sample_rate=sample_rate
            )

    # ============== ✅ 新增: 参考音频管理 ============== #

    def save_reference_audio(
            self,
            video_path: Path,
            source_audio_path: Path
    ) -> Path:
        """
        保存参考音频（持久化Gradio临时文件或视频提取的音频）

        Args:
            video_path: 关联的视频路径
            source_audio_path: 源音频路径

        Returns:
            持久化后的参考音频路径
        """
        import hashlib
        import shutil

        # 创建参考音频目录
        ref_audio_dir = self.base_dir.parent / "reference_audio"
        ref_audio_dir.mkdir(parents=True, exist_ok=True)

        # 生成唯一文件名
        video_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
        video_name = video_path.stem
        file_ext = source_audio_path.suffix or ".wav"

        persistent_path = ref_audio_dir / f"{video_name}_{video_hash}_ref{file_ext}"

        # 复制文件
        shutil.copy2(source_audio_path, persistent_path)

        print(f"✅ 参考音频已持久化:")
        print(f"   源路径: {source_audio_path}")
        print(f"   持久路径: {persistent_path}")

        return persistent_path

    def load_reference_audio(
            self,
            video_path: Path
    ) -> Optional[Path]:
        """
        加载参考音频路径

        Args:
            video_path: 关联的视频路径

        Returns:
            参考音频路径，不存在则返回None
        """
        import hashlib

        ref_audio_dir = self.base_dir.parent / "reference_audio"
        if not ref_audio_dir.exists():
            return None

        video_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
        video_name = video_path.stem

        # 查找匹配的文件（支持多种格式）
        pattern = f"{video_name}_{video_hash}_ref.*"
        matches = list(ref_audio_dir.glob(pattern))

        if matches:
            return matches[0]

        return None

    def delete_reference_audio(
            self,
            video_path: Path
    ) -> bool:
        """删除参考音频"""
        ref_audio_path = self.load_reference_audio(video_path)

        if ref_audio_path and ref_audio_path.exists():
            try:
                ref_audio_path.unlink()
                print(f"🗑️  已删除参考音频: {ref_audio_path}")
                return True
            except Exception as e:
                print(f"⚠️  删除失败: {e}")
                return False

        return False