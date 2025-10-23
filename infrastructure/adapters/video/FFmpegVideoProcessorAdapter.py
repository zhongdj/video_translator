"""
Infrastructure Layer - 增强的FFmpeg视频处理适配器

✅ 需求1: 支持可配置的参考音频起始偏移和时长
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

from domain.entities import Video, AudioTrack, Subtitle
from domain.ports import VideoProcessor


class FFmpegVideoProcessorAdapter(VideoProcessor):
    """FFmpeg 视频处理适配器（增强版）"""

    def extract_audio(self, video: Video) -> Path:
        """从视频提取音频"""
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(temp_file.name)
        temp_file.close()

        subprocess.run([
            'ffmpeg', '-y',
            '-i', str(video.path),
            '-ac', '1',
            '-ar', '16000',
            '-vn',
            str(output_path)
        ], check=True, capture_output=True)

        return output_path

    def extract_reference_audio(
            self,
            video: Video,
            duration: float,
            start_offset: float = 0.0  # ✅ 新增参数
    ) -> Path:
        """
        提取参考音频片段（增强版）

        Args:
            video: 视频对象
            duration: 提取时长（秒）
            start_offset: 起始偏移（秒），默认从头开始

        特性:
        1. 支持指定起始位置（如跳过前30秒）
        2. 使用VAD检测最佳语音片段（如果start_offset=0）
        3. 自动调整时长不超过视频长度
        """
        import tempfile

        # 先提取完整音频
        full_audio = self.extract_audio(video)

        try:
            # ✅ 情况1: 指定了起始偏移（直接提取，不使用VAD）
            if start_offset > 0:
                print(f"📍 使用指定偏移: {start_offset}s")
                extract_start = start_offset
                extract_duration = min(duration, video.duration - start_offset)

                if extract_duration <= 0:
                    raise ValueError(
                        f"起始偏移 {start_offset}s 超出视频长度 {video.duration}s"
                    )

            # ✅ 情况2: 未指定偏移（使用VAD检测最佳片段）
            else:
                print(f"🔍 使用VAD检测最佳语音片段...")
                try:
                    model, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=False
                    )
                    (get_speech_timestamps, _, read_audio, *_) = utils

                    wav = read_audio(str(full_audio), sampling_rate=16000)
                    speech_timestamps = get_speech_timestamps(
                        wav, model,
                        sampling_rate=16000,
                        threshold=0.5
                    )

                    # 选择最长且能量最高的片段
                    if speech_timestamps:
                        best_segment = max(
                            speech_timestamps,
                            key=lambda ts: (ts['end'] - ts['start']) * torch.sum(wav[ts['start']:ts['end']] ** 2)
                        )

                        extract_start = best_segment['start'] / 16000
                        extract_duration = min(
                            (best_segment['end'] - best_segment['start']) / 16000,
                            duration
                        )
                        print(f"✅ VAD检测到最佳片段: {extract_start:.2f}s")
                    else:
                        # VAD未检测到，使用默认策略
                        extract_start = 0
                        extract_duration = min(duration, video.duration)
                        print(f"⚠️  VAD未检测到语音，使用默认片段")

                except Exception as e:
                    print(f"⚠️  VAD检测失败: {e}，使用默认策略")
                    extract_start = 0
                    extract_duration = min(duration, video.duration)

            # 提取指定片段
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()

            print(f"🎵 提取参考音频: {extract_start:.2f}s - {extract_start + extract_duration:.2f}s")

            subprocess.run([
                'ffmpeg', '-y',
                '-ss', str(extract_start),
                '-t', str(extract_duration),
                '-i', str(full_audio),
                '-ac', '1',
                '-ar', '24000',  # TTS使用24kHz
                str(output_path)
            ], check=True, capture_output=True)

            return output_path

        finally:
            # 清理临时文件
            if full_audio.exists():
                full_audio.unlink()

    def merge_audio_video(
            self,
            video: Video,
            audio_track: AudioTrack,
            output_path: Path
    ) -> Path:
        """合并音频和视频"""
        import tempfile

        # 保存音频到临时文件
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = Path(temp_audio.name)
        temp_audio.close()

        # 转换 AudioSample 为 wav 文件
        audio_array = np.array(audio_track.audio.samples, dtype=np.float32)
        torchaudio.save(
            temp_audio_path,
            torch.from_numpy(audio_array).unsqueeze(0),
            audio_track.audio.sample_rate
        )

        # 使用 ffmpeg 合并
        subprocess.run([
            'ffmpeg', '-y',
            '-i', str(video.path),
            '-i', str(temp_audio_path),
            '-c:v', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            str(output_path)
        ], check=True, capture_output=True)

        # 清理临时文件
        temp_audio_path.unlink()

        return output_path

    def burn_subtitles(
            self,
            video: Video,
            subtitle: Subtitle,
            output_path: Path
    ) -> Path:
        """烧录字幕到视频"""
        import tempfile
        import shutil

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                temp_video = tmpdir_path / "video.mp4"
                temp_subtitle = tmpdir_path / "subtitle.ass"
                temp_output = tmpdir_path / "output.mp4"

                assert subtitle.path is not None, "字幕文件路径不能为空"

                shutil.copy2(video.path, temp_video)
                shutil.copy2(subtitle.path, temp_subtitle)

                subtitle_filter = "subtitles=subtitle.ass"

                # 尝试硬件编码
                cmd_hardware = [
                    'ffmpeg', '-y',
                    '-i', 'video.mp4',
                    '-vf', subtitle_filter,
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',
                    '-cq', '23',
                    '-rc', 'vbr',
                    '-c:a', 'copy',
                    '-y', 'output.mp4'
                ]

                try:
                    result = subprocess.run(
                        cmd_hardware,
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if result.returncode == 0:
                        shutil.copy2(temp_output, output_path)
                        print(f"✅ 硬件编码成功: {output_path}")
                        return output_path
                    else:
                        raise RuntimeError("硬件编码失败")

                except (subprocess.CalledProcessError, RuntimeError):
                    print("🔄 硬件编码失败，尝试软件编码...")

                    cmd_software = [
                        'ffmpeg', '-y',
                        '-i', 'video.mp4',
                        '-vf', subtitle_filter,
                        '-c:v', 'libx264',
                        '-crf', '23',
                        '-preset', 'medium',
                        '-c:a', 'copy',
                        'output.mp4'
                    ]

                    result_software = subprocess.run(
                        cmd_software,
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if result_software.returncode == 0:
                        shutil.copy2(temp_output, output_path)
                        print(f"✅ 软件编码成功: {output_path}")
                        return output_path
                    else:
                        raise RuntimeError(f"软件编码失败: {result_software.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg 处理超时")
        except Exception as e:
            raise RuntimeError(f"烧录字幕失败: {str(e)}")