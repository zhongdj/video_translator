import subprocess
from typing import Union

import numpy as np
import torch
import torchaudio

from domain.entities import *
from domain.ports import VideoProcessor


class FFmpegVideoProcessorAdapter(VideoProcessor):
    """FFmpeg 视频处理适配器"""

    def extract_audio(self, video: Video) -> Path:
        """从视频提取音频"""
        import tempfile

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
            duration: float
    ) -> Path:
        """提取参考音频片段（使用 VAD 检测最佳片段）"""
        import tempfile

        # 先提取完整音频
        full_audio = self.extract_audio(video)

        try:
            # 使用 silero-vad 检测语音区域
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

                start_sec = best_segment['start'] / 16000
                extract_duration = min((best_segment['end'] - best_segment['start']) / 16000, duration)
            else:
                # 如果没检测到，使用开头部分
                start_sec = 0
                extract_duration = min(duration, video.duration)

        except Exception as e:
            print(f"⚠️ VAD 检测失败: {e}，使用默认策略")
            start_sec = 0
            extract_duration = min(duration, video.duration)

        # 提取指定片段
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(temp_file.name)
        temp_file.close()

        subprocess.run([
            'ffmpeg', '-y',
            '-ss', str(start_sec),
            '-t', str(extract_duration),
            '-i', str(full_audio),
            '-ac', '1',
            '-ar', '24000',  # F5-TTS 使用 24kHz
            str(output_path)
        ], check=True, capture_output=True)

        # 清理临时文件
        full_audio.unlink()

        return output_path

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
        from domain.ports import SubtitleWriter

        try:
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 使用临时目录工作，避免路径问题
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # 复制视频文件到临时目录
                temp_video = tmpdir_path / "video.mp4"
                temp_subtitle = tmpdir_path / "subtitle.ass"
                temp_output = tmpdir_path / "output.mp4"

                # 使用断言确保路径存在
                assert subtitle.path is not None, "字幕文件路径不能为空"

                shutil.copy2(video.path, temp_video)
                shutil.copy2(subtitle.path, temp_subtitle)

                # 使用相对路径
                subtitle_filter = "subtitles=subtitle.ass"

                # 尝试硬件编码
                cmd_hardware = [
                    'ffmpeg', '-y',
                    '-i', 'video.mp4',
                    '-vf', subtitle_filter,
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',  # p1最快，p7最慢
                    '-cq', '23',  # 恒定质量模式
                    '-rc', 'vbr',  # 可变比特率
                    '-c:a', 'copy',
                    '-y', 'output.mp4'
                ]

                print(f"执行 FFmpeg 命令: {' '.join(cmd_hardware)} (在临时目录: {tmpdir})")

                try:
                    result = subprocess.run(
                        cmd_hardware,
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if result.returncode == 0:
                        # 复制输出文件
                        shutil.copy2(temp_output, output_path)
                        print(f"✅ 硬件编码成功: {output_path}")
                        return output_path
                    else:
                        print(f"硬件编码失败: {result.stderr}")
                        raise RuntimeError("硬件编码失败")

                except subprocess.CalledProcessError:
                    # 回退到软件编码
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