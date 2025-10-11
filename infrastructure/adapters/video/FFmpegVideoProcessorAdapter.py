import subprocess

import numpy as np
import torch
import torchaudio

from domain.entities import *

class FFmpegVideoProcessorAdapter:
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

        # 先写字幕文件
        temp_srt = tempfile.NamedTemporaryFile(mode='w', suffix=".ass", delete=False, encoding='utf-8')
        temp_srt_path = Path(temp_srt.name)

        # 使用 SubtitleWriter 写入
        writer = PySRTSubtitleWriterAdapter()
        writer.write_ass(subtitle, temp_srt_path)

        # 烧录字幕
        subprocess.run([
            'ffmpeg', '-y',
            '-i', str(video.path),
            '-vf', f"ass={temp_srt_path.name}",
            '-c:v', 'libx264',
            '-crf', '23',
            '-c:a', 'copy',
            str(output_path)
        ], check=True, capture_output=True, cwd=temp_srt_path.parent)

        # 清理
        temp_srt_path.unlink()

        return output_path