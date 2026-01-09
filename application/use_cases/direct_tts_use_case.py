"""
Application Layer - 直接文本转语音用例
给定文本和参考音频，直接合成语音
"""
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from domain.entities import VoiceProfile, AudioSample
from domain.value_objects import LanguageCode
from domain.ports import TTSProvider


@dataclass(frozen=True)
class DirectTTSResult:
    """直接TTS结果"""
    audio_sample: AudioSample
    text: str
    reference_audio_path: Path
    synthesis_time: float
    language: LanguageCode


def direct_tts_use_case(
        text: str,
        reference_audio_path: Path,
        tts_provider: TTSProvider,
        language: LanguageCode = LanguageCode.CHINESE,
        target_duration: Optional[float] = None,
        reference_duration: float = 10.0,
        progress: Optional[Callable[[float, str], None]] = None
) -> DirectTTSResult:
    """
    直接文本转语音用例（纯函数）

    Args:
        text: 要合成的文本
        reference_audio_path: 参考音频路径
        tts_provider: TTS提供者
        language: 目标语言
        target_duration: 目标时长（秒），可选
        reference_duration: 参考音频时长（秒）
        progress: 进度回调

    Returns:
        DirectTTSResult: 包含合成的音频和元数据
    """
    start_time = time.perf_counter()

    if progress:
        progress(0.0, "准备合成...")

    # 验证输入
    if not text or not text.strip():
        raise ValueError("文本不能为空")

    if not reference_audio_path.exists():
        raise FileNotFoundError(f"参考音频不存在: {reference_audio_path}")

    # 创建语音配置
    if progress:
        progress(0.2, "创建语音配置...")

    voice_profile = VoiceProfile(
        reference_audio_path=reference_audio_path,
        language=language,
        duration=reference_duration
    )

    # 执行合成
    if progress:
        progress(0.4, f"合成语音: {len(text)} 字符...")

    audio_sample = tts_provider.synthesize(
        text=text,
        voice_profile=voice_profile,
        target_duration=target_duration
    )

    synthesis_time = time.perf_counter() - start_time

    if progress:
        progress(1.0, "合成完成")

    return DirectTTSResult(
        audio_sample=audio_sample,
        text=text,
        reference_audio_path=reference_audio_path,
        synthesis_time=synthesis_time,
        language=language
    )


def batch_direct_tts_use_case(
        texts: list[str],
        reference_audio_path: Path,
        tts_provider: TTSProvider,
        language: LanguageCode = LanguageCode.CHINESE,
        target_durations: Optional[list[float]] = None,
        reference_duration: float = 10.0,
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[DirectTTSResult, ...]:
    """
    批量文本转语音用例

    Args:
        texts: 文本列表
        reference_audio_path: 参考音频路径
        tts_provider: TTS提供者
        language: 目标语言
        target_durations: 目标时长列表（可选）
        reference_duration: 参考音频时长
        progress: 进度回调

    Returns:
        DirectTTSResult元组
    """
    start_time = time.perf_counter()

    if progress:
        progress(0.0, f"批量合成 {len(texts)} 段文本...")

    if not reference_audio_path.exists():
        raise FileNotFoundError(f"参考音频不存在: {reference_audio_path}")

    if target_durations and len(target_durations) != len(texts):
        raise ValueError("target_durations长度必须与texts一致")

    # 批量合成
    if progress:
        progress(0.2, "执行批量合成...")

    audio_samples = tts_provider.batch_synthesize(
        texts=texts,
        reference_audio_path=reference_audio_path,
        language=language,
        target_durations=target_durations
    )

    synthesis_time = time.perf_counter() - start_time

    # 构建结果
    results = []
    for idx, (text, audio_sample) in enumerate(zip(texts, audio_samples)):
        result = DirectTTSResult(
            audio_sample=audio_sample,
            text=text,
            reference_audio_path=reference_audio_path,
            synthesis_time=synthesis_time / len(texts),
            language=language
        )
        results.append(result)

        if progress:
            progress(0.2 + (idx + 1) / len(texts) * 0.8, f"完成 {idx + 1}/{len(texts)}")

    if progress:
        progress(1.0, f"批量合成完成，耗时 {synthesis_time:.1f}秒")

    return tuple(results)


# ============== 辅助函数：保存音频 ============== #

def save_audio_to_file(
        audio_sample: AudioSample,
        output_path: Path,
        format: str = "wav"
) -> Path:
    """
    保存音频到文件

    Args:
        audio_sample: 音频样本
        output_path: 输出路径
        format: 音频格式（wav/mp3）

    Returns:
        保存的文件路径
    """
    import numpy as np
    import soundfile as sf

    # ✅ 确保父目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ 使用绝对路径
    output_path = output_path.resolve()

    # 转换为numpy数组
    audio_data = np.array(audio_sample.samples, dtype=np.float32)

    # 保存文件
    if format.lower() == "wav":
        sf.write(
            str(output_path),
            audio_data,
            audio_sample.sample_rate,
            format='WAV',
            subtype='PCM_16'  # ✅ 明确指定子类型
        )
        print(f"✅ WAV 已保存: {output_path}")

    elif format.lower() == "mp3":
        # 需要安装 pydub 和 ffmpeg
        try:
            from pydub import AudioSegment

            # ✅ 先保存为临时WAV（使用绝对路径）
            temp_wav = output_path.parent / f"temp_{output_path.stem}.wav"
            sf.write(str(temp_wav), audio_data, audio_sample.sample_rate, subtype='PCM_16')

            # ✅ 验证临时文件
            if not temp_wav.exists():
                raise FileNotFoundError(f"临时 WAV 文件创建失败: {temp_wav}")

            print(f"📝 临时 WAV: {temp_wav} ({temp_wav.stat().st_size} bytes)")

            # 转换为MP3
            audio = AudioSegment.from_wav(str(temp_wav))
            audio.export(str(output_path), format="mp3", bitrate="192k")

            print(f"✅ MP3 已保存: {output_path}")

            # 删除临时文件
            if temp_wav.exists():
                temp_wav.unlink()
                print(f"🗑️  删除临时文件: {temp_wav}")

        except ImportError as e:
            print(f"⚠️  MP3 转换失败: {e}")
            print("   请安装: pip install pydub")
            print("   并确保安装了 ffmpeg")
            # ✅ 降级为 WAV 格式
            wav_path = output_path.with_suffix('.wav')
            sf.write(str(wav_path), audio_data, audio_sample.sample_rate, subtype='PCM_16')
            print(f"✅ 降级为 WAV: {wav_path}")
            return wav_path

        except Exception as e:
            print(f"❌ MP3 转换错误: {e}")
            # 清理可能存在的临时文件
            temp_wav = output_path.parent / f"temp_{output_path.stem}.wav"
            if temp_wav.exists():
                temp_wav.unlink()
            raise
    else:
        raise ValueError(f"不支持的格式: {format}")

    # ✅ 验证最终文件是否存在
    if not output_path.exists():
        raise FileNotFoundError(f"文件保存失败: {output_path}")

    return output_path