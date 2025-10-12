from application import *

# 导入领域层
from domain.entities import (
    # Entities
    VoiceProfile, TextSegment,
    # Value Objects
    TimeRange, LanguageCode, )

from domain.ports import (
# Ports
    ASRProvider, TranslationProvider, TTSProvider,
)


def test_component_use_case(
        component_type: str,
        test_input: dict,
        asr_provider: Optional[ASRProvider] = None,
        translation_provider: Optional[TranslationProvider] = None,
        tts_provider: Optional[TTSProvider] = None,
        progress: ProgressCallback = None
) -> dict:
    """
    测试组件用例（纯函数）

    用于在 WebUI 中测试各个组件的效果

    Args:
        component_type: "asr" | "translation" | "tts"
        test_input: 测试输入数据

    Returns:
        测试结果字典
    """
    if progress:
        progress(0.0, f"开始测试 {component_type}")

    if component_type == "asr" and asr_provider:
        # 测试 ASR
        audio_path = Path(test_input["audio_path"])
        language = LanguageCode(test_input.get("language", "auto"))

        if progress:
            progress(0.3, "ASR 识别中")

        segments, detected_lang = asr_provider.transcribe(audio_path, language)
        asr_provider.unload()
        if progress:
            progress(1.0, "ASR 测试完成")

        return {
            "detected_language": detected_lang.value,
            "segments": [
                {"text": seg.text, "start": seg.time_range.start_seconds, "end": seg.time_range.end_seconds}
                for seg in segments
            ],
            "total_segments": len(segments)
        }

    elif component_type == "translation" and translation_provider:
        # 测试翻译
        text = test_input["text"]
        source_lang = LanguageCode(test_input["source_language"])
        target_lang = LanguageCode(test_input["target_language"])

        # 创建临时文本片段
        segment = TextSegment(
            text=text,
            time_range=TimeRange(0, 1),
            language=source_lang
        )

        if progress:
            progress(0.5, "翻译中")

        translated = translation_provider.translate(
            (segment,),
            source_lang,
            target_lang
        )

        if progress:
            progress(1.0, "翻译测试完成")

        return {
            "original": text,
            "translated": translated[0].text,
            "source_language": source_lang.value,
            "target_language": target_lang.value
        }

    elif component_type == "tts" and tts_provider:
        # 测试 TTS
        text = test_input["text"]
        reference_audio = Path(test_input["reference_audio"])
        target_duration = test_input.get("target_duration", None)

        voice_profile = VoiceProfile(
            reference_audio_path=reference_audio,
            language=LanguageCode.CHINESE,
            duration=10.0
        )

        if progress:
            progress(0.5, "合成语音")

        audio = tts_provider.synthesize(text, voice_profile, target_duration)

        if progress:
            progress(1.0, "TTS 测试完成")

        return {
            "text": text,
            "duration": audio.duration,
            "sample_rate": audio.sample_rate,
            "samples_count": len(audio.samples)
        }

    else:
        raise ValueError(f"Unsupported component type: {component_type}")
