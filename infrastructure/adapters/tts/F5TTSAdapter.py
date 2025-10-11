# 导入领域层
import numpy as np
import torch

from domain.ports import *

from domain.services import *

class F5TTSAdapter:
    """F5-TTS (IndexTTS 2.0) 适配器"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def _load_model(self):
        """懒加载模型"""
        if self._model is None:
            try:
                # 导入 F5-TTS
                from f5_tts.api import F5TTS

                print(f"🔄 加载 F5-TTS 模型")
                self._model = F5TTS(
                    ckpt_file=self.model_path,
                    device=self.device
                )
            except ImportError:
                raise ImportError(
                    "请安装 F5-TTS:\n"
                    "pip install f5-tts"
                )

        return self._model

    def synthesize(
            self,
            text: str,
            voice_profile: VoiceProfile,
            target_duration: Optional[float] = None
    ) -> AudioSample:
        """实现 TTSProvider 接口"""
        model = self._load_model()
        speed: float = 1.0,
        seed: Optional[int] = None

        # 构建推理参数
        infer_kwargs = {
            "ref_file": str(voice_profile.reference_audio_path),
            "ref_text": "",
            "gen_text": text,
            "target_rms": 0.1,
            "speed": speed,
        }

        # 可选参数
        if seed is not None:
            infer_kwargs["seed"] = seed
        if target_duration is not None:
            infer_kwargs["fix_duration"] = target_duration

        print(f"🎯 开始语音合成...")
        print(f"   参考音频: {infer_kwargs['ref_file']}")
        print(f"   生成文本: {text}")

        # 调用 F5-TTS 合成
        wav, sr, spec = model.infer(**infer_kwargs)

        # 确保wav是numpy数组
        import numpy as np
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav)

        print(f"✅ 成功生成音频: {len(wav)}采样点, {sr}Hz采样率")

        # 计算时长
        duration = len(wav) / sr
        print(f"⏱️ 音频时长: {duration:.2f}秒")

        # 检查音频数据
        if wav.size == 0:
            print("❌ 警告: 生成的音频数据为空!")
            raise ValueError("生成的音频数据为空")
        elif np.all(wav == 0):
            print("⚠️ 警告: 生成的音频数据全为零!")

        print(f"📊 音频数据范围: [{np.min(wav):.4f}, {np.max(wav):.4f}]")

        # 调试：直接保存音频文件以验证数据
        debug_audio_path = "debug_audio_output.wav"
        # try:
        #     import soundfile as sf
        #     sf.write(debug_audio_path, wav, sr)
        #     print(f"🔧 调试音频已保存到: {debug_audio_path}")
        #
        #     # 检查文件大小
        #     import os
        #     file_size = os.path.getsize(debug_audio_path)
        #     print(f"📁 调试文件大小: {file_size} 字节 ({file_size / 1024:.2f} KB)")
        #
        #     if file_size == 0:
        #         print("❌ 调试文件也是0KB，问题出现在保存过程中")
        #     else:
        #         print("✅ 调试文件保存成功，问题可能在测试代码中")
        #
        # except Exception as e:
        #     print(f"❌ 保存调试音频时出错: {e}")

        # 将numpy数组转换为tuple以符合AudioSample的要求
        # 注意: 对于大型音频数据，这可能会占用较多内存
        audio_tuple = tuple(wav.tolist())
        print(f"🔄 转换音频数据: ndarray -> tuple (长度: {len(audio_tuple)})")

        # 返回 AudioSample - 使用tuple格式
        return AudioSample(
            samples=audio_tuple,  # 使用转换后的tuple
            sample_rate=sr
        )

    def unload(self):
        """卸载模型"""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 F5-TTS 模型已卸载")