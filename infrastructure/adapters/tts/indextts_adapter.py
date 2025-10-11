# infrastructure/adapters/tts/indextts_adapter.py
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torchaudio

from domain.ports import TTSProvider
from domain.entities import VoiceProfile, AudioSample


class IndexTTSAdapter(TTSProvider):
    """IndexTTS2 适配器 - 简化版"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self._model = None

        # 设置 IndexTTS2 路径
        current_dir = Path(__file__).parent
        self.indextts2_path = current_dir.parent.parent.parent.parent / "indextts2"

        # 基础配置参数
        self.temperature = 0.8
        self.top_p = 0.8
        self.speed = 1.0

    def _load_model(self):
        """加载 IndexTTS2 模型"""
        if self._model is not None:
            return self._model

        print(f"🔄 加载 IndexTTS2 模型...")
        print(f"   IndexTTS2 路径: {self.indextts2_path}")

        if not self.indextts2_path.exists():
            raise ImportError(f"❌ IndexTTS2 目录不存在: {self.indextts2_path}")

        try:
            # 添加路径到 Python 路径
            if str(self.indextts2_path) not in sys.path:
                sys.path.insert(0, str(self.indextts2_path))

            # 尝试导入 IndexTTS2
            try:
                from indextts2.api import IndexTTS2
                print("✅ 从 indextts2.api 导入成功")
            except ImportError:
                # 尝试备用导入路径
                try:
                    from api import IndexTTS2
                    print("✅ 从 api 导入成功")
                except ImportError:
                    # 最后尝试直接导入
                    try:
                        sys.path.append(str(self.indextts2_path))
                        from infer import IndexTTS2
                        print("✅ 从 infer 导入成功")
                    except ImportError as e:
                        raise ImportError(f"无法导入 IndexTTS2: {e}")

            # 初始化模型
            checkpoints_dir = self.indextts2_path / "checkpoints"
            if not checkpoints_dir.exists():
                raise FileNotFoundError(f"模型检查点目录不存在: {checkpoints_dir}")

            self._model = IndexTTS2(
                model_dir=str(checkpoints_dir),
                device=self.device
            )

            print("✅ IndexTTS2 模型加载完成")
            return self._model

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 创建虚拟模型用于开发和测试
            self._model = self._create_dummy_model()
            return self._model

    def _create_dummy_model(self):
        """创建虚拟模型用于测试"""

        class DummyModel:
            def synthesize(self, text, reference_audio, speed=1.0):
                print(f"🔊 虚拟合成: '{text}' (参考音频: {reference_audio}, 语速: {speed})")
                # 生成1秒的测试音频 (22050 采样点)
                duration = max(1.0, len(text) * 0.1)  # 基于文本长度估算
                samples = int(duration * 22050)
                t = np.linspace(0, duration * 2 * np.pi, samples)
                audio_data = 0.3 * np.sin(440 * t) * np.exp(-0.001 * t)  # 衰减的正弦波
                return audio_data, 22050

        return DummyModel()

    def synthesize(
            self,
            text: str,
            voice_profile: VoiceProfile,
            target_duration: Optional[float] = None
    ) -> AudioSample:
        """实现 TTSProvider 接口"""
        model = self._load_model()

        print(f"🎯 开始语音合成...")
        print(f"   参考音频: {voice_profile.reference_audio_path}")
        print(f"   生成文本: {text}")

        try:
            # 调用 IndexTTS2 合成
            audio_data, sample_rate = model.synthesize(
                text=text,
                reference_audio=str(voice_profile.reference_audio_path),
                speed=self.speed
            )

            # 确保音频数据是 numpy 数组
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)

            print(f"✅ 成功生成音频: {len(audio_data)}采样点, {sample_rate}Hz采样率")

            # 计算时长
            duration = len(audio_data) / sample_rate
            print(f"⏱️ 音频时长: {duration:.2f}秒")

            # 检查音频数据
            if audio_data.size == 0:
                raise ValueError("生成的音频数据为空")

            print(f"📊 音频数据范围: [{np.min(audio_data):.4f}, {np.max(audio_data):.4f}]")

            # 转换为 tuple 以符合 AudioSample 的要求
            audio_tuple = tuple(audio_data.tolist())
            print(f"🔄 转换音频数据: ndarray -> tuple (长度: {len(audio_tuple)})")

            return AudioSample(
                samples=audio_tuple,
                sample_rate=sample_rate
            )

        except Exception as e:
            print(f"❌ 语音合成失败: {e}")
            # 返回静音作为降级方案
            silent_samples = int(22050 * 2.0)  # 2秒静音
            silent_audio = tuple(np.zeros(silent_samples).tolist())
            return AudioSample(
                samples=silent_audio,
                sample_rate=22050
            )

    def update_config(self, temperature: float = None, top_p: float = None, speed: float = None):
        """更新配置参数"""
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if speed is not None:
            self.speed = speed

    def unload(self):
        """卸载模型"""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 IndexTTS2 模型已卸载")