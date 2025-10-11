# infrastructure/adapters/tts/indextts_adapter.py
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torchaudio

from domain.ports import TTSProvider
from domain.entities import VoiceProfile, AudioSample


class IndexTTSAdapter(TTSProvider):
    """IndexTTS2 适配器 - 修复版"""

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

        print(f"🔍 检测 IndexTTS2 路径:")
        print(f"   IndexTTS2 主目录: {self.indextts2_path}")
        print(f"   主目录存在: {self.indextts2_path.exists()}")

        # 检查关键目录
        self.checkpoints_path = self.indextts2_path / "checkpoints"
        print(f"   checkpoints 目录: {self.checkpoints_path.exists()}")

        if self.checkpoints_path.exists():
            print(f"   checkpoints 内容: {list(self.checkpoints_path.glob('*'))}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        # 添加WebUI参数
        self.gpt_config = {
            'do_sample': True,
            'temperature': 0.8,
            'top_p': 0.8,
            'top_k': 30,
            'num_beams': 3,
            'repetition_penalty': 10,
            'length_penalty': 0,
        }
        self.max_mel_tokens = 1500
        self.split_max_tokens = 120

        # 创建输出目录
        self.output_dir = Path("temp_output")
        self.output_dir.mkdir(exist_ok=True)

    def _load_model(self):
        """加载 IndexTTS 2.0 模型"""
        if not self.indextts2_path.exists():
            raise ImportError(f"❌ IndexTTS2 目录不存在: {self.indextts2_path}")

        if self.model is not None:
            return self.model

        print(f"🔄 加载 IndexTTS 2.0 模型...")

        try:
            # 添加路径到 sys.path
            if str(self.indextts2_path) not in sys.path:
                sys.path.insert(0, str(self.indextts2_path))

            # 尝试不同的导入方式
            IndexTTS2 = None
            try:
                from indextts.infer_v2 import IndexTTS2
                print("✅ 从 indextts.infer_v2 导入 IndexTTS2 成功")
            except ImportError as e:
                print(f"❌ 第一种导入方式失败: {e}")
                try:
                    from infer_v2 import IndexTTS2
                    print("✅ 从 infer_v2 导入 IndexTTS2 成功")
                except ImportError as e:
                    print(f"❌ 第二种导入方式失败: {e}")
                    try:
                        sys.path.append(str(self.indextts2_path))
                        from indextts2.infer_v2 import IndexTTS2
                        print("✅ 从 indextts2.infer_v2 导入 IndexTTS2 成功")
                    except ImportError as e:
                        print(f"❌ 所有导入方式都失败: {e}")
                        return self._create_dummy_model()

            # 初始化模型
            config_path = self.checkpoints_path / "config.yaml"
            if not config_path.exists():
                # 尝试寻找其他配置文件
                config_files = list(self.checkpoints_path.glob("*.yaml")) + list(self.checkpoints_path.glob("*.yml"))
                if config_files:
                    config_path = config_files[0]
                    print(f"🔧 使用备用配置文件: {config_path}")
                else:
                    raise FileNotFoundError(f"没有找到配置文件 in {self.checkpoints_path}")

            print(f"🔧 使用配置文件: {config_path}")
            print(f"🔧 模型目录: {self.checkpoints_path}")

            self.model = IndexTTS2(
                model_dir=str(self.checkpoints_path),
                cfg_path=str(config_path),
                device=self.device,
                use_fp16=(self.device != "cpu")
            )

            print("✅ IndexTTS 2.0 模型加载完成")
            return self.model

        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            return self._create_dummy_model()

    def _create_dummy_model(self):
        """创建虚拟模型用于测试"""

        class DummyModel:
            def infer(self, text, reference_audio, speed=1.0):
                print(f"🔊 虚拟合成: '{text}' (参考音频: {reference_audio}, 语速: {speed})")
                # 生成基于文本长度的测试音频
                duration = max(1.0, len(text) * 0.1)
                samples = int(duration * 22050)
                t = np.linspace(0, duration * 2 * np.pi, samples)
                audio_data = 0.3 * np.sin(440 * t) * np.exp(-0.001 * t)
                return (audio_data, 22050)  # 返回元组

        return DummyModel()

    def _handle_model_output(self, result, output_path: Path) -> Tuple[np.ndarray, int]:
        """处理模型输出，适应不同的返回类型"""
        print(f"🔧 处理模型输出，类型: {type(result)}")

        # 如果返回的是路径，读取音频文件
        if isinstance(result, (str, Path)):
            audio_path = Path(result)
            if audio_path.exists():
                print(f"📁 从文件读取音频: {audio_path}")
                audio_data, sample_rate = torchaudio.load(audio_path)
                audio_data = audio_data.numpy()[0]  # 转换为numpy并取第一个通道
                return audio_data, sample_rate
            else:
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 如果返回的是元组 (audio_data, sample_rate)
        elif isinstance(result, tuple) and len(result) == 2:
            audio_data, sample_rate = result

            # 处理torch tensor
            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().numpy()

            # 确保是单声道
            if audio_data.ndim > 1:
                audio_data = audio_data[0]  # 取第一个通道

            return audio_data, sample_rate

        # 如果返回的是单个值（可能是音频数据），假设采样率为22050
        else:
            print(f"⚠️ 未知返回类型，尝试处理...")
            audio_data = result
            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().numpy()

            if audio_data.ndim > 1:
                audio_data = audio_data[0]

            return audio_data, 22050

    def synthesize(
            self,
            text: str,
            voice_profile: VoiceProfile,
            target_duration: Optional[float] = None
    ) -> AudioSample:
        """实现 TTSProvider 接口"""
        if self.model is None:
            self.model = self._load_model()

        print(f"🎯 开始语音合成...")
        print(f"   参考音频: {voice_profile.reference_audio_path}")
        print(f"   生成文本: {text}")

        try:
            # 准备输出路径
            import time
            output_path = self.output_dir / f"tts_output_{int(time.time())}_{hash(text) % 100000}.wav"

            # 调用 IndexTTS2 合成
            if hasattr(self.model, 'infer_with_config'):
                print("🔧 使用 infer_with_config 方法")
                result = self.model.infer_with_config(
                    spk_audio_prompt=str(voice_profile.reference_audio_path),
                    text=text,
                    output_path=str(output_path),
                    emotion_prompt=None,
                    do_sample=self.gpt_config['do_sample'],
                    temperature=self.gpt_config['temperature'],
                    top_p=self.gpt_config['top_p'],
                    top_k=self.gpt_config['top_k'],
                    num_beams=self.gpt_config['num_beams'],
                    repetition_penalty=self.gpt_config['repetition_penalty'],
                    length_penalty=self.gpt_config['length_penalty'],
                    max_mel_tokens=self.max_mel_tokens
                )
            else:
                print("🔧 使用 infer 方法")
                result = self.model.infer(
                    spk_audio_prompt=str(voice_profile.reference_audio_path),
                    text=text,
                    output_path=str(output_path),
                    verbose=True
                )

            # 处理模型输出
            audio_data, sample_rate = self._handle_model_output(result, output_path)

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
            import traceback
            traceback.print_exc()

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
            self.gpt_config['temperature'] = temperature
        if top_p is not None:
            self.top_p = top_p
            self.gpt_config['top_p'] = top_p
        if speed is not None:
            self.speed = speed

    def unload(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 IndexTTS2 模型已卸载")