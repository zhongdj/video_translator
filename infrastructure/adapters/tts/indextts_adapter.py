# infrastructure/adapters/tts/indextts_adapter.py
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torchaudio

from domain.ports import TTSProvider
from domain.entities import *


class IndexTTSAdapter(TTSProvider):
    """IndexTTS2 适配器 - 修复版"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.enable_auto_recovery = True
        self.default_batch_size = 16
        self._is_loaded = None
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
        # 性能统计
        self.stats = {
            "total_texts": 0,
            "total_batches": 0,
            "total_time": 0.0,
            "oom_count": 0,
            "peak_memory_gb": 0.0
        }


    def load(self):
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
            self._is_loaded = True

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

    def unload(self) -> None:
        """卸载模型"""
        if not self._is_loaded:
            return

        # 打印统计信息
        if self.stats["total_texts"] > 0:
            avg_time = self.stats["total_time"] / self.stats["total_texts"]
            print(f"\n📊 性能统计:")
            print(f"   总文本数: {self.stats['total_texts']}")
            print(f"   总批次数: {self.stats['total_batches']}")
            print(f"   总耗时: {self.stats['total_time']:.2f}秒")
            print(f"   平均: {avg_time:.3f}秒/文本")
            print(f"   峰值内存: {self.stats['peak_memory_gb']:.2f}GB")
            if self.stats['oom_count'] > 0:
                print(f"   ⚠️ OOM次数: {self.stats['oom_count']}")

        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_loaded = False
        print(f"✅ IndexTTS2 模型已卸载")

    def synthesize(
            self,
            text: str,
            voice_profile: VoiceProfile,
            target_duration: Optional[float] = None
    ) -> AudioSample:
        """单句合成（兼容旧接口）"""
        if not self._is_loaded:
            self.load()

        #调用批量接口（batch_size=1）
        results = self.batch_synthesize(
            texts=[text],
            reference_audio_path=voice_profile.reference_audio_path,
            language=voice_profile.language,
            batch_size=8
        )
        return results[0]



    def suggest_batch_size(self, texts: list[str]) -> int:
        """
        根据文本特征建议 batch_size

        考虑因素：
        1. 平均文本长度
        2. 最长文本长度
        3. 文本数量
        """
        if not texts:
            return self.default_batch_size

        avg_len = sum(len(t) for t in texts) / len(texts)
        max_len = max(len(t) for t in texts)

        # 启发式规则
        if max_len > 200 or avg_len > 100:
            # 长文本：小批量
            suggested = 8
        elif max_len > 100 or avg_len > 50:
            # 中等文本：中批量
            suggested = 16
        else:
            # 短文本：大批量
            suggested = 24

        # 考虑总数量
        if len(texts) < suggested:
            suggested = len(texts)

        return max(1, int(suggested/4))

    def batch_synthesize(
            self,
            texts: list[str],
            reference_audio_path: Path,
            language: LanguageCode,
            batch_size: Optional[int] = None
    ) -> tuple[AudioSample, ...]:
        """
        批量合成（自适应 batch_size）

        Args:
            texts: 待合成文本列表
            reference_audio_path: 参考音频路径
            language: 目标语言
            batch_size: 批量大小（None 则自动建议）
        """
        if not self._is_loaded:
            self.load()

        total_texts = len(texts)
        print(f"  📝 批量合成: {total_texts} 个文本片段，batch_size={batch_size}")

        # 尝试批量合成，失败则自动降级
        try:
            return self._batch_synthesize_with_recovery(
                texts=texts,
                reference_audio_path=reference_audio_path
            )
        except Exception as e:
            print(f"❌ 批量合成失败: {e}")
            raise

    def _batch_synthesize_with_recovery(
            self,
            texts: list[str],
            reference_audio_path: Path
    ) -> tuple[AudioSample, ...]:
        """带 OOM 自动恢复的批量合成"""

        try:
            return self._do_batch_synthesize(
                texts=texts,
                reference_audio_path=reference_audio_path
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.enable_auto_recovery:
                # OOM 发生，尝试恢复
                self.stats["oom_count"] += 1
            raise
        finally:
            # 清理内存
            torch.cuda.empty_cache()

    def _do_batch_synthesize(
            self,
            texts: list[str],
            reference_audio_path: Path
    ) -> tuple[AudioSample, ...]:
        """实际执行批量合成"""

        print(f"  ⚠️  reference audio path: {str(reference_audio_path)}")
        total_texts = len(texts)
        all_audio_samples = []

        batch_start_time = time.perf_counter()

        # 🔥 关键修复：直接调用 batch_infer_same_speaker，不做任何分批
        print(f"    批次处理: {len(texts)} 个片段")

        iter_start = time.perf_counter()

        # 调用批量推理 - 关键：output_paths=None，让它返回音频数据而非文件路径
        batch_results = self.model.batch_infer_same_speaker(
            texts=texts,
            spk_audio_prompt=str(reference_audio_path),
            output_paths=None,  # 🔥 重要：返回音频数据而非保存文件
            emo_audio_prompt=None,
            emo_alpha=1.0,
            interval_silence=0,  # 🔥 重要：不插入静音，保持原始音频
            verbose=True,
            max_text_tokens_per_segment=120,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=0.8,
            length_penalty=0.0,
            num_beams=3,
            repetition_penalty=10.0,
            max_mel_tokens=1500
        )

        iter_time = time.perf_counter() - iter_start

        # 记录内存峰值
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            self.stats["peak_memory_gb"] = max(self.stats["peak_memory_gb"], peak_memory)
            print(f" ✓ {iter_time:.2f}秒 (GPU: {peak_memory:.1f}GB)")
        else:
            print(f" ✓ {iter_time:.2f}秒")

        # 🔥 关键修复：正确处理返回值格式
        for result in batch_results:
            # batch_infer_same_speaker 返回 (sampling_rate, wav_data_numpy)
            if isinstance(result, tuple) and len(result) == 2:
                sampling_rate, wav_data = result

                # wav_data 已经是 numpy array，形状为 (samples, channels) 或 (samples,)
                # 需要转换为一维数组
                if wav_data.ndim == 2:
                    wav_data = wav_data[:, 0]  # 取第一个声道
                elif wav_data.ndim == 1:
                    pass  # 已经是一维
                else:
                    print(f"⚠️ 未预期的音频维度: {wav_data.shape}")
                    wav_data = wav_data.flatten()

                # 🔥 重要：保持原始 int16 格式，不要做额外的归一化
                # IndexTTS2 已经做了 torch.clamp(32767 * wav, -32767.0, 32767.0)
                # 转换为 float 只需要除以 32767
                wav_float = wav_data.astype(np.float32) / 32767.0

                audio_sample = AudioSample(
                    samples=tuple(float(s) for s in wav_float),
                    sample_rate=sampling_rate
                )
                all_audio_samples.append(audio_sample)
            else:
                print(f"⚠️ 未预期的返回格式: {type(result)}")
                continue

        # 主动清理 CUDA 缓存
        del batch_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 更新统计
        total_time = time.perf_counter() - batch_start_time
        self.stats["total_texts"] += total_texts
        self.stats["total_batches"] += 1
        self.stats["total_time"] += total_time

        avg_time = total_time / total_texts
        print(f"  ✅ 完成: {total_texts} 个片段, 总耗时 {total_time:.2f}秒 (平均 {avg_time:.3f}秒/片段)")

        return tuple(all_audio_samples)

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

