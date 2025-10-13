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

        # 调用批量接口（batch_size=1）
        # results = self.batch_synthesize(
        #     texts=[text],
        #     reference_audio_path=voice_profile.reference_audio_path,
        #     language=voice_profile.language,
        #     batch_size=None
        # )
        import time
        output_path = self.output_dir / f"tts_output_{int(time.time())}_{hash(text) % 100000}.wav"
        print(f" **** {output_path}")
        # result = self.model.infer(
        #     spk_audio_prompt=str(voice_profile.reference_audio_path),
        #     text=text,
        #     output_path=str(output_path),
        #     verbose=True
        # )
        #
        # # return results[0]
        # audio_data, sample_rate = self._handle_model_output(result, output_path)
        # return AudioSample(tuple(audio_data.tolist()), sample_rate)

        # 🔥 关键修复
        def synthesize(
                self,
                text: str,
                voice_profile: VoiceProfile,
                target_duration: Optional[float] = None
        ) -> AudioSample:
            """单句合成（兼容旧接口）- 保存文件版本"""
            if not self._is_loaded:
                self.load()

            import time

            # 生成输出文件路径
            output_path = self.output_dir / f"tts_output_{int(time.time())}_{hash(text) % 100000}.wav"

            # 🔥 如果指定 output_paths，返回的是文件路径列表
            batch_results = self.model.batch_infer_same_speaker(
                texts=[text],
                spk_audio_prompt=str(voice_profile.reference_audio_path),
                output_paths=[str(output_path)],  # ✅ 指定输出路径
                emo_audio_prompt=str(voice_profile.reference_audio_path),
                emo_alpha=1.0,
                interval_silence=0,
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

            # batch_results[0] 现在是文件路径字符串
            output_file = batch_results[0]

            print(f"📁 音频已保存到: {output_file}")

            # 从文件加载音频
            import torchaudio
            wav_tensor, sampling_rate = torchaudio.load(output_file)

            print(f"\n📊 加载的音频信息:")
            print(f"   采样率: {sampling_rate}")
            print(f"   Tensor shape: {wav_tensor.shape}")
            print(f"   Tensor dtype: {wav_tensor.dtype}")

            # 转换为 numpy
            wav_data = wav_tensor.numpy()

            # 🔧 维度处理
            if wav_data.ndim == 2:
                if wav_data.shape[0] < wav_data.shape[1]:
                    # (channels, samples)
                    wav_data = wav_data[0, :]
                    print(f"   取第一声道: {wav_data.shape}")
                else:
                    # (samples, channels)
                    wav_data = wav_data[:, 0]
                    print(f"   取第一列: {wav_data.shape}")

            # 🔧 归一化（如果是 int16）
            if wav_data.dtype == np.int16:
                print(f"   🔧 int16 -> float64 归一化")
                wav_data = wav_data.astype(np.float64) / 32767.0

            # 🔧 验证范围
            max_val = np.abs(wav_data).max()
            if max_val > 1.0:
                print(f"   ⚠️  音频峰值 {max_val:.1f} 超出范围，归一化")
                wav_data = wav_data / max_val

            print(f"   最终范围: [{wav_data.min():.4f}, {wav_data.max():.4f}]")

            # 创建 AudioSample
            audio_sample = AudioSample(
                samples=tuple(float(s) for s in wav_data.flatten()),
                sample_rate=sampling_rate
            )

            print(f"✅ AudioSample 创建成功: {len(audio_sample.samples)} 样本\n")

            return audio_sample


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

        # 自动建议 batch_size
        if batch_size is None:
            batch_size = self.suggest_batch_size(texts)
            print(f"  💡 自动建议 batch_size={batch_size} (基于文本长度分析)")

        total_texts = len(texts)
        print(f"  📝 批量合成: {total_texts} 个文本片段，batch_size={batch_size}")

        # 尝试批量合成，失败则自动降级
        try:
            return self._batch_synthesize_with_recovery(
                texts=texts,
                reference_audio_path=reference_audio_path,
                language=language,
                batch_size=batch_size
            )
        except Exception as e:
            print(f"❌ 批量合成失败: {e}")
            raise

    def _batch_synthesize_with_recovery(
            self,
            texts: list[str],
            reference_audio_path: Path,
            language: LanguageCode,
            batch_size: int
    ) -> tuple[AudioSample, ...]:
        """带 OOM 自动恢复的批量合成"""

        try:
            return self._do_batch_synthesize(
                texts=texts,
                reference_audio_path=reference_audio_path,
                language=language,
                batch_size=batch_size
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.enable_auto_recovery:
                # OOM 发生，尝试恢复
                self.stats["oom_count"] += 1
                new_batch_size = max(1, batch_size // 2)

                print(f"  ⚠️  GPU OOM! 自动降级: batch_size {batch_size} → {new_batch_size}")

                # 清理内存
                torch.cuda.empty_cache()

                # 递归重试（更小的 batch_size）
                return self._batch_synthesize_with_recovery(
                    texts=texts,
                    reference_audio_path=reference_audio_path,
                    language=language,
                    batch_size=new_batch_size
                )
            else:
                raise

    def _do_batch_synthesize(
            self,
            texts: list[str],
            reference_audio_path: Path,
            language: LanguageCode,
            batch_size: int
    ) -> tuple[AudioSample, ...]:
        """实际执行批量合成"""

        print(f"  ⚠️  refence audio path: {str(reference_audio_path)}")
        total_texts = len(texts)
        all_audio_samples = []
        num_batches = (total_texts + batch_size - 1) // batch_size

        batch_start_time = time.perf_counter()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_texts)
            batch_texts = texts[start_idx:end_idx]

            print(f"    批次 {batch_idx + 1}/{num_batches}: 处理 {len(batch_texts)} 个片段 [{start_idx}:{end_idx}]",
                  end="")

            iter_start = time.perf_counter()

            # 调用批量推理
            batch_results = self.model.batch_infer_same_speaker(
                texts=batch_texts,
                spk_audio_prompt=str(reference_audio_path),
                output_paths=None,
                emo_audio_prompt=None,
                emo_alpha=1.0,
                interval_silence=0,
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

            # 转换为 AudioSample
            for sampling_rate, wav_data in batch_results:
                if wav_data.ndim == 2:
                    wav_data = wav_data[:, 0]

                audio_sample = AudioSample(
                    samples=tuple(float(s) for s in wav_data.flatten()),
                    sample_rate=sampling_rate
                )
                all_audio_samples.append(audio_sample)

            # 🔧 清理批次中间结果，释放显存
            del batch_results

            # 主动清理 CUDA 缓存（每个批次后）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 更新统计
        total_time = time.perf_counter() - batch_start_time
        self.stats["total_texts"] += total_texts
        self.stats["total_batches"] += num_batches
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
