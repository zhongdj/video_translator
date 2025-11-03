# infrastructure/adapters/tts/indextts_adapter.py
import sys
import time

import numpy as np
import torch

from domain.entities import *
from domain.ports import TTSProvider


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
        # 速度配置
        self.speed = 1.0  # 默认速度
        self.speed_index = 0  # IndexTTS2 使用整数索引控制速度

        # 速度映射表 (根据 IndexTTS2 的实际实现)
        # 通常: 0=正常, 1=稍快, 2=快, -1=稍慢, -2=慢
        self.speed_mapping = {
            0.5: -2,  # 0.5x -> 很慢
            0.75: -1,  # 0.75x -> 稍慢
            1.0: 0,  # 1.0x -> 正常
            1.25: 1,  # 1.25x -> 稍快
            1.5: 2,  # 1.5x -> 快
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

    def _speed_to_index(self, speed: float) -> int:
        """将速度因子转换为 IndexTTS2 的速度索引"""
        # 找到最接近的预定义速度
        closest_speed = min(self.speed_mapping.keys(),
                            key=lambda x: abs(x - speed))
        return self.speed_mapping[closest_speed]


    def synthesize(
            self,
            text: str,
            voice_profile: VoiceProfile,
            target_duration: Optional[float] = None
    ) -> AudioSample:
        """单句合成(兼容旧接口)"""
        if not self._is_loaded:
            self.load()

        # 如果指定了 target_duration,先试合成一次估算时长
        if target_duration is not None:
            # 第一次合成(使用默认速度)
            results = self.batch_synthesize(
                texts=[text],
                reference_audio_path=voice_profile.reference_audio_path,
                language=voice_profile.language,
                batch_size=8,
                speed_factor=1.0,  # 明确传递速度
                target_durations=[target_duration],
            )
            audio = results[0]
            actual_duration = len(audio.samples) / audio.sample_rate

            # 如果超时,调整语速重新合成
            if actual_duration > target_duration:
                speed_factor = actual_duration / (0.95 * target_duration)
                adjusted_speed = min(speed_factor, 2.0)  # 最大2倍速

                print(f"  ⚡ 音频过长 ({actual_duration:.2f}s > {target_duration:.2f}s)")
                print(f"     调整语速至 {adjusted_speed:.2f}x 重新合成")

                # 重新合成(使用调整后的速度)
                results = self.batch_synthesize(
                    texts=[text],
                    reference_audio_path=voice_profile.reference_audio_path,
                    language=voice_profile.language,
                    batch_size=8,
                    speed_factor=adjusted_speed,
                    target_durations=[target_duration],
                )

            return results[0]

        # 没有 target_duration,直接合成
        results = self.batch_synthesize(
            texts=[text],
            reference_audio_path=voice_profile.reference_audio_path,
            language=voice_profile.language,
            batch_size=8,
            speed_factor=1.0  # 默认速度
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
            batch_size: Optional[int] = None,
            speed_factor: float = 1.0,  # 🔥 新增速度参数
            target_durations: Optional[list[float]] = None  # ✅ 新增
    ) -> tuple[AudioSample, ...]:
        """
        批量合成(自适应 batch_size)

        Args:
            speed_factor: 语速因子 (1.0=正常, >1.0=加快, <1.0=减慢)
        """
        if not self._is_loaded:
            self.load()

        total_texts = len(texts)
        print(f"  📝 批量合成: {total_texts} 个文本片段, speed={speed_factor}x")

        try:
            return self._batch_synthesize_with_recovery(
                texts=texts,
                reference_audio_path=reference_audio_path,
                speed_factor=speed_factor,  # 🔥 传递速度参数
                target_durations=target_durations
            )
        except Exception as e:
            print(f"❌ 批量合成失败: {e}")
            raise

    def _batch_synthesize_with_recovery(
            self,
            texts: list[str],
            reference_audio_path: Path,
            speed_factor: float = 1.0,  # 🔥 新增参数
            target_durations: Optional[list[float]] = None  # ✅
    ) -> tuple[AudioSample, ...]:
        """带 OOM 自动恢复的批量合成"""
        try:
            return self._do_batch_synthesize(
                texts=texts,
                reference_audio_path=reference_audio_path,
                speed_factor=speed_factor,  # 🔥 传递参数
                target_durations=target_durations  # ✅
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.enable_auto_recovery:
                self.stats["oom_count"] += 1
            raise
        finally:
            torch.cuda.empty_cache()

    def _do_batch_synthesize(
            self,
            texts: list[str],
            reference_audio_path: Path,
            speed_factor: float = 1.0,
            target_durations: Optional[list[float]] = None  # (每条 text 对应目标秒数)
    ) -> tuple[AudioSample, ...]:
        """实际执行批量合成（按条计算每条的 max_text_tokens_per_segment & max_mel_tokens）"""
        print(f"  ⚠️  reference audio path: {str(reference_audio_path)}")
        print(f"  ⚡ speed factor: {speed_factor}x")

        total_texts = len(texts)
        all_audio_samples = []
        batch_start_time = time.perf_counter()

        # 转换速度因子为 IndexTTS2 的速度索引
        speed_index = self._speed_to_index(speed_factor)
        print(f"  📊 speed_index={speed_index} (mapped from {speed_factor}x)")

        # 估算参数的基准值（可调整或改成从 token_calculator 导入）
        token_per_sec = 12.8
        mel_per_sec = 55
        margin = 1.05  # 稍微放宽一点，避免截断

        def estimate_text_tokens(sec):
            if sec is None:
                return min(self.split_max_tokens, 120)
            val = int(sec * token_per_sec * margin)
            val = max(20, val)
            return min(val, int(self.split_max_tokens))

        def estimate_mel_tokens(sec):
            if sec is None:
                return int(self.max_mel_tokens)
            val = int(sec * mel_per_sec * margin)
            val = max(50, val)
            return min(val, int(self.max_mel_tokens))

        # 如果给了 target_durations 长度校验
        if target_durations and len(target_durations) != len(texts):
            raise ValueError("target_durations length must equal texts length")

        # 逐条合成（如果底层支持批量传入 per-item 参数，可改为一次性传入 list）
        for idx, text in enumerate(texts):
            tgt_sec = None
            if target_durations:
                tgt_sec = target_durations[idx]

            max_text_tokens_per_segment = 120 #estimate_text_tokens(tgt_sec)
            max_mel_tokens = 1500 #estimate_mel_tokens(tgt_sec)

            print(
                f"  ▶ [{idx}] tgt_sec={tgt_sec} -> max_text_tokens_per_segment={max_text_tokens_per_segment}, max_mel_tokens={max_mel_tokens}")

            # 调用底层（单条或小批量模式）
            results = self.model.batch_infer_same_speaker(
                texts=[text],
                spk_audio_prompt=str(reference_audio_path),
                output_paths=None,
                emo_audio_prompt=None,
                emo_alpha=1.0,
                interval_silence=0,
                verbose=True,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                speed_index=speed_index,
                # generation kwargs
                do_sample=self.gpt_config.get('do_sample', True),
                top_p=self.top_p,
                top_k=self.gpt_config.get('top_k', 30),
                temperature=self.temperature,
                length_penalty=self.gpt_config.get('length_penalty', 0.0),
                num_beams=self.gpt_config.get('num_beams', 3),
                repetition_penalty=10.0,
                max_mel_tokens=max_mel_tokens
            )

            # 处理返回值（batch_infer_same_speaker 返回的可能是 list/tuple）
            if not results:
                print(f"⚠️ [{idx}] 无返回结果")
                continue

            # 取第一个结果（因为我们传入单条 text）
            sampling_rate, wav_data = results[0]
            if wav_data.ndim == 2:
                wav = wav_data[:, 0]
            else:
                wav = wav_data

            wav_float = wav.astype(np.float32) / 32767.0
            audio_sample = AudioSample(samples=tuple(float(s) for s in wav_float), sample_rate=sampling_rate)

            # 打印实际时长用于调试与二次调整参考
            actual_dur = len(audio_sample.samples) / audio_sample.sample_rate
            if tgt_sec is not None:
                diff = actual_dur - tgt_sec
                print(f"    ⏱ 实际时长: {actual_dur:.3f}s (目标 {tgt_sec:.3f}s, 偏差 {diff:+.3f}s)")
            else:
                print(f"    ⏱ 实际时长: {actual_dur:.3f}s")

            all_audio_samples.append(audio_sample)

        # 统计与缓存清理
        iter_time = time.perf_counter() - batch_start_time
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            self.stats["peak_memory_gb"] = max(self.stats["peak_memory_gb"], peak_memory)
            print(f" ✓ {iter_time:.2f}s (GPU: {peak_memory:.1f}GB)")
        else:
            print(f" ✓ {iter_time:.2f}s")

        # 更新统计
        total_time = time.perf_counter() - batch_start_time
        self.stats["total_texts"] += total_texts
        self.stats["total_batches"] += 1
        self.stats["total_time"] += total_time

        avg_time = total_time / max(1, total_texts)
        print(f"  ✅ 完成: {total_texts} 个片段, 总耗时 {total_time:.2f}s (平均 {avg_time:.3f}s/片段)")

        return tuple(all_audio_samples)

    def update_config(self, temperature: float = None, top_p: float = None, speed: float = None, length_penalty: float = None) -> None :
        """更新配置参数"""
        if temperature is not None and temperature > 0:
            self.temperature = temperature
            self.gpt_config['temperature'] = temperature
        if top_p is not None and 0 < top_p <= 1:
            self.top_p = top_p
            self.gpt_config['top_p'] = top_p
        if speed is not None and speed > 0:
            self.speed = speed
        if length_penalty is not None:
            self.gpt_config['length_penalty'] = length_penalty

