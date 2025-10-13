import torch

# 导入领域层
from domain.entities import (
    # Entities
    TextSegment,
    # Value Objects
    LanguageCode, )


class QwenTranslationAdapter:
    """Qwen 翻译适配器"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """懒加载模型"""
        if self._model is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            print(f"🔄 加载 Qwen 翻译模型: {self.model_name}")

            # 8bit 量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='left'
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

        return self._model, self._tokenizer

    def translate(
            self,
            segments: tuple[TextSegment, ...],
            source_lang: LanguageCode,
            target_lang: LanguageCode
    ) -> tuple[TextSegment, ...]:
        """实现 TranslationProvider 接口"""
        model, tokenizer = self._load_model()

        # 批量翻译
        batch_size = len(segments)
        translated_segments = []

        print(f" Translating {len(segments)} segments in {batch_size} batch")
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]

            # 构造 prompt
            prompts = [
                f"Translate the following {source_lang.value} text into {target_lang.value}. "
                f"Output only the translation.\n{source_lang.value}: {seg.text}\n{target_lang.value}:"
                for seg in batch
            ]

            # 编码
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(model.device)

            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # 解码
            for seg, output_ids in zip(batch, outputs):
                # 只取新生成的 token
                new_tokens = output_ids[inputs.input_ids.shape[1]:]
                translated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                translated_text = translated_text.strip().split('\n')[0]  # 取第一行

                translated_segment = TextSegment(
                    text=translated_text,
                    time_range=seg.time_range,
                    language=target_lang
                )
                translated_segments.append(translated_segment)

            # 清理中间张量
            del inputs, outputs
            torch.cuda.empty_cache()

        return tuple(translated_segments)

    def unload(self):
        """卸载模型"""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 Qwen 翻译模型已卸载")