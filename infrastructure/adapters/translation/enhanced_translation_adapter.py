"""
Infrastructure Layer - 增强的翻译适配器（支持上下文）
"""
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from domain.entities import TextSegment, LanguageCode, TimeRange
from domain.ports import TranslationProvider


class EnhancedQwenTranslationAdapter(TranslationProvider):
    """增强的Qwen翻译适配器，支持系统提示词和术语表"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 可配置的翻译上下文
        self.system_prompt: Optional[str] = None
        self.terminology: dict[str, str] = {}



    def _load_model(self):
        """加载模型"""
        print(f"🔄 加载翻译模型: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print(f"✅ 翻译模型已加载到 {self.device}")

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt

    def set_terminology(self, terminology: dict[str, str]):
        """设置术语表"""
        self.terminology = terminology

    def _build_system_message(self, source_lang: LanguageCode, target_lang: LanguageCode) -> str:
        """构建系统消息"""
        lang_names = {
            LanguageCode.CHINESE: "中文",
            LanguageCode.ENGLISH: "英文",
            LanguageCode.PORTUGUESE: "葡萄牙语",
            LanguageCode.JAPANESE: "日语",
        }

        src_name = lang_names.get(source_lang, source_lang.value)
        tgt_name = lang_names.get(target_lang, target_lang.value)

        # 如果有自定义系统提示词，使用它
        if self.system_prompt:
            base_prompt = self.system_prompt
        else:
            base_prompt = f"你是一位专业的{src_name}到{tgt_name}翻译专家。"

        # 添加术语表说明
        if self.terminology:
            term_list = "\n".join([
                f"  - {src} → {tgt}"
                for src, tgt in self.terminology.items()
            ])
            base_prompt += f"\n\n专业术语对照表：\n{term_list}"

        base_prompt += f"\n\n请将以下{src_name}文本翻译成{tgt_name}。只输出译文，不要添加任何解释。"

        return base_prompt

    def _translate_single(
            self,
            text: str,
            source_lang: LanguageCode,
            target_lang: LanguageCode
    ) -> str:
        """翻译单个文本"""
        if not text.strip():
            return ""

        system_message = self._build_system_message(source_lang, target_lang)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ]

        # 使用tokenizer的chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        if self.model is None:
            self._load_model()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def _translate_batch(
            self,
            texts: list[str],
            source_lang: LanguageCode,
            target_lang: LanguageCode,
            batch_size: int = 4
    ) -> list[str]:
        """批量翻译（分批处理）"""
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []

            for text in batch:
                translated = self._translate_single(text, source_lang, target_lang)
                batch_results.append(translated)

            results.extend(batch_results)

            # 打印进度
            progress = min(i + batch_size, len(texts))
            print(f"  翻译进度: {progress}/{len(texts)}")

        return results

    def translate(
            self,
            segments: tuple[TextSegment, ...],
            source_lang: LanguageCode,
            target_lang: LanguageCode
    ) -> tuple[TextSegment, ...]:
        """翻译文本片段"""
        if not segments:
            return tuple()

        print(f"🌐 翻译: {source_lang.value} → {target_lang.value}")
        print(f"   片段数: {len(segments)}")
        if self.system_prompt:
            print(f"   使用自定义提示词: 是")
        if self.terminology:
            print(f"   术语表条目: {len(self.terminology)}")

        # 提取文本
        texts = [seg.text for seg in segments]

        # 批量翻译
        translated_texts = self._translate_batch(texts, source_lang, target_lang)

        # 重建片段
        translated_segments = tuple(
            TextSegment(
                text=translated_text,
                time_range=original_seg.time_range,
                language=target_lang
            )
            for translated_text, original_seg in zip(translated_texts, segments)
        )

        print(f"✅ 翻译完成")

        return translated_segments

    def unload(self):
        """卸载模型释放内存"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 重置上下文
        self.system_prompt = None
        self.terminology = {}

        print("✅ 翻译模型已卸载")


# ============== 工厂函数 ============== #

def create_enhanced_translation_provider(
        model_name: str = "Qwen/Qwen2.5-7B-Instruct"
) -> TranslationProvider:
    """创建增强的翻译提供者"""
    return EnhancedQwenTranslationAdapter(model_name)