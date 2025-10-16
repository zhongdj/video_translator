import json
from pathlib import Path
from typing import Optional

from domain.translation import TranslationContext
from domain.translation_context_repository import TranslationContextRepository


class TranslationContextRepositoryAdapter(TranslationContextRepository):
    """翻译上下文仓储"""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_dir.mkdir(exist_ok=True)
        self._ensure_default_configs()

    def _ensure_default_configs(self):
        """确保默认配置存在"""
        default_contexts = {
            "general": TranslationContext(
                domain="general",
                system_prompt="""你是一位专业的视频字幕翻译专家。
    翻译要求：
    1. 准确传达原文含义，符合目标语言习惯
    2. 保持口语化风格，适合视频观看
    3. 简洁明了，避免冗长表达
    4. 保留专业术语的准确性""",
                terminology={}
            ),
            "inline_skating": TranslationContext(
                domain="inline_skating",
                system_prompt="""你是一位专业的轮滑（Inline Skating）视频字幕翻译专家。
    翻译要求：
    1. 准确使用轮滑专业术语
    2. 保持教学内容的清晰性和准确性
    3. 动作描述要精确，避免歧义
    4. 保持口语化，适合教学视频观看""",
                terminology={
                    "inline skating": "轮滑",
                    "inline skates": "直排轮",
                    "crossover": "交叉步",
                    "t-stop": "T字刹",
                    "powerslide": "侧刹",
                    "edge": "刃",
                    "stride": "滑步",
                    "frame": "刀架",
                    "wheel": "轮子",
                    "bearing": "轴承",
                    "cuff": "鞋腰",
                    "liner": "内胆",
                    "heel brake": "后刹",
                    "slalom": "平地花式"
                }
            )
        }

        for name, context in default_contexts.items():
            config_file = self.config_dir / f"{name}.json"
            if not config_file.exists():
                self.save(name, context)

    def save(self, name: str, context: TranslationContext):
        """保存翻译上下文"""
        config_file = self.config_dir / f"{name}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(context.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, name: str) -> Optional[TranslationContext]:
        """加载翻译上下文"""
        config_file = self.config_dir / f"{name}.json"
        if not config_file.exists():
            return None

        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return TranslationContext.from_dict(data)

    def list_contexts(self) -> list[str]:
        """列出所有可用的上下文"""
        return [f.stem for f in self.config_dir.glob("*.json")]