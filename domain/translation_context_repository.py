from typing import Optional

from domain.translation import TranslationContext


class TranslationContextRepository:

    def save(self, name: str, context: TranslationContext):
        """保存翻译上下文"""
        ...

    def load(self, name: str) -> Optional[TranslationContext]:
        """加载翻译上下文"""
        ...
    def list_contexts(self) -> list[str]:
        """列出所有可用的上下文"""
        ...
