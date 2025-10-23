"""
Application Layer - 缓存服务
封装所有缓存相关的业务逻辑，避免UI层直接调用Domain Service
"""

from pathlib import Path
from typing import Optional, Dict, Any

from domain.entities import Video, Subtitle, LanguageCode, TextSegment, TimeRange
from domain.ports import CacheRepository
from domain.services import calculate_cache_key


class CacheService:
    """缓存服务（应用层）"""

    def __init__(self, cache_repo: CacheRepository):
        self.cache_repo = cache_repo

    # ============== 字幕缓存 ============== #

    def get_subtitle_cache_key(
            self,
            video_path: Path,
            target_language: LanguageCode,
            source_language: Optional[LanguageCode] = None,
            context_domain: Optional[str] = None
    ) -> str:
        """获取字幕缓存键（统一入口）"""
        params = {
            "target_language": target_language.value,
            "source_language": source_language.value if source_language else None
        }

        if context_domain:
            params["context_domain"] = context_domain

        return calculate_cache_key(video_path, "subtitles_v2", params)

    def save_subtitle_cache(
            self,
            video_path: Path,
            original_subtitle: Subtitle,
            chinese_subtitle: Subtitle,
            english_subtitle: Subtitle,
            detected_language: LanguageCode,
            source_language: Optional[LanguageCode] = None,
            context_domain: Optional[str] = None
    ) -> None:
        """保存字幕到缓存"""
        cache_key = self.get_subtitle_cache_key(
            video_path,
            LanguageCode.CHINESE,
            source_language,
            context_domain
        )

        cache_data = {
            "detected_language": detected_language.value,
            "zh_segments": self._serialize_segments(chinese_subtitle.segments),
            "en_segments": self._serialize_segments(english_subtitle.segments),
        }

        # 如果原始语言不是中英文，也保存
        if detected_language not in [LanguageCode.CHINESE, LanguageCode.ENGLISH]:
            cache_data[f"{detected_language.value}_segments"] = self._serialize_segments(
                original_subtitle.segments
            )

        self.cache_repo.set(cache_key, cache_data)

    def load_subtitle_cache(
            self,
            video_path: Path,
            source_language: Optional[LanguageCode] = None,
            context_domain: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """从缓存加载字幕"""
        cache_key = self.get_subtitle_cache_key(
            video_path,
            LanguageCode.CHINESE,
            source_language,
            context_domain
        )

        if not self.cache_repo.exists(cache_key):
            return None

        cached = self.cache_repo.get(cache_key)
        if not cached:
            return None

        detected_lang = LanguageCode(cached["detected_language"])

        # 重建字幕对象
        zh_subtitle = self._rebuild_subtitle(
            cached.get("zh_segments", []),
            LanguageCode.CHINESE
        )

        en_subtitle = self._rebuild_subtitle(
            cached.get("en_segments", []),
            LanguageCode.ENGLISH
        )

        # 原始字幕
        if detected_lang == LanguageCode.CHINESE:
            original_subtitle = zh_subtitle
        elif detected_lang == LanguageCode.ENGLISH:
            original_subtitle = en_subtitle
        else:
            original_subtitle = self._rebuild_subtitle(
                cached.get(f"{detected_lang.value}_segments", []),
                detected_lang
            ) or zh_subtitle

        return {
            "original_subtitle": original_subtitle,
            "chinese_subtitle": zh_subtitle,
            "english_subtitle": en_subtitle,
            "detected_language": detected_lang,
        }

    def update_chinese_subtitle(
            self,
            video_path: Path,
            updated_subtitle: Subtitle,
            source_language: Optional[LanguageCode] = None,
            context_domain: Optional[str] = None
    ) -> None:
        """更新缓存中的中文字幕（用于编辑后保存）"""
        cache_key = self.get_subtitle_cache_key(
            video_path,
            LanguageCode.CHINESE,
            source_language,
            context_domain
        )

        # 读取现有缓存
        cached = self.cache_repo.get(cache_key) or {}

        # 只更新中文字幕部分
        cached["zh_segments"] = self._serialize_segments(updated_subtitle.segments)

        # 写回
        self.cache_repo.set(cache_key, cached)

    def invalidate_downstream_caches(
            self,
            video_path: Path,
            detected_language: LanguageCode
    ) -> None:
        """使下游缓存失效（语音、视频）"""
        # 删除语音克隆缓存
        voice_key = calculate_cache_key(
            video_path,
            "clone_voice",
            {
                "target_language": LanguageCode.CHINESE.value,
                "source_language": detected_language.value,
            }
        )
        self.cache_repo.delete(voice_key)

        # 删除视频合成缓存
        synth_key = calculate_cache_key(
            video_path,
            "synthesize_video",
            {
                "target_language": LanguageCode.CHINESE.value,
                "source_language": detected_language.value,
            }
        )
        self.cache_repo.delete(synth_key)

    # ============== 私有辅助方法 ============== #

    def _serialize_segments(self, segments: tuple[TextSegment, ...]) -> list:
        """序列化片段"""
        return [
            {
                "text": seg.text,
                "start": seg.time_range.start_seconds,
                "end": seg.time_range.end_seconds
            }
            for seg in segments
        ]

    def _rebuild_subtitle(
            self,
            segments_data: list,
            language: LanguageCode
    ) -> Optional[Subtitle]:
        """重建字幕对象"""
        if not segments_data:
            return None

        segments = tuple(
            TextSegment(
                text=seg["text"],
                time_range=TimeRange(seg["start"], seg["end"]),
                language=language
            )
            for seg in segments_data
        )

        return Subtitle(segments, language)