import json
from pathlib import Path
from typing import Optional


class FileCacheRepositoryAdapter:
    """文件缓存仓储适配器"""

    def __init__(self, cache_dir: Path = Path(".cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get(self, key: str) -> Optional[dict]:
        """获取缓存"""
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def set(self, key: str, value: dict) -> None:
        """设置缓存"""
        cache_file = self.cache_dir / f"{key}.json"

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        cache_file = self.cache_dir / f"{key}.json"
        return cache_file.exists() and cache_file.stat().st_size > 0

    def delete(self, key: str) -> bool:
        """删除缓存"""
        cache_file = self.cache_dir / f"{key}.json"
        try:
            if cache_file.exists():
                cache_file.unlink()
                return True
            return False
        except Exception:
            return False