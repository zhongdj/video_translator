from pathlib import Path


class LocalFileStorageAdapter:
    """本地文件存储适配器"""

    def save(self, data: bytes, path: Path) -> Path:
        """保存文件"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def load(self, path: Path) -> bytes:
        """加载文件"""
        return path.read_bytes()

    def exists(self, path: Path) -> bool:
        """检查文件是否存在"""
        return path.exists() and path.is_file()