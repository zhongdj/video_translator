from application import *

# 导入领域层
from domain.entities import (
    # Entities
    Subtitle,  # Value Objects
)

from domain.ports import (
# Ports
    SubtitleWriter, )

from domain.services import (
    # Domain Services
    merge_bilingual_subtitles, )

def create_bilingual_subtitle_use_case(
        primary_subtitle: Subtitle,
        secondary_subtitle: Subtitle,
        subtitle_writer: SubtitleWriter,
        output_path: Path,
        format: str = "ass"
) -> Path:
    """
    创建双语字幕用例（纯函数）
    """
    bilingual = merge_bilingual_subtitles(primary_subtitle, secondary_subtitle)

    if format == "ass":
        return subtitle_writer.write_ass(bilingual, output_path)
    else:
        return subtitle_writer.write_srt(bilingual, output_path)