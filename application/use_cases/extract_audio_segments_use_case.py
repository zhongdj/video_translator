from application import *

# 导入领域层
from domain.entities import (
    # Entities
    Video, Subtitle,  # Value Objects
)

from domain.ports import (
# Ports
    VideoProcessor, )


def extract_audio_segments_use_case(
        video: Video,
        subtitle: Subtitle,
        video_processor: VideoProcessor,
        output_dir: Path
) -> tuple[Path, ...]:
    """
    提取音频片段用例（纯函数）

    按字幕时间轴切割音频，用于分析或训练
    """
    # 提取完整音频
    full_audio_path = video_processor.extract_audio(video)

    # 这里实际需要音频处理能力，简化实现
    # 返回分段音频路径列表
    segment_paths = []
    for idx, segment in enumerate(subtitle.segments):
        segment_path = output_dir / f"segment_{idx:04d}.wav"
        segment_paths.append(segment_path)

    return tuple(segment_paths)