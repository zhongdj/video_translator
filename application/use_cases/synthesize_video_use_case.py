from application import *

# 导入领域层
from domain.entities import (
    # Entities
    Video, Subtitle, AudioTrack,  # Value Objects
)

from domain.ports import (
# Ports
    VideoProcessor, SubtitleWriter, )


def synthesize_video_use_case(
        video: Video,
        subtitles: tuple[Subtitle, ...],
        audio_track: Optional[AudioTrack],
        video_processor: VideoProcessor,
        subtitle_writer: SubtitleWriter,
        output_dir: Path,
        formats: tuple[str, ...] = ("srt", "ass"),
        burn_subtitles: bool = False,
        progress: ProgressCallback = None
) -> VideoSynthesisResult:
    """
    合成视频用例（纯函数）

    流程:
    1. 写字幕文件
    2. 如果有音轨，合并音视频
    3. 如果需要，烧录字幕
    """
    import time
    start_time = time.time()

    if progress:
        progress(0.0, "开始视频合成")

    output_paths = []

    # 1. 写字幕文件
    if progress:
        progress(0.2, "生成字幕文件")

    for subtitle in subtitles:
        lang_suffix = subtitle.language.value

        if "srt" in formats:
            srt_path = output_dir / f"{video.path.stem}.{lang_suffix}.srt"
            subtitle_writer.write_srt(subtitle, srt_path)
            output_paths.append(srt_path)

        if "ass" in formats:
            ass_path = output_dir / f"{video.path.stem}.{lang_suffix}.ass"
            subtitle_writer.write_ass(subtitle, ass_path)
            output_paths.append(ass_path)

    # 2. 合并音视频
    if audio_track is not None:
        if progress:
            progress(0.5, "合并音频和视频")

        voiced_output = output_dir / f"{video.path.stem}_voiced.mp4"
        video_processor.merge_audio_video(
            video,
            audio_track,
            voiced_output
        )
        output_paths.append(voiced_output)

    # 3. 烧录字幕
    if burn_subtitles and subtitles:
        if progress:
            progress(0.8, "烧录字幕")

        # 使用第一个字幕（通常是翻译后的）
        burned_output = output_dir / f"{video.path.stem}_subtitled.mp4"
        video_processor.burn_subtitles(
            video,
            subtitles[0],
            burned_output
        )
        output_paths.append(burned_output)

    if progress:
        progress(1.0, "视频合成完成")

    processing_time = time.time() - start_time

    return VideoSynthesisResult(
        output_paths=tuple(output_paths),
        processing_time=processing_time
    )
