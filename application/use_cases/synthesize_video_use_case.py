from application import *

# 导入领域层
from domain.entities import (
    # Entities
    Video, Subtitle, AudioTrack,  # Value Objects
)

from domain.ports import (
# Ports
    VideoProcessor, SubtitleWriter, )


def generate_subtitle_files_with_paths(
        subtitles: tuple[Subtitle, ...],
        video: Video,
        subtitle_writer: SubtitleWriter,
        output_dir: Path,
        formats: tuple[str, ...] = ("srt", "ass")
) -> tuple[Subtitle, ...]:
    """
    为字幕列表生成文件并返回带路径的 Subtitle 对象

    Args:
        subtitles: 原始字幕对象列表
        video: 视频对象
        subtitle_writer: 字幕写入器
        output_dir: 输出目录
        formats: 要生成的格式

    Returns:
        带有路径属性的新 Subtitle 对象列表
    """

    base_name = video.path.stem
    subtitles_with_paths = []

    for subtitle in subtitles:
        lang_code = subtitle.language.value

        # 检查是否是双语字幕（文本中包含换行符）
        is_bilingual = any('\n' in seg.text for seg in subtitle.segments)

        if is_bilingual:
            # 双语字幕：命名为 zh_en
            file_prefix = "zh_en"
        else:
            # 单语字幕：直接使用语言代码
            file_prefix = lang_code

        subtitle_path = None

        # 生成指定格式的字幕文件
        if "srt" in formats:
            srt_path = output_dir / f"{base_name}.{file_prefix}.srt"
            subtitle_writer.write_srt(subtitle, srt_path)
            subtitle_path = srt_path  # 优先使用 srt 路径
            print(f"📝 生成字幕: {srt_path.name}")

        if "ass" in formats and subtitle_path is None:
            # 如果没有生成 srt，则使用 ass 路径
            ass_path = output_dir / f"{base_name}.{file_prefix}.ass"
            subtitle_writer.write_ass(subtitle, ass_path)
            subtitle_path = ass_path
            print(f"📝 生成字幕: {ass_path.name}")

        # 创建带有路径的新 Subtitle 对象
        if subtitle_path:
            new_subtitle = subtitle.with_path(subtitle_path)
            subtitles_with_paths.append(new_subtitle)
        else:
            # 如果没有生成任何文件，保持原样
            subtitles_with_paths.append(subtitle)

    return tuple(subtitles_with_paths)

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
    合成视频用例（纯函数）- 修复版

    流程:
    1. 写字幕文件（使用明确的命名规则）
    2. 如果有音轨，合并音视频
    3. 如果需要，烧录字幕
    """
    import time
    start_time = time.time()

    if progress:
        progress(0.0, "开始视频合成")

    output_paths = []

    # 1. 写字幕文件 - 使用明确的命名规则
    if progress:
        progress(0.2, "生成字幕文件")

    # 根据语言代码判断字幕类型
    # 新规范: 始终生成 zh, en, zh_en 三种字幕
    base_name = video.path.stem

    # 为双语字幕也生成文件
    bilingual_with_paths = generate_subtitle_files_with_paths(
        subtitles=subtitles,
        video=video,
        subtitle_writer=subtitle_writer,
        output_dir=output_dir,
        formats=("srt", "ass")  # 双语字幕生成所有格式
    )

    bilingual_with_path = bilingual_with_paths[2]
    print(bilingual_with_path)

    # 2. 合并音视频（如果有配音）
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
        print(f"🎤 生成配音视频: {voiced_output.name}")

        # 为配音视频烧录双语字幕（如果有双语字幕）
        if burn_subtitles and len(subtitles) >= 3:
            if progress:
                progress(0.7, "为配音视频烧录双语字幕")

            voiced_subtitled = output_dir / f"{video.path.stem}_voiced_subtitled.mp4"
            video_processor.burn_subtitles(
                Video(path = voiced_output,
                      duration=video.duration,
                      has_audio=True),  # 基于配音视频
                bilingual_with_path,
                voiced_subtitled
            )
            output_paths.append(voiced_subtitled)
            print(f"🎬 生成配音+双语字幕视频: {voiced_subtitled.name}")



    if progress:
        progress(1.0, "视频合成完成")

    processing_time = time.time() - start_time

    return VideoSynthesisResult(
        output_paths=tuple(output_paths),
        processing_time=processing_time
    )
