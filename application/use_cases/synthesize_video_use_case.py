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

        if "srt" in formats:
            srt_path = output_dir / f"{base_name}.{file_prefix}.srt"
            subtitle_writer.write_srt(subtitle, srt_path)
            output_paths.append(srt_path)
            print(f"📝 生成字幕: {srt_path.name}")

        if "ass" in formats:
            ass_path = output_dir / f"{base_name}.{file_prefix}.ass"
            subtitle_writer.write_ass(subtitle, ass_path)
            output_paths.append(ass_path)
            print(f"📝 生成字幕: {ass_path.name}")

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

            bilingual_subtitle = subtitles[2]  # 第三个是双语字幕
            voiced_subtitled = output_dir / f"{video.path.stem}_voiced_subtitled.mp4"
            video_processor.burn_subtitles(
                voiced_output,  # 基于配音视频
                bilingual_subtitle,
                voiced_subtitled
            )
            output_paths.append(voiced_subtitled)
            print(f"🎬 生成配音+双语字幕视频: {voiced_subtitled.name}")

    # 3. 烧录字幕到原始视频（仅中文字幕）
    if burn_subtitles and subtitles:
        if progress:
            progress(0.8, "烧录字幕到原始视频")

        burned_output = output_dir / f"{video.path.stem}_subtitled.mp4"
        video_processor.burn_subtitles(
            video,
            subtitles[0],  # 使用中文字幕
            burned_output
        )
        output_paths.append(burned_output)
        print(f"🎬 生成硬字幕视频（中文）: {burned_output.name}")

    if progress:
        progress(1.0, "视频合成完成")

    processing_time = time.time() - start_time

    return VideoSynthesisResult(
        output_paths=tuple(output_paths),
        processing_time=processing_time
    )
