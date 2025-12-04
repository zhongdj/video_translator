"""
Application Layer - 仅字幕合成用例
不生成语音，只生成字幕文件和烧录字幕视频
"""
import time
from pathlib import Path
from typing import Optional, Callable

from domain.entities import Video, Subtitle, ProcessedVideo
from domain.ports import VideoProcessor, SubtitleWriter
from domain.services import merge_bilingual_subtitles


def subtitle_only_synthesis_use_case(
        video: Video,
        target_subtitle: Subtitle,
        secondary_subtitle: Optional[Subtitle],
        video_processor: VideoProcessor,
        subtitle_writer: SubtitleWriter,
        output_dir: Path,
        enable_bilingual: bool = True,
        burn_subtitles: bool = True,
        formats: tuple[str, ...] = ("srt", "ass"),
        progress: Optional[Callable[[float, str], None]] = None
) -> tuple[tuple[Path, ...], str]:
    """
    仅字幕合成用例（纯函数）

    Args:
        video: 视频对象
        target_subtitle: 目标语言字幕（中文）
        secondary_subtitle: 次要语言字幕（英文），可选
        video_processor: 视频处理器
        subtitle_writer: 字幕写入器
        output_dir: 输出目录
        enable_bilingual: 是否生成双语字幕
        burn_subtitles: 是否烧录字幕到视频
        formats: 字幕格式
        progress: 进度回调

    Returns:
        (output_paths, status_message): 输出文件列表和状态信息
    """
    start_time = time.perf_counter()

    if progress:
        progress(0.0, "开始仅字幕模式合成")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    base_name = video.path.stem

    # 1. 生成目标语言字幕文件
    if progress:
        progress(0.2, f"生成{target_subtitle.language.value}字幕文件")

    if "srt" in formats:
        target_srt = output_dir / f"{base_name}.{target_subtitle.language.value}.srt"
        subtitle_writer.write_srt(target_subtitle, target_srt)
        output_paths.append(target_srt)
        print(f"📝 生成字幕: {target_srt.name}")

    if "ass" in formats:
        target_ass = output_dir / f"{base_name}.{target_subtitle.language.value}.ass"
        subtitle_writer.write_ass(target_subtitle, target_ass)
        output_paths.append(target_ass)
        print(f"📝 生成字幕: {target_ass.name}")

    # 2. 生成次要语言字幕（如果提供）
    if secondary_subtitle:
        if progress:
            progress(0.3, f"生成{secondary_subtitle.language.value}字幕文件")

        if "srt" in formats:
            secondary_srt = output_dir / f"{base_name}.{secondary_subtitle.language.value}.srt"
            subtitle_writer.write_srt(secondary_subtitle, secondary_srt)
            output_paths.append(secondary_srt)
            print(f"📝 生成字幕: {secondary_srt.name}")

        if "ass" in formats:
            secondary_ass = output_dir / f"{base_name}.{secondary_subtitle.language.value}.ass"
            subtitle_writer.write_ass(secondary_subtitle, secondary_ass)
            output_paths.append(secondary_ass)
            print(f"📝 生成字幕: {secondary_ass.name}")

    # 3. 生成双语字幕（如果启用且有次要字幕）
    bilingual_subtitle = None
    if enable_bilingual and secondary_subtitle:
        if progress:
            progress(0.4, "生成双语字幕")

        bilingual_subtitle = merge_bilingual_subtitles(
            target_subtitle,
            secondary_subtitle
        )

        bilingual_srt = output_dir / f"{base_name}.zh_en.srt"
        subtitle_writer.write_srt(bilingual_subtitle, bilingual_srt)
        output_paths.append(bilingual_srt)
        print(f"📝 生成双语字幕: {bilingual_srt.name}")

        bilingual_ass = output_dir / f"{base_name}.zh_en.ass"
        subtitle_writer.write_ass(bilingual_subtitle, bilingual_ass)
        output_paths.append(bilingual_ass)
        print(f"📝 生成双语字幕: {bilingual_ass.name}")

    # 4. 烧录字幕到视频（如果启用）
    if burn_subtitles:
        if progress:
            progress(0.6, "烧录字幕到视频")

        # 选择要烧录的字幕
        if bilingual_subtitle:
            subtitle_to_burn = bilingual_subtitle.with_path(
                output_dir / f"{base_name}.zh_en.ass"
            )
            subtitle_type = "双语"
        else:
            subtitle_to_burn = target_subtitle.with_path(
                output_dir / f"{base_name}.{target_subtitle.language.value}.ass"
            )
            subtitle_type = "单语"

        # 烧录字幕
        subtitled_video = output_dir / f"{base_name}_subtitled.mp4"
        video_processor.burn_subtitles(
            video,
            subtitle_to_burn,
            subtitled_video
        )
        output_paths.append(subtitled_video)
        print(f"🎬 生成{subtitle_type}字幕视频: {subtitled_video.name}")

    processing_time = time.perf_counter() - start_time

    if progress:
        progress(1.0, "仅字幕模式完成")

    # 生成状态报告
    status = f"""
✅ 仅字幕模式完成!

📦 输出文件 ({len(output_paths)} 个):
"""

    for path in output_paths:
        file_type = "字幕文件" if path.suffix in ['.srt', '.ass'] else "视频文件"
        status += f"   - {file_type}: {path.name}\n"

    status += f"""
⚙️  配置:
   字幕模式: {'双语' if enable_bilingual and secondary_subtitle else '单语'}
   烧录字幕: {'是' if burn_subtitles else '否'}
   字幕格式: {', '.join(formats)}

⏱️  处理时间: {processing_time:.1f} 秒

💡 提示: 未生成配音，原视频音频保持不变
"""

    return tuple(output_paths), status