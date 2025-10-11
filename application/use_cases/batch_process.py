from application import *
from application.use_cases.clone_voice import clone_voice_use_case
from application.use_cases.generate_subtitles import generate_subtitles_use_case
from application.use_cases.synthesize_video_use_case import synthesize_video_use_case

# 导入领域层
from domain.entities import (
    # Entities
    Video, ProcessedVideo,  # Value Objects
    LanguageCode, )

from domain.ports import (
# Ports
    ASRProvider, TranslationProvider, TTSProvider,
    VideoProcessor, SubtitleWriter, CacheRepository,

)

from domain.services import (
    # Domain Services
    merge_bilingual_subtitles, )

def batch_process_use_case(
        videos: tuple[Video, ...],
        asr_provider: ASRProvider,
        translation_provider: TranslationProvider,
        tts_provider: TTSProvider,
        video_processor: VideoProcessor,
        subtitle_writer: SubtitleWriter,
        cache_repo: CacheRepository,
        output_dir: Path,
        enable_voice_cloning: bool = True,
        target_language: LanguageCode = LanguageCode.CHINESE,
        progress: ProgressCallback = None
) -> tuple[ProcessedVideo, ...]:
    """
    批量处理用例（纯函数）

    编排多个用例，批量处理视频
    """
    if progress:
        progress(0.0, f"开始批量处理 {len(videos)} 个视频")

    results = []
    total_videos = len(videos)

    for idx, video in enumerate(videos):
        video_progress_start = idx / total_videos
        video_progress_range = 1.0 / total_videos

        def video_progress(p: float, desc: str):
            if progress:
                overall_progress = video_progress_start + (p * video_progress_range)
                progress(overall_progress, f"视频 {idx + 1}/{total_videos}: {desc}")

        # 1. 生成字幕
        subtitle_result = generate_subtitles_use_case(
            video=video,
            asr_provider=asr_provider,
            translation_provider=translation_provider,
            video_processor=video_processor,
            cache_repo=cache_repo,
            target_language=target_language,
            progress=lambda p, d: video_progress(p * 0.4, d)
        )

        # 2. 克隆语音（如果启用）
        audio_track = None
        if enable_voice_cloning:
            voice_result = clone_voice_use_case(
                video=video,
                subtitle=subtitle_result.translated_subtitle,
                tts_provider=tts_provider,
                video_processor=video_processor,
                cache_repo=cache_repo,
                progress=lambda p, d: video_progress(0.4 + p * 0.4, d)
            )
            audio_track = voice_result.audio_track

        # 3. 合成视频
        # 合并双语字幕
        bilingual = merge_bilingual_subtitles(
            subtitle_result.translated_subtitle,
            subtitle_result.original_subtitle
        )

        synthesis_result = synthesize_video_use_case(
            video=video,
            subtitles=(
                subtitle_result.translated_subtitle,
                subtitle_result.original_subtitle,
                bilingual
            ),
            audio_track=audio_track,
            video_processor=video_processor,
            subtitle_writer=subtitle_writer,
            output_dir=output_dir,
            burn_subtitles=True,
            progress=lambda p, d: video_progress(0.8 + p * 0.2, d)
        )

        # 构建结果
        processed = ProcessedVideo(
            original_video=video,
            subtitles=(
                subtitle_result.translated_subtitle,
                subtitle_result.original_subtitle,
                bilingual
            ),
            audio_tracks=(audio_track,) if audio_track else tuple(),
            output_paths=synthesis_result.output_paths
        )

        results.append(processed)

    if progress:
        progress(1.0, "批量处理完成")

    return tuple(results)