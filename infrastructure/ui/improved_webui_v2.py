"""
Infrastructure Layer - 重构后的WebUI V2（完整版）

✅ 需求1: 支持可配置的参考音频起始偏移
✅ 需求2: 支持双语字幕可选
✅ 需求3: 使用AudioFileRepository管理参考音频
"""

from pathlib import Path
from typing import Optional, Dict

import gradio as gr

from application.use_cases.incremental_voice_cloning import (
    incremental_voice_cloning_use_case,
    regenerate_modified_segments_use_case
)
from domain.entities import (
    Video, Subtitle, LanguageCode,
    AudioSegment, SegmentReviewStatus
)
from infrastructure.config.dependency_injection import container

# 初始化仓储
audio_segment_repo = container.audio_segment_repo
audio_file_repo = container.audio_file_repo  # ✅ 使用音频文件仓储
cache_service = container.cache_service


# ============== 会话状态 ============== #

class TranslationSessionV2:
    """翻译会话状态"""

    def __init__(self):
        self.translation_context = None
        self.video: Optional[Video] = None
        self.original_subtitle: Optional[Subtitle] = None
        self.translated_subtitle: Optional[Subtitle] = None
        self.english_subtitle: Optional[Subtitle] = None
        self.detected_language: Optional[LanguageCode] = None
        self.source_language: Optional[LanguageCode] = None
        self.quality_report = None
        self.audio_segments: Dict[int, AudioSegment] = {}
        self.segment_review_status: Dict[int, SegmentReviewStatus] = {}
        self.edited_segments: Dict[int, str] = {}
        self.modified_indices: set[int] = set()
        self.reference_audio_path: Optional[Path] = None
        self.approved = False


current_session = TranslationSessionV2()


# ============== 辅助函数 ============== #

def _load_cached_audio_segments(video: Video, subtitle: Subtitle) -> Dict[int, AudioSegment]:
    """从磁盘加载已缓存的音频片段"""
    cached_segments = {}

    for idx, text_seg in enumerate(subtitle.segments):
        try:
            audio_seg = audio_segment_repo.load_segment(
                segment_index=idx,
                video_path=video.path,
                text_segment=text_seg
            )
            if audio_seg:
                cached_segments[idx] = audio_seg
        except Exception:
            continue

    return cached_segments


def _source_language_cache_format(source_language: str) -> Optional[LanguageCode]:
    """转换源语言格式"""
    return LanguageCode(source_language) if source_language != "auto" else None


def _prepare_review_data_v2():
    """准备审核数据"""
    if not current_session.translated_subtitle:
        return None

    data = []
    for idx, (orig_seg, trans_seg) in enumerate(
        zip(current_session.original_subtitle.segments,
            current_session.translated_subtitle.segments)
    ):
        en_text = (
            current_session.english_subtitle.segments[idx].text
            if current_session.english_subtitle
               and idx < len(current_session.english_subtitle.segments)
            else orig_seg.text
        )

        audio_seg = current_session.audio_segments.get(idx)
        if audio_seg:
            actual_duration = len(audio_seg.audio.samples) / audio_seg.audio.sample_rate
            audio_status = "✅ 已缓存"
            duration_str = f"{actual_duration:.2f}s"
        else:
            audio_status = "未生成"
            duration_str = "-"

        data.append([
            idx,
            f"{trans_seg.time_range.start_seconds:.2f}s",
            en_text,
            trans_seg.text,
            f"{trans_seg.time_range.duration:.2f}s",
            duration_str,
            audio_status,
            "",
            "⏳ 待审核"
        ])

    return data


def _apply_edits_to_subtitle_v2():
    """应用编辑到字幕对象"""
    if not current_session.edited_segments:
        return

    from domain.entities import TextSegment

    new_segments = []
    for idx, seg in enumerate(current_session.translated_subtitle.segments):
        if idx in current_session.edited_segments:
            new_seg = TextSegment(
                text=current_session.edited_segments[idx],
                time_range=seg.time_range,
                language=seg.language
            )
            new_segments.append(new_seg)
        else:
            new_segments.append(seg)

    current_session.translated_subtitle = Subtitle(
        segments=tuple(new_segments),
        language=current_session.translated_subtitle.language
    )


def get_video_duration(video_path: Path) -> float:
    """获取视频时长"""
    import subprocess
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ], capture_output=True, text=True)
    return float(result.stdout.strip())


# ============== 步骤1: 生成字幕 ============== #

def step1_generate_and_check_v2(
    video_file,
    whisper_model: str,
    translation_model: str,
    translation_context_name: str,
    source_language: str,
    progress=gr.Progress()
):
    """步骤1: 生成字幕"""
    if not video_file:
        return None, "❌ 请上传视频", gr.update(visible=False)

    try:
        global current_session
        current_session = TranslationSessionV2()

        video_path = Path(video_file.name)
        current_session.video = Video(
            path=video_path,
            duration=get_video_duration(video_path),
            has_audio=True
        )

        translation_context = container.translator_context_repo.load(
            translation_context_name
        )

        src_lang = _source_language_cache_format(source_language)

        progress(0.1, "检查缓存...")
        cached_result = cache_service.load_subtitle_cache(
            video_path=video_path,
            source_language=src_lang,
            context_domain=translation_context.domain if translation_context else None
        )

        if cached_result:
            current_session.original_subtitle = cached_result["original_subtitle"]
            current_session.translated_subtitle = cached_result["chinese_subtitle"]
            current_session.english_subtitle = cached_result["english_subtitle"]
            current_session.detected_language = cached_result["detected_language"]
            current_session.source_language = src_lang
            current_session.translation_context = translation_context

            status_report = f"""
✅ 字幕缓存命中

📊 基本信息:
   视频: {video_path.name}
   检测语言: {cached_result['detected_language'].value}
   总片段数: {len(cached_result['chinese_subtitle'].segments)}
"""
            review_data = _prepare_review_data_v2()
            return review_data, status_report, gr.update(visible=True)

        progress(0.2, "生成字幕...")

        from application.use_cases.improved_generate_subtitles import improved_generate_subtitles_use_case

        result = improved_generate_subtitles_use_case(
            video=current_session.video,
            asr_provider=container.get_asr(whisper_model),
            translation_provider=container.get_translator(),
            video_processor=container.video_processor,
            cache_repo=container.cache_repo,
            translation_context=translation_context,
            target_language=LanguageCode.CHINESE,
            source_language=src_lang,
            enable_quality_check=True,
            progress=lambda p, d: progress(p, d)
        )

        container.get_translator().unload()

        current_session.original_subtitle = result.original_subtitle
        current_session.translated_subtitle = result.translated_subtitle
        current_session.detected_language = result.detected_language
        current_session.translation_context = translation_context
        current_session.source_language = src_lang

        cached_result = cache_service.load_subtitle_cache(
            video_path=video_path,
            source_language=src_lang,
            context_domain=translation_context.domain if translation_context else None
        )

        if cached_result and cached_result["english_subtitle"]:
            current_session.english_subtitle = cached_result["english_subtitle"]

        status_report = f"""
✅ 字幕生成完成

📊 基本信息:
   视频: {video_path.name}
   检测语言: {result.detected_language.value}
   总片段数: {len(result.translated_subtitle.segments)}
"""

        review_data = _prepare_review_data_v2()
        return review_data, status_report, gr.update(visible=True)

    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, gr.update(visible=False)


# ============== 步骤2A: 增量语音克隆 ============== #

def step2_incremental_voice_cloning(
    reference_audio_file,
    ref_audio_duration: float,  # ✅ 新增参数
    ref_audio_start_offset: float,  # ✅ 新增参数
    progress=gr.Progress()
):
    """步骤2A: 增量语音克隆（重构版）"""
    global current_session

    if not current_session.video or not current_session.translated_subtitle:
        return "❌ 错误: 会话状态丢失", gr.update()

    try:
        # ✅ 修复: 准备参考音频（使用AudioFileRepository）
        if reference_audio_file:
            # 用户上传了参考音频
            progress(0.05, "保存参考音频...")

            # 使用AudioFileRepository持久化Gradio临时文件
            ref_audio_path = audio_file_repo.save_reference_audio(
                video_path=current_session.video.path,
                source_audio_path=Path(reference_audio_file.name)
            )
            current_session.reference_audio_path = ref_audio_path
            print(f"📁 使用用户上传的参考音频: {ref_audio_path}")

        else:
            # 先尝试加载已存在的参考音频
            existing_ref_audio = audio_file_repo.load_reference_audio(
                current_session.video.path
            )

            if existing_ref_audio and existing_ref_audio.exists():
                ref_audio_path = existing_ref_audio
                current_session.reference_audio_path = ref_audio_path
                print(f"📁 复用已有参考音频: {ref_audio_path}")
            else:
                # 从视频提取参考音频
                progress(0.05, f"从视频提取参考音频（偏移: {ref_audio_start_offset}s, 时长: {ref_audio_duration}s）...")

                temp_ref_audio = container.video_processor.extract_reference_audio(
                    video=current_session.video,
                    duration=ref_audio_duration,
                    start_offset=ref_audio_start_offset  # ✅ 使用可配置的偏移
                )

                # 持久化提取的音频
                ref_audio_path = audio_file_repo.save_reference_audio(
                    video_path=current_session.video.path,
                    source_audio_path=temp_ref_audio
                )
                current_session.reference_audio_path = ref_audio_path

                # 清理临时文件
                if temp_ref_audio.exists():
                    temp_ref_audio.unlink()

                print(f"📁 从视频提取参考音频: {ref_audio_path}")

        # 实时进度回调
        def segment_progress(ratio, msg, idx, audio_seg):
            progress(ratio, msg)
            if audio_seg:
                current_session.audio_segments[idx] = audio_seg
                status = current_session.segment_review_status.get(idx)
                if status and not status.subtitle_modified:
                    current_session.segment_review_status[idx] = SegmentReviewStatus(
                        segment_index=idx,
                        subtitle_approved=False,
                        audio_approved=True,
                        subtitle_modified=False,
                        needs_regeneration=False
                    )

        # 执行增量合成
        result = incremental_voice_cloning_use_case(
            video=current_session.video,
            subtitle=current_session.translated_subtitle,
            tts_provider=container.get_tts(),
            video_processor=container.video_processor,
            audio_repo=audio_segment_repo,
            cache_repo=container.cache_repo,
            reference_audio_path=ref_audio_path,
            progress=segment_progress
        )

        # 更新所有音频片段到会话
        for audio_seg in result.audio_segments:
            current_session.audio_segments[audio_seg.segment_index] = audio_seg

        status = f"""
✅ 增量语音克隆完成!

📊 统计信息:
   总片段数: {result.total_segments}
   缓存命中: {result.cached_segments}
   新生成: {result.regenerated_segments}
   耗时: {result.synthesis_time:.1f} 秒

📁 参考音频: {ref_audio_path.name}
   起始偏移: {ref_audio_start_offset}s
   时长: {ref_audio_duration}s

💡 提示: 
   - 参考音频已持久化，修改字幕后可安全重新生成
   - 点击表格中的行查看和播放音频
"""

        updated_data = _prepare_review_data_v2()
        return status, gr.update(value=updated_data)

    except Exception as e:
        import traceback
        error_msg = f"❌ 语音克隆失败: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, gr.update()


# ============== 步骤2B: 保存编辑 ============== #

def step2_save_edits_and_regenerate(review_dataframe):
    """保存编辑并标记需要重新生成的片段"""
    global current_session

    if hasattr(review_dataframe, "values"):
        review_dataframe = review_dataframe.values.tolist()

    if not review_dataframe:
        return "⚠️ 没有可保存的修改", gr.update()

    if review_dataframe and isinstance(review_dataframe[0][0], str):
        review_dataframe = review_dataframe[1:]

    edited_count = 0

    for row in review_dataframe:
        try:
            idx = int(row[0])
        except (ValueError, IndexError):
            continue

        if idx >= len(current_session.translated_subtitle.segments):
            continue

        original_text = current_session.translated_subtitle.segments[idx].text
        edited_text = row[3]

        if edited_text != original_text:
            current_session.edited_segments[idx] = edited_text
            current_session.modified_indices.add(idx)
            edited_count += 1

    if edited_count:
        _apply_edits_to_subtitle_v2()

        cache_service.update_chinese_subtitle(
            video_path=current_session.video.path,
            updated_subtitle=current_session.translated_subtitle,
            source_language=current_session.source_language,
            context_domain=current_session.translation_context.domain
            if current_session.translation_context else None
        )

        cache_service.invalidate_downstream_caches(
            video_path=current_session.video.path,
            detected_language=current_session.detected_language
        )

        updated_data = _prepare_review_data_v2()

        return (
            f"✅ 已保存 {edited_count} 处修改（已同步到缓存）\n"
            f"⚠️ 需要重新生成 {len(current_session.modified_indices)} 个音频片段",
            gr.update(value=updated_data)
        )
    else:
        return "ℹ️ 未检测到修改", gr.update()


# ============== 步骤2C: 重新生成 ============== #

def step2_regenerate_modified():
    """重新生成修改过的片段（修复版）"""
    global current_session

    if not current_session.modified_indices:
        return "ℹ️ 没有需要重新生成的片段", gr.update()

    # ✅ 修复: 智能获取参考音频
    ref_audio_path = None

    # 1. 优先使用会话中的路径
    if current_session.reference_audio_path and current_session.reference_audio_path.exists():
        ref_audio_path = current_session.reference_audio_path
        print(f"📁 使用会话中的参考音频: {ref_audio_path}")

    # 2. 尝试从仓储加载
    else:
        ref_audio_path = audio_file_repo.load_reference_audio(current_session.video.path)
        if ref_audio_path and ref_audio_path.exists():
            current_session.reference_audio_path = ref_audio_path
            print(f"📁 从仓储加载参考音频: {ref_audio_path}")

    # 3. 都失败了，提示用户
    if not ref_audio_path:
        return (
            "❌ 错误: 缺少参考音频\n\n"
            "💡 解决方案:\n"
            "   1. 重新执行步骤2A\n"
            "   2. 上传参考音频或让系统从视频提取",
            gr.update()
        )

    try:
        print(f"\n🔄 重新生成修改片段:")
        print(f"   修改片段数: {len(current_session.modified_indices)}")
        print(f"   参考音频: {ref_audio_path}")

        result = regenerate_modified_segments_use_case(
            video=current_session.video,
            original_subtitle=current_session.original_subtitle,
            modified_subtitle=current_session.translated_subtitle,
            modified_indices=current_session.modified_indices,
            tts_provider=container.get_tts(),
            video_processor=container.video_processor,
            audio_repo=audio_segment_repo,
            reference_audio_path=ref_audio_path,
            progress=None
        )

        for audio_seg in result.audio_segments:
            current_session.audio_segments[audio_seg.segment_index] = audio_seg

        current_session.modified_indices.clear()

        updated_data = _prepare_review_data_v2()

        return (
            f"✅ 重新生成完成!\n"
            f"   重新生成: {result.regenerated_segments} 个片段\n"
            f"   耗时: {result.synthesis_time:.1f} 秒",
            gr.update(value=updated_data)
        )

    except Exception as e:
        import traceback
        return f"❌ 重新生成失败: {str(e)}\n{traceback.format_exc()}", gr.update()


# ============== 片段预览 ============== #

def preview_segment(evt: gr.SelectData):
    """预览选中的片段"""
    global current_session

    if evt is None or not current_session.video or not current_session.translated_subtitle:
        return None, "⚠️ 无效的会话状态", "", ""

    try:
        if isinstance(evt.index, (tuple, list)):
            selected_row_index = int(evt.index[0])
        else:
            selected_row_index = int(evt.index)

        total_segments = len(current_session.translated_subtitle.segments)
        if selected_row_index < 0 or selected_row_index >= total_segments:
            return None, f"❌ 无效的片段索引: {selected_row_index}", "", ""

        idx = selected_row_index
        text_seg = current_session.translated_subtitle.segments[idx]
        max_duration = text_seg.time_range.duration

        audio_seg = current_session.audio_segments.get(idx)
        if not audio_seg:
            audio_seg = audio_segment_repo.load_segment(
                segment_index=idx,
                video_path=current_session.video.path,
                text_segment=text_seg
            )
            if audio_seg:
                current_session.audio_segments[idx] = audio_seg

        actual_duration = None
        if audio_seg and audio_seg.file_path and audio_seg.file_path.exists():
            audio_path = str(audio_seg.file_path)
            actual_duration = len(audio_seg.audio.samples) / audio_seg.audio.sample_rate
            duration_diff = actual_duration - max_duration
            diff_sign = "+" if duration_diff > 0 else ""
            audio_status = f"✅ 音频已生成 ({(actual_duration / max_duration * 100):.1f}%)"
        else:
            audio_path = None
            audio_status = "⚠️ 音频未生成"

        if actual_duration:
            duration_diff = actual_duration - max_duration
            diff_sign = "+" if duration_diff > 0 else ""
            text_info = f"""
片段 #{idx}
━━━━━━━━━━━━━━━━━━━━
⏱️  时间轴: {text_seg.time_range.start_seconds:.2f}s - {text_seg.time_range.end_seconds:.2f}s

📏 时长信息:
   • 最大允许: {max_duration:.2f}s
   • 实际生成: {actual_duration:.2f}s
   • 差异: {diff_sign}{duration_diff:.2f}s ({diff_sign}{(duration_diff / max_duration * 100):.1f}%)

📊 状态: {'✅ 正常' if abs(duration_diff) < 0.5 else '⚠️ 偏差较大'}
"""
        else:
            text_info = f"""
片段 #{idx}
━━━━━━━━━━━━━━━━━━━━
⏱️  时间轴: {text_seg.time_range.start_seconds:.2f}s - {text_seg.time_range.end_seconds:.2f}s

📏 时长信息:
   • 最大允许: {max_duration:.2f}s
   • 实际生成: 未生成
"""

        subtitle_text = text_seg.text
        return audio_path, audio_status, text_info, subtitle_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 预览失败: {e}", "", ""


# ============== 步骤3: 最终合成 ============== #

def step3_final_synthesis(
    enable_bilingual: bool,  # ✅ 新增参数：是否启用双语字幕
    progress=gr.Progress()
):
    """步骤3: 最终视频合成（支持双语字幕可选）"""
    global current_session
    container.get_tts().unload()

    if not current_session.video:
        return None, None, None, "❌ 错误: 会话状态丢失"

    total_segments = len(current_session.translated_subtitle.segments)
    audio_ready = len(current_session.audio_segments)

    print(f"\n🔍 最终合成前检查:")
    print(f"   总片段数: {total_segments}")
    print(f"   音频已生成: {audio_ready}")
    print(f"   缺失片段: {total_segments - audio_ready}")
    print(f"   双语字幕: {'启用' if enable_bilingual else '禁用'}")

    unreviewed = [
        idx for idx in range(total_segments)
        if idx not in current_session.audio_segments
    ]

    if unreviewed and len(unreviewed) > total_segments * 0.3:
        return None, None, None, f"⚠️ 还有 {len(unreviewed)} 个片段未完成音频生成,请先完成步骤2"

    try:
        progress(0.1, "准备合成...")

        output_dir = current_session.video.path.parent / "output"
        output_dir.mkdir(exist_ok=True)

        # 合并音频片段
        progress(0.2, "合并音频片段...")

        from domain.entities import AudioTrack
        from domain.value_objects import AudioSample

        sample_rate = list(current_session.audio_segments.values())[0].audio.sample_rate
        total_samples = int(current_session.video.duration * sample_rate)
        buffer = [0.0] * total_samples

        for idx, audio_seg in current_session.audio_segments.items():
            text_seg = audio_seg.text_segment
            start_idx = int(text_seg.time_range.start_seconds * sample_rate)

            for i, sample in enumerate(audio_seg.audio.samples):
                target_idx = start_idx + i
                if target_idx < total_samples:
                    buffer[target_idx] = sample

        full_audio = AudioSample(
            samples=tuple(buffer),
            sample_rate=sample_rate
        )

        audio_track = AudioTrack(full_audio, current_session.translated_subtitle.language)

        # 视频合成
        progress(0.5, "合成视频...")

        from application.use_cases.synthesize_video_use_case import synthesize_video_use_case
        from domain.services import merge_bilingual_subtitles

        # ✅ 需求2: 根据用户选择决定字幕方案
        if enable_bilingual and current_session.english_subtitle:
            # 启用双语字幕
            zh_en_subtitle = merge_bilingual_subtitles(
                current_session.translated_subtitle,
                current_session.english_subtitle
            )
            subtitles_tuple = (
                current_session.translated_subtitle,
                current_session.english_subtitle,
                zh_en_subtitle
            )
            subtitle_mode = "双语（中文+英文）"
        else:
            # 只用中文字幕
            subtitles_tuple = (current_session.translated_subtitle,)
            subtitle_mode = "单语（仅中文）"

        synthesis_result = synthesize_video_use_case(
            video=current_session.video,
            subtitles=subtitles_tuple,
            audio_track=audio_track,
            video_processor=container.video_processor,
            subtitle_writer=container.subtitle_writer,
            output_dir=output_dir,
            formats=("srt", "ass"),
            burn_subtitles=True,
            progress=lambda p, d: progress(0.5 + p * 0.5, d)
        )

        # 查找输出文件
        def find_file(patterns: list[str], suffix: str = None) -> Optional[str]:
            for pattern in patterns:
                matches = [
                    p for p in synthesis_result.output_paths
                    if pattern in p.name and (suffix is None or p.suffix == suffix)
                ]
                if matches:
                    return str(matches[0])
            return None

        zh_srt = find_file(['zh.srt'], '.srt')
        zh_en_ass = find_file(['zh_en'], '.ass') if enable_bilingual else None
        voiced_video = find_file(['_voiced_subtitled.mp4'])

        status = f"""
✅ 最终合成完成!

📦 输出文件:
   - 中文字幕: {zh_srt.split('/')[-1] if zh_srt else '❌'}
   - 双语字幕: {zh_en_ass.split('/')[-1] if zh_en_ass else '未启用'}
   - 配音视频: {voiced_video.split('/')[-1] if voiced_video else '❌'}

⚙️  字幕模式: {subtitle_mode}
⏱️  处理时间: {synthesis_result.processing_time:.1f} 秒

📊 统计信息:
   总片段数: {len(current_session.audio_segments)}
   使用缓存: {audio_ready - len(current_session.modified_indices)}
   重新生成: {len(current_session.modified_indices) if hasattr(current_session, 'modified_indices') else 0}
"""

        return zh_srt, zh_en_ass, voiced_video, status

    except Exception as e:
        import traceback
        error_msg = f"❌ 合成失败: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, error_msg


# ============== UI 构建 ============== #

def build_ui_v2():
    """构建增强 UI V2（完整重构版）"""

    with gr.Blocks(
        title="视频翻译工厂 Pro V2",
        css="""
        .gradio-container {max-width: 1800px !important}
        .segment-preview {border: 1px solid #ddd; padding: 10px; border-radius: 5px;}
        """
    ) as demo:
        gr.Markdown("""
        # 🎬 视频翻译工厂 Pro V2 - 完整重构版

        ## ✨ 新增特性
        - 🎵 **可配置参考音频**: 支持自定义起始偏移和时长
        - 📝 **双语字幕可选**: 合成视频前选择是否烧录双语字幕
        - 🏗️ **架构优化**: 参考音频管理下沉到Infrastructure层
        
        ## 📋 工作流程
        1. **生成字幕** → 2A. **增量语音克隆** → 2B. **审核修改** → 2C. **重新生成** → 3. **最终合成**
        """)

        with gr.Tab("🎬 单视频处理 V2"):
            # ========== 步骤1 ========== #
            with gr.Accordion("🔍 步骤1: 生成字幕", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.File(
                            label="📹 上传视频",
                            file_types=[".mp4", ".avi", ".mov", ".mkv"]
                        )

                        whisper_model = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large", "large-v3"],
                            value="medium",
                            label="🎙️ Whisper 模型"
                        )

                        translation_model = gr.Dropdown(
                            choices=["Qwen/Qwen2.5-7B"],
                            value="Qwen/Qwen2.5-7B",
                            label="🌐 翻译模型"
                        )

                        translation_context = gr.Dropdown(
                            choices=container.translator_context_repo.list_contexts(),
                            value="general",
                            label="📚 翻译上下文"
                        )

                        source_lang = gr.Dropdown(
                            choices=["auto", "en", "zh", "pt", "ja"],
                            value="auto",
                            label="🗣️ 源语言"
                        )

                        step1_btn = gr.Button("▶️ 生成字幕", variant="primary")

                    with gr.Column(scale=1):
                        step1_status = gr.Textbox(
                            label="📊 生成状态",
                            lines=12
                        )

            # ========== 步骤2 ========== #
            with gr.Accordion("🎤 步骤2: 增量语音克隆", open=False) as step2_accordion:
                gr.Markdown("""
                ### 工作流程
                1. **2A. 增量语音克隆**: 逐片段生成音频并缓存
                2. **2B. 审核预览**: 试听音频,修改字幕
                3. **2C. 重新生成**: 只重新生成修改过的片段
                """)

                # 2A: 语音克隆
                with gr.Group():
                    gr.Markdown("### 2A. 增量语音克隆")

                    reference_audio = gr.File(
                        label="🎵 参考音频(可选，留空则从视频提取)",
                        file_types=[".wav", ".mp3"]
                    )

                    # ✅ 新增: 参考音频配置
                    with gr.Row():
                        ref_duration_slider = gr.Slider(
                            minimum=5,
                            maximum=60,
                            value=10,
                            step=5,
                            label="⏱️ 参考音频时长（秒）",
                            info="提取或使用的参考音频长度"
                        )

                        ref_offset_slider = gr.Slider(
                            minimum=0,
                            maximum=120,
                            value=0,
                            step=5,
                            label="📍 起始偏移（秒）",
                            info="从视频的第几秒开始提取，0表示从头开始（或使用VAD检测）"
                        )

                    clone_btn = gr.Button("🎤 开始增量语音克隆", variant="primary")
                    clone_status = gr.Textbox(label="克隆状态", lines=10)

                # 2B: 审核表格
                with gr.Group():
                    gr.Markdown("### 2B. 审核和预览")

                    review_dataframe = gr.Dataframe(
                        headers=[
                            "索引", "时间", "原文", "翻译",
                            "最大长度", "已生成长度", "音频", "问题", "状态"
                        ],
                        datatype=[
                            "number", "str", "str", "str",
                            "str", "str", "str", "str", "str"
                        ],
                        col_count=(9, "fixed"),
                        row_count=(10, "dynamic"),
                        interactive=True,
                        wrap=True,
                        label="字幕审核表格 (点击行预览音频)"
                    )

                    with gr.Row():
                        save_edits_btn = gr.Button("💾 保存修改", variant="secondary")
                        regenerate_btn = gr.Button("🔄 重新生成修改的片段", variant="primary")

                    edit_status = gr.Textbox(label="编辑状态", lines=3)

                # 片段预览区
                with gr.Group():
                    gr.Markdown("### 👂 片段预览 (点击表格行预览)")

                    with gr.Row():
                        with gr.Column(scale=1):
                            preview_audio = gr.Audio(
                                label="🔊 音频播放",
                                type="filepath"
                            )
                            preview_status = gr.Textbox(label="状态", lines=1)

                        with gr.Column(scale=1):
                            preview_info = gr.Textbox(label="片段信息", lines=3)
                            preview_text = gr.Textbox(label="字幕文本", lines=4)

            # ========== 步骤3 ========== #
            with gr.Accordion("🎬 步骤3: 最终合成", open=False):
                gr.Markdown("""
                ### 提示
                - 确保所有关键片段都已审核通过
                - 选择是否烧录双语字幕
                """)

                # ✅ 新增: 双语字幕选项
                enable_bilingual_checkbox = gr.Checkbox(
                    label="📝 烧录双语字幕（中文+英文）",
                    value=True,
                    info="取消勾选则只烧录中文字幕"
                )

                final_btn = gr.Button("▶️ 生成最终视频", variant="primary", size="lg")
                final_status = gr.Textbox(label="合成状态", lines=12)

                with gr.Row():
                    zh_srt_output = gr.File(label="中文字幕")
                    zh_en_ass_output = gr.File(label="双语字幕")
                    final_video_output = gr.File(label="最终视频")

            # ========== 事件绑定 ========== #

            # 步骤1
            step1_btn.click(
                step1_generate_and_check_v2,
                inputs=[
                    video_input, whisper_model, translation_model,
                    translation_context, source_lang
                ],
                outputs=[review_dataframe, step1_status, step2_accordion]
            ).then(
                lambda: gr.update(open=True),
                outputs=[step2_accordion]
            )

            # 步骤2A: 语音克隆
            clone_btn.click(
                step2_incremental_voice_cloning,
                inputs=[
                    reference_audio,
                    ref_duration_slider,  # ✅ 新增
                    ref_offset_slider  # ✅ 新增
                ],
                outputs=[clone_status, review_dataframe]
            )

            # 步骤2B: 编辑保存
            save_edits_btn.click(
                step2_save_edits_and_regenerate,
                inputs=[review_dataframe],
                outputs=[edit_status, review_dataframe]
            )

            # 步骤2C: 重新生成
            regenerate_btn.click(
                step2_regenerate_modified,
                outputs=[edit_status, review_dataframe]
            )

            # 表格选择事件
            review_dataframe.select(
                preview_segment,
                outputs=[preview_audio, preview_status, preview_info, preview_text]
            )

            # 步骤3: 最终合成
            final_btn.click(
                step3_final_synthesis,
                inputs=[enable_bilingual_checkbox],  # ✅ 新增
                outputs=[zh_srt_output, zh_en_ass_output, final_video_output, final_status]
            )

    return demo


def main():
    """启动 WebUI V2"""
    demo = build_ui_v2()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False
    )


if __name__ == "__main__":
    main()