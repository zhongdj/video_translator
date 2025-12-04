"""
Infrastructure Layer - 增强 WebUI V3

新增功能:
1. ✅ 多说话人语音合成支持
2. ✅ 仅字幕模式（不生成语音）
"""

from pathlib import Path
from typing import Optional, Dict, List

import gradio as gr

from application.use_cases.incremental_voice_cloning import (
    incremental_voice_cloning_use_case,
    regenerate_modified_segments_use_case
)
from application.use_cases.multi_speaker_voice_cloning import (
    multi_speaker_voice_cloning_use_case
)
from application.use_cases.subtitle_only_synthesis import (
    subtitle_only_synthesis_use_case
)
from domain.entities import (
    Video, Subtitle, LanguageCode,
    AudioSegment, SegmentReviewStatus
)
from domain.multi_speaker import (
    SpeakerId, MultiSpeakerVoiceProfile,
    SegmentSpeakerAssignment, MultiSpeakerConfig
)
from infrastructure.config.dependency_injection import container

# 初始化仓储
audio_segment_repo = container.audio_segment_repo
audio_file_repo = container.audio_file_repo
cache_service = container.cache_service


# ============== 会话状态 ============== #

class TranslationSessionV3:
    """增强翻译会话状态"""

    def __init__(self):
        # 原有字段
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
        self.length_penalty: float = 0.0
        self.duration_stats: Dict[int, dict] = {}

        # ✅ 新增：多说话人支持
        self.synthesis_mode: str = "single_speaker"  # "single_speaker" | "multi_speaker" | "subtitle_only"
        self.speaker_profiles: Dict[str, MultiSpeakerVoiceProfile] = {}  # speaker_id -> profile
        self.segment_speaker_map: Dict[int, str] = {}  # segment_index -> speaker_id
        self.default_speaker_id: Optional[str] = None


current_session = TranslationSessionV3()


# ============== 导入原有辅助函数 ============== #

def _source_language_cache_format(source_language: str) -> Optional[LanguageCode]:
    """转换源语言格式"""
    return LanguageCode(source_language) if source_language != "auto" else None


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


def _prepare_review_data_v3(filter_over_limit: bool = False):
    """
    准备审核数据（V3增强版，支持多说话人）
    """
    if not current_session.translated_subtitle:
        return None

    data = []
    for idx, (orig_seg, trans_seg) in enumerate(
            zip(current_session.original_subtitle.segments,
                current_session.translated_subtitle.segments)
    ):
        # 获取英文字幕
        en_text = (
            current_session.english_subtitle.segments[idx].text
            if current_session.english_subtitle
               and idx < len(current_session.english_subtitle.segments)
            else orig_seg.text
        )

        # 获取说话人信息
        speaker_info = "默认"
        if current_session.synthesis_mode == "multi_speaker":
            speaker_id = current_session.segment_speaker_map.get(
                idx,
                current_session.default_speaker_id
            )
            if speaker_id and speaker_id in current_session.speaker_profiles:
                speaker_info = current_session.speaker_profiles[speaker_id].speaker_id.name

        # 获取音频信息
        audio_seg = current_session.audio_segments.get(idx)
        target_duration = trans_seg.time_range.duration

        if audio_seg:
            actual_duration = len(audio_seg.audio.samples) / audio_seg.audio.sample_rate
            duration_error = actual_duration - target_duration
            duration_ratio = (actual_duration / target_duration * 100) if target_duration > 0 else 0

            audio_status = "✅ 已生成"
            duration_str = f"{actual_duration:.2f}s"

            if duration_error > 0.5:
                duration_status = f"⚠️ 超时 {duration_error:.2f}s ({duration_ratio:.0f}%)"
            elif duration_error > 0.1:
                duration_status = f"⚡ 略超 {duration_error:.2f}s ({duration_ratio:.0f}%)"
            elif duration_error < -0.5:
                duration_status = f"📉 过短 {duration_error:.2f}s ({duration_ratio:.0f}%)"
            else:
                duration_status = f"✅ 正常 ({duration_ratio:.0f}%)"

            if filter_over_limit and duration_error <= 0.1:
                continue
        else:
            audio_status = "未生成" if current_session.synthesis_mode != "subtitle_only" else "N/A"
            duration_str = "-"
            duration_status = "⏳ 待生成" if current_session.synthesis_mode != "subtitle_only" else "N/A"

            if filter_over_limit:
                continue

        data.append([
            idx,
            f"{trans_seg.time_range.start_seconds:.2f}s",
            speaker_info,  # ✅ 新增列：说话人
            en_text,
            trans_seg.text,
            f"{target_duration:.2f}s",
            duration_str,
            duration_status,
            audio_status,
            "⏳ 待审核"
        ])

    return data


# ============== 步骤1: 生成字幕（复用原有逻辑）============== #

def step1_generate_and_check_v3(
        video_file,
        whisper_model: str,
        translation_model: str,
        translation_context_name: str,
        source_language: str,
        progress=gr.Progress()
):
    """步骤1: 生成字幕（复用V2逻辑）"""
    if not video_file:
        return None, "❌ 请上传视频", gr.update(visible=False)

    try:
        global current_session
        current_session = TranslationSessionV3()

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
            review_data = _prepare_review_data_v3()
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

        review_data = _prepare_review_data_v3()
        return review_data, status_report, gr.update(visible=True)

    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, gr.update(visible=False)


# ============== ✅ 新增：多说话人管理 ============== #

def add_speaker_profile(
        speaker_name: str,
        reference_audio_file,
        ref_duration: float
):
    """添加说话人配置"""
    global current_session

    if not speaker_name or not speaker_name.strip():
        return "❌ 请输入说话人名称", gr.update()

    if not reference_audio_file:
        return "❌ 请上传参考音频", gr.update()

    try:
        # 生成唯一ID
        speaker_id_str = f"speaker_{len(current_session.speaker_profiles) + 1}"
        speaker_id = SpeakerId(id=speaker_id_str, name=speaker_name)

        # 持久化参考音频
        ref_audio_path = audio_file_repo.save_reference_audio(
            video_path=current_session.video.path,
            source_audio_path=Path(reference_audio_file.name)
        )

        # 创建配置
        profile = MultiSpeakerVoiceProfile(
            speaker_id=speaker_id,
            reference_audio_path=ref_audio_path,
            language=LanguageCode.CHINESE,
            duration=ref_duration
        )

        current_session.speaker_profiles[speaker_id_str] = profile

        # 设置第一个为默认说话人
        if not current_session.default_speaker_id:
            current_session.default_speaker_id = speaker_id_str

        # 更新说话人列表显示
        speaker_list = "\n".join([
            f"- {p.speaker_id.name} ({p.speaker_id.id})" +
            (" [默认]" if sid == current_session.default_speaker_id else "")
            for sid, p in current_session.speaker_profiles.items()
        ])

        return (
            f"✅ 已添加说话人: {speaker_name}\n\n当前说话人列表:\n{speaker_list}",
            gr.update(
                choices=list(current_session.speaker_profiles.keys()),
                value=speaker_id_str
            )
        )

    except Exception as e:
        import traceback
        return f"❌ 添加失败: {str(e)}\n{traceback.format_exc()}", gr.update()


def assign_speaker_to_segments(
        segment_indices_str: str,
        speaker_id: str
):
    """为片段分配说话人"""
    global current_session

    if not segment_indices_str or not speaker_id:
        return "❌ 请输入片段索引和选择说话人"

    try:
        # 解析片段索引（支持 "1,2,3" 或 "1-5"）
        indices = []
        for part in segment_indices_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))

        # 分配说话人
        for idx in indices:
            if 0 <= idx < len(current_session.translated_subtitle.segments):
                current_session.segment_speaker_map[idx] = speaker_id

        # 更新审核表格
        updated_data = _prepare_review_data_v3()

        speaker_name = current_session.speaker_profiles[speaker_id].speaker_id.name

        return (
            f"✅ 已为 {len(indices)} 个片段分配说话人: {speaker_name}",
            gr.update(value=updated_data)
        )

    except Exception as e:
        return f"❌ 分配失败: {str(e)}", gr.update()


# ============== 步骤2: 语音合成（支持多模式）============== #

def step2_voice_synthesis_multi_mode(
        synthesis_mode: str,
        reference_audio_file,
        ref_audio_duration: float,
        ref_audio_start_offset: float,
        length_penalty: float,
        progress=gr.Progress()
):
    """
    步骤2: 语音合成（多模式支持）

    模式:
    - single_speaker: 单说话人
    - multi_speaker: 多说话人
    - subtitle_only: 仅字幕（跳过语音合成）
    """
    global current_session

    if not current_session.video or not current_session.translated_subtitle:
        return "❌ 错误: 会话状态丢失", gr.update(), ""

    current_session.synthesis_mode = synthesis_mode

    # ✅ 模式1: 仅字幕模式（跳过语音合成）
    if synthesis_mode == "subtitle_only":
        progress(1.0, "仅字幕模式：跳过语音合成")

        status = """
✅ 仅字幕模式

📋 提示:
   - 已跳过语音合成步骤
   - 可直接进入步骤3生成字幕文件
   - 不会生成配音视频
"""

        updated_data = _prepare_review_data_v3()
        return status, gr.update(value=updated_data), ""

    # ✅ 模式2: 单说话人模式
    if synthesis_mode == "single_speaker":
        try:
            # 准备参考音频
            if reference_audio_file:
                ref_audio_path = audio_file_repo.save_reference_audio(
                    video_path=current_session.video.path,
                    source_audio_path=Path(reference_audio_file.name)
                )
                current_session.reference_audio_path = ref_audio_path
            else:
                existing_ref_audio = audio_file_repo.load_reference_audio(
                    current_session.video.path
                )
                if existing_ref_audio and existing_ref_audio.exists():
                    ref_audio_path = existing_ref_audio
                else:
                    temp_ref_audio = container.video_processor.extract_reference_audio(
                        video=current_session.video,
                        duration=ref_audio_duration,
                        start_offset=ref_audio_start_offset
                    )
                    ref_audio_path = audio_file_repo.save_reference_audio(
                        video_path=current_session.video.path,
                        source_audio_path=temp_ref_audio
                    )
                    if temp_ref_audio.exists():
                        temp_ref_audio.unlink()

                current_session.reference_audio_path = ref_audio_path

            # 更新TTS配置
            tts = container.get_tts()
            if hasattr(tts, 'update_config'):
                tts.update_config(length_penalty=length_penalty)

            # 执行单说话人合成
            def segment_progress(ratio, msg, idx, audio_seg):
                progress(ratio, msg)
                if audio_seg:
                    current_session.audio_segments[idx] = audio_seg

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

            for audio_seg in result.audio_segments:
                current_session.audio_segments[audio_seg.segment_index] = audio_seg

            status = f"""
✅ 单说话人语音克隆完成!

📊 统计:
   总片段: {result.total_segments}
   缓存命中: {result.cached_segments}
   新生成: {result.regenerated_segments}
   耗时: {result.synthesis_time:.1f}秒

⚙️ 配置:
   length_penalty: {length_penalty}
   参考音频: {ref_audio_path.name}
"""

            updated_data = _prepare_review_data_v3()
            return status, gr.update(value=updated_data), ""

        except Exception as e:
            import traceback
            return f"❌ 合成失败: {str(e)}\n{traceback.format_exc()}", gr.update(), ""

    # ✅ 模式3: 多说话人模式
    if synthesis_mode == "multi_speaker":
        if not current_session.speaker_profiles:
            return "❌ 请先添加至少一个说话人配置", gr.update(), ""

        if not current_session.default_speaker_id:
            return "❌ 请设置默认说话人", gr.update(), ""

        try:
            # 构建多说话人配置
            voice_profiles = tuple(current_session.speaker_profiles.values())

            assignments = tuple(
                SegmentSpeakerAssignment(
                    segment_index=idx,
                    speaker_id=profile.speaker_id
                )
                for idx, speaker_id_str in current_session.segment_speaker_map.items()
                for profile in current_session.speaker_profiles.values()
                if profile.speaker_id.id == speaker_id_str
            )

            default_speaker = current_session.speaker_profiles[
                current_session.default_speaker_id
            ].speaker_id

            multi_speaker_config = MultiSpeakerConfig(
                voice_profiles=voice_profiles,
                segment_assignments=assignments,
                default_speaker_id=default_speaker
            )

            # 更新TTS配置
            tts = container.get_tts()
            if hasattr(tts, 'update_config'):
                tts.update_config(length_penalty=length_penalty)

            # 执行多说话人合成
            def segment_progress(ratio, msg, idx, audio_seg):
                progress(ratio, msg)
                if audio_seg:
                    current_session.audio_segments[idx] = audio_seg

            result = multi_speaker_voice_cloning_use_case(
                video=current_session.video,
                subtitle=current_session.translated_subtitle,
                multi_speaker_config=multi_speaker_config,
                tts_provider=container.get_tts(),
                video_processor=container.video_processor,
                audio_repo=audio_segment_repo,
                cache_repo=container.cache_repo,
                progress=segment_progress
            )

            for audio_seg in result.audio_segments:
                current_session.audio_segments[audio_seg.segment_index] = audio_seg

            status = f"""
✅ 多说话人语音克隆完成!

📊 统计:
   总片段: {result.total_segments}
   说话人数: {len(current_session.speaker_profiles)}
   缓存命中: {result.cached_segments}
   新生成: {result.regenerated_segments}
   耗时: {result.synthesis_time:.1f}秒

⚙️ 配置:
   length_penalty: {length_penalty}
"""

            updated_data = _prepare_review_data_v3()
            return status, gr.update(value=updated_data), ""

        except Exception as e:
            import traceback
            return f"❌ 合成失败: {str(e)}\n{traceback.format_exc()}", gr.update(), ""


# ============== 步骤3: 最终合成（支持仅字幕模式）============== #

def step3_final_synthesis_v3(
        enable_bilingual: bool,
        burn_subtitles: bool,
        progress=gr.Progress()
):
    """步骤3: 最终合成（支持仅字幕模式）"""
    global current_session

    if not current_session.video:
        return None, None, None, "❌ 错误: 会话状态丢失"

    try:
        output_dir = current_session.video.path.parent / "output"
        output_dir.mkdir(exist_ok=True)

        # ✅ 仅字幕模式
        if current_session.synthesis_mode == "subtitle_only":
            progress(0.5, "仅字幕模式：生成字幕文件...")

            output_paths, status = subtitle_only_synthesis_use_case(
                video=current_session.video,
                target_subtitle=current_session.translated_subtitle,
                secondary_subtitle=current_session.english_subtitle,
                video_processor=container.video_processor,
                subtitle_writer=container.subtitle_writer,
                output_dir=output_dir,
                enable_bilingual=enable_bilingual,
                burn_subtitles=burn_subtitles,
                formats=("srt", "ass"),
                progress=lambda p, d: progress(p, d)
            )

            # 查找输出文件
            zh_srt = next((str(p) for p in output_paths if 'zh.srt' in p.name), None)
            zh_en_ass = next((str(p) for p in output_paths if 'zh_en' in p.name), None) if enable_bilingual else None
            subtitled_video = next((str(p) for p in output_paths if p.suffix == '.mp4'),
                                   None) if burn_subtitles else None

            return zh_srt, zh_en_ass, subtitled_video, status

        # ✅ 语音合成模式（单说话人/多说话人）
        total_segments = len(current_session.translated_subtitle.segments)
        audio_ready = len(current_session.audio_segments)

        if audio_ready < total_segments * 0.7:
            return None, None, None, f"⚠️ 音频片段不足（{audio_ready}/{total_segments}），请先完成步骤2"

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

        progress(0.5, "合成视频...")

        from application.use_cases.synthesize_video_use_case import synthesize_video_use_case
        from domain.services import merge_bilingual_subtitles

        if enable_bilingual and current_session.english_subtitle:
            zh_en_subtitle = merge_bilingual_subtitles(
                current_session.translated_subtitle,
                current_session.english_subtitle
            )
            subtitles_tuple = (
                current_session.translated_subtitle,
                current_session.english_subtitle,
                zh_en_subtitle
            )
            subtitle_mode = "双语"
        else:
            subtitles_tuple = (current_session.translated_subtitle,)
            subtitle_mode = "单语"

        synthesis_result = synthesize_video_use_case(
            video=current_session.video,
            subtitles=subtitles_tuple,
            audio_track=audio_track,
            video_processor=container.video_processor,
            subtitle_writer=container.subtitle_writer,
            output_dir=output_dir,
            formats=("srt", "ass"),
            burn_subtitles=burn_subtitles,
            progress=lambda p, d: progress(0.5 + p * 0.5, d)
        )

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
        voiced_video = find_file(['_voiced_subtitled.mp4']) if burn_subtitles else find_file(['_voiced.mp4'])

        synthesis_mode_name = "多说话人" if current_session.synthesis_mode == "multi_speaker" else "单说话人"

        status = f"""
✅ 最终合成完成!

📦 输出文件:
   - 中文字幕: {zh_srt.split('/')[-1] if zh_srt else '❌'}
   - 双语字幕: {zh_en_ass.split('/')[-1] if zh_en_ass else '未启用'}
   - 配音视频: {voiced_video.split('/')[-1] if voiced_video else '❌'}

⚙️ 配置:
   合成模式: {synthesis_mode_name}
   字幕模式: {subtitle_mode}
   烧录字幕: {'是' if burn_subtitles else '否'}
   处理时间: {synthesis_result.processing_time:.1f}秒

📊 统计:
   总片段: {len(current_session.audio_segments)}
   使用缓存: {audio_ready - len(current_session.modified_indices)}
   重新生成: {len(current_session.modified_indices) if hasattr(current_session, 'modified_indices') else 0}
"""

        return zh_srt, zh_en_ass, voiced_video, status

    except Exception as e:
        import traceback
        error_msg = f"❌ 合成失败: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, error_msg


# ============== 片段预览（复用V2逻辑）============== #

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
        if not audio_seg and current_session.synthesis_mode != "subtitle_only":
            audio_seg = audio_segment_repo.load_segment(
                segment_index=idx,
                video_path=current_session.video.path,
                text_segment=text_seg
            )
            if audio_seg:
                current_session.audio_segments[idx] = audio_seg

        # 获取说话人信息
        speaker_info = ""
        if current_session.synthesis_mode == "multi_speaker":
            speaker_id = current_session.segment_speaker_map.get(
                idx,
                current_session.default_speaker_id
            )
            if speaker_id and speaker_id in current_session.speaker_profiles:
                speaker_name = current_session.speaker_profiles[speaker_id].speaker_id.name
                speaker_info = f"\n🎤 说话人: {speaker_name}"

        actual_duration = None
        if audio_seg and audio_seg.file_path and audio_seg.file_path.exists():
            audio_path = str(audio_seg.file_path)
            actual_duration = len(audio_seg.audio.samples) / audio_seg.audio.sample_rate
            duration_diff = actual_duration - max_duration
            diff_sign = "+" if duration_diff > 0 else ""
            audio_status = f"✅ 音频已生成 ({(actual_duration / max_duration * 100):.1f}%)"
        else:
            audio_path = None
            if current_session.synthesis_mode == "subtitle_only":
                audio_status = "N/A (仅字幕模式)"
            else:
                audio_status = "⚠️ 音频未生成"

        if actual_duration:
            duration_diff = actual_duration - max_duration
            diff_sign = "+" if duration_diff > 0 else ""
            text_info = f"""
片段 #{idx}
━━━━━━━━━━━━━━━━━━━━
⏱️  时间轴: {text_seg.time_range.start_seconds:.2f}s - {text_seg.time_range.end_seconds:.2f}s{speaker_info}

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
⏱️  时间轴: {text_seg.time_range.start_seconds:.2f}s - {text_seg.time_range.end_seconds:.2f}s{speaker_info}

📏 时长信息:
   • 最大允许: {max_duration:.2f}s
   • 实际生成: {'未生成' if current_session.synthesis_mode != 'subtitle_only' else 'N/A'}
"""

        subtitle_text = text_seg.text
        return audio_path, audio_status, text_info, subtitle_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 预览失败: {e}", "", ""


# ============== UI 构建 ============== #

def build_ui_v3():
    """构建增强 UI V3"""

    with gr.Blocks(
            title="视频翻译工厂 Pro V3",
            css="""
        .gradio-container {max-width: 1900px !important}
        .segment-preview {border: 1px solid #ddd; padding: 10px; border-radius: 5px;}
        .speaker-config {background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0;}
        """
    ) as demo:
        gr.Markdown("""
        # 🎬 视频翻译工厂 Pro V3 - 多说话人 & 仅字幕支持

        ## ✨ V3 新增功能
        - 🎭 **多说话人语音合成**: 支持为不同片段指定不同说话人
        - 📝 **仅字幕模式**: 只生成字幕文件，跳过语音合成
        - 🎛️ **灵活的合成模式**: 单说话人 / 多说话人 / 仅字幕 三种模式可选

        ## 📋 工作流程
        1. **生成字幕** → 2. **选择合成模式** → 3. **配置参数** → 4. **最终合成**
        """)

        with gr.Tab("🎬 单视频处理 V3"):
            # ========== 步骤1: 生成字幕 ========== #
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

            # ========== 步骤2: 合成模式选择 ========== #
            with gr.Accordion("🎛️ 步骤2: 选择合成模式", open=False) as step2_accordion:
                gr.Markdown("""
                ### 选择合成模式

                - **单说话人**: 所有片段使用同一个参考音频
                - **多说话人**: 为不同片段分配不同说话人（如对话场景）
                - **仅字幕**: 只生成字幕文件，不合成语音
                """)

                synthesis_mode = gr.Radio(
                    choices=[
                        ("单说话人", "single_speaker"),
                        ("多说话人", "multi_speaker"),
                        ("仅字幕（不生成语音）", "subtitle_only")
                    ],
                    value="single_speaker",
                    label="合成模式"
                )

                # ✅ 单说话人配置
                with gr.Group(visible=True) as single_speaker_config:
                    gr.Markdown("### 单说话人配置")

                    reference_audio = gr.File(
                        label="🎵 参考音频(可选)",
                        file_types=[".wav", ".mp3"]
                    )

                    with gr.Row():
                        ref_duration_slider = gr.Slider(
                            minimum=5, maximum=60, value=10, step=5,
                            label="⏱️ 参考音频时长（秒）"
                        )
                        ref_offset_slider = gr.Slider(
                            minimum=0, maximum=120, value=0, step=5,
                            label="📍 起始偏移（秒）"
                        )

                    length_penalty_slider = gr.Slider(
                        minimum=-2.0, maximum=2.0, value=0.0, step=0.1,
                        label="⚙️ length_penalty"
                    )

                # ✅ 多说话人配置
                with gr.Group(visible=False) as multi_speaker_config:
                    gr.Markdown("### 多说话人配置")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 1. 添加说话人")

                            speaker_name_input = gr.Textbox(
                                label="说话人名称",
                                placeholder="例如: 主讲人, 旁白, 角色A"
                            )

                            speaker_audio_input = gr.File(
                                label="参考音频",
                                file_types=[".wav", ".mp3"]
                            )

                            speaker_duration_input = gr.Slider(
                                minimum=5, maximum=60, value=10, step=5,
                                label="参考音频时长"
                            )

                            add_speaker_btn = gr.Button("➕ 添加说话人", variant="secondary")

                            speaker_status = gr.Textbox(
                                label="说话人列表",
                                lines=8,
                                placeholder="尚未添加说话人"
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("#### 2. 分配片段")

                            segment_indices_input = gr.Textbox(
                                label="片段索引",
                                placeholder="例如: 0,1,2 或 0-5",
                                info="逗号分隔或使用连字符表示范围"
                            )

                            speaker_selector = gr.Dropdown(
                                label="选择说话人",
                                choices=[],
                                interactive=True
                            )

                            assign_speaker_btn = gr.Button("✅ 分配说话人", variant="secondary")

                            assign_status = gr.Textbox(
                                label="分配状态",
                                lines=3
                            )

                            gr.Markdown("""
                            **提示**:
                            - 未分配的片段将使用第一个添加的说话人
                            - 可以多次分配，覆盖之前的设置
                            """)

                            multi_length_penalty = gr.Slider(
                                minimum=-2.0, maximum=2.0, value=0.0, step=0.1,
                                label="⚙️ length_penalty"
                            )

                # ✅ 仅字幕提示
                with gr.Group(visible=False) as subtitle_only_info:
                    gr.Markdown("""
                    ### 仅字幕模式

                    ✅ 将跳过语音合成步骤

                    **输出内容**:
                    - 中文字幕文件 (.srt, .ass)
                    - 英文字幕文件 (.srt, .ass)
                    - 双语字幕文件 (.srt, .ass)
                    - 烧录字幕的视频（可选）

                    **优势**:
                    - 处理速度快
                    - 不需要GPU
                    - 保留原始音频
                    """)

                # 开始合成按钮
                start_synthesis_btn = gr.Button(
                    "▶️ 开始合成",
                    variant="primary",
                    size="lg"
                )

                synthesis_status = gr.Textbox(
                    label="合成状态",
                    lines=12
                )

            # ========== 审核表格 ========== #
            with gr.Accordion("📋 步骤3: 审核和预览", open=False):
                review_dataframe = gr.Dataframe(
                    headers=[
                        "索引", "时间", "说话人", "原文", "翻译",
                        "目标长度", "实际长度", "时长状态", "音频", "审核"
                    ],
                    datatype=[
                        "number", "str", "str", "str", "str",
                        "str", "str", "str", "str", "str"
                    ],
                    col_count=(10, "fixed"),
                    row_count=(10, "dynamic"),
                    interactive=True,
                    wrap=True,
                    label="字幕审核表格 (点击行预览)"
                )

                # 片段预览区
                with gr.Group():
                    gr.Markdown("### 👂 片段预览")

                    with gr.Row():
                        with gr.Column(scale=1):
                            preview_audio = gr.Audio(
                                label="🔊 音频播放",
                                type="filepath"
                            )
                            preview_status = gr.Textbox(label="状态", lines=1)

                        with gr.Column(scale=1):
                            preview_info = gr.Textbox(label="片段信息", lines=5)
                            preview_text = gr.Textbox(label="字幕文本", lines=4)

            # ========== 步骤4: 最终合成 ========== #
            with gr.Accordion("🎬 步骤4: 最终合成", open=False):
                gr.Markdown("""
                ### 输出配置
                """)

                with gr.Row():
                    enable_bilingual_checkbox = gr.Checkbox(
                        label="📝 生成双语字幕",
                        value=True,
                        info="中文+英文双语字幕"
                    )

                    burn_subtitles_checkbox = gr.Checkbox(
                        label="🔥 烧录字幕到视频",
                        value=True,
                        info="将字幕硬编码到视频中"
                    )

                final_btn = gr.Button("▶️ 生成最终输出", variant="primary", size="lg")
                final_status = gr.Textbox(label="合成状态", lines=12)

                with gr.Row():
                    zh_srt_output = gr.File(label="中文字幕")
                    zh_en_ass_output = gr.File(label="双语字幕")
                    final_video_output = gr.File(label="最终视频")

            # ========== 事件绑定 ========== #

            # 步骤1: 生成字幕
            step1_btn.click(
                step1_generate_and_check_v3,
                inputs=[
                    video_input, whisper_model, translation_model,
                    translation_context, source_lang
                ],
                outputs=[review_dataframe, step1_status, step2_accordion]
            ).then(
                lambda: gr.update(open=True),
                outputs=[step2_accordion]
            )

            # 模式切换
            def toggle_synthesis_mode(mode):
                return (
                    gr.update(visible=(mode == "single_speaker")),
                    gr.update(visible=(mode == "multi_speaker")),
                    gr.update(visible=(mode == "subtitle_only"))
                )

            synthesis_mode.change(
                toggle_synthesis_mode,
                inputs=[synthesis_mode],
                outputs=[single_speaker_config, multi_speaker_config, subtitle_only_info]
            )

            # 多说话人：添加说话人
            add_speaker_btn.click(
                add_speaker_profile,
                inputs=[speaker_name_input, speaker_audio_input, speaker_duration_input],
                outputs=[speaker_status, speaker_selector]
            )

            # 多说话人：分配片段
            assign_speaker_btn.click(
                assign_speaker_to_segments,
                inputs=[segment_indices_input, speaker_selector],
                outputs=[assign_status, review_dataframe]
            )

            # 开始合成（根据模式选择参数）
            def dispatch_synthesis(mode, ref_audio, ref_dur, ref_offset, lp, mlp, progress=gr.Progress()):
                # 根据模式选择正确的 length_penalty
                final_lp = lp if mode != "multi_speaker" else mlp

                return step2_voice_synthesis_multi_mode(
                    synthesis_mode=mode,
                    reference_audio_file=ref_audio,
                    ref_audio_duration=ref_dur,
                    ref_audio_start_offset=ref_offset,
                    length_penalty=final_lp,
                    progress=progress
                )

            start_synthesis_btn.click(
                dispatch_synthesis,
                inputs=[
                    synthesis_mode,
                    reference_audio,
                    ref_duration_slider,
                    ref_offset_slider,
                    length_penalty_slider,
                    multi_length_penalty
                ],
                outputs=[synthesis_status, review_dataframe, synthesis_status]
            )

            # 表格选择事件
            review_dataframe.select(
                preview_segment,
                outputs=[preview_audio, preview_status, preview_info, preview_text]
            )

            # 步骤4: 最终合成
            final_btn.click(
                step3_final_synthesis_v3,
                inputs=[enable_bilingual_checkbox, burn_subtitles_checkbox],
                outputs=[zh_srt_output, zh_en_ass_output, final_video_output, final_status]
            )

    return demo


def main():
    """启动 WebUI V3"""
    demo = build_ui_v3()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False
    )


if __name__ == "__main__":
    main()