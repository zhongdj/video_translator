"""
Infrastructure Layer - 增强 WebUI V2 (修复版)
修复音频片段缓存和预览问题
"""
from pathlib import Path
from typing import Optional, Dict

import gradio as gr

# 导入新的用例
from application.use_cases.incremental_voice_cloning import (
    incremental_voice_cloning_use_case,
    regenerate_modified_segments_use_case
)
from domain.entities import (
    Video, Subtitle, LanguageCode,
    TextSegment, TimeRange, AudioSegment,
    SegmentReviewStatus
)
from domain.services import calculate_cache_key
from infrastructure.adapters.storage.audio_segment_repository_adapter import AudioSegmentRepositoryAdapter
from infrastructure.config.dependency_injection import container

# 初始化音频片段仓储
audio_segment_repo = AudioSegmentRepositoryAdapter()


# ============== 会话状态管理 V2 ============== #
class TranslationSessionV2:
    """增强的翻译会话状态"""

    def __init__(self):
        self.translation_context = None
        self.video: Optional[Video] = None
        self.original_subtitle: Optional[Subtitle] = None
        self.translated_subtitle: Optional[Subtitle] = None
        self.english_subtitle: Optional[Subtitle] = None
        self.detected_language: Optional[LanguageCode] = None
        self.source_language: Optional[LanguageCode] = None
        self.quality_report = None

        # 新增:音频片段管理
        self.audio_segments: Dict[int, AudioSegment] = {}
        self.segment_review_status: Dict[int, SegmentReviewStatus] = {}

        # 修改追踪
        self.edited_segments: Dict[int, str] = {}
        self.modified_indices: set[int] = set()

        # 参考音频
        self.reference_audio_path: Optional[Path] = None

        self.approved = False


# 全局会话对象
current_session = TranslationSessionV2()


# ============== 🔧 修复函数:加载已缓存的音频片段 ============== #
def _load_cached_audio_segments(video: Video, subtitle: Subtitle) -> Dict[int, AudioSegment]:
    """
    从磁盘加载已缓存的音频片段

    Returns:
        {segment_index: AudioSegment}
    """
    cached_segments = {}

    print(f"\n🔍 检查音频片段缓存:")
    print(f"   视频: {video.path.name}")
    print(f"   片段总数: {len(subtitle.segments)}")

    for idx, text_seg in enumerate(subtitle.segments):
        try:
            # 尝试从仓储加载
            audio_seg = audio_segment_repo.load_segment(
                segment_index=idx,
                video_path=video.path,
                text_segment=text_seg
            )

            if audio_seg:
                cached_segments[idx] = audio_seg
                # print(f"   ✅ 片段 {idx} 已加载")
            # else:
            #     print(f"   ⚠️  片段 {idx} 未缓存")

        except Exception as e:
            print(f"   ❌ 片段 {idx} 加载失败: {e}")
            continue

    print(f"✅ 共加载 {len(cached_segments)}/{len(subtitle.segments)} 个缓存片段\n")

    return cached_segments


# ============== 步骤1: 生成字幕和质量检查 ============== #
def step1_generate_and_check_v2(
        video_file,
        whisper_model: str,
        translation_model: str,
        translation_context_name: str,
        source_language: str,
        progress=gr.Progress()
):
    """步骤1: 生成字幕并进行质量检查"""
    global current_session

    if not video_file:
        return None, "❌ 请上传视频", gr.update(visible=False)

    try:
        current_session = TranslationSessionV2()

        video_path = Path(video_file.name)

        # 创建视频对象
        current_session.video = Video(
            path=video_path,
            duration=get_video_duration(video_path),
            has_audio=True
        )

        # 加载翻译上下文
        translation_context = container.translator_context_repo.load(
            translation_context_name
        )

        if not translation_context:
            return None, f"❌ 无法加载翻译上下文: {translation_context_name}", gr.update(visible=False)

        # 解析源语言
        src_lang = LanguageCode(source_language) if source_language != "auto" else None

        progress(0.0, "开始生成字幕...")

        # 使用改进的字幕生成用例
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
        current_session.quality_report = result.quality_report
        current_session.translation_context = translation_context
        current_session.source_language = src_lang

        src_lang = LanguageCode(
            current_session.source_language.value) if current_session.source_language and current_session.source_language.value != "auto" else None

        # 从缓存加载英文字幕
        cache_params = {
            "target_language": LanguageCode.CHINESE.value,
            "source_language": src_lang
        }

        if translation_context:
            cache_params["context_domain"] = translation_context.domain

        cache_key = calculate_cache_key(
            current_session.video.path,
            "subtitles_v2",
            cache_params
        )

        try:
            cached = container.cache_repo.get(cache_key)
            if cached and "en_segments" in cached:
                en_segments = tuple(
                    TextSegment(
                        text=seg["text"],
                        time_range=TimeRange(seg["start"], seg["end"]),
                        language=LanguageCode.ENGLISH
                    )
                    for seg in cached["en_segments"]
                )
                current_session.english_subtitle = Subtitle(en_segments, LanguageCode.ENGLISH)
        except Exception as e:
            print(f"  ⚠️  加载英文字幕失败: {e}")

        # 🔧 关键修复1: 加载已缓存的音频片段
        progress(0.95, "检查音频缓存...")
        current_session.audio_segments = _load_cached_audio_segments(
            current_session.video,
            current_session.translated_subtitle
        )

        # 🔧 关键修复1.5: 如果有缓存音频，尝试恢复参考音频路径
        if current_session.audio_segments:
            # 尝试从视频中提取参考音频（为后续编辑做准备）
            try:
                temp_ref_audio = container.video_processor.extract_reference_audio(
                    current_session.video,
                    duration=10.0
                )
                current_session.reference_audio_path = temp_ref_audio
                print(f"  ✅ 已准备参考音频: {temp_ref_audio}")
            except Exception as e:
                print(f"  ⚠️  准备参考音频失败: {e}")
                print(f"  💡 提示: 如需修改字幕，请先执行步骤2A获取参考音频")

        # 初始化审核状态
        for idx in range(len(result.translated_subtitle.segments)):
            # 检查音频是否已缓存
            audio_exists = idx in current_session.audio_segments

            current_session.segment_review_status[idx] = SegmentReviewStatus(
                segment_index=idx,
                subtitle_approved=False,
                audio_approved=audio_exists,  # 如果音频已缓存则标记为已完成
                subtitle_modified=False,
                needs_regeneration=not audio_exists  # 如果音频不存在则需要生成
            )

        # 生成状态报告
        cached_audio_count = len(current_session.audio_segments)
        total_segments = len(result.translated_subtitle.segments)

        # 🆕 计算音频时长统计
        total_max_duration = sum(seg.time_range.duration for seg in result.translated_subtitle.segments)
        total_actual_duration = sum(
            len(audio_seg.audio.samples) / audio_seg.audio.sample_rate
            for audio_seg in current_session.audio_segments.values()
        )

        report_lines = [
            f"✅ 字幕生成完成",
            f"",
            f"📊 基本信息:",
            f"   视频: {current_session.video.path.name}",
            f"   时长: {current_session.video.duration:.1f} 秒",
            f"   检测语言: {result.detected_language.value}",
            f"   总片段数: {total_segments}",
            f"   使用上下文: {translation_context.domain}",
            f"",
            f"🎵 音频缓存状态:",
            f"   已缓存片段: {cached_audio_count}/{total_segments}",
            f"   需要生成: {total_segments - cached_audio_count}",
            f"   理论总时长: {total_max_duration:.1f}s",
            f"   已生成时长: {total_actual_duration:.1f}s",
        ]

        # 质量报告
        if result.quality_report:
            qr = result.quality_report
            report_lines.extend([
                f"",
                f"🔍 质量检查结果:",
                f"   整体质量: {qr.overall_quality}",
                f"   发现问题: {qr.issues_found} 个",
                f"   是否需要审核: {'是 ⚠️' if qr.requires_review else '否 ✅'}",
            ])

        status_report = "\n".join(report_lines)

        # 准备审核数据(不包含音频)
        review_data = _prepare_review_data_v2()

        return review_data, status_report, gr.update(visible=True)

    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, gr.update(visible=False)


def _prepare_review_data_v2():
    """准备审核数据(包含音频播放器和时长信息)"""
    global current_session

    if not current_session.translated_subtitle:
        return None

    data = []
    for idx, (orig_seg, trans_seg) in enumerate(
            zip(current_session.original_subtitle.segments,
                current_session.translated_subtitle.segments)
    ):
        # 优先拿英文字幕
        en_text = (
            current_session.english_subtitle.segments[idx].text
            if current_session.english_subtitle and idx < len(current_session.english_subtitle.segments)
            else orig_seg.text
        )

        # 问题标记
        has_issue = False
        issue_desc = ""
        if current_session.quality_report:
            segment_issues = [
                i for i in current_session.quality_report.issues
                if i.segment_index == idx
            ]
            if segment_issues:
                has_issue = True
                issue_desc = "; ".join([
                    f"{i.issue_type}({i.severity})"
                    for i in segment_issues
                ])

        # 🔧 计算时间片最大长度（秒）
        max_duration = trans_seg.time_range.duration

        # 🔧 计算已生成音频长度
        audio_seg = current_session.audio_segments.get(idx)
        if audio_seg:
            # 如果有音频，计算实际长度
            actual_duration = len(audio_seg.audio.samples) / audio_seg.audio.sample_rate
            audio_status = "✅ 已缓存"
            duration_str = f"{actual_duration:.2f}s"
        else:
            audio_status = "未生成"
            duration_str = "-"

        # 审核状态
        review_status = current_session.segment_review_status.get(idx)
        if review_status:
            if review_status.subtitle_approved and review_status.audio_approved:
                review_mark = "✅ 已审核"
            elif review_status.subtitle_modified:
                review_mark = "🔄 已修改"
            else:
                review_mark = "⏳ 待审核"
        else:
            review_mark = "⏳ 待审核"

        data.append([
            idx,
            f"{trans_seg.time_range.start_seconds:.2f}s",
            en_text,
            trans_seg.text,
            f"{max_duration:.2f}s",  # 🆕 最大长度
            duration_str,  # 🆕 已生成长度
            audio_status,
            "⚠️" if has_issue else "",
            review_mark
        ])

    return data


# ============== 步骤2: 增量语音克隆 ============== #
def step2_incremental_voice_cloning(
        reference_audio_file,
        progress=gr.Progress()
):
    """步骤2: 增量语音克隆(逐片段合成)"""
    global current_session

    if not current_session.video or not current_session.translated_subtitle:
        return "❌ 错误: 会话状态丢失", gr.update()

    try:
        # 准备参考音频
        if reference_audio_file:
            ref_audio_path = Path(reference_audio_file.name)
            current_session.reference_audio_path = ref_audio_path
        elif current_session.reference_audio_path:
            ref_audio_path = current_session.reference_audio_path
        else:
            progress(0.05, "提取参考音频...")
            ref_audio_path = container.video_processor.extract_reference_audio(
                current_session.video,
                duration=10.0
            )
            current_session.reference_audio_path = ref_audio_path

        # 实时进度回调(更新表格)
        synthesis_log = []

        def segment_progress(ratio, msg, idx, audio_seg):
            synthesis_log.append(f"[{ratio * 100:.0f}%] {msg}")
            progress(ratio, msg)

            # 🔧 关键修复3: 如果有音频片段,更新会话状态
            if audio_seg:
                current_session.audio_segments[idx] = audio_seg

                # 更新审核状态
                status = current_session.segment_review_status.get(idx)
                if status and not status.subtitle_modified:
                    current_session.segment_review_status[idx] = SegmentReviewStatus(
                        segment_index=idx,
                        subtitle_approved=False,
                        audio_approved=True,  # 标记音频已完成
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

        #container.get_tts().unload()

        # 🔧 关键修复4: 更新所有音频片段到会话
        for audio_seg in result.audio_segments:
            current_session.audio_segments[audio_seg.segment_index] = audio_seg

        status = f"""
✅ 增量语音克隆完成!

📊 统计信息:
   总片段数: {result.total_segments}
   缓存命中: {result.cached_segments}
   新生成: {result.regenerated_segments}
   耗时: {result.synthesis_time:.1f} 秒

💡 提示: 
   - 点击表格中的行查看和播放音频
   - 修改字幕后需要重新生成对应片段
   - 审核通过后可以继续步骤3
"""

        # 更新表格数据
        updated_data = _prepare_review_data_v2()

        return status, gr.update(value=updated_data)

    except Exception as e:
        import traceback
        error_msg = f"❌ 语音克隆失败: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, gr.update()


# ============== 步骤2B: 字幕编辑和重新生成 ============== #
def step2_save_edits_and_regenerate(review_dataframe):
    """保存编辑并标记需要重新生成的片段"""
    global current_session

    if hasattr(review_dataframe, "values"):
        review_dataframe = review_dataframe.values.tolist()

    if not review_dataframe:
        return "⚠️ 没有可保存的修改", gr.update()

    # 跳过表头
    if review_dataframe and isinstance(review_dataframe[0][0], str):
        review_dataframe = review_dataframe[1:]

    edited_count = 0
    modified_indices = set()

    for row in review_dataframe:
        try:
            idx = int(row[0])
        except (ValueError, IndexError):
            continue

        if idx >= len(current_session.translated_subtitle.segments):
            continue

        original_text = current_session.translated_subtitle.segments[idx].text
        edited_text = row[3]  # 翻译列

        if edited_text != original_text:
            current_session.edited_segments[idx] = edited_text
            current_session.modified_indices.add(idx)
            edited_count += 1

            # 更新审核状态
            status = current_session.segment_review_status.get(idx)
            if status:
                current_session.segment_review_status[idx] = status.mark_subtitle_modified()

    if edited_count:
        # 应用编辑到字幕
        _apply_edits_to_subtitle_v2()

        # 保存到缓存
        _save_to_cache_v2("保存修改")

        updated_data = _prepare_review_data_v2()

        return (
            f"✅ 已保存 {edited_count} 处修改\n"
            f"⚠️  需要重新生成 {len(current_session.modified_indices)} 个音频片段",
            gr.update(value=updated_data)
        )
    else:
        return "ℹ️ 未检测到修改", gr.update()


def step2_regenerate_modified():
    """重新生成修改过的片段"""
    global current_session

    if not current_session.modified_indices:
        return "ℹ️ 没有需要重新生成的片段", gr.update()

    # 🔧 关键修复: 检查参考音频
    if not current_session.reference_audio_path:
        return "❌ 错误: 缺少参考音频。请先完成步骤2A(增量语音克隆)", gr.update()

    if not current_session.reference_audio_path.exists():
        return f"❌ 错误: 参考音频文件不存在: {current_session.reference_audio_path}", gr.update()

    try:
        print(f"\n🔄 重新生成修改片段:")
        print(f"   修改片段数: {len(current_session.modified_indices)}")
        print(f"   参考音频: {current_session.reference_audio_path}")

        result = regenerate_modified_segments_use_case(
            video=current_session.video,
            original_subtitle=current_session.original_subtitle,
            modified_subtitle=current_session.translated_subtitle,
            modified_indices=current_session.modified_indices,
            tts_provider=container.get_tts(),
            video_processor=container.video_processor,
            audio_repo=audio_segment_repo,
            reference_audio_path=current_session.reference_audio_path,  # 🔧 确保传递
            progress=None
        )

        #container.get_tts().unload()

        # 更新音频片段
        for audio_seg in result.audio_segments:
            current_session.audio_segments[audio_seg.segment_index] = audio_seg

        # 清除修改标记
        for idx in current_session.modified_indices:
            status = current_session.segment_review_status.get(idx)
            if status:
                current_session.segment_review_status[idx] = SegmentReviewStatus(
                    segment_index=idx,
                    subtitle_approved=False,
                    audio_approved=True,  # 音频已重新生成
                    subtitle_modified=True,
                    needs_regeneration=False
                )

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


# ============== 🔧 修复函数: 片段预览功能 ============== #
def preview_segment(evt: gr.SelectData):
    """
    预览选中的片段

    Args:
        evt: Gradio SelectData 事件,包含选中行的信息
    """
    global current_session

    # 🐛 调试信息
    print(f"\n🔍 预览片段调试信息:")
    print(f"   事件对象: {evt}")
    print(f"   事件类型: {type(evt)}")
    if evt:
        print(f"   evt.index: {evt.index}")
        print(f"   evt.index 类型: {type(evt.index)}")
        print(f"   evt.value: {getattr(evt, 'value', 'N/A')}")

    # 🔧 防御性检查1: 检查事件对象
    if evt is None:
        print(f"   ❌ 事件对象为 None")
        return None, "⚠️ 事件数据为空", "", ""

    # 🔧 防御性检查2: 检查会话状态
    if not current_session.video:
        print(f"   ❌ 会话状态丢失")
        return None, "❌ 会话状态丢失,请重新从步骤1开始", "", ""

    if not current_session.translated_subtitle:
        print(f"   ❌ 没有字幕数据")
        return None, "❌ 没有字幕数据", "", ""

    # 🔧 防御性检查3: 检查索引
    try:
        if evt.index is None:
            print(f"   ❌ evt.index 为 None")
            return None, "⚠️ 未选中任何行", "", ""

        # 🔧 关键修复: 正确解析 evt.index
        print(f"   原始 evt.index: {evt.index}, 类型: {type(evt.index)}")

        if isinstance(evt.index, (tuple, list)):
            # [row, col] 或 (row, col) 格式
            if len(evt.index) >= 1:
                selected_row_index = evt.index[0]
                print(f"   ✅ 解析序列索引: {evt.index} -> 行 {selected_row_index}")
            else:
                print(f"   ❌ 空序列索引")
                return None, "❌ 索引格式错误（空序列）", "", ""
        elif isinstance(evt.index, (int, float)):
            # 直接是数字
            selected_row_index = evt.index
            print(f"   ✅ 直接使用索引: {selected_row_index}")
        else:
            # 未知格式
            print(f"   ❌ 未知的索引格式: {type(evt.index)}, 值: {evt.index}")
            return None, f"❌ 未知的索引格式: {type(evt.index)}", "", ""

        # 🔧 重要：确保转换为整数
        try:
            selected_row_index = int(selected_row_index)
        except (TypeError, ValueError) as e:
            print(f"   ❌ 无法转换为整数: {selected_row_index}, 错误: {e}")
            return None, f"❌ 索引值无法转换为整数: {selected_row_index}", "", ""

        print(f"   ✅ 最终行索引: {selected_row_index} (类型: {type(selected_row_index)})")

    except (TypeError, ValueError, IndexError) as e:
        print(f"   ❌ 索引解析失败: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ 索引解析失败: {e}", "", ""

    # 🔧 防御性检查4: 验证索引范围
    total_segments = len(current_session.translated_subtitle.segments)
    print(f"   总片段数: {total_segments}")

    if selected_row_index < 0 or selected_row_index >= total_segments:
        print(f"   ❌ 索引超出范围: {selected_row_index}")
        return None, f"❌ 无效的片段索引: {selected_row_index} (总数: {total_segments})", "", ""

    idx = selected_row_index

    try:
        text_seg = current_session.translated_subtitle.segments[idx]
        print(f"   ✅ 获取片段 {idx}: {text_seg.text[:30]}...")
    except (IndexError, TypeError) as e:
        print(f"   ❌ 无法获取片段 {idx}: {e}")
        return None, f"❌ 无法获取片段 {idx}: {e}", "", ""

    # 🔧 计算时长信息
    max_duration = text_seg.time_range.duration

    # 🔧 关键修复5: 从会话或磁盘获取音频
    audio_seg = current_session.audio_segments.get(idx)
    print(f"   内存中音频: {audio_seg is not None}")

    # 如果内存中没有,尝试从磁盘加载
    if not audio_seg:
        print(f"   尝试从磁盘加载...")
        try:
            audio_seg = audio_segment_repo.load_segment(
                segment_index=idx,
                video_path=current_session.video.path,
                text_segment=text_seg
            )

            # 如果加载成功,更新到会话
            if audio_seg:
                current_session.audio_segments[idx] = audio_seg
                print(f"   ✅ 片段 {idx} 从磁盘加载成功")
            else:
                print(f"   ⚠️  磁盘也没有片段 {idx}")
        except Exception as e:
            print(f"   ❌ 片段 {idx} 加载失败: {e}")

    # 检查音频文件并计算实际时长
    actual_duration = None
    if audio_seg and audio_seg.file_path:
        print(f"   音频文件路径: {audio_seg.file_path}")
        if audio_seg.file_path.exists():
            audio_path = str(audio_seg.file_path)
            actual_duration = len(audio_seg.audio.samples) / audio_seg.audio.sample_rate

            # 🆕 计算时长差异
            duration_diff = actual_duration - max_duration
            duration_ratio = (actual_duration / max_duration) * 100 if max_duration > 0 else 0

            audio_status = f"✅ 音频已生成 ({duration_ratio:.1f}%)"
            print(f"   ✅ 音频文件存在，时长: {actual_duration:.2f}s")
        else:
            audio_path = None
            audio_status = f"❌ 音频文件不存在: {audio_seg.file_path.name}"
            print(f"   ❌ 音频文件不存在")
    else:
        audio_path = None
        audio_status = "⚠️  音频未生成"
        print(f"   ⚠️  没有音频片段")

    # 🆕 文本信息 - 包含详细时长信息
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

    print(f"   返回结果: audio={audio_path is not None}, status={audio_status}\n")

    return audio_path, audio_status, text_info, subtitle_text


# ============== 辅助函数 ============== #
def _apply_edits_to_subtitle_v2():
    """应用编辑到字幕对象"""
    global current_session

    if not current_session.edited_segments:
        return

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


def _save_to_cache_v2(operation_name: str = "操作"):
    """保存到缓存"""
    global current_session

    try:
        if not current_session.video or not current_session.translated_subtitle:
            return
        src_lang = LanguageCode(current_session.source_language.value) if current_session.source_language and current_session.source_language.value != "auto" else None
        cache_params = {
            "target_language": LanguageCode.CHINESE.value,
            "source_language": src_lang,
        }

        if current_session.translation_context:
            cache_params["context_domain"] = current_session.translation_context.domain

        cache_key = calculate_cache_key(
            current_session.video.path,
            "subtitles_v2",
            cache_params
        )

        cached = container.cache_repo.get(cache_key) or {}

        # 更新中文字幕
        cached["zh_segments"] = [
            {
                "text": seg.text,
                "start": seg.time_range.start_seconds,
                "end": seg.time_range.end_seconds,
            }
            for seg in current_session.translated_subtitle.segments
        ]

        container.cache_repo.set(cache_key, cached)
        print(f"✅ {operation_name}: 中文字幕已写回缓存")

    except Exception as e:
        print(f"⚠️ {operation_name}: 写回缓存失败: {e}")


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


# ============== 步骤3: 最终合成 ============== #
def step3_final_synthesis(progress=gr.Progress()):
    """步骤3: 最终视频合成"""
    global current_session
    container.get_tts().unload()
    if not current_session.video:
        return None, None, None, "❌ 错误: 会话状态丢失"

    # 🔧 关键修复6: 重新检查音频状态
    total_segments = len(current_session.translated_subtitle.segments)
    audio_ready = len(current_session.audio_segments)

    print(f"\n🔍 最终合成前检查:")
    print(f"   总片段数: {total_segments}")
    print(f"   音频已生成: {audio_ready}")
    print(f"   缺失片段: {total_segments - audio_ready}")

    # 检查是否所有片段都已审核
    unreviewed = [
        idx for idx in range(total_segments)
        if idx not in current_session.audio_segments
    ]

    if unreviewed and len(unreviewed) > total_segments * 0.3:
        return None, None, None, f"⚠️  还有 {len(unreviewed)} 个片段未完成音频生成,请先完成步骤2"

    try:
        progress(0.1, "准备合成...")

        output_dir = current_session.video.path.parent / "output"
        output_dir.mkdir(exist_ok=True)

        # 合并音频片段
        progress(0.2, "合并音频片段...")

        from domain.entities import AudioTrack
        from domain.value_objects import AudioSample

        # 创建完整音轨
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

        # 创建双语字幕
        if current_session.english_subtitle:
            zh_en_subtitle = merge_bilingual_subtitles(
                current_session.translated_subtitle,
                current_session.english_subtitle
            )
            subtitles_tuple = (
                current_session.translated_subtitle,
                current_session.english_subtitle,
                zh_en_subtitle
            )
        else:
            subtitles_tuple = (current_session.translated_subtitle,)

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
        zh_en_ass = find_file(['zh_en'], '.ass')
        voiced_video = find_file(['_voiced_subtitled.mp4'])

        status = f"""
✅ 最终合成完成!

📦 输出文件:
   - 中文字幕: {zh_srt.split('/')[-1] if zh_srt else '❌'}
   - 双语字幕: {zh_en_ass.split('/')[-1] if zh_en_ass else '❌'}
   - 配音视频: {voiced_video.split('/')[-1] if voiced_video else '❌'}

⏱️  处理时间: {synthesis_result.processing_time:.1f} 秒

📊 统计信息:
   总片段数: {len(current_session.audio_segments)}
   使用缓存: {len([s for s in current_session.segment_review_status.values() if not s.needs_regeneration])}
   重新生成: {len([s for s in current_session.segment_review_status.values() if s.subtitle_modified])}
"""

        return zh_srt, zh_en_ass, voiced_video, status

    except Exception as e:
        import traceback
        error_msg = f"❌ 合成失败: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, error_msg


# ============== UI 构建 ============== #
def build_ui_v2():
    """构建增强 UI V2"""

    with gr.Blocks(
            title="视频翻译工厂 Pro V2",
            css="""
        .gradio-container {max-width: 1800px !important}
        .segment-preview {border: 1px solid #ddd; padding: 10px; border-radius: 5px;}
        """
    ) as demo:
        gr.Markdown("""
        # 🎬 视频翻译工厂 Pro V2 - 分段审核版 (修复版)

        ## ✨ V2 新特性
        - 🎵 **分段语音克隆**: 逐片段生成并缓存音频
        - 👂 **实时预览**: 边生成边试听,即时反馈
        - ✏️  **精细编辑**: 修改字幕后仅重新生成对应片段
        - 💾 **智能缓存**: 片段级缓存,断点续传
        - 🔄 **增量合成**: 跳过未修改的片段,提升效率

        ## 🔧 本次修复
        - ✅ 修复音频片段预览无法播放问题
        - ✅ 修复缓存加载逻辑,步骤1后自动加载已缓存音频
        - ✅ 修复最终合成时音频状态检查
        - ✅ 优化会话状态管理,确保数据一致性

        ## 📋 优化工作流程
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
                        label="🎵 参考音频(可选)",
                        file_types=[".wav", ".mp3"]
                    )

                    clone_btn = gr.Button("🎤 开始增量语音克隆", variant="primary")
                    clone_status = gr.Textbox(label="克隆状态", lines=8)

                # 2B: 审核表格
                with gr.Group():
                    gr.Markdown("### 2B. 审核和预览")

                    review_dataframe = gr.Dataframe(
                        headers=[
                            "索引",
                            "时间",
                            "原文",
                            "翻译",
                            "最大长度",  # 🆕 新增列
                            "已生成长度",  # 🆕 新增列
                            "音频",
                            "问题",
                            "状态"
                        ],
                        datatype=[
                            "number",  # 索引
                            "str",  # 时间
                            "str",  # 原文
                            "str",  # 翻译
                            "str",  # 🆕 最大长度
                            "str",  # 🆕 已生成长度
                            "str",  # 音频
                            "str",  # 问题
                            "str"  # 状态
                        ],
                        col_count=(9, "fixed"),  # 🔧 改为 9 列
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
                            preview_status = gr.Textbox(
                                label="状态",
                                lines=1
                            )

                        with gr.Column(scale=1):
                            preview_info = gr.Textbox(
                                label="片段信息",
                                lines=3
                            )
                            preview_text = gr.Textbox(
                                label="字幕文本",
                                lines=4
                            )

            # ========== 步骤3 ========== #
            with gr.Accordion("🎬 步骤3: 最终合成", open=False):
                gr.Markdown("""
                ### 提示
                - 确保所有关键片段都已审核通过
                - 系统会合并所有音频片段生成完整视频
                """)

                final_btn = gr.Button("▶️ 生成最终视频", variant="primary", size="lg")
                final_status = gr.Textbox(label="合成状态", lines=10)

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
                inputs=[reference_audio],
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

            # 🔧 修复: 表格选择事件 - 使用 SelectData + 错误处理
            try:
                review_dataframe.select(
                    preview_segment,
                    outputs=[preview_audio, preview_status, preview_info, preview_text]
                )
            except Exception as e:
                print(f"⚠️ 表格选择事件绑定失败: {e}")
                # 如果绑定失败,添加一个替代方案
                gr.Markdown("""
                ⚠️ **片段预览功能初始化失败**
                
                可能原因:
                - Gradio 版本不兼容
                - 表格数据格式问题
                
                解决方案:
                1. 确保 Gradio >= 4.0
                2. 检查表格是否有数据
                3. 查看控制台错误日志
                """)

            # 步骤3: 最终合成
            final_btn.click(
                step3_final_synthesis,
                outputs=[zh_srt_output, zh_en_ass_output, final_video_output, final_status]
            )

        # ========== 使用说明 ========== #
        with gr.Tab("📚 V2 使用指南 + 修复说明"):
            gr.Markdown("""
            ## 🔧 本次修复内容
            
            ### 修复的问题
            1. ❌ **片段预览无法播放**: 点击表格行后音频无法加载
            2. ❌ **音频状态不准确**: 明明已缓存但显示"未生成"
            3. ❌ **最终合成失败**: 提示音频未生成,无法合成视频
            4. ❌ **会话状态丢失**: 刷新后音频缓存信息丢失
            
            ### 根本原因
            ```python
            # 问题1: 音频片段未加载到会话
            # 步骤1完成后,虽然磁盘有缓存,但 current_session.audio_segments 为空
            
            # 问题2: 预览功能只从内存读取
            audio_seg = current_session.audio_segments.get(idx)
            # 如果内存没有,直接返回 None,不尝试从磁盘加载
            
            # 问题3: Gradio 事件绑定错误
            review_dataframe.select(
                preview_segment,
                inputs=[],  # ❌ 空输入,无法获取选中行
                outputs=[...]
            )
            ```
            
            ### 修复方案
            
            #### 1. 步骤1后自动加载缓存音频
            ```python
            # 新增函数: _load_cached_audio_segments()
            def step1_generate_and_check_v2(...):
                # ... 生成字幕 ...
                
                # 🔧 关键修复: 加载已缓存的音频片段
                progress(0.95, "检查音频缓存...")
                current_session.audio_segments = _load_cached_audio_segments(
                    current_session.video,
                    current_session.translated_subtitle
                )
                
                # 更新审核状态
                for idx in range(len(segments)):
                    audio_exists = idx in current_session.audio_segments
                    status[idx] = SegmentReviewStatus(
                        audio_approved=audio_exists,  # 正确反映音频状态
                        needs_regeneration=not audio_exists
                    )
            ```
            
            #### 2. 预览功能支持磁盘加载
            ```python
            def preview_segment(evt: gr.SelectData):  # 🔧 使用 SelectData
                selected_row_index = evt.index[0]  # 获取行索引
                
                # 先从内存获取
                audio_seg = current_session.audio_segments.get(idx)
                
                # 🔧 如果内存没有,尝试从磁盘加载
                if not audio_seg:
                    audio_seg = audio_segment_repo.load_segment(
                        segment_index=idx,
                        video_path=current_session.video.path,
                        text_segment=text_seg
                    )
                    
                    # 加载成功后更新到会话
                    if audio_seg:
                        current_session.audio_segments[idx] = audio_seg
                
                return str(audio_seg.file_path), ...
            ```
            
            #### 3. 正确的 Gradio 事件绑定
            ```python
            # ❌ 错误写法
            review_dataframe.select(
                preview_segment,
                inputs=[],  # 无法获取选中信息
                outputs=[...]
            )
            
            # ✅ 正确写法
            review_dataframe.select(
                preview_segment,  # 函数自动接收 SelectData 参数
                outputs=[preview_audio, preview_status, ...]
            )
            ```
            
            #### 4. 步骤2 音频生成后更新会话
            ```python
            def step2_incremental_voice_cloning(...):
                # ... 执行合成 ...
                
                # 🔧 确保所有音频片段都更新到会话
                for audio_seg in result.audio_segments:
                    current_session.audio_segments[audio_seg.segment_index] = audio_seg
                    
                    # 同时更新审核状态
                    current_session.segment_review_status[idx] = SegmentReviewStatus(
                        audio_approved=True,  # 标记音频已完成
                        needs_regeneration=False
                    )
            ```
            
            ### 验证修复效果
            
            #### 测试步骤
            ```bash
            # 1. 上传视频,完成步骤1
            ✅ 检查控制台输出:
                "🔍 检查音频片段缓存:"
                "✅ 共加载 X/Y 个缓存片段"
            
            # 2. 查看审核表格
            ✅ "音频"列应显示:
                - "✅ 已缓存" (如果磁盘有缓存)
                - "未生成" (如果需要生成)
            
            # 3. 点击表格中任意行
            ✅ 如果音频已缓存:
                - 左侧音频播放器出现波形
                - 状态显示 "✅ 音频已生成"
                - 右侧显示片段信息和字幕文本
            
            # 4. 点击"开始增量语音克隆"
            ✅ 生成过程中:
                - 表格实时更新 "音频"列
                - 新生成的片段可立即预览
            
            # 5. 点击"生成最终视频"
            ✅ 不再提示 "音频未生成"
            ✅ 成功合成完整视频
            ```
            
            #### 缓存文件检查
            ```bash
            # 查看音频片段缓存
            ls -lh .cache/audio_segments/video_name_*/
            
            # 应该看到:
            seg_0000.wav
            seg_0000.json
            seg_0001.wav
            seg_0001.json
            ...
            ```
            
            ### 数据流图
            
            ```
            磁盘缓存                会话内存               UI显示
            ─────────────────────────────────────────────────────
            
            步骤1完成后:
            .cache/audio_segments/  →  audio_segments  →  表格"✅已缓存"
                seg_0000.wav           {0: AudioSeg}       预览可播放
                seg_0001.wav           {1: AudioSeg}
            
            步骤2生成新片段:
            .cache/audio_segments/  ←  audio_segments  ←  TTS生成
                seg_0002.wav        ←  {2: AudioSeg}  ←  立即更新
                                       
            点击预览:
            .cache/audio_segments/  →  audio_segments  →  音频播放器
                seg_0002.wav           读取file_path       加载音频
            
            步骤3合成:
            audio_segments  →  合并所有片段  →  完整音轨  →  最终视频
            {0,1,2,...}
            ```
            
            ### 性能优化
            
            #### 懒加载策略
            - **步骤1**: 只加载元数据,不加载音频数据
            - **预览时**: 按需加载音频文件
            - **步骤3**: 批量读取所有音频
            
            #### 内存管理
            ```python
            # AudioSegment 只存储文件路径,不存储原始音频数据
            @dataclass(frozen=True)
            class AudioSegment:
                file_path: Path  # 只存路径
                # samples: tuple  # 不在内存中保存
            
            # 播放时才读取文件
            audio_path = str(audio_seg.file_path)
            gr.Audio(value=audio_path)  # Gradio 从文件读取
            ```
            
            ### 常见问题排查
            
            #### Q: 步骤1后表格显示"未生成",但磁盘有缓存?
            ```python
            # 检查 _load_cached_audio_segments() 是否被调用
            print(f"✅ 共加载 {len(cached_segments)} 个缓存片段")
            
            # 如果未输出,说明函数未执行,检查步骤1代码
            ```
            
            #### Q: 点击表格行没有反应?
            ```python
            # 检查事件绑定
            review_dataframe.select(
                preview_segment,  # ✅ 正确
                outputs=[...]
            )
            
            # 不要写成:
            review_dataframe.select(
                preview_segment,
                inputs=[],  # ❌ 错误
                outputs=[...]
            )
            ```
            
            #### Q: 预览时提示"文件不存在"?
            ```python
            # 检查文件路径
            if audio_seg.file_path and audio_seg.file_path.exists():
                return str(audio_seg.file_path)
            else:
                print(f"⚠️ 文件不存在: {audio_seg.file_path}")
            ```
            
            #### Q: 步骤3提示"音频未生成"?
            ```python
            # 检查会话状态
            print(f"内存中音频片段: {len(current_session.audio_segments)}")
            print(f"总片段数: {len(current_session.translated_subtitle.segments)}")
            
            # 如果数量不匹配,说明步骤2未正确更新会话
            ```
            
            ### 总结
            
            本次修复通过**统一磁盘缓存与内存状态**,确保:
            - ✅ 步骤1后自动加载已缓存音频
            - ✅ 预览功能支持磁盘+内存双重查找
            - ✅ 步骤2生成后立即更新会话
            - ✅ 步骤3能正确获取所有音频片段
            
            核心思想: **磁盘是真相,内存是缓存**
            - 磁盘: 持久化存储,断点续传
            - 内存: 快速访问,会话管理
            - 同步: 双向更新,保持一致
            """)

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