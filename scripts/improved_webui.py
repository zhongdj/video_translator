"""
Infrastructure Layer - 改进的WebUI（带翻译审核功能）
"""
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import torch
import torchaudio

from domain.entities import Video, Subtitle, LanguageCode, TextSegment, TimeRange
from domain.services import calculate_cache_key
from domain.translation import TranslationContext
from infrastructure.config.dependency_injection import container


# 初始化翻译上下文仓储
context_repo = container.translator_context_repo


# ============== 会话状态管理 ============== #
class TranslationSession:
    """翻译会话状态"""
    def __init__(self):
        self.translation_context = None
        self.video: Optional[Video] = None
        self.original_subtitle: Optional[Subtitle] = None
        self.translated_subtitle: Optional[Subtitle] = None
        self.english_subtitle: Optional[Subtitle] = None  # ⭐ 新增：英文字幕
        self.detected_language: Optional[LanguageCode] = None  # ⭐ 关键：检测到的语言
        self.quality_report = None
        self.edited_segments = {}  # {index: edited_text}
        self.approved = False


# 全局会话对象
current_session = TranslationSession()


# ============== 步骤1：生成字幕和质量检查 ============== #
def step1_generate_and_check(
    video_file,
    whisper_model: str,
    translation_model: str,
    translation_context_name: str,
    source_language: str,
    progress=gr.Progress()
):
    """步骤1：生成字幕并进行质量检查"""
    if not video_file:
        return None, "❌ 请上传视频", gr.update(visible=False)

    try:
        global current_session
        current_session = TranslationSession()

        video_path = Path(video_file.name)

        # 创建视频对象
        current_session.video = Video(
            path=video_path,
            duration=get_video_duration(video_path),
            has_audio=True
        )

        # 加载翻译上下文
        translation_context = context_repo.load(translation_context_name)

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

        current_session.original_subtitle = result.original_subtitle
        current_session.translated_subtitle = result.translated_subtitle
        current_session.detected_language = result.detected_language
        current_session.quality_report = result.quality_report

        # ⭐ 从缓存加载英文字幕
        cache_key = calculate_cache_key(
            current_session.video.path,
            "subtitles_v2",
            {
                "target_language": LanguageCode.CHINESE.value,
                "source_language": src_lang.value if src_lang else "auto",
                "context_domain": translation_context.domain if translation_context else "general"
            }
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
                print(f"  ✅ 已加载英文字幕: {len(en_segments)} 片段")
            else:
                print(f"  ⚠️  缓存中未找到英文字幕")
        except Exception as e:
            print(f"  ⚠️  加载英文字幕失败: {e}")

        # 生成状态报告
        report_lines = [
            f"✅ 字幕生成完成",
            f"",
            f"📊 基本信息:",
            f"   视频: {current_session.video.path.name}",
            f"   时长: {current_session.video.duration:.1f} 秒",
            f"   检测语言: {result.detected_language.value} ⭐",
            f"   总片段数: {len(result.translated_subtitle.segments)}",
            f"   使用上下文: {translation_context.domain}",
            f"   缓存状态: {'命中' if result.cache_hit else '新生成'}",
        ]

        # 质量报告
        if result.quality_report:
            qr = result.quality_report
            report_lines.extend([
                f"",
                f"🔍 质量检查结果:",
                f"   整体质量: {qr.overall_quality}",
                f"   发现问题: {qr.issues_found} 个",
                f"   - 高严重度: {qr.high_severity_count}",
                f"   - 中严重度: {qr.medium_severity_count}",
                f"   - 低严重度: {qr.low_severity_count}",
                f"",
                f"   是否需要审核: {'是 ⚠️' if qr.requires_review else '否 ✅'}",
            ])

            # 显示前5个问题
            if qr.issues_found > 0:
                report_lines.append(f"")
                report_lines.append(f"⚠️  主要问题预览（前5个）:")
                for i, issue in enumerate(list(qr.issues)[:5], 1):
                    report_lines.append(
                        f"   {i}. [片段{issue.segment_index}] {issue.issue_type} ({issue.severity})"
                    )
                    report_lines.append(f"      {issue.description}")
        elif result.cache_hit:
            # ⭐ 缓存命中时，没有质量报告，但仍然可以审核
            report_lines.extend([
                f"",
                f"🔍 质量检查:",
                f"   缓存命中，跳过质量检查",
                f"   如需审核翻译，请展开步骤2",
            ])

        status_report = "\n".join(report_lines)

        # ⭐ 准备审核数据（无论缓存命中与否都生成）
        review_data = _prepare_review_data()

        # ⭐ 决定是否显示审核界面
        # 条件1: 质量报告存在且需要审核
        # 条件2: 用户可能想手动审核（缓存命中但无质量报告）
        show_review = False
        if result.quality_report and result.quality_report.requires_review:
            show_review = True
        elif result.cache_hit:
            # 缓存命中时，默认不显示审核，但可以手动展开
            show_review = False
        else:
            # 新生成但质量很好，也默认不显示
            show_review = False

        return review_data, status_report, gr.update(visible=True)  # ⭐ 始终可见，但可以折叠

    except Exception as e:
        import traceback
        error_msg = f"❌ 生成失败: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, gr.update(visible=False)


def _prepare_review_data():
    """准备审核数据"""
    global current_session

    if not current_session.translated_subtitle:
        return None

    # 创建可编辑的数据格式
    data = []
    for idx, (orig_seg, trans_seg) in enumerate(
        zip(current_session.original_subtitle.segments,
            current_session.translated_subtitle.segments)
    ):
        # 检查是否有问题标记
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
                    f"{i.issue_type}({i.severity}): {i.description}"
                    for i in segment_issues
                ])

        data.append([
            idx,
            f"{orig_seg.time_range.start_seconds:.2f}s",
            orig_seg.text,
            trans_seg.text,
            "⚠️" if has_issue else "✅",
            issue_desc
        ])

    return data


# ============== 步骤2：人工审核和修改 ============== #
def step2_review_and_edit(review_dataframe):
    # 1. 把 Pandas DataFrame → 纯 Python 二维列表
    if hasattr(review_dataframe, "values"):
        review_dataframe = review_dataframe.values.tolist()

    if not review_dataframe:          # 现在可以安全判断空
        return "⚠️ 没有可保存的修改", gr.update(), gr.update()

    # 2. 跳过表头（若第一行第一列是字符串）
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
        edited_text   = row[3]
        if edited_text != original_text:
            current_session.edited_segments[idx] = edited_text
            edited_count += 1

    if edited_count:
        _apply_edits_to_subtitle()
        current_session.approved = False
        updated_data = _prepare_review_data()
        return f"✅ 已保存 {edited_count} 处修改", gr.update(value=updated_data), gr.update(interactive=True)
    else:
        return "ℹ️ 未检测到修改", gr.update(), gr.update()


def _apply_edits_to_subtitle():
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


def step2_approve_translation():
    """步骤2：批准翻译，进入下一步"""
    global current_session
    current_session.approved = True

    # ⭐ 把审核后的字幕写回缓存，保证下次命中的是“人工校对版”
    try:
        cache_key = calculate_cache_key(
            current_session.video.path,
            "subtitles_v2",
            {
                "target_language": LanguageCode.CHINESE.value,
                "source_language": current_session.detected_language.value,
                "context_domain": current_session.translation_context.domain
            }
        )
        cached = container.cache_repo.get(cache_key) or {}  # 原缓存对象
        cached["zh_segments"] = [
            {
                "text": seg.text,
                "start": seg.time_range.start_seconds,
                "end": seg.time_range.end_seconds,
            }
            for seg in current_session.translated_subtitle.segments
        ]
        container.cache_repo.save(cache_key, cached)  # 写回
        print(f"✅ 审核后的中文字幕已写回缓存：{cache_key}")
        # ⭐ 让下游语音克隆+视频合成缓存失效
        try:
            # 1. 删除语音克隆缓存
            clone_key = calculate_cache_key(
                current_session.video.path,
                "clone_voice",
                {
                    "target_language": LanguageCode.CHINESE.value,
                    "source_language": current_session.detected_language.value,
                    "reference_audio_hash": "default"  # 与你生成时保持一致即可
                }
            )
            container.cache_repo.delete(clone_key)
            print(f"🗑️ 已删除语音克隆缓存：{clone_key}")
        except Exception as e:
            print(f"⚠️ 删除语音克隆缓存失败：{e}")

        try:
            # 2. 删除视频合成缓存
            synth_key = calculate_cache_key(
                current_session.video.path,
                "synthesize_video",
                {
                    "target_language": LanguageCode.CHINESE.value,
                    "source_language": current_session.detected_language.value,
                    "burn_subtitles": True
                }
            )
            container.cache_repo.delete(synth_key)
            print(f"🗑️ 已删除视频合成缓存：{synth_key}")
        except Exception as e:
            print(f"⚠️ 删除视频合成缓存失败：{e}")
    except Exception as e:
        print(f"⚠️ 写回缓存失败：{e}")


    return "✅ 翻译已批准，可以继续步骤3", gr.update(open=True)  # ⭐ 自动展开步骤3


# ============== 步骤3：语音合成和视频生成 ============== #
def step3_synthesize_video(
    enable_voice: bool,
    reference_audio_file,
    progress=gr.Progress()
):
    """步骤3：语音合成和视频生成"""
    global current_session

    # ⭐ 检查必要的会话状态
    if not current_session.video:
        return None, None, None, None, "❌ 错误：会话状态丢失，请重新从步骤1开始"

    if not current_session.translated_subtitle:
        return None, None, None, None, "❌ 错误：没有翻译结果，请先完成步骤1"

    if not current_session.approved and current_session.quality_report:
        if current_session.quality_report.requires_review:
            return None, None, None, None, "⚠️  请先完成翻译审核并批准"

    try:
        output_dir = current_session.video.path.parent / "output"
        output_dir.mkdir(exist_ok=True)

        # ⭐ 获取英文字幕
        if not current_session.english_subtitle:
            progress(0.05, "生成英文字幕...")
            # 如果会话中没有，尝试重新加载
            from domain.services import calculate_cache_key
            cache_key = calculate_cache_key(
                current_session.video.path,
                "subtitles_v2",
                {
                    "target_language": LanguageCode.CHINESE.value,
                    "source_language": "auto"
                }
            )

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
            else:
                # 如果还是没有，可能需要重新翻译
                print("⚠️  警告：未找到英文字幕，将仅生成中文字幕")

        en_subtitle = current_session.english_subtitle
        zh_subtitle = current_session.translated_subtitle
        if zh_subtitle is None:
            try:
                cache_key = calculate_cache_key(
                    current_session.video.path,
                    "subtitles_v2",
                    {
                        "target_language": LanguageCode.CHINESE.value,
                        "source_language": current_session.detected_language.value,
                    }
                )
                cached = container.cache_repo.get(cache_key)
                if cached and "zh_segments" in cached:
                    zh_segments = tuple(
                        TextSegment(
                            text=seg["text"],
                            time_range=TimeRange(seg["start"], seg["end"]),
                            language=LanguageCode.CHINESE
                        )
                        for seg in cached["zh_segments"]
                    )
                    zh_subtitle = Subtitle(zh_segments, LanguageCode.CHINESE)
                    print("⚠️  session 字幕丢失，已从缓存兜底加载中文字幕")
                else:
                    raise RuntimeError("缓存中也没有中文字幕")
            except Exception as e:
                return None, None, None, None, f"❌ 无法获取中文字幕：{e}"
        # 语音克隆
        audio_track = None
        if enable_voice:
            ref_audio_path = Path(reference_audio_file.name) if reference_audio_file else None

            progress(0.1, f"开始语音克隆（目标语言：{current_session.detected_language.value}）...")

            from application.use_cases.clone_voice import clone_voice_use_case

            voice_result = clone_voice_use_case(
                video=current_session.video,
                subtitle=zh_subtitle,
                tts_provider=container.get_tts(),
                video_processor=container.video_processor,
                cache_repo=container.cache_repo,
                reference_audio_path=ref_audio_path,
                progress=lambda p, d: progress(0.1 + p * 0.5, d)
            )
            audio_track = voice_result.audio_track
            print(f"✅ 语音克隆完成")

        # 创建双语字幕
        progress(0.6, "创建双语字幕...")
        from domain.services import merge_bilingual_subtitles

        if en_subtitle:
            zh_en_subtitle = merge_bilingual_subtitles(
                current_session.translated_subtitle,  # 中文在上
                en_subtitle  # 英文在下
            )
            print(f"✅ 双语字幕创建完成")
        else:
            zh_en_subtitle = current_session.translated_subtitle
            print(f"⚠️  仅使用中文字幕")

        progress(0.7, "合成视频...")

        # 视频合成
        from application.use_cases.synthesize_video_use_case import synthesize_video_use_case

        # ⭐ 准备字幕元组（根据是否有英文字幕）
        if en_subtitle:
            subtitles_tuple = (
                current_session.translated_subtitle,  # 中文
                en_subtitle,                          # 英文
                zh_en_subtitle                        # 双语
            )
        else:
            subtitles_tuple = (
                current_session.translated_subtitle,  # 仅中文
            )

        synthesis_result = synthesize_video_use_case(
            video=current_session.video,
            subtitles=subtitles_tuple,
            audio_track=audio_track,
            video_processor=container.video_processor,
            subtitle_writer=container.subtitle_writer,
            output_dir=output_dir,
            formats=("srt", "ass"),
            burn_subtitles=True,
            progress=lambda p, d: progress(0.7 + p * 0.3, d)
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
        zh_en_ass = find_file(['zh_en'], '.ass') if en_subtitle else None
        voiced_video = find_file(['_voiced.mp4']) if audio_track else None
        voiced_subtitled = find_file(['_voiced_subtitled.mp4']) if audio_track else None

        # ⭐ 生成详细状态报告
        status = f"""
✅ 处理完成！

📊 处理信息:
   视频: {current_session.video.path.name}
   原始语言: {current_session.detected_language.value}
   目标语言: zh (中文)
   
📦 生成文件:
   - 中文字幕: {zh_srt.split('/')[-1] if zh_srt else '❌'}
   - 双语字幕: {zh_en_ass.split('/')[-1] if zh_en_ass else '❌ (英文字幕缺失)'}
   - 配音视频: {voiced_video.split('/')[-1] if voiced_video else '未启用'}
   - 配音+字幕: {voiced_subtitled.split('/')[-1] if voiced_subtitled else '未启用'}

⏱️  处理时间: {synthesis_result.processing_time:.1f} 秒
"""

        if not en_subtitle:
            status += "\n⚠️  提示：英文字幕未生成，可能影响双语字幕输出"

        progress(1.0, "完成！")

        return zh_srt, zh_en_ass, voiced_video, voiced_subtitled, status

    except Exception as e:
        import traceback
        error_msg = f"❌ 合成失败: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, None, error_msg


# ============== 辅助函数 ============== #
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


# ============== 上下文管理界面 ============== #
def list_translation_contexts():
    """列出所有翻译上下文"""
    return context_repo.list_contexts()


def load_context_for_editing(context_name: str):
    """加载上下文进行编辑"""
    context = context_repo.load(context_name)
    if not context:
        return "", "", "❌ 加载失败"

    # 格式化术语表
    terminology_text = "\n".join([
        f"{k} = {v}" for k, v in context.terminology.items()
    ])

    return context.system_prompt, terminology_text, f"✅ 已加载 {context_name}"


def save_custom_context(
    context_name: str,
    system_prompt: str,
    terminology_text: str
):
    """保存自定义上下文"""
    if not context_name.strip():
        return "❌ 请输入上下文名称"

    # 解析术语表
    terminology = {}
    for line in terminology_text.strip().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            terminology[key.strip()] = value.strip()


    context = TranslationContext(
        domain=context_name,
        system_prompt=system_prompt,
        terminology=terminology
    )

    context_repo.save(context_name, context)

    return f"✅ 已保存上下文: {context_name}"


# ============== Gradio UI 构建 ============== #
def build_improved_ui():
    """构建改进的UI"""

    with gr.Blocks(
        title="视频翻译工厂 - 质量优先版",
        css="""
        .gradio-container {max-width: 1600px !important}
        .quality-excellent {color: #10b981;}
        .quality-good {color: #3b82f6;}
        .quality-fair {color: #f59e0b;}
        .quality-poor {color: #ef4444;}
        """
    ) as demo:
        gr.Markdown("""
        # 🎬 视频翻译工厂 Pro - 质量优先版
        
        ## ✨ 新特性
        - 🎯 **翻译上下文管理**: 领域专属提示词 + 术语表
        - 🔍 **智能质量检查**: AI 辅助发现翻译问题
        - ✏️  **人工审核界面**: 可视化编辑和审批流程
        - 💾 **增量处理**: 审核通过后才进行语音合成
        
        ## 📋 工作流程
        1. **生成和检查** → 2. **审核修改**（可选）→ 3. **语音合成**
        """)

        with gr.Tab("🎬 单视频处理（改进版）"):
            gr.Markdown("""
            ### 三步式工作流
            此流程确保翻译质量后再进行耗时的语音合成
            """)

            # ========== 步骤1：生成和检查 ========== #
            with gr.Accordion("📝 步骤1: 生成字幕和质量检查", open=True) as step1_accordion:
                with gr.Row():
                    with gr.Column():
                        video_input = gr.File(
                            label="📹 上传视频",
                            file_types=[".mp4", ".avi", ".mov", ".mkv"]
                        )

                        with gr.Row():
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
                            choices=list_translation_contexts(),
                            value="general",
                            label="📚 翻译上下文",
                            info="选择领域专属的提示词和术语表"
                        )

                        source_lang = gr.Dropdown(
                            choices=["auto", "en", "zh", "pt", "ja"],
                            value="auto",
                            label="🗣️ 源语言"
                        )

                        step1_btn = gr.Button("▶️ 步骤1: 生成和检查", variant="primary", size="lg")

                    with gr.Column():
                        step1_status = gr.Textbox(
                            label="📊 质量报告",
                            lines=15,
                            max_lines=20
                        )

            # ========== 步骤2：审核界面 ========== #
            with gr.Accordion("✏️  步骤2: 审核和修改（可选）", open=False) as step2_accordion:
                gr.Markdown("""
                ### 翻译审核说明
                - ✅ 表示该片段无明显问题
                - ⚠️ 表示检测到潜在问题，请重点检查
                - 可以直接在"翻译"列中编辑文本
                - 编辑完成后点击"保存修改"
                - **无论缓存命中与否，都可以审核和修改翻译**
                """)

                review_dataframe = gr.Dataframe(
                    headers=["索引", "时间", "原文", "翻译", "状态", "问题说明"],
                    datatype=["number", "str", "str", "str", "str", "str"],
                    col_count=(6, "fixed"),
                    row_count=(10, "fixed"),  # 固定行数，不再动态
                    interactive=True,
                    wrap=True,
                    label="审核表格"
                )

                with gr.Row():
                    save_edits_btn = gr.Button("💾 保存修改", variant="secondary")
                    approve_btn = gr.Button("✅ 批准翻译，继续下一步", variant="primary")

                step2_status = gr.Textbox(label="操作状态", lines=2)

            # ========== 步骤3：语音合成 ========== #
            with gr.Accordion("🎤 步骤3: 语音合成和视频生成", open=False) as step3_accordion:
                gr.Markdown("""
                ### 提示
                - 如果质量检查未发现严重问题，可以直接点击"批准翻译"跳过步骤2
                - 如果缓存命中，步骤2仍然可用于查看和修改翻译
                - 批准后才能开始语音合成
                """)

                with gr.Row():
                    with gr.Column():
                        enable_voice = gr.Checkbox(
                            label="🎤 启用语音克隆",
                            value=False
                        )

                        reference_audio = gr.File(
                            label="🎵 参考音频（可选）",
                            file_types=[".wav", ".mp3"],
                            visible=False
                        )

                        enable_voice.change(
                            lambda x: gr.update(visible=x),
                            inputs=[enable_voice],
                            outputs=[reference_audio]
                        )

                        step3_btn = gr.Button("▶️ 步骤3: 开始合成", variant="primary", size="lg")

                    with gr.Column():
                        step3_status = gr.Textbox(label="📊 合成状态", lines=10)

                gr.Markdown("### 📦 输出文件")

                with gr.Row():
                    zh_srt_output = gr.File(label="中文字幕")
                    zh_en_ass_output = gr.File(label="双语字幕")

                with gr.Row():
                    voiced_output = gr.File(label="配音视频")
                    voiced_subtitled_output = gr.File(label="配音+字幕视频")

            # ========== 事件绑定 ========== #
            step1_btn.click(
                step1_generate_and_check,
                inputs=[
                    video_input, whisper_model, translation_model,
                    translation_context, source_lang
                ],
                outputs=[review_dataframe, step1_status, step2_accordion]
            ).then(
                # ⭐ 步骤1完成后，自动展开步骤2（用户可以选择是否使用）
                lambda: gr.update(open=True),
                inputs=[],
                outputs=[step2_accordion]
            )

            save_edits_btn.click(
                step2_review_and_edit,
                inputs=[review_dataframe],
                outputs=[step2_status]
            )

            approve_btn.click(
                step2_approve_translation,
                inputs=[],
                outputs=[step2_status, step3_accordion]
            ).then(
                # ⭐ 批准后自动展开步骤3
                lambda: gr.update(open=True),
                inputs=[],
                outputs=[step3_accordion]
            )

            step3_btn.click(
                step3_synthesize_video,
                inputs=[enable_voice, reference_audio],
                outputs=[
                    zh_srt_output, zh_en_ass_output,
                    voiced_output, voiced_subtitled_output,
                    step3_status
                ]
            )

        # ========== 上下文管理标签页 ========== #
        with gr.Tab("📚 翻译上下文管理"):
            gr.Markdown("""
            ### 自定义翻译上下文
            
            为不同领域配置专属的：
            - **系统提示词**: 指导AI如何翻译
            - **术语表**: 确保专业术语翻译准确
            
            示例领域：轮滑、编程、烹饪、医学等
            """)

            with gr.Row():
                with gr.Column():
                    context_name_input = gr.Textbox(
                        label="上下文名称",
                        placeholder="例如: cooking, programming, medicine"
                    )

                    existing_contexts = gr.Dropdown(
                        choices=list_translation_contexts(),
                        label="或选择现有上下文进行编辑"
                    )

                    load_context_btn = gr.Button("📂 加载上下文")

                    system_prompt_input = gr.Textbox(
                        label="系统提示词",
                        lines=8,
                        placeholder="描述如何翻译这个领域的内容..."
                    )

                    terminology_input = gr.Textbox(
                        label="术语表（每行一个，格式: 英文 = 中文）",
                        lines=10,
                        placeholder="inline skating = 轮滑\ncrossover = 交叉步"
                    )

                    save_context_btn = gr.Button("💾 保存上下文", variant="primary")

                with gr.Column():
                    context_status = gr.Textbox(label="操作状态", lines=3)

                    gr.Markdown("""
                    ### 💡 提示词编写建议
                    
                    1. **明确角色**: "你是XX领域的专业翻译"
                    2. **具体要求**: 列出3-5条翻译规则
                    3. **强调风格**: 正式/口语化、简洁/详细
                    4. **特殊处理**: 如何处理专业术语
                    
                    ### 📝 术语表格式
                    ```
                    source term = 目标术语
                    another term = 另一个术语
                    ```
                    """)

            load_context_btn.click(
                load_context_for_editing,
                inputs=[existing_contexts],
                outputs=[system_prompt_input, terminology_input, context_status]
            )

            save_context_btn.click(
                save_custom_context,
                inputs=[context_name_input, system_prompt_input, terminology_input],
                outputs=[context_status]
            )

        # ========== 架构说明 ========== #
        with gr.Tab("📚 改进说明"):
            gr.Markdown("""
            ## 🎯 质量优先的设计理念
            
            ### 问题分析
            1. **翻译质量不稳定**: 缺乏领域知识和术语规范
            2. **错误发现太晚**: 语音合成后才发现翻译问题，浪费时间
            3. **无法干预**: 自动化流程缺少人工检查点
            
            ### 解决方案
            
            #### 1. 翻译上下文系统
            ```python
            TranslationContext:
              - domain: 领域名称
              - system_prompt: 专业提示词
              - terminology: 术语表
            ```
            
            #### 2. 分阶段处理流程
            ```
            阶段1: ASR + 翻译 + 质量检查 (快速)
                ↓
            阶段2: 人工审核和修改 (按需)
                ↓
            阶段3: 语音合成 + 视频生成 (耗时)
            ```
            
            #### 3. 智能质量检查
            - 术语使用检查
            - 长度异常检测
            - 空白翻译检测
            - 严重度分级
            
            ### 优势
            - ✅ **早期发现问题**: 翻译完成后立即检查
            - ✅ **节省时间**: 避免对错误翻译进行语音合成
            - ✅ **灵活控制**: 可选择跳过审核或详细编辑
            - ✅ **持续改进**: 积累领域知识到上下文库
            
            ### 使用建议
            1. **首次处理新领域**: 使用默认上下文，记录常见问题
            2. **创建专属上下文**: 根据问题编写提示词和术语表
            3. **迭代优化**: 持续完善上下文配置
            4. **批量应用**: 同领域视频复用相同上下文
            """)

    return demo


def main():
    """启动改进的WebUI"""
    demo = build_improved_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False
    )


if __name__ == "__main__":
    main()