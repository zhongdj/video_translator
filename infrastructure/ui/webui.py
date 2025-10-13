"""
Infrastructure Layer - WebUI
基于 Gradio 的 Web 用户界面
"""
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio

from application.use_cases.synthesize_video_use_case import synthesize_video_use_case
# 导入应用层用例
from application.use_cases.test_component_use_case import *
from application.use_cases.batch_process import batch_process_use_case
from application.use_cases.clone_voice import clone_voice_use_case
from application.use_cases.generate_subtitles import generate_subtitles_use_case
# 导入领域层
from domain.entities import (
    Video, LanguageCode,
)
from infrastructure.config.dependency_injection import container


# ============== UI 处理函数 ============== #
def process_single_video_ui(
        video_file,
        whisper_model: str,
        translation_model: str,
        enable_voice: bool,
        reference_audio_file,
        source_language: str,
        progress=gr.Progress()
):
    """
    单视频处理 UI 处理函数 - 始终输出中英双语

    输出规范:
    - 中文字幕 (zh.srt)
    - 英文字幕 (en.srt)
    - 中英双语字幕 (zh_en.ass)
    - 中文配音视频（无字幕）
    - 中文配音+双语字幕视频
    - 原始音频+中文硬字幕视频
    """
    if not video_file:
        return None, None, None, None, None, "❌ 请上传视频"

    try:
        video_path = Path(video_file.name)

        # 创建输出目录
        output_dir = video_path.parent / "output"
        output_dir.mkdir(exist_ok=True)

        # 创建领域对象
        video = Video(
            path=video_path,
            duration=get_video_duration(video_path),
            has_audio=True
        )

        # 解析源语言
        src_lang = LanguageCode(source_language) if source_language and source_language != "auto" else None

        # 进度回调
        def prog_callback(p: float, desc: str):
            progress(p, desc=desc)

        print(f"\n{'=' * 60}")
        print(f"🎬 开始处理视频: {video.path.name}")
        print(f"{'=' * 60}")

        # ============== 1. 生成字幕（会同时生成中英文）============== #
        subtitle_result = generate_subtitles_use_case(
            video=video,
            asr_provider=container.get_asr(whisper_model),
            translation_provider=container.get_translator(),
            video_processor=container.video_processor,
            cache_repo=container.cache_repo,
            target_language=LanguageCode.CHINESE,
            source_language=src_lang,
            progress=lambda p, d: prog_callback(p * 0.5, d)
        )

        detected_lang = subtitle_result.detected_language
        zh_subtitle = subtitle_result.translated_subtitle  # 中文字幕

        print(f"\n📝 字幕生成完成:")
        print(f"   检测语言: {detected_lang.value}")
        print(f"   中文字幕: {len(zh_subtitle.segments)} 片段")

        # ============== 2. 从缓存获取英文字幕 ============== #
        # 因为 generate_subtitles_use_case 已经生成了英文版本并缓存
        from domain.services import calculate_cache_key
        cache_key = calculate_cache_key(
            video.path,
            "subtitles",
            {
                "target_language": LanguageCode.CHINESE.value,
                "source_language": src_lang.value if src_lang else "auto"
            }
        )

        cached = container.cache_repo.get(cache_key)
        en_segments = tuple(
            TextSegment(
                text=seg["text"],
                time_range=TimeRange(seg["start"], seg["end"]),
                language=LanguageCode.ENGLISH
            )
            for seg in cached.get("en_segments", [])
        )
        en_subtitle = Subtitle(en_segments, LanguageCode.ENGLISH)
        print(f"   英文字幕: {len(en_subtitle.segments)} 片段")

        # ============== 3. 语音克隆（使用中文字幕）============== #
        audio_track = None
        if enable_voice:
            ref_audio_path = Path(reference_audio_file.name) if reference_audio_file else None

            print(f"\n🎤 开始语音克隆（中文配音）:")
            if ref_audio_path:
                print(f"   参考音频: {ref_audio_path.name}")

            voice_result = clone_voice_use_case(
                video=video,
                subtitle=zh_subtitle,  # 使用中文字幕配音
                tts_provider=container.get_tts(),
                video_processor=container.video_processor,
                cache_repo=container.cache_repo,
                reference_audio_path=ref_audio_path,
                progress=lambda p, d: prog_callback(0.5 + p * 0.3, d)
            )
            audio_track = voice_result.audio_track
            print(f"✅ 中文配音完成")

        # ============== 4. 合成视频 ============== #
        # 创建中英双语字幕（中文在上，英文在下）
        from domain.services import merge_bilingual_subtitles
        zh_en_subtitle = merge_bilingual_subtitles(
            zh_subtitle,  # 中文（上）
            en_subtitle  # 英文（下）
        )
        print(f"\n📝 中英双语字幕创建完成")

        synthesis_result = synthesize_video_use_case(
            video=video,
            subtitles=(
                zh_subtitle,  # zh.srt / zh.ass
                en_subtitle,  # en.srt / en.ass
                zh_en_subtitle  # zh_en.ass (双语)
            ),
            audio_track=audio_track,
            video_processor=container.video_processor,
            subtitle_writer=container.subtitle_writer,
            output_dir=output_dir,
            formats=("srt", "ass"),
            burn_subtitles=True,
            progress=lambda p, d: prog_callback(0.8 + p * 0.2, d)
        )

        # ============== 5. 查找输出文件 ============== #
        print(f"\n🔍 查找输出文件:")
        for path in synthesis_result.output_paths:
            print(f"   - {path.name}")

        # 智能查找文件
        def find_file(patterns: list[str], suffix: str = None) -> Optional[str]:
            for pattern in patterns:
                matches = [
                    p for p in synthesis_result.output_paths
                    if pattern in p.name and (suffix is None or p.suffix == suffix)
                ]
                if matches:
                    print(f"   ✅ {pattern}: {matches[0].name}")
                    return str(matches[0])
            print(f"   ⚠️  未找到匹配 {patterns}")
            return None

        # 查找各类文件
        zh_srt = find_file(['zh.srt', 'translated.zh'], '.srt')
        en_srt = find_file(['en.srt', 'translated.en'], '.srt')
        zh_en_ass = find_file(['zh_en', 'bilingual'], '.ass')

        # 配音视频（纯配音，无字幕）
        voiced_video = find_file(['_voiced.mp4']) if audio_track else None

        # 配音+双语字幕视频
        voiced_subtitled_video = find_file(['_voiced_subtitled.mp4']) if audio_track else None

        # 原始视频+中文硬字幕
        subtitled_video = find_file(['_subtitled.mp4'])

        # ============== 6. 生成状态报告 ============== #
        status_lines = [
            f"✅ 处理完成！耗时 {synthesis_result.processing_time:.1f} 秒",
            f"",
            f"📊 字幕信息:",
            f"   检测语言: {detected_lang.value}",
            f"   中文字幕: {len(zh_subtitle.segments)} 片段",
            f"   英文字幕: {len(en_subtitle.segments)} 片段",
            f"",
            f"📦 生成文件: {len(synthesis_result.output_paths)} 个"
        ]

        if subtitle_result.cache_hit:
            status_lines.append("💾 字幕缓存命中")

        # 文件检查
        file_status = []
        if zh_srt:
            file_status.append(f"✅ 中文字幕")
        else:
            file_status.append(f"❌ 中文字幕缺失")

        if en_srt:
            file_status.append(f"✅ 英文字幕")
        else:
            file_status.append(f"❌ 英文字幕缺失")

        if zh_en_ass:
            file_status.append(f"✅ 中英双语字幕")
        else:
            file_status.append(f"❌ 中英双语字幕缺失")

        if voiced_video:
            file_status.append(f"✅ 中文配音视频（无字幕）")
        elif enable_voice:
            file_status.append(f"❌ 中文配音视频缺失")

        if voiced_subtitled_video:
            file_status.append(f"✅ 中文配音+双语字幕视频")
        elif enable_voice:
            file_status.append(f"❌ 配音字幕视频缺失")

        if subtitled_video:
            file_status.append(f"✅ 原始音频+中文硬字幕")
        else:
            file_status.append(f"❌ 硬字幕视频缺失")

        if file_status:
            status_lines.append("")
            status_lines.append("📁 文件状态:")
            status_lines.extend([f"   {s}" for s in file_status])

        status = "\n".join(status_lines)
        print(f"\n{status}")
        print(f"{'=' * 60}\n")

        return zh_srt, en_srt, zh_en_ass, voiced_video, subtitled_video, status

    except Exception as e:
        import traceback
        error_msg = f"❌ 处理失败: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, None, None, error_msg


def batch_process_videos_ui(
        video_files,
        whisper_model: str,
        translation_model: str,
        enable_voice: bool,
        reference_audio_file: Path,
        progress=gr.Progress()
):
    """批量处理 UI 处理函数"""
    if not video_files:
        return None, "❌ 请上传视频文件"

    try:
        # 创建临时输出目录
        output_dir = Path(tempfile.mkdtemp(prefix="video_batch_"))

        # 转换为 Video 对象
        videos = []
        for vf in video_files:
            video_path = Path(vf.name)
            video = Video(
                path=video_path,
                duration=get_video_duration(video_path),
                has_audio=True
            )
            videos.append(video)

        # 进度回调
        log_lines = []

        def log_callback(line: str):
            log_lines.append(line)
            if len(log_lines) > 100:
                log_lines.pop(0)

        def prog_callback(p: float, desc: str):
            progress(p, desc=desc)
            log_callback(desc)

        # 执行批量处理
        results = batch_process_use_case(
            videos=tuple(videos),
            asr_provider=container.get_asr(whisper_model),
            translation_provider=container.get_translator(),
            tts_provider=container.get_tts() if enable_voice else None,
            video_processor=container.video_processor,
            subtitle_writer=container.subtitle_writer,
            cache_repo=container.cache_repo,
            output_dir=output_dir,
            enable_voice_cloning=enable_voice,
            reference_audio_file=reference_audio_file,
            progress=prog_callback
        )

        # 打包结果
        import zipfile
        zip_path = output_dir / "batch_results.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for result in results:
                for out_path in result.output_paths:
                    if out_path.exists():
                        zipf.write(out_path, out_path.name)

        log_callback(f"📦 打包完成！共处理 {len(results)} 个视频")
        final_log = "\n".join(log_lines)

        return str(zip_path), final_log

    except Exception as e:
        import traceback
        error_msg = f"❌ 批量处理失败: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


def test_asr_ui(audio_file, whisper_model: str, language: str, progress=gr.Progress()):
    """测试 ASR 组件"""
    if not audio_file:
        return "❌ 请上传音频文件"

    try:
        progress(0.1, "检查设备环境...")

        # 添加设备检查
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            device_info = f"GPU ({torch.cuda.get_device_name()})"
        else:
            device = "cpu"
            device_info = "CPU"

        progress(0.2, f"使用设备: {device_info}")

        # 确保容器使用正确的设备
        asr_provider = container.get_asr(whisper_model, device=device)

        progress(0.3, "语音识别中...")

        result = test_component_use_case(
            component_type="asr",
            test_input={
                "audio_path": audio_file.name,
                "language": language
            },
            asr_provider=asr_provider,
            progress=None
        )

        # 格式化输出
        output = f"""
✅ ASR 测试完成 ({device_info})

检测语言: {result['detected_language']}
片段数量: {result['total_segments']}

前 5 个片段:
"""
        for i, seg in enumerate(result['segments'][:5], 1):
            output += f"\n{i}. [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}"

        return output

    except Exception as e:
        return f"❌ 测试失败: {str(e)}"

def test_translation_ui(text: str, source_lang: str, target_lang: str, progress=gr.Progress()):
    """测试翻译组件"""
    if not text.strip():
        return "❌ 请输入文本"

    try:
        progress(0.5, "翻译中...")

        result = test_component_use_case(
            component_type="translation",
            test_input={
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang
            },
            translation_provider=container.get_translator(),
            progress=lambda p, d: progress(p, d)
        )

        output = f"""
✅ 翻译测试完成

原文 ({result['source_language']}): {result['original']}

译文 ({result['target_language']}): {result['translated']}
"""
        return output

    except Exception as e:
        return f"❌ 测试失败: {str(e)}"


def test_tts_ui(text: str, reference_audio, progress=gr.Progress()):
    """测试 TTS 组件"""
    if not text.strip():
        return None, "❌ 请输入文本"

    if not reference_audio:
        return None, "❌ 请上传参考音频"

    try:
        progress(0.5, "合成语音中...")

        result = test_component_use_case(
            component_type="tts",
            test_input={
                "text": text,
                "reference_audio": reference_audio.name,
                "target_duration": None
            },
            tts_provider=container.get_tts(),
            progress=lambda p, d: progress(p, d)
        )

        # 保存音频到临时文件
        import tempfile
        import torchaudio
        import torch

        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_audio.name)
        print(f"temp_path:{temp_path}")
        temp_audio.close()

        # 这里需要从 result 获取音频数据并保存
        # 简化实现，实际需要根据 F5-TTS 返回格式调整

        status = f"""
✅ TTS 测试完成

文本: {result['text']}
时长: {result['duration']:.2f} 秒
采样率: {result['sample_rate']} Hz
"""
        return str(temp_path), status

    except Exception as e:
        return None, f"❌ 测试失败: {str(e)}"


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


# 在 webui.py 中添加以下函数

def test_tts_simple(
        text: str,
        reference_audio,
        speed: float = 1.0,
        progress=gr.Progress()
):
    """简化的 TTS 测试界面"""
    if not text.strip():
        return None, "❌ 请输入文本"

    if not reference_audio:
        return None, "❌ 请上传参考音频"

    try:
        progress(0.3, "初始化 TTS 引擎...")

        # 创建 IndexTTSAdapter 实例
        from infrastructure.adapters.tts.indextts_adapter import IndexTTSAdapter
        tts_adapter = IndexTTSAdapter()
        tts_adapter.update_config(speed=speed)

        progress(0.5, "合成语音中...")

        # 创建 VoiceProfile
        from domain.entities import VoiceProfile
        voice_profile = VoiceProfile(
            reference_audio_path=Path(reference_audio.name),
            language=LanguageCode.CHINESE,
            duration=0.1
        )

        # 执行合成
        audio_sample = tts_adapter.synthesize(
            text=text,
            voice_profile=voice_profile
        )

        progress(0.8, "保存音频文件...")

        # 保存音频到临时文件
        import tempfile
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_audio.name)
        temp_audio.close()

        # 使用 torchaudio 保存
        audio_array = np.array(audio_sample.samples, dtype=np.float32)
        torchaudio.save(
            str(temp_path),
            torch.from_numpy(audio_array).unsqueeze(0),
            audio_sample.sample_rate
        )

        status = f"""
✅ TTS 测试完成

文本: {text}
时长: {audio_sample.duration:.2f} 秒
采样率: {audio_sample.sample_rate} Hz
语速: {speed}
"""

        return str(temp_path), status

    except Exception as e:
        import traceback
        error_msg = f"❌ TTS 测试失败: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg
# ============== Gradio UI 构建 ============== #

def build_ui():
    """构建 Gradio UI"""

    with gr.Blocks(
            title="视频翻译工厂 - 洋葱架构版",
            css=".gradio-container {max-width: 1400px !important}"
    ) as demo:
        gr.Markdown("""
        # 🎬 视频翻译工厂（洋葱架构版）

        基于 **领域驱动设计** + **洋葱架构** + **函数式编程** 构建

        ### ✨ 核心特性
        - 🎯 纯函数核心，易于测试和维护
        - 🔌 可插拔组件，支持多种模型
        - 💾 智能缓存，断点续传
        - 🎤 F5-TTS 语音克隆（IndexTTS 2.0）
        """)

        # 在 build_ui() 函数中，更新单视频处理的输出部分

        with gr.Tab("🎬 单视频处理"):
            gr.Markdown("""
            ### 处理流程
            1. 上传视频
            2. 选择模型配置
            3. （可选）上传参考音频进行语音克隆
            4. 开始处理

            ### 输出说明
            - 始终生成**中文**和**英文**字幕
            - 如果启用语音克隆，生成**中文配音视频**
            - 配音视频会自动烧录**中英双语字幕** ⭐
            """)

            with gr.Row():
                with gr.Column():
                    video_input = gr.File(
                        label="📹 上传视频",
                        file_types=[".mp4", ".avi", ".mov", ".mkv"]
                    )

                    with gr.Row():
                        whisper_dropdown = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large", "large-v3"],
                            value="medium",
                            label="🎙️ Whisper 模型"
                        )

                        translation_dropdown = gr.Dropdown(
                            choices=["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"],
                            value="Qwen/Qwen2.5-7B",
                            label="🌐 翻译模型"
                        )

                    with gr.Row():
                        source_lang_input = gr.Dropdown(
                            choices=["auto", "en", "zh", "pt", "ja", "ko"],
                            value="auto",
                            label="🗣️ 源语言"
                        )

                    enable_voice_checkbox = gr.Checkbox(
                        label="🎤 启用语音克隆（生成中文配音）",
                        value=False
                    )

                    reference_audio_input = gr.File(
                        label="🎵 参考音频（可选，留空则自动提取）",
                        file_types=[".wav", ".mp3"],
                        visible=False
                    )

                    # 动态显示参考音频上传
                    enable_voice_checkbox.change(
                        lambda x: gr.update(visible=x),
                        inputs=[enable_voice_checkbox],
                        outputs=[reference_audio_input]
                    )

                    process_btn = gr.Button("▶️ 开始处理", variant="primary", size="lg")

                with gr.Column():
                    status_output = gr.Textbox(label="📊 处理状态", lines=8)

                    gr.Markdown("### 📝 字幕文件")
                    with gr.Row():
                        zh_srt_output = gr.File(label="中文字幕 (SRT)")
                        en_srt_output = gr.File(label="英文字幕 (SRT)")

                    zh_en_ass_output = gr.File(label="中英双语字幕 (ASS)")

                    gr.Markdown("### 🎬 视频文件")

                    with gr.Accordion("💡 查看文件说明", open=False):
                        gr.Markdown("""
                        - **配音视频（无字幕）**: 仅含中文配音，无烧录字幕
                        - **配音+双语字幕视频** ⭐: 中文配音 + 中英双语硬字幕（推荐观看）
                        - **原音+中文字幕视频**: 保留原始音频 + 中文硬字幕
                        """)

                    voiced_output = gr.File(label="中文配音视频（无字幕）")
                    voiced_subtitled_output = gr.File(label="⭐ 中文配音+双语字幕视频（推荐）")
                    subtitled_output = gr.File(label="原音+中文硬字幕视频")

            process_btn.click(
                process_single_video_ui,
                inputs=[
                    video_input,
                    whisper_dropdown,
                    translation_dropdown,
                    enable_voice_checkbox,
                    reference_audio_input,
                    source_lang_input
                ],
                outputs=[
                    zh_srt_output,  # 中文字幕
                    en_srt_output,  # 英文字幕
                    zh_en_ass_output,  # 双语字幕
                    voiced_output,  # 配音视频（无字幕）
                    voiced_subtitled_output,  # 配音+双语字幕（推荐）
                    #subtitled_output,  # 原音+中文字幕
                    status_output  # 状态信息
                ]
            )

        with gr.Tab("🎞️ 批量处理"):
            gr.Markdown("""
            ### 批量处理说明
            - 支持同时处理多个视频
            - 模型只加载一次，效率更高
            - 所有结果打包为 ZIP 下载
            """)

            with gr.Row():
                with gr.Column():
                    batch_videos = gr.File(
                        label="📹 上传多个视频",
                        file_count="multiple",
                        file_types=[".mp4", ".avi", ".mov", ".mkv"]
                    )

                    with gr.Row():
                        batch_whisper = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large"],
                            value="medium",
                            label="🎙️ Whisper 模型"
                        )

                        batch_translation = gr.Dropdown(
                            choices=["Qwen/Qwen2.5-7B"],
                            value="Qwen/Qwen2.5-7B",
                            label="🌐 翻译模型"
                        )

                    with gr.Row():
                        batch_enable_voice = gr.Checkbox(
                            label="🎤 启用语音克隆",
                            value=False
                        )
                        reference_audio_input = gr.File(
                            label="🎵 参考音频（可选，留空则自动提取）",
                            file_types=[".wav", ".mp3"],
                            visible=False
                        )

                    # 动态显示参考音频上传
                    batch_enable_voice.change(
                        lambda x: gr.update(visible=x),
                        inputs=[batch_enable_voice],
                        outputs=[reference_audio_input]
                    )

                    batch_btn = gr.Button("▶️ 开始批量处理", variant="primary", size="lg")

                with gr.Column():
                    batch_log = gr.Textbox(label="📊 处理日志", lines=15)
                    batch_output = gr.File(label="📦 下载结果（ZIP）")

            batch_btn.click(
                batch_process_videos_ui,
                inputs=[
                    batch_videos,
                    batch_whisper,
                    batch_translation,
                    batch_enable_voice,
                    reference_audio_input
                ],
                outputs=[batch_output, batch_log]
            )

        with gr.Tab("🧪 组件测试"):
            gr.Markdown("""
            ### 组件测试工具
            在集成到主流程前，先测试各个组件的效果和参数
            """)

            with gr.Tab("🎙️ 测试 ASR"):
                with gr.Row():
                    with gr.Column():
                        test_asr_audio = gr.File(label="上传音频", file_types=[".wav", ".mp3"])
                        test_asr_model = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large"],
                            value="medium",
                            label="Whisper 模型"
                        )
                        test_asr_lang = gr.Dropdown(
                            choices=["auto", "en", "zh", "pt"],
                            value="auto",
                            label="语言"
                        )
                        test_asr_btn = gr.Button("测试 ASR")

                    with gr.Column():
                        test_asr_output = gr.Textbox(label="ASR 结果", lines=15)

                test_asr_btn.click(
                    test_asr_ui,
                    inputs=[test_asr_audio, test_asr_model, test_asr_lang],
                    outputs=[test_asr_output]
                )

            with gr.Tab("🌐 测试翻译"):
                with gr.Row():
                    with gr.Column():
                        test_trans_text = gr.Textbox(label="输入文本", lines=5)
                        test_trans_src = gr.Dropdown(
                            choices=["en", "zh", "pt"],
                            value="en",
                            label="源语言"
                        )
                        test_trans_tgt = gr.Dropdown(
                            choices=["en", "zh", "pt"],
                            value="zh",
                            label="目标语言"
                        )
                        test_trans_btn = gr.Button("测试翻译")

                    with gr.Column():
                        test_trans_output = gr.Textbox(label="翻译结果", lines=10)

                test_trans_btn.click(
                    test_translation_ui,
                    inputs=[test_trans_text, test_trans_src, test_trans_tgt],
                    outputs=[test_trans_output]
                )

            with gr.Tab("🎤 测试 TTS"):
                gr.Markdown("""
                ### IndexTTS2 语音合成测试
                基于声音参考进行语音合成
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        test_tts_text = gr.Textbox(
                            label="输入文本",
                            lines=3,
                            placeholder="请输入要合成的文本内容...",
                            value="你好，这是一个语音合成测试。"
                        )
                        test_tts_ref = gr.File(
                            label="参考音频",
                            file_types=[".wav", ".mp3"]
                        )

                        with gr.Accordion("⚙️ 参数设置", open=False):
                            speed_slider = gr.Slider(
                                minimum=0.5, maximum=2.0, value=1.0,
                                label="语速",
                                info="调整合成语音的速度"
                            )

                        test_tts_btn = gr.Button("🎵 测试语音合成", variant="primary")

                    with gr.Column(scale=1):
                        test_tts_audio = gr.Audio(
                            label="合成音频",
                            type="filepath",
                            interactive=False
                        )
                        test_tts_status = gr.Textbox(
                            label="合成状态",
                            lines=6,
                            max_lines=8
                        )

                test_tts_btn.click(
                    test_tts_simple,
                    inputs=[
                        test_tts_text,
                        test_tts_ref,
                        speed_slider
                    ],
                    outputs=[test_tts_audio, test_tts_status]
                )

        with gr.Tab("📚 架构说明"):
            gr.Markdown("""
            ## 🏗️ 洋葱架构设计

            ### 层次结构
            ```
            ┌─────────────────────────────────────┐
            │   Infrastructure Layer (外层)        │
            │   - WebUI (本界面)                   │
            │   - 模型适配器 (Whisper, Qwen, F5-TTS)│
            │   - 文件系统、缓存                    │
            └──────────────┬──────────────────────┘
                           │ 实现接口
            ┌──────────────▼──────────────────────┐
            │   Application Layer (应用层)         │
            │   - 用例编排（纯函数）                │
            │   - 业务流程定义                      │
            └──────────────┬──────────────────────┘
                           │ 使用
            ┌──────────────▼──────────────────────┐
            │   Domain Layer (领域核心)            │
            │   - 实体、值对象（不可变）            │
            │   - 领域服务（纯函数）                │
            │   - 接口定义（Port）                  │
            └──────────────────────────────────────┘
            ```

            ### 核心原则
            1. **依赖倒置**: 外层依赖内层，内层定义接口
            2. **纯函数**: 核心层和应用层无副作用
            3. **不可变性**: 领域对象使用 `@dataclass(frozen=True)`
            4. **可测试**: 通过接口注入，易于 mock 测试

            ### 优势
            - ✅ 业务逻辑与技术实现解耦
            - ✅ 轻松替换底层实现（如更换 TTS 引擎）
            - ✅ 高可测试性
            - ✅ 清晰的代码组织
            """)

    return demo


# ============== 启动应用 ============== #

def main():
    """启动 WebUI"""
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False
    )


if __name__ == "__main__":
    main()
