from .entities import *

def merge_bilingual_subtitles(
        primary: Subtitle,
        secondary: Subtitle
) -> Subtitle:
    """
    合并双语字幕（纯函数）

    Args:
        primary: 主字幕（通常是翻译后的）
        secondary: 次字幕（通常是原文）

    Returns:
        合并后的双语字幕
    """
    if len(primary.segments) != len(secondary.segments):
        raise ValueError("Subtitle segments count mismatch")

    merged_segments = []
    for p_seg, s_seg in zip(primary.segments, secondary.segments):
        # 合并文本：主字幕在上，次字幕在下
        merged_text = f"{p_seg.text}\n{s_seg.text}"
        merged_segment = TextSegment(
            text=merged_text,
            time_range=p_seg.time_range,
            language=primary.language  # 使用主字幕语言
        )
        merged_segments.append(merged_segment)

    return Subtitle(
        segments=tuple(merged_segments),
        language=primary.language
    )


def calculate_speed_adjustment(
        current_duration: float,
        target_duration: float
) -> tuple[float, str]:
    """
    计算语速调整策略（纯函数）

    Returns:
        (speed_ratio, strategy)
        - speed_ratio: 速度比例 (0.8-1.2)
        - strategy: "speed_up" | "slow_down" | "pad" | "trim" | "keep"
    """
    if target_duration <= 0:
        raise ValueError("target_duration must be positive")

    ratio = current_duration / target_duration

    if ratio < 0.8:
        # 太快，需要减速
        return 0.9, "slow_down"
    elif ratio > 1.2:
        # 太慢，需要加速
        return 1.1, "speed_up"
    elif 0.95 <= ratio <= 1.05:
        # 差异很小，直接填充或裁剪
        return 1.0, "pad" if ratio < 1.0 else "trim"
    else:
        # 可接受范围，保持原样
        return 1.0, "keep"


def calculate_cache_key(
        video_path: Path,
        operation: str,
        params: dict
) -> str:
    """
    计算缓存键（纯函数）

    Args:
        video_path: 视频路径
        operation: 操作名称
        params: 参数字典

    Returns:
        缓存键字符串
    """
    import hashlib
    import json

    # 使用文件名和大小生成视频标识
    video_id = f"{video_path.name}_{video_path.stat().st_size if video_path.exists() else 0}"

    # 序列化参数
    params_str = json.dumps(params, sort_keys=True)

    # 生成哈希
    combined = f"{video_id}_{operation}_{params_str}"
    cache_key = hashlib.md5(combined.encode()).hexdigest()

    return f"{operation}_{cache_key[:16]}"


def split_audio_by_segments(
        audio: AudioSample,
        segments: tuple[TextSegment, ...]
) -> tuple[tuple[AudioSample, TextSegment], ...]:
    """
    按文本片段切割音频（纯函数）

    Returns:
        (AudioSample, TextSegment) 元组列表
    """
    results = []

    for segment in segments:
        start_sample = int(segment.time_range.start_seconds * audio.sample_rate)
        end_sample = int(segment.time_range.end_seconds * audio.sample_rate)

        # 切割音频
        segment_samples = audio.samples[start_sample:end_sample]

        segment_audio = AudioSample(
            samples=tuple(segment_samples),
            sample_rate=audio.sample_rate
        )

        results.append((segment_audio, segment))

    return tuple(results)


def validate_video_path(path: Path) -> bool:
    """验证视频路径（纯函数）"""
    if not isinstance(path, Path):
        return False

    # 检查扩展名
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    return path.suffix.lower() in valid_extensions


def estimate_processing_time(
        video_duration: float,
        enable_voice_cloning: bool,
        model_size: str
) -> float:
    """
    估算处理时间（纯函数）

    Returns:
        预估时间（秒）
    """
    # 基础时间：每分钟视频约需 30 秒处理
    base_time = video_duration * 0.5

    # 模型大小系数
    model_factors = {
        "tiny": 0.3,
        "small": 0.5,
        "medium": 1.0,
        "large": 2.0,
        "large-v3": 2.5
    }
    model_factor = model_factors.get(model_size, 1.0)

    # 语音克隆额外时间
    voice_factor = 2.5 if enable_voice_cloning else 1.0

    return base_time * model_factor * voice_factor