from .entities import *


"""
修复缓存键生成 - 确保稳定性和可重现性
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def calculate_cache_key_stable(
        video_path: Path,
        operation: str,
        params: Optional[Dict[str, Any]] = None
) -> str:
    """
    稳定的缓存键生成算法

    关键原则：
    1. 使用文件内容哈希（而非路径）
    2. 确保字典顺序一致（sorted）
    3. 排除易变的参数（如时间戳）

    Args:
        video_path: 视频文件路径
        operation: 操作类型（如 "subtitles_v2"）
        params: 额外参数（必须是可序列化的）

    Returns:
        稳定的缓存键（MD5哈希）
    """

    # 1. 文件内容哈希（最稳定）
    file_hash = _get_file_content_hash(video_path)

    # 2. 参数标准化（确保顺序一致）
    normalized_params = _normalize_params(params or {})

    # 3. 组合生成最终缓存键
    cache_key_components = {
        "file_hash": file_hash,
        "operation": operation,
        "params": normalized_params
    }

    # 使用 JSON 序列化（sort_keys=True 确保顺序）
    cache_key_string = json.dumps(cache_key_components, sort_keys=True)

    # 返回 MD5 哈希
    return hashlib.md5(cache_key_string.encode('utf-8')).hexdigest()


def _get_file_content_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """
    计算文件内容哈希（只读取前后各1MB，节省时间）

    对于视频文件，读取：
    - 前 1MB（包含元数据）
    - 后 1MB（包含尾部信息）
    - 文件大小
    """
    hasher = hashlib.md5()
    file_size = file_path.stat().st_size

    # 添加文件大小到哈希（快速检查）
    hasher.update(str(file_size).encode())

    # 小文件：全部读取
    if file_size <= 2 * 1024 * 1024:  # 2MB
        hasher.update(file_path.read_bytes())
        return hasher.hexdigest()

    # 大文件：只读取头尾
    sample_size = 1024 * 1024  # 1MB

    with open(file_path, 'rb') as f:
        # 读取前 1MB
        hasher.update(f.read(sample_size))

        # 跳到文件末尾前 1MB
        f.seek(file_size - sample_size)
        hasher.update(f.read(sample_size))

    return hasher.hexdigest()


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化参数字典

    1. 排除不稳定的键（时间戳等）
    2. 递归处理嵌套字典
    3. 确保所有值都是可序列化的
    """

    # 要排除的不稳定参数
    EXCLUDE_KEYS = {
        'timestamp',
        'created_at',
        'updated_at',
        'random_seed',
        'session_id',
        'request_id'
    }

    normalized = {}

    for key, value in params.items():
        # 跳过不稳定的键
        if key in EXCLUDE_KEYS:
            continue

        # 递归处理嵌套字典
        if isinstance(value, dict):
            normalized[key] = _normalize_params(value)

        # 转换 Path 为字符串
        elif isinstance(value, Path):
            normalized[key] = str(value)

        # 处理对象（提取关键属性）
        elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool)):
            # 对于自定义对象，只保留关键属性
            if hasattr(value, 'to_dict'):
                normalized[key] = value.to_dict()
            elif hasattr(value, 'domain'):  # TranslationContext
                normalized[key] = getattr(value, 'domain', str(value))
            else:
                normalized[key] = str(value)

        # 基础类型直接保留
        else:
            normalized[key] = value

    return normalized

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
    # import hashlib
    # import json
    #
    # # 使用文件名和大小生成视频标识
    # video_id = f"{video_path.name}_{video_path.stat().st_size if video_path.exists() else 0}"
    #
    # # 序列化参数
    # params_str = json.dumps(params, sort_keys=True)
    #
    # # 生成哈希
    # combined = f"{video_id}_{operation}_{params_str}"
    # cache_key = hashlib.md5(combined.encode()).hexdigest()
    #
    # # return f"{operation}_{cache_key[:16]}"

    return calculate_cache_key_stable(video_path, operation, params)

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