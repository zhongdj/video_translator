"""
Domain Layer - IndexTTS2 Token 计算服务
精确控制合成音频时长
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TokenCountConfig:
    """Token 计数配置（值对象）"""
    base_token_per_second: float = 12.8  # 中文正常语速基准
    min_token_per_second: float = 10.0  # 慢速阈值
    max_token_per_second: float = 16.0  # 快速阈值

    def __post_init__(self):
        if not (self.min_token_per_second <= self.base_token_per_second <= self.max_token_per_second):
            raise ValueError("token_per_second 必须在 min 和 max 之间")


@dataclass(frozen=True)
class TokenCountResult:
    """Token 计数结果（值对象）"""
    token_count: int
    target_duration: float
    effective_token_per_second: float
    adjustment_reason: Optional[str] = None


def calculate_token_count(
        target_duration: float,
        config: Optional[TokenCountConfig] = None,
        reference_speed: Optional[float] = None
) -> TokenCountResult:
    """
    计算目标时长对应的 token 数（纯函数）

    Args:
        target_duration: 目标时长（秒）
        config: Token 配置（可选）
        reference_speed: 参考音频语速（字/分，可选）

    Returns:
        TokenCountResult: 包含 token_count 和元数据

    算法：
    1. 基准公式：token_count = target_duration × token_per_second
    2. 语速校正：根据参考音频语速微调 token_per_second
    3. 边界保护：确保 token_count 在合理范围内
    """
    if config is None:
        config = TokenCountConfig()

    if target_duration <= 0:
        raise ValueError(f"target_duration 必须为正数，当前值: {target_duration}")

    # 1. 基础计算
    base_token_per_second = config.base_token_per_second
    adjustment_reason = None

    # 2. 语速校正（如果提供了参考音频语速）
    if reference_speed is not None:
        effective_tps = _adjust_for_reference_speed(
            base_token_per_second,
            reference_speed,
            config
        )

        if abs(effective_tps - base_token_per_second) > 0.5:
            adjustment_reason = (
                f"参考语速 {reference_speed:.0f} 字/分 "
                f"→ 调整 token/s 至 {effective_tps:.2f}"
            )
    else:
        effective_tps = base_token_per_second

    # 3. 计算 token_count（取整）
    raw_token_count = target_duration * effective_tps
    token_count = int(round(raw_token_count))

    # 4. 边界保护（避免极端值）
    min_tokens = max(1, int(target_duration * config.min_token_per_second))
    max_tokens = int(target_duration * config.max_token_per_second * 1.2)

    if token_count < min_tokens:
        token_count = min_tokens
        adjustment_reason = (adjustment_reason or "") + f" [下限保护: {min_tokens}]"
    elif token_count > max_tokens:
        token_count = max_tokens
        adjustment_reason = (adjustment_reason or "") + f" [上限保护: {max_tokens}]"

    return TokenCountResult(
        token_count=token_count,
        target_duration=target_duration,
        effective_token_per_second=effective_tps,
        adjustment_reason=adjustment_reason
    )


def _adjust_for_reference_speed(
        base_tps: float,
        reference_speed: float,
        config: TokenCountConfig
) -> float:
    """
    根据参考音频语速调整 token_per_second

    Args:
        base_tps: 基准 token/s
        reference_speed: 参考语速（字/分）
        config: 配置

    Returns:
        调整后的 token_per_second

    规则：
    - 正常语速范围：120-300 字/分 → 不调整
    - 慢速 <120 → 乘以 0.9
    - 快速 >300 → 乘以 1.1
    """
    if reference_speed < 120:
        # 慢速：减少 token 数，让模型有更多时间
        adjusted = base_tps * 0.9
    elif reference_speed > 300:
        # 快速：增加 token 数，压缩节奏
        adjusted = base_tps * 1.1
    else:
        # 正常语速：不调整
        adjusted = base_tps

    # 确保在合理范围内
    return max(
        config.min_token_per_second,
        min(adjusted, config.max_token_per_second)
    )


def estimate_reference_speed(
        text: str,
        audio_duration: float
) -> float:
    """
    从参考音频估算语速（纯函数）

    Args:
        text: 参考音频的文本
        audio_duration: 参考音频时长（秒）

    Returns:
        语速（字/分）
    """
    if audio_duration <= 0:
        raise ValueError("audio_duration 必须为正数")

    char_count = len(text.strip())
    chars_per_second = char_count / audio_duration
    chars_per_minute = chars_per_second * 60

    return chars_per_minute


def calibrate_token_per_second(
        actual_token_count: int,
        actual_duration: float
) -> float:
    """
    校准 token_per_second（用于模型微调）

    Args:
        actual_token_count: 实际使用的 token 数
        actual_duration: 实际生成的音频时长（秒）

    Returns:
        校准后的 token_per_second

    用法示例：
    ```python
    # 第一次合成
    result = calculate_token_count(target_duration=5.0)
    wav = tts.synthesize(text="测试文本", token_count=result.token_count)

    # 测量实际时长
    actual_duration = len(wav) / sample_rate  # 假设是 4.8s

    # 校准
    calibrated_tps = calibrate_token_per_second(
        actual_token_count=result.token_count,
        actual_duration=actual_duration
    )

    # 后续使用校准值
    config = TokenCountConfig(base_token_per_second=calibrated_tps)
    ```
    """
    if actual_duration <= 0:
        raise ValueError("actual_duration 必须为正数")

    return actual_token_count / actual_duration


# ============== 批量处理优化 ============== #

def calculate_batch_token_counts(
        target_durations: list[float],
        config: Optional[TokenCountConfig] = None,
        reference_speed: Optional[float] = None
) -> list[TokenCountResult]:
    """
    批量计算 token_count（纯函数）

    Args:
        target_durations: 目标时长列表
        config: 配置
        reference_speed: 参考语速（所有片段共享）

    Returns:
        TokenCountResult 列表
    """
    return [
        calculate_token_count(duration, config, reference_speed)
        for duration in target_durations
    ]


def suggest_token_config_from_sample(
        sample_text: str,
        sample_duration: float,
        sample_token_count: int
) -> TokenCountConfig:
    """
    从实际样本推荐 TokenCountConfig

    Args:
        sample_text: 样本文本
        sample_duration: 样本实际时长
        sample_token_count: 样本使用的 token 数

    Returns:
        推荐的配置
    """
    # 计算实际 token/s
    actual_tps = calibrate_token_per_second(
        sample_token_count,
        sample_duration
    )

    # 估算语速
    reference_speed = estimate_reference_speed(
        sample_text,
        sample_duration
    )

    # 构建配置
    return TokenCountConfig(
        base_token_per_second=actual_tps,
        min_token_per_second=actual_tps * 0.8,
        max_token_per_second=actual_tps * 1.2
    )