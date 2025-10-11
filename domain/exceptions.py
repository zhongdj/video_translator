class DomainException(Exception):
    """领域异常基类"""
    pass


class InvalidTimeRangeError(DomainException):
    """无效时间范围异常"""
    pass


class InvalidSubtitleError(DomainException):
    """无效字幕异常"""
    pass


class InvalidAudioError(DomainException):
    """无效音频异常"""
    pass


class InvalidVideoError(DomainException):
    """无效视频异常"""
    pass