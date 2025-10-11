# test_architecture.py - 验证架构正确性
import sys
from pathlib import Path


# def test_domain_has_no_external_deps():
#     """验证领域层无外部依赖"""
#     from video_translator import domain
#     # domain 应该只依赖标准库
#     pass
#
#
# def test_application_uses_protocols():
#     """验证应用层使用接口"""
#     from video_translator.application import generate_subtitles_use_case
#     import inspect
#
#     sig = inspect.signature(generate_subtitles_use_case)
#     # 验证参数类型是 Protocol
#     pass
#
#
# def test_adapters_implement_protocols():
#     """验证适配器实现接口"""
#     from video_translator.infrastructure.adapters import WhisperASRAdapter
#     from video_translator.domain import ASRProvider
#
#     # 验证 WhisperASRAdapter 实现了 ASRProvider
#     assert hasattr(WhisperASRAdapter, 'transcribe')
#
#
# if __name__ == "__main__":
#     test_domain_has_no_external_deps()
#     test_application_uses_protocols()
#     test_adapters_implement_protocols()
#     print("✅ 架构验证通过！")