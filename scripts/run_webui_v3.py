#!/usr/bin/env python3
"""
启动脚本 - WebUI V3
支持多说话人和仅字幕模式
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from infrastructure.ui.enhanced_webui_v3 import main

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 启动视频翻译工厂 Pro V3")
    print("=" * 70)
    print()
    print("✨ V3 新特性:")
    print("  🎭 多说话人支持")
    print("     - 为不同片段分配不同说话人")
    print("     - 适合对话、访谈等多人场景")
    print()
    print("  📝 仅字幕模式")
    print("     - 跳过语音合成，只生成字幕文件")
    print("     - 处理速度快，不需要GPU")
    print("     - 保留原始音频")
    print()
    print("  🎛️ 灵活的合成模式")
    print("     - 单说话人：传统模式")
    print("     - 多说话人：高级对话场景")
    print("     - 仅字幕：快速字幕生成")
    print()
    print("📂 缓存位置:")
    print(f"  - 字幕缓存: .cache/")
    print(f"  - 音频片段: .cache/audio_segments/")
    print(f"  - 参考音频: .cache/reference_audio/")
    print()
    print("🌐 访问地址: http://localhost:7860")
    print("=" * 70)
    print()

    main()