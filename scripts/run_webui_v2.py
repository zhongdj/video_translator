#!/usr/bin/env python3
"""
启动脚本 - WebUI V2
支持分段语音克隆和增量合成
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from infrastructure.ui.improved_webui_v2 import main

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动视频翻译工厂 Pro V2")
    print("=" * 60)
    print()
    print("✨ V2 新特性:")
    print("  - 🎵 分段语音克隆（逐片段生成）")
    print("  - 👂 实时预览（边生成边试听）")
    print("  - ✏️  精细编辑（仅重新生成修改的片段）")
    print("  - 💾 智能缓存（片段级缓存）")
    print("  - 🔄 增量合成（跳过未修改片段）")
    print()
    print("📂 缓存位置:")
    print(f"  - 字幕缓存: .cache/")
    print(f"  - 音频片段: .cache/audio_segments/")
    print()
    print("🌐 访问地址: http://localhost:7860")
    print("=" * 60)
    print()

    main()