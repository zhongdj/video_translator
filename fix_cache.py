#!/usr/bin/env python
"""
缓存修复脚本

使用方法:
    python fix_cache.py              # 扫描并显示统计
    python fix_cache.py --clean      # 清理无效缓存
    python fix_cache.py --clean-all  # 清理所有缓存
"""

import sys
import json
from pathlib import Path


def validate_audio_cache(cached):
    """验证音频缓存"""
    if cached is None:
        return False

    required = ["audio_samples", "sample_rate", "reference_audio", "reference_duration"]
    if not all(key in cached for key in required):
        return False

    if not isinstance(cached["audio_samples"], (list, tuple)):
        return False

    if len(cached["audio_samples"]) == 0:
        return False

    if not isinstance(cached["sample_rate"], int) or cached["sample_rate"] <= 0:
        return False

    return True


def validate_subtitle_cache(cached):
    """验证字幕缓存"""
    if cached is None:
        return False

    required = ["detected_language", "zh_segments", "en_segments"]
    if not all(key in cached for key in required):
        return False

    if not isinstance(cached["zh_segments"], list) or len(cached["zh_segments"]) == 0:
        return False

    if not isinstance(cached["en_segments"], list) or len(cached["en_segments"]) == 0:
        return False

    # 检查第一个 segment 的结构
    for seg in cached["zh_segments"][:1]:
        if not all(key in seg for key in ["text", "start", "end"]):
            return False

    return True


def scan_cache_dir(cache_dir: Path):
    """扫描缓存目录"""
    stats = {
        "total": 0,
        "valid_audio": 0,
        "valid_subtitle": 0,
        "invalid": 0,
        "corrupt": 0,
        "total_size_mb": 0.0
    }

    invalid_files = []

    if not cache_dir.exists():
        print(f"❌ 缓存目录不存在: {cache_dir}")
        return stats, invalid_files

    cache_files = list(cache_dir.glob("*.json"))
    stats["total"] = len(cache_files)

    print(f"\n🔍 扫描缓存目录: {cache_dir}")
    print(f"   发现 {len(cache_files)} 个缓存文件\n")

    for cache_file in cache_files:
        try:
            # 统计大小
            file_size = cache_file.stat().st_size / (1024 * 1024)
            stats["total_size_mb"] += file_size

            # 读取并验证
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)

            # 判断类型并验证
            if "audio_samples" in cached:
                if validate_audio_cache(cached):
                    stats["valid_audio"] += 1
                    print(f"  ✅ 音频: {cache_file.name} ({file_size:.2f} MB)")
                else:
                    stats["invalid"] += 1
                    invalid_files.append(cache_file)
                    print(f"  ❌ 音频损坏: {cache_file.name}")
            elif "zh_segments" in cached:
                if validate_subtitle_cache(cached):
                    stats["valid_subtitle"] += 1
                    print(f"  ✅ 字幕: {cache_file.name} ({file_size:.2f} MB)")
                else:
                    stats["invalid"] += 1
                    invalid_files.append(cache_file)
                    print(f"  ❌ 字幕损坏: {cache_file.name}")
            else:
                stats["invalid"] += 1
                invalid_files.append(cache_file)
                print(f"  ❓ 未知类型: {cache_file.name}")

        except json.JSONDecodeError:
            stats["corrupt"] += 1
            invalid_files.append(cache_file)
            print(f"  💥 JSON 损坏: {cache_file.name}")
        except Exception as e:
            stats["corrupt"] += 1
            invalid_files.append(cache_file)
            print(f"  ❌ 读取失败: {cache_file.name} - {e}")

    return stats, invalid_files


def print_stats(stats):
    """打印统计信息"""
    print(f"\n{'=' * 60}")
    print(f"📊 缓存统计")
    print(f"{'=' * 60}")
    print(f"  总文件数:    {stats['total']}")
    print(f"  有效音频:    {stats['valid_audio']}")
    print(f"  有效字幕:    {stats['valid_subtitle']}")
    print(f"  无效文件:    {stats['invalid']}")
    print(f"  损坏文件:    {stats['corrupt']}")
    print(f"  总大小:      {stats['total_size_mb']:.2f} MB")
    print(f"{'=' * 60}\n")

    if stats['invalid'] > 0 or stats['corrupt'] > 0:
        print(f"⚠️  发现 {stats['invalid'] + stats['corrupt']} 个问题文件")
        print(f"   运行 'python fix_cache.py --clean' 清理")


def clean_invalid_files(invalid_files):
    """清理无效文件"""
    if not invalid_files:
        print("✅ 没有需要清理的文件")
        return 0

    print(f"\n🗑️  开始清理 {len(invalid_files)} 个无效文件...")

    cleaned = 0
    for cache_file in invalid_files:
        try:
            cache_file.unlink()
            cleaned += 1
            print(f"  ✅ 已删除: {cache_file.name}")
        except Exception as e:
            print(f"  ❌ 删除失败: {cache_file.name} - {e}")

    print(f"\n✅ 清理完成: 删除了 {cleaned}/{len(invalid_files)} 个文件")
    return cleaned


def clean_all_cache(cache_dir: Path):
    """清理所有缓存"""
    if not cache_dir.exists():
        print(f"❌ 缓存目录不存在: {cache_dir}")
        return 0

    cache_files = list(cache_dir.glob("*.json"))

    if not cache_files:
        print("✅ 缓存目录已经是空的")
        return 0

    print(f"\n⚠️  确认要删除所有 {len(cache_files)} 个缓存文件吗？")
    print("   这将导致下次处理需要重新生成所有缓存")
    confirm = input("   输入 'yes' 确认: ")

    if confirm.lower() != 'yes':
        print("❌ 已取消")
        return 0

    print(f"\n🗑️  开始清理所有缓存...")

    cleaned = 0
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            cleaned += 1
        except Exception as e:
            print(f"  ❌ 删除失败: {cache_file.name} - {e}")

    print(f"\n✅ 清理完成: 删除了 {cleaned}/{len(cache_files)} 个文件")
    return cleaned


def main():
    """主函数"""
    # 默认缓存目录
    cache_dir = Path(".cache")

    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clean":
            # 清理无效缓存
            stats, invalid_files = scan_cache_dir(cache_dir)
            print_stats(stats)

            if invalid_files:
                clean_invalid_files(invalid_files)

                # 再次扫描显示结果
                print(f"\n{'=' * 60}")
                print("🔍 清理后重新扫描...")
                stats, invalid_files = scan_cache_dir(cache_dir)
                print_stats(stats)

        elif sys.argv[1] == "--clean-all":
            # 清理所有缓存
            clean_all_cache(cache_dir)

        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("""
缓存修复脚本

使用方法:
    python fix_cache.py              # 扫描并显示统计
    python fix_cache.py --clean      # 清理无效缓存
    python fix_cache.py --clean-all  # 清理所有缓存（需确认）
    python fix_cache.py --help       # 显示帮助
            """)
        else:
            print(f"❌ 未知参数: {sys.argv[1]}")
            print("   运行 'python fix_cache.py --help' 查看帮助")
    else:
        # 默认：只扫描不清理
        stats, invalid_files = scan_cache_dir(cache_dir)
        print_stats(stats)

        if invalid_files:
            print("\n💡 提示:")
            print(f"   运行 'python fix_cache.py --clean' 清理 {len(invalid_files)} 个无效文件")


if __name__ == "__main__":
    main()