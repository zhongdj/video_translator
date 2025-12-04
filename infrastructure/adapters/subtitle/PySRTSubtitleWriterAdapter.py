from pathlib import Path

from domain.entities import Subtitle


class PySRTSubtitleWriterAdapter:
    """PySRT 字幕写入适配器"""

    def write_srt(self, subtitle: Subtitle, output_path: Path) -> Path:
        """写入 SRT 格式"""
        import pysrt

        subs = pysrt.SubRipFile()
        for idx, seg in enumerate(subtitle.segments, 1):
            start = pysrt.SubRipTime(seconds=seg.time_range.start_seconds)
            end = pysrt.SubRipTime(seconds=seg.time_range.end_seconds)
            subs.append(pysrt.SubRipItem(idx, start, end, seg.text))

        subs.save(str(output_path), encoding='utf-8')
        return output_path

    def write_ass(self, subtitle: Subtitle, output_path: Path) -> Path:
        """写入 ASS 格式"""
        # ASS 头部
        ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,42,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        lines = [ass_header]

        for seg in subtitle.segments:
            start = self._format_ass_time(seg.time_range.start_seconds)
            end = self._format_ass_time(seg.time_range.end_seconds)
            text = seg.text.replace('\n', '\\N')

            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

        output_path.write_text('\n'.join(lines), encoding='utf-8')
        return output_path

    def _format_ass_time(self, seconds: float) -> str:
        """格式化 ASS 时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
