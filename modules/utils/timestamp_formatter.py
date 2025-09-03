# ===============================================================
# Utility: Timestamp formatter
# ===============================================================
class TimestampFormatter:
    @staticmethod
    def format(seconds: float, srt: bool = False) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        if srt:
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        return f"[{h:02d}:{m:02d}:{s:02d}]"