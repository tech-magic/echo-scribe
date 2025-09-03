import os
from modules.utils.timestamp_formatter import TimestampFormatter

# ===============================================================
# Component: Transcript Writer
# ===============================================================
class TranscriptWriter:
    def __init__(self, base_session_dir, session_id):
        session_dir = os.path.join(base_session_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        self.txt_file = open(os.path.join(session_dir, f"{session_id}_transcript.txt"), "w", encoding="utf-8")
        self.srt_file = open(os.path.join(session_dir, f"{session_id}_subtitles.srt"), "w", encoding="utf-8")
        self.srt_index = 1

    def write(self, start, end, speaker, text):
        line = f"📝 {TimestampFormatter.format(start)} -> {TimestampFormatter.format(end)} {speaker}: {text}"
        print(line)

        self.txt_file.write(line + "\n")
        self.txt_file.flush()

        self.srt_file.write(f"{self.srt_index}\n")
        self.srt_file.write(
            f"{TimestampFormatter.format(start, srt=True)} --> {TimestampFormatter.format(end, srt=True)}\n"
        )
        self.srt_file.write(f"{speaker}: {text.strip()}\n\n")
        self.srt_file.flush()

        self.srt_index += 1
        return line

    def close(self):
        self.txt_file.close()
        self.srt_file.close()