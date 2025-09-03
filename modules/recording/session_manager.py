import os
from modules.utils.file_utils import FileUtils

# ===============================================================
# Component: Session Manager
# ===============================================================
class SessionManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def list_sessions(self):
        rows = []
        if not os.path.exists(self.base_dir):
            return []

        for session_id in sorted(os.listdir(self.base_dir)):
            session_path = os.path.join(self.base_dir, session_id)
            if os.path.isdir(session_path):
                rows.append([session_id, FileUtils.get_relative_if_exists(session_path, f"{session_id}_transcript.txt"), FileUtils.get_relative_if_exists(session_path, f"{session_id}_subtitles.srt")])
        return rows