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

        for session_dir in sorted(os.listdir(self.base_dir)):
            session_path = os.path.join(self.base_dir, session_dir)
            if os.path.isdir(session_path):
                rows.append([session_dir, FileUtils.get_relative_if_exists(session_path, "transcript.txt"), FileUtils.get_relative_if_exists(session_path, "transcript.srt")])
        return rows