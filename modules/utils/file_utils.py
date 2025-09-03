import os

# ===============================================================
# Utility: File Utils
# ===============================================================

class FileUtils:
    @staticmethod
    def get_relative_if_exists(session_path: str, filename: str) -> str | None:
        """
        Check if a file exists in the given session_path.
        If it exists, return the relative path (from current working directory).
        Otherwise, return None.
        """
        file_path = os.path.join(session_path, filename)
        if os.path.exists(file_path):
            return os.path.relpath(file_path)
        return None