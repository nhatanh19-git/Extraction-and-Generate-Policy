import os
import json

class FileHandler:
    @staticmethod
    def read_acp_file(file_path):
        """Đọc file TXT, mỗi dòng là một câu ACP."""
        sentences = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File không tồn tại: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    sentences.append(stripped)
        return sentences

    @staticmethod
    def save_json(data, file_path):
        """Lưu dữ liệu thành file JSON."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
