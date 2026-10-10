import json
from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QApplication
from PySide6.QtGui import QFont

class JsonViewerDialog(QDialog):
    def __init__(self, json_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kết quả JSON")
        self.resize(700, 500)
        
        layout = QVBoxLayout(self)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 10))
        
        if isinstance(json_data, (dict, list)):
            formatted_json = json.dumps(json_data, ensure_ascii=False, indent=4)
        else:
            formatted_json = str(json_data)
            
        self.text_edit.setPlainText(formatted_json)
        layout.addWidget(self.text_edit)
        
        btn_layout = QHBoxLayout()
        self.btn_copy = QPushButton("Sao chép JSON")
        self.btn_close = QPushButton("Đóng")
        
        self.btn_copy.clicked.connect(self.copy_to_clipboard)
        self.btn_close.clicked.connect(self.close)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_copy)
        btn_layout.addWidget(self.btn_close)
        
        layout.addLayout(btn_layout)
        
    def copy_to_clipboard(self):
        cb = QApplication.clipboard()
        cb.setText(self.text_edit.toPlainText())
