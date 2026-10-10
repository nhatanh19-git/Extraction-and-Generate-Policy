import os
import json
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QGroupBox, QLabel, QTextEdit, QPushButton, QFileDialog, 
                               QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, 
                               QSplitter, QProgressBar, QLineEdit, QApplication)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QColor

from backend.abac_pipeline.pipeline import ABACPipeline
from file_handler import FileHandler
from json_viewer import JsonViewerDialog

class ExtractionThread(QThread):
    progress = Signal(int, int)
    result_ready = Signal(dict)
    finished_batch = Signal(list)
    
    def __init__(self, pipeline, sentences):
        super().__init__()
        self.pipeline = pipeline
        self.sentences = sentences
        
    def run(self):
        results = []
        total = len(self.sentences)
        for i, sentence in enumerate(self.sentences):
            try:
                res = self.pipeline.process_policy(sentence)
                
                record = {
                    "id": i + 1,
                    "policy_text": sentence,
                    "rule": {
                        "effect": res.rule.effect,
                        "subjects": [{"attribute": s.attribute, "operator": s.operator, "values": s.values} for s in res.rule.subjects],
                        "actions": res.rule.actions,
                        "objects": [{"attribute": o.attribute, "operator": o.operator, "values": o.values} for o in res.rule.objects],
                        "constraints": [{"attribute": c.attribute, "operator": c.operator, "values": c.values, "attribute_ref": c.attribute_ref} for c in res.rule.constraints]
                    }
                }
                
                out = {"status": "success", "record": record}
                self.result_ready.emit(out)
                results.append(out)
            except Exception as e:
                out = {"status": "failed", "sentence": sentence, "error": str(e)}
                self.result_ready.emit(out)
                results.append(out)
            
            self.progress.emit(i + 1, total)
            
        self.finished_batch.emit(results)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABAC Policy Extractor")
        self.resize(1000, 800)
        self.apply_stylesheet()
        
        self.pipeline = None
        self.current_results = []
        
        self.init_ui()
        self.load_pipeline()
        
    def apply_stylesheet(self):
        self.setStyleSheet('''
            QMainWindow { background-color: #F0F0F0; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #B0B0B0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: #0A192F;
            }
            QPushButton {
                background-color: #1e6a8e;
                color: white;
                border: 1px solid #14496b;
                border-radius: 3px;
                padding: 6px 15px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2481ad; }
            QPushButton:disabled { background-color: #A0A0A0; border: 1px solid #808080; }
            QLineEdit, QTextEdit, QTableWidget {
                border: 1px solid #8ba1a8;
                border-radius: 3px;
                background-color: #FFFFFF;
                padding: 4px;
            }
        ''')

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # --- Khu vực A: Tiêu đề ---
        title_widget = QWidget()
        title_widget.setStyleSheet("background-color: #0A192F; color: white; border-radius: 5px;")
        title_layout = QVBoxLayout(title_widget)
        title_layout.setContentsMargins(10, 15, 10, 15)
        title_label = QLabel("ABAC Policy Extractor")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        desc_label = QLabel("Natural Language Access Control Policy Extraction")
        desc_label.setFont(QFont("Segoe UI", 10))
        title_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        title_layout.addWidget(desc_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(title_widget)
        
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # --- Vùng Nhập Liệu ---
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        group_single = QGroupBox("Đầu vào ACP (Nhập trực tiếp)")
        single_layout = QVBoxLayout(group_single)
        single_layout.setSpacing(10)
        self.txt_single = QTextEdit()
        self.txt_single.setPlaceholderText("Nhập một câu chính sách (VD: An on-call senior nurse may change the list of approved lab procedures.)")
        self.txt_single.setMaximumHeight(80)
        single_layout.addWidget(self.txt_single)
        
        btn_layout_single = QHBoxLayout()
        self.btn_extract_single = QPushButton("Trích xuất câu hiện tại")
        self.btn_clear_single = QPushButton("Xóa nội dung")
        self.btn_clear_single.setStyleSheet("background-color: #6c757d; border: 1px solid #5a6268;")
        btn_layout_single.addWidget(self.btn_extract_single)
        btn_layout_single.addWidget(self.btn_clear_single)
        btn_layout_single.addStretch()
        single_layout.addLayout(btn_layout_single)
        input_layout.addWidget(group_single)
        
        group_batch = QGroupBox("Nhập hàng loạt từ file TXT")
        batch_layout = QVBoxLayout(group_batch)
        batch_layout.setSpacing(10)
        
        file_select_layout = QHBoxLayout()
        self.txt_file_path = QLineEdit()
        self.txt_file_path.setReadOnly(True)
        self.txt_file_path.setPlaceholderText("Đường dẫn file...")
        self.btn_select_file = QPushButton("Chọn file TXT")
        file_select_layout.addWidget(self.txt_file_path)
        file_select_layout.addWidget(self.btn_select_file)
        batch_layout.addLayout(file_select_layout)
        
        self.lbl_batch_info = QLabel("Mỗi dòng trong file TXT tương ứng với một câu ACP.")
        batch_layout.addWidget(self.lbl_batch_info)
        
        self.batch_sentences = []
        input_layout.addWidget(group_batch)
        splitter.addWidget(input_widget)
        
        # --- Khu vực D: Kết quả ---
        res_widget = QWidget()
        res_layout = QVBoxLayout(res_widget)
        res_layout.setContentsMargins(0, 0, 0, 0)
        group_res = QGroupBox("Kết quả trích xuất")
        group_res_layout = QVBoxLayout(group_res)
        group_res_layout.setSpacing(10)
        
        self.table_res = QTableWidget(0, 6)
        self.table_res.setHorizontalHeaderLabels(["Câu ACP", "Subject", "Action", "Object", "Constraint", "Trạng thái"])
        self.table_res.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_res.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_res.setSelectionMode(QTableWidget.SingleSelection)
        group_res_layout.addWidget(self.table_res)
        
        detail_layout = QHBoxLayout()
        detail_layout.setSpacing(10)
        
        def make_detail_box(title):
            v = QVBoxLayout()
            v.addWidget(QLabel(title))
            txt = QTextEdit()
            txt.setReadOnly(True)
            txt.setMaximumHeight(80)
            v.addWidget(txt)
            return v, txt
            
        subj_layout, self.txt_res_subj = make_detail_box("Subject")
        act_layout, self.txt_res_act = make_detail_box("Action")
        obj_layout, self.txt_res_obj = make_detail_box("Object")
        cons_layout, self.txt_res_cons = make_detail_box("Constraint")
        
        detail_layout.addLayout(subj_layout)
        detail_layout.addLayout(act_layout)
        detail_layout.addLayout(obj_layout)
        detail_layout.addLayout(cons_layout)
        group_res_layout.addLayout(detail_layout)
        
        res_layout.addWidget(group_res)
        splitter.addWidget(res_widget)
        
        # --- Khu vực E: Trạng thái ---
        status_layout = QHBoxLayout()
        self.lbl_status = QLabel("Trạng thái: Sẵn sàng")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.progress_bar)
        main_layout.addLayout(status_layout)
        
        # --- Khu vực F: Các nút thao tác chung ---
        action_layout = QHBoxLayout()
        self.btn_extract_batch = QPushButton("Trích xuất hàng loạt")
        self.btn_extract_batch.setEnabled(False)
        self.btn_export = QPushButton("Xuất JSON")
        self.btn_view_json = QPushButton("Xem JSON")
        self.btn_clear_res = QPushButton("Xóa kết quả hiển thị")
        self.btn_clear_res.setStyleSheet("background-color: #dc3545; border: 1px solid #c82333;")
        
        action_layout.addWidget(self.btn_extract_batch)
        action_layout.addWidget(self.btn_export)
        action_layout.addWidget(self.btn_view_json)
        action_layout.addWidget(self.btn_clear_res)
        main_layout.addLayout(action_layout)
        
        # Connect signals
        self.btn_clear_single.clicked.connect(self.txt_single.clear)
        self.btn_select_file.clicked.connect(self.select_file)
        self.btn_extract_single.clicked.connect(self.extract_single)
        self.btn_extract_batch.clicked.connect(self.extract_batch)
        self.btn_clear_res.clicked.connect(self.clear_results)
        self.table_res.itemSelectionChanged.connect(self.display_selected_detail)
        self.btn_export.clicked.connect(self.export_json)
        self.btn_view_json.clicked.connect(self.view_json)
        
    def load_pipeline(self):
        self.lbl_status.setText("Trạng thái: Đang tải mô hình NLP...")
        QApplication.processEvents()
        try:
            self.pipeline = ABACPipeline()
            self.lbl_status.setText("Trạng thái: Tải mô hình thành công. Sẵn sàng.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tải mô hình: {e}")
            self.lbl_status.setText("Trạng thái: Lỗi tải mô hình.")
            
    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file TXT", "", "Text Files (*.txt)")
        if path:
            try:
                sentences = FileHandler.read_acp_file(path)
                self.txt_file_path.setText(path)
                self.batch_sentences = sentences
                self.lbl_batch_info.setText(f"Đã đọc {len(sentences)} câu hợp lệ từ file.")
                self.btn_extract_batch.setEnabled(len(sentences) > 0)
            except Exception as e:
                QMessageBox.warning(self, "Lỗi đọc file", f"Không thể đọc file TXT:\n{e}")
                self.txt_file_path.clear()
                self.batch_sentences = []
                self.btn_extract_batch.setEnabled(False)

    def extract_single(self):
        text = self.txt_single.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Chú ý", "Vui lòng nhập câu ACP!")
            return
            
        if not self.pipeline:
            QMessageBox.warning(self, "Lỗi", "Mô hình NLP chưa sẵn sàng.")
            return
            
        self.lbl_status.setText("Trạng thái: Đang trích xuất...")
        QApplication.processEvents()
        
        self.current_results.clear()
        self.table_res.setRowCount(0)
        self.run_extraction([text])
        
    def extract_batch(self):
        if not self.batch_sentences:
            QMessageBox.warning(self, "Chú ý", "Chưa có danh sách câu để trích xuất.")
            return
            
        self.current_results.clear()
        self.table_res.setRowCount(0)
        self.run_extraction(self.batch_sentences)
        
    def run_extraction(self, sentences):
        self.btn_extract_single.setEnabled(False)
        self.btn_extract_batch.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(sentences))
        self.lbl_status.setText("Trạng thái: Đang xử lý...")
        
        self.thread = ExtractionThread(self.pipeline, sentences)
        self.thread.progress.connect(self.update_progress)
        self.thread.result_ready.connect(self.add_result_to_table)
        self.thread.finished_batch.connect(self.extraction_finished)
        self.thread.start()
        
    def update_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.lbl_status.setText(f"Trạng thái: Đang xử lý ({current}/{total})...")
        
    def add_result_to_table(self, out):
        self.current_results.append(out)
        
        row = self.table_res.rowCount()
        self.table_res.insertRow(row)
        
        if out["status"] == "success":
            record = out["record"]
            rule = record["rule"]
            self.table_res.setItem(row, 0, QTableWidgetItem(record["policy_text"]))
            
            subj_texts = [s.get("values", [""])[0] for s in rule.get("subjects", [])] 
            obj_texts = [o.get("values", [""])[0] for o in rule.get("objects", [])]
            cons_texts = [c.get("values", [""])[0] for c in rule.get("constraints", [])]
            
            self.table_res.setItem(row, 1, QTableWidgetItem(", ".join(subj_texts) if subj_texts else "(none)"))
            self.table_res.setItem(row, 2, QTableWidgetItem(", ".join(rule.get("actions", [])) if rule.get("actions") else "(none)"))
            self.table_res.setItem(row, 3, QTableWidgetItem(", ".join(obj_texts) if obj_texts else "(none)"))
            self.table_res.setItem(row, 4, QTableWidgetItem(", ".join(cons_texts) if cons_texts else "(none)"))
            
            item_status = QTableWidgetItem("Thành công")
            item_status.setForeground(QColor("green"))
            self.table_res.setItem(row, 5, item_status)
        else:
            self.table_res.setItem(row, 0, QTableWidgetItem(out.get("sentence", "")))
            for i in range(1, 5):
                self.table_res.setItem(row, i, QTableWidgetItem("---"))
            item_status = QTableWidgetItem("Lỗi")
            item_status.setForeground(QColor("red"))
            self.table_res.setItem(row, 5, item_status)

    def extraction_finished(self, results):
        self.btn_extract_single.setEnabled(True)
        if self.batch_sentences:
            self.btn_extract_batch.setEnabled(True)
            
        self.progress_bar.setVisible(False)
        
        success = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - success
        self.lbl_status.setText(f"Trạng thái: Hoàn tất. Thành công: {success}, Thất bại: {failed}.")
        
        if len(results) > 1:
            QMessageBox.information(self, "Hoàn tất", f"Trích xuất hoàn tất!\nTổng số câu: {len(results)}\nThành công: {success}\nThất bại: {failed}")
        
    def display_selected_detail(self):
        items = self.table_res.selectedItems()
        if not items:
            return
            
        row = items[0].row()
        if row < len(self.current_results):
            out = self.current_results[row]
            if out["status"] == "success":
                rule = out["record"]["rule"]
                self.txt_res_subj.setPlainText(json.dumps(rule.get("subjects", []), ensure_ascii=False, indent=2))
                self.txt_res_act.setPlainText(json.dumps(rule.get("actions", []), ensure_ascii=False, indent=2))
                self.txt_res_obj.setPlainText(json.dumps(rule.get("objects", []), ensure_ascii=False, indent=2))
                self.txt_res_cons.setPlainText(json.dumps(rule.get("constraints", []), ensure_ascii=False, indent=2))
            else:
                self.txt_res_subj.setPlainText("LỖI")
                self.txt_res_act.setPlainText(str(out.get("error", "")))
                self.txt_res_obj.setPlainText("LỖI")
                self.txt_res_cons.setPlainText("LỖI")
                
    def clear_results(self):
        self.table_res.setRowCount(0)
        self.txt_res_subj.clear()
        self.txt_res_act.clear()
        self.txt_res_obj.clear()
        self.txt_res_cons.clear()
        self.current_results.clear()
        self.lbl_status.setText("Trạng thái: Đã xóa kết quả hiển thị.")
        
    def get_export_data(self):
        return [r["record"] for r in self.current_results if r["status"] == "success"]
        
    def export_json(self):
        data = self.get_export_data()
        if not data:
            QMessageBox.warning(self, "Chú ý", "Chưa có kết quả hợp lệ để xuất JSON.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Lưu file JSON", "abac_extraction_results.json", "JSON Files (*.json)")
        if path:
            try:
                FileHandler.save_json(data, path)
                self.lbl_status.setText(f"Trạng thái: Đã lưu JSON tại {path}")
                QMessageBox.information(self, "Thành công", f"File JSON đã được lưu tại:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi lưu file", f"Không thể lưu file JSON.\nChi tiết: {e}")
                
    def view_json(self):
        data = self.get_export_data()
        if not data:
            QMessageBox.warning(self, "Chú ý", "Chưa có kết quả hợp lệ để hiển thị.")
            return
            
        dialog = JsonViewerDialog(data, self)
        dialog.exec()
