# ABAC NLP Pipeline

Một quy trình Xử lý Ngôn ngữ Tự nhiên (NLP) từ đầu đến cuối để trích xuất các chính sách Kiểm soát Truy cập dựa trên Thuộc tính (ABAC) từ văn bản ngôn ngữ tự nhiên.

Dự án này sử dụng sự kết hợp của Phân tích cú pháp phụ thuộc (Dependency Parsing), Độ tương đồng ngữ nghĩa (Semantic Similarity), và một mô hình gán nhãn chuỗi BiLSTM-CRF đã được huấn luyện để chuyển đổi các câu chính sách bằng tiếng Anh thành các quy tắc ABAC có cấu trúc.

## Kiến trúc

Quy trình bao gồm các giai đoạn mô-đun sau:

1. **Tiền xử lý (`core/preprocessor.py`)**: Sử dụng `spaCy` (`en_core_web_sm`) để phân tích văn bản và trích xuất các từ loại (POS tags), bổ đề (lemmas), và cây phụ thuộc (dependency trees).
2. **Trích xuất SAO (`sao_extraction/`)**: Trích xuất các thực thể Chủ thể (Subject), Hành động (Action), và Đối tượng/Tài nguyên (Object/Resource) cốt lõi bằng cách sử dụng các heuristic phụ thuộc dựa trên luật.
3. **Trích xuất Ràng buộc (`bilstm/`)**: Sử dụng mạng nơ-ron BiLSTM-CRF PyTorch được huấn luyện trên các tập dữ liệu ABAC-Lab để trích xuất chính xác các cụm ràng buộc phức tạp (ví dụ: `"that are assigned to them"`, `"in their department"`).
4. **Ngữ nghĩa Ràng buộc (`semantic/constraint_analyzer.py`)**: Duyệt qua đường dẫn phụ thuộc ngắn nhất (SDP) để giải quyết các toán hạng và toán tử chuẩn cho các ràng buộc được trích xuất. Xử lý phân giải đại từ (`reference_resolver.py`) cho các từ như "he", "she", "their", và "own".
5. **Ánh xạ Thuộc tính (`mapping/attribute_mapper.py`)**: Sử dụng sự kết hợp của Khớp chuỗi mờ (Fuzzy String Matching - `thefuzz`) và Độ tương đồng ngữ nghĩa (Semantic Similarity - vector từ `spaCy en_core_web_md`) để ánh xạ các thực thể thô thành các thuộc tính chuẩn được định nghĩa trong `config/abac_schema.yaml`.
6. **Tạo Quy tắc (`pipeline.py` & `json_builder.py`)**: Lắp ráp các thành phần đã phân tích thành một định dạng Quy tắc ABAC hoàn chỉnh, mang tính quyết định và tuần tự hóa nó thành JSON.

## Ví dụ

**Chính sách đầu vào:**
```text
"Technicians can only view and complete tasks that are assigned to them."
```

**Cấu trúc đầu ra:**
```json
{
  "effect": "Permit",
  "subjects": [
    { "attribute": "position", "operator": "IN", "values": ["technicians"] }
  ],
  "resources": [
    { "attribute": "type", "operator": "IN", "values": ["tasks"] }
  ],
  "actions": ["view", "complete"],
  "constraints": [
    { "attribute": "task.assigned_to", "operator": "EQUALS", "values": ["technician"], "attribute_ref": null }
  ]
}
```

## Cấu trúc Dự án

```text
abac_pipeline/
├── bilstm/                     # Mô-đun mạng nơ-ron BiLSTM-CRF
│   ├── config.py               # Cấu hình siêu tham số (Hyperparameter)
│   ├── dataset.py              # PyTorch Dataset & Collate (Padding động)
│   ├── embeddings.py           # Trình tải vector GloVe/FastText/spaCy
│   ├── model.py                # Triển khai BiLSTM-CRF PyTorch
│   ├── trainer.py              # Vòng lặp huấn luyện với các số liệu (metrics) seqeval
│   └── vocabulary.py           # Ánh xạ chỉ mục Từ/Nhãn (Word/Tag)
├── checkpoints/                # Mô hình và từ vựng đã huấn luyện được lưu trữ
├── config/                     # Định nghĩa Schema ABAC chuẩn
├── core/                       # Mô hình dữ liệu cốt lõi & Tiền xử lý
├── data/                       # Trình tạo tập dữ liệu & Trình phân tích cú pháp
├── dataset/                    # Các tập dữ liệu chuỗi CoNLL được tạo
├── evaluation/                 # Đánh giá chéo miền (LODO)
├── mapping/                    # Xếp hạng ứng viên ngữ nghĩa
├── sao_extraction/             # Các heuristic trích xuất Subject-Action-Object (SAO) cũ
├── semantic/                   # Phân tích ràng buộc dựa trên phụ thuộc
├── pipeline.py                 # Trình điều phối quy trình (Orchestrator) từ đầu đến cuối
├── train_bilstm.py             # Script để huấn luyện mô hình BiLSTM
└── test_pipeline.py            # Script kiểm tra độ chính xác từ đầu đến cuối
```

## Tải và Cài đặt (Setup & Installation)

### Yêu cầu hệ thống
- Python 3.10+
- PyTorch
- spaCy
- pytorch-crf
- seqeval
- thefuzz

### Cách Tải và Cài đặt

1. **Tải chương trình (Clone repository)**:
Nếu bạn sử dụng Git, hãy clone kho lưu trữ này về máy:
```powershell
git clone <đường-dẫn-tới-repo-của-bạn>
cd <thư-mục-repo>
```
*(Nếu bạn không dùng Git, có thể tải file `.zip` trực tiếp từ GitHub và giải nén, sau đó mở Terminal/PowerShell tại thư mục vừa giải nén).*

2. **Cài đặt các thư viện phụ thuộc (Dependencies)**:
Khuyến nghị tạo môi trường ảo (virtual environment) trước khi cài đặt:
```powershell
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
pip install thefuzz python-Levenshtein
```

3. **Tải các mô hình ngôn ngữ `spaCy` cần thiết** (Được sử dụng cho Tiền xử lý và Nhúng ngữ nghĩa):
```powershell
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

## Cách Sử dụng (Usage)

### 1. Chạy Chương trình Thủ công trên Terminal

Bạn có thể tương tác trực tiếp với chương trình bằng cách chạy file `main.py`. Chương trình sẽ cho phép bạn nhập các câu chính sách từ bàn phím và in ra kết quả ngay lập tức:

```powershell
python main.py
```

*Khi chạy, chương trình sẽ yêu cầu bạn nhập "Tên Policy" để tạo tệp lưu tự động. Sau khi mô hình tải xong, bạn chỉ việc gõ câu tiếng Anh (ví dụ: "Only members of the sales department can send invoices."), nhấn Enter và xem kết quả phân tích. Quá trình trích xuất này được tự động lưu tịnh tiến vào thư mục `result/`.*
*Để kết thúc quá trình nhập câu và dừng chương trình, bạn chỉ cần gõ chữ `done` (hoặc `exit`, `quit`) rồi nhấn Enter.*

### 2. Tích hợp trong Mã Python

Bạn cũng có thể tích hợp toàn bộ quy trình vào các dự án khác thông qua mã Python:

```python
from backend.abac_pipeline.pipeline import ABACPipeline

pipeline = ABACPipeline()
result = pipeline.process_policy("Only members of the sales department can send invoices.")

print(f"Effect: {result.rule.effect}")
print(f"Subjects: {result.rule.subjects}")
print(f"Actions: {result.rule.actions}")
print(f"Resources: {result.rule.resources}")
print(f"Constraints: {result.rule.constraints}")
```

### 3. Kiểm thử Toàn bộ Hệ thống (End-to-End Testing)

Chạy script kiểm thử đi kèm để xác minh rằng các kiến trúc cốt lõi đang hoạt động chính xác trên các câu đại diện của ABAC-Lab.

```powershell
python test_pipeline.py
```

### 3. Huấn luyện Mô hình Trích xuất Ràng buộc BiLSTM

Mô hình BiLSTM-CRF xử lý nhiệm vụ phức tạp là xác định ranh giới ràng buộc trong một câu.

Để tạo lại các tập dữ liệu huấn luyện và huấn luyện mô hình từ đầu:

```powershell
# 1. Tạo lại các tập dữ liệu CoNLL từ các quy tắc của ABAC-Lab
python data/generate_dataset.py

# 2. Huấn luyện mô hình (Tự động sử dụng vector spaCy en_core_web_md)
python train_bilstm.py --epochs 80 --batch_size 8 --lr 0.005
```

### 4. Đánh giá Chéo miền (Cross-Domain Evaluation)

Để kiểm tra độ mạnh mẽ của quy trình bằng cách sử dụng chiến lược Bỏ lại một miền (Leave-One-Domain-Out - LODO) trên tất cả 5 tập dữ liệu ABAC-Lab (eDocument, Healthcare, Project Management, University, Workforce):

```powershell
python evaluation/evaluate.py
```
