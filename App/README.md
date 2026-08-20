# Vietnamese ACP to ABAC/XACML

Ứng dụng desktop nghiên cứu NLP, dùng để phân tích câu chính sách truy cập tự nhiên (ACP) bằng tiếng Việt và chuyển đổi thành:

- Subject: chủ thể thực hiện hành động
- Action: hành động truy cập
- Object: tài nguyên được truy cập
- ABAC Policy: chính sách với Effect mặc định là `Permit`
- XACML Policy: XML hợp lệ mô tả chính sách ABAC

## Ví dụ

Đầu vào:

```text
Bác sĩ có thể xem hồ sơ bệnh án.
```

Kết quả:

```text
Subject: Bác sĩ
Action: xem
Object: hồ sơ bệnh án
Effect: Permit
```

## Cấu trúc dự án

```text
.
├── main.py
├── requirements.txt
├── README.md
├── ui/
│   └── demo.ui
├── backend/
│   ├── nlp_engine.py
│   ├── extractor.py
│   ├── policy_model.py
│   ├── policy_generator.py
│   └── xacml_generator.py
└── tests/
    ├── test_extractor.py
    ├── test_policy.py
    └── test_xacml_generator.py
```

## Kiến trúc xử lý

```text
Người dùng nhập ACP
        ↓
Qt Designer UI (ui/demo.ui)
        ↓
main.py
        ↓
NLP Engine
        ↓
Attribute Extractor
        ↓
Policy Model
        ↓
XACML Generator
        ↓
Hiển thị kết quả trên giao diện
```

## Thành phần chính

### `main.py`

- Load giao diện có sẵn từ `ui/demo.ui` bằng `QUiLoader`.
- Kết nối các nút trong UI với backend.
- Hiển thị Subject, Action, Object.
- Sinh ABAC Policy và XACML.
- Hiển thị lỗi nhập liệu hoặc lỗi extraction.

### `backend/nlp_engine.py`

- Xử lý câu ACP.
- Tự động tìm model spaCy tiếng Việt nếu model đã được cài.
- Cung cấp token, lemma, POS và dependency khi model khả dụng.
- Sử dụng tokenizer fallback nếu chưa có model tiếng Việt.

### `backend/extractor.py`

Trích xuất ba thuộc tính chính:

- Subject: cụm danh từ đứng trước cấu trúc cấp quyền.
- Action: hành động truy cập, hỗ trợ cả cụm động từ như `tải xuống`.
- Object: cụm tài nguyên đứng sau Action.

Extractor có thể xử lý các dạng diễn đạt như:

- `có thể`
- `được phép`
- `được quyền`
- `có quyền`
- `được cho phép`

### `backend/policy_model.py`

Định nghĩa model ABAC bằng `dataclass`:

```python
Policy(
    subject="Bác sĩ",
    action="xem",
    object="hồ sơ bệnh án",
    effect="Permit",
)
```

Effect hiện hỗ trợ:

- `Permit`
- `Deny`

### `backend/xacml_generator.py`

- Sinh XACML bằng `xml.etree.ElementTree`.
- Escape đúng các giá trị Unicode và ký tự XML đặc biệt.
- Sử dụng các category chuẩn cho Subject, Action và Resource.

## Yêu cầu môi trường

- Python 3.10 trở lên
- Windows, Linux hoặc macOS
- PySide6
- spaCy
- pytest nếu muốn chạy test

## Cài đặt

Mở PowerShell tại thư mục dự án:

```powershell
cd "D:\Thực tập cơ sở\App"
```

Tạo virtual environment:

```powershell
python -m venv .venv
```

Kích hoạt virtual environment trên Windows:

```powershell
.venv\Scripts\activate
```

Cài dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Model tiếng Việt spaCy

Ứng dụng không bắt buộc phải có model tiếng Việt. Khi chưa có model, ứng dụng tự động dùng fallback rule-based.

Để sử dụng POS tagging và dependency parsing tiếng Việt đầy đủ, cài một model tiếng Việt tương thích với spaCy, nếu model đó có sẵn trong môi trường nghiên cứu của bạn:

```powershell
python -m spacy download vi_core_news_sm
```

Sau khi cài, khởi động lại ứng dụng. Nếu model không tồn tại hoặc không tải được, ứng dụng vẫn có thể chạy bằng fallback.

## Chạy chương trình

Sau khi kích hoạt virtual environment:

```powershell
python main.py
```

Các bước sử dụng:

1. Nhập câu ACP tiếng Việt vào ô `ACP sentence`.
2. Nhấn nút `Analyst` để phân tích.
3. Kiểm tra các trường Subject, Action và Object.
4. Nhấn `GENERATE POLICY` để tạo ABAC Policy.
5. Nhấn `GENERATE XACML` để sinh và hiển thị XML XACML.

## Chạy kiểm thử

Chạy toàn bộ test:

```powershell
python -m pytest -q
```

Các test bao gồm:

- Trích xuất Subject, Action, Object từ các câu ACP tiếng Việt.
- Xử lý câu không hợp lệ mà không làm chương trình crash.
- Kiểm tra ABAC Policy mặc định có Effect là `Permit`.
- Kiểm tra XACML XML parse được và escape ký tự đặc biệt đúng.

## Các câu kiểm thử mẫu

```text
Bác sĩ có thể xem hồ sơ bệnh án.
Quản trị viên được phép xóa tài khoản người dùng.
Nhân viên có thể tải báo cáo tài chính.
Giảng viên được phép xem điểm của sinh viên.
abc xyz
```

Câu `abc xyz` được xem là không hợp lệ và phải trả thông báo lỗi extraction thay vì làm ứng dụng crash.

## Giao diện Qt Designer

File giao diện hiện tại là:

```text
ui/demo.ui
```

File này được load trực tiếp bởi `main.py` và không bị chỉnh sửa. Backend sử dụng các widget có sẵn:

| Object name | Chức năng |
|---|---|
| `txtACP_sentence` | Nhập câu ACP |
| `buttonAnalyst` | Phân tích câu |
| `txtSubject` | Hiển thị Subject |
| `txtAction` | Hiển thị Action |
| `txtObject` | Hiển thị Object |
| `buttonGeneratePolicy` | Tạo ABAC Policy |
| `buttonGenerateXACML` | Tạo XACML |
| `txtXACML` | Hiển thị XACML XML |
| `statusbar` | Hiển thị trạng thái và lỗi |

## Lưu ý

- Không cần LLM hoặc API bên ngoài để chạy pipeline extraction.
- Kết quả extraction bằng fallback là rule-based và phù hợp cho các mẫu ACP thông dụng.
- Với dữ liệu tiếng Việt đa dạng hoặc câu phức tạp, nên cài model spaCy tiếng Việt và mở rộng bộ test.
- File `.ui` không được sửa trong quá trình phát triển backend.
