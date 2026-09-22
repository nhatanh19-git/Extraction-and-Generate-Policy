# Tại sao cần tải mô hình ABAC Pipeline khi khởi chạy?

Khi chạy chương trình (`main.py`), bạn sẽ thấy thông báo: *"Đang tải mô hình ABAC Pipeline, vui lòng đợi..."*. Việc khởi tạo này sẽ mất một khoảng thời gian ngắn (vài giây đến vài chục giây tùy cấu hình máy tính) trước khi bạn có thể nhập câu.

Dưới đây là lời giải thích lý do tại sao chương trình cần thực hiện bước này:

## 1. Khởi tạo các Mô hình Ngôn ngữ Tự nhiên (NLP)
Chương trình sử dụng thư viện `spaCy` (các mô hình ngôn ngữ tiếng Anh như `en_core_web_sm` hoặc `en_core_web_md`) làm cốt lõi cho bước tiền xử lý. 
- **Phân tích Cú pháp (Dependency Parsing):** Máy tính cần tải bộ quy tắc và mô hình thống kê vào bộ nhớ RAM để có thể hiểu được chủ ngữ, vị ngữ, bổ ngữ và từ loại (POS tagging) trong câu.
- **Mạng Nhúng Từ (Word Embeddings):** Để hiểu được độ tương đồng ngữ nghĩa (Semantic Similarity), mô hình phải nạp bộ vector từ vựng khổng lồ. Kích thước của những tệp dữ liệu này có thể lên tới hàng chục hoặc hàng trăm Megabyte.

## 2. Tải Mạng Nơ-ron Học sâu (BiLSTM-CRF)
Để trích xuất các "Ràng buộc" (Constraints) phức tạp trong câu truy cập, chương trình tích hợp một mạng học sâu BiLSTM-CRF đã được huấn luyện.
- **Nạp Trọng số (Weights):** Hàng triệu tham số và ma trận trọng số của mạng nơ-ron phải được đọc từ ổ cứng (file `.pt` trong thư mục `checkpoints/`) và đưa vào RAM.
- **Tải Từ vựng (Vocabulary):** Bộ từ điển dùng để chuyển đổi văn bản sang dạng số (index mapping) cũng cần được nạp song song.

## 3. Khởi tạo Môi trường Ánh xạ Thuộc tính (Attribute Mapping)
Chương trình cần đọc cấu hình tệp `abac_schema.yaml` để nạp các thuộc tính và giá trị chuẩn (canonical attributes), đồng thời chuẩn bị các thuật toán so khớp chuỗi (như Fuzzy String Matching) để đối chiếu thông tin trích xuất.

## 💡 Tổng kết: Tối ưu Hóa Hiệu suất
Nếu chương trình không thực hiện bước **Tải trước (Pre-loading)** này, thì **mỗi khi bạn gõ một câu** để phân tích, máy tính sẽ phải lặp lại toàn bộ quá trình đọc hàng trăm Megabyte từ ổ đĩa. Quá trình đó sẽ khiến mỗi câu phân tích mất đến hàng chục giây.

Bằng cách nạp tất cả mô hình vào bộ nhớ RAM ngay từ đầu **chỉ một lần duy nhất**, chương trình chỉ phải chịu độ trễ lúc khởi động. Bù lại, tốc độ phản hồi khi bạn nhập từng câu phía sau sẽ gần như **tức thì (real-time)**, mang lại trải nghiệm mượt mà hơn.
