# 🛡️ Ứng dụng Phát hiện Giao dịch Gian lận trên Nền tảng Streamlit

Ứng dụng web tương tác thông minh này được chuyển đổi tự động từ quy trình huấn luyện và kiểm định mô hình Học máy trong file Notebook (`phat_hien_giao_dich_gian_lan.ipynb`). Ứng dụng được thiết kế nhằm mục đích hỗ trợ phân tích quản trị rủi ro, nhận diện tự động và phân loại các giao dịch gian lận/bất thường từ dữ liệu hệ thống.

## 📊 Mô hình & Thuật toán tích hợp
- **Thuật toán sử dụng:** Rừng Ngẫu nhiên (`RandomForestClassifier` thuộc thư viện Scikit-learn).
- **Cơ chế tiền xử lý đặc trưng:** Chuẩn hóa phân phối chuẩn `StandardScaler` dựa trên tập biến đầu vào số liên tục từ **X_1 đến X_14**.
- **Biến mục tiêu nhận diện:** Cột `default` mang tính chất nhị phân phân loại (0: Giao dịch bình thường hợp lệ, 1: Giao dịch giả mạo/rủi ro cao).

## 🗂️ Cấu trúc Yêu cầu cho Dữ liệu Đầu vào
Tệp tin dữ liệu mẫu tải lên (`.csv` hoặc `.xlsx`) yêu cầu bắt buộc định dạng chứa đúng cấu trúc tiêu đề các cột như sau:
- **Các biến đầu vào (Features):** `X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11, X_12, X_13, X_14` (Kiểu số thực/số nguyên liên tục).
- **Biến phân lớp mục tiêu (Target):** Cột `default` nhận hai giá trị duy nhất là `0` hoặc `1`.

---

## 🛠️ Hướng dẫn Cài đặt và Khởi chạy Ứng dụng

### Bước 1: Khởi tạo và thiết lập môi trường Python độc lập (Khuyến nghị)
Bạn nên thiết lập môi trường ảo (`venv` hoặc `conda`) sử dụng phiên bản Python thích hợp (Từ `3.9` đến `3.12`) để tránh xung đột hệ thống.

```bash
# Tạo môi trường ảo mới tên là env_fraud
python -m venv env_fraud

# Kích hoạt môi trường trên Windows:
env_fraud\Scripts\activate

# Kích hoạt môi trường trên macOS/Linux:
source env_fraud/bin/activate
