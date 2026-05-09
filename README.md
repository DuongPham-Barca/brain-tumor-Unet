# Brain Tumor Segmentation using U-Net and Django
Dự án sử dụng mô hình AI U-Net để phân đoạn khối u não từ ảnh MRI, triển khai trên nền tảng Web bằng Django.


## 📚 Cơ sở lý thuyết Deep Learning
Dự án áp dụng các kiến thức cốt lõi về mạng Neural nhân tạo và xử lý ảnh y tế:

### 1. Thành phần mạng và Tối ưu hóa
Weight & Bias: Các tham số học được để điều chỉnh mối quan hệ dữ liệu.

Hàm Activation (ReLU): Giúp mạng học các đặc trưng phi tuyến tính.

Quy tắc Chain Rule: Cơ sở để tính đạo hàm trong lan truyền ngược.

Gradient Descent: Thuật toán tối ưu hóa để cập nhật trọng số mạng.

### 2. Kiến trúc mạng Convolutional Neural Network (CNN)
Convolutional Layer: Trích xuất các đặc trưng (Feature Map) từ ảnh MRI.

Pooling (Maxpooling & Average pooling): Giảm kích thước dữ liệu và giữ lại thông tin quan trọng.

Fully Connected Layer: Kết nối các neuron để đưa ra dự đoán.

Skip Connection: Đặc trưng của U-Net giúp bảo toàn chi tiết không gian khi phân đoạn ảnh.

### 3. Huấn luyện và Đánh giá
Epoch & Batch size: Các tham số điều khiển quá trình huấn luyện.

Hàm Loss (Binary Cross Entropy): Đo lường sai số cho bài toán phân đoạn nhị phân.

Hàm tối ưu hóa (Adam): Giúp mô hình hội tụ nhanh hơn.

Metric đánh giá: Sử dụng Dice Metric và IoU Metric để đo độ chính xác vùng phân đoạn.

Vấn đề huấn luyện: Kiểm soát Overfitting và Underfitting để đảm bảo tính tổng quát của mô hình.

## 🛠 Công nghệ sử dụng
Python, Django

PyTorch (U-Net model)

OpenCV (cv2), Albumentations