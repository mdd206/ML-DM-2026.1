# Giao việc: EDA và tiền xử lý dữ liệu cảm biến

## 1. Phân công

| Người làm | Nhiệm vụ |
|---|---|
| Dũng | EDA + tiền xử lý dữ liệu IMU (`acc_*`, `rot_*`) |
| Huy | EDA + tiền xử lý dữ liệu ToF (`tof_*`) và nhiệt (`thm_*`) |

Trước khi bắt đầu, hãy đọc notebook `notebooks/01_eda_overview.ipynb` để nắm bối cảnh chung.

## 2. Sản phẩm cần nộp

| | Dũng | Huy |
|---|---|---|
| Notebook EDA | `notebooks/02_eda_imu.ipynb` | `notebooks/03_eda_tof_thm.ipynb` |
| Code tiền xử lý | hàm `preprocess_imu` trong `src/preprocess_imu.py` | hàm `preprocess_tof` trong `src/preprocess_tof.py` |

- Notebook EDA kết thúc bằng mục **"Tóm tắt phát hiện và quyết định tiền xử lý"**.
- Khung code đã có sẵn. Hãy điền phần xử lý vào chỗ có chú thích `TODO`.

## 3. Quy ước bắt buộc

1. **Đầu vào** của hàm là dữ liệu gốc (train hoặc test) và bảng nhân khẩu học.
2. **Đầu ra** giữ nguyên số dòng, có cột `sequence_id`, `sequence_counter` và các cột đã xử lý thuộc phần của mình.
3. Tên cột không được trùng với phần của bạn kia.
4. Không bỏ dòng hay chuỗi nào.
5. Chỉ làm các bước xử lý trên **từng chuỗi riêng lẻ**. Các bước cần "học" từ cả tập dữ liệu thì để lại cho bước huấn luyện.
6. Hàm phải chạy được cho cả tập test.
7. Kiểm tra bằng lệnh `python src/preprocess.py` (lệnh này tạo ra file `data/processed/train.parquet`).
8. Code đơn giản, chú thích bằng tiếng Anh.
9. Nếu một quyết định của mình có thể ảnh hưởng tới phần của bạn kia, hai bạn cần trao đổi với nhau trước.

## 4. Yêu cầu giải thích

Hai bạn được **tự do chọn cách làm**. Nhưng mọi bước đã làm đều phải giải thích được:

- **Làm gì?**
- **Vì sao?** Dựa trên phát hiện EDA nào, hoặc lý do nào.
- **Kiểm chứng thế nào?** Nếu có thể, bằng hình vẽ hoặc số liệu.
