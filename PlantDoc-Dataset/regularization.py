import os
import cv2
from tqdm import tqdm

# Thư mục chứa dữ liệu huấn luyện
data_dir = "val"  # Thay bằng đường dẫn thực tế

# Kích thước đích cho ảnh
target_size = (224, 224)  # Bạn có thể thay đổi tùy ý

# Duyệt qua toàn bộ thư mục và xử lý ảnh
for root, _, files in os.walk(data_dir):
    for file in tqdm(files, desc=f"Processing {root}"):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Lỗi đọc ảnh: {img_path}")
                continue

            # Resize ảnh nhưng vẫn giữ nguyên tỷ lệ (padding nếu cần)
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            # Lưu đè ảnh
            cv2.imwrite(img_path, img_resized)

print("Xử lý hoàn tất!")
