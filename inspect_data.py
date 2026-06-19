import numpy as np

# 1. Load data
data = np.load(r'C:\Users\as\OneDrive\Desktop\Thesis\processed\lstm_data.npz')

# 2. Xem các khóa (tên mảng) có trong file
print("Các thành phần trong file npz:", data.files)

# 3. Lấy ra biến X_train
X_train = data['X_train']
y_train = data['y_train']

print(f"\n--- KÍCH THƯỚC DỮ LIỆU ---")
print(f"X_train shape: {X_train.shape} -> (số lượng mẫu, số giờ làm input, số lượng bài viết)")
print(f"y_train shape: {y_train.shape} -> (số lượng mẫu, số lượng bài viết)")

# 4. Xem thử dữ liệu của mẫu đầu tiên (index 0)
print("\n--- MẪU DỮ LIỆU ĐẦU TIÊN ---")
# Lấy input của 5 bài viết đầu tiên trong 24 giờ của mẫu số 0
print("X_train (5 bài viết đầu tiên) đã được log1p:\n", X_train[0, :, :5])

# Xem thử biến y (nhãn dự đoán)
print("\ny_train (Label của mẫu đầu tiên):", y_train[0])
print("Số bài lọt top 100 trong nhãn:", np.sum(y_train[0]))
