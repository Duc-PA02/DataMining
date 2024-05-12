import csv
import os
import numpy as np
import matplotlib.pyplot as plt

chart_counter = 3

# Đọc dữ liệu từ file CSV, bỏ qua hàng đầu tiên (tiêu đề)
def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Bỏ qua hàng đầu tiên (tiêu đề)
        for row in csv_reader:
            data.append([float(x) for x in row[:6]])  # Lấy 5 cột đầu tiên
    return np.array(data)

# Chọn chỉ 5 thuộc tính: Area, Perimeter, MajorAxisLength, MinorAxisLength, và AspectRation
def select_features(data):
    return data[:, :6]  # Lấy 5 cột đầu tiên

# Khởi tạo các điểm trung tâm ban đầu là hàng 1, 50, 150, 200, 250
def initialize_centroids(data, k):
    return data[[1, 50, 150, 200, 250]]

# Tính toán khoảng cách giữa các điểm dữ liệu và các điểm trung tâm
def compute_distances(data, centroids):
    distances = []
    for centroid in centroids:
        distances.append(np.linalg.norm(data - centroid, axis=1))
    return np.array(distances).T

# Gán các điểm dữ liệu vào các cụm
def assign_clusters(distances):
    return np.argmin(distances, axis=1)

# Cập nhật các điểm trung tâm của các cụm
def update_centroids(data, clusters, k):
    centroids = []
    for i in range(k):
        cluster_data = data[clusters == i]
        if len(cluster_data) > 0:
            centroids.append(np.mean(cluster_data, axis=0))
        else:
            centroids.append(np.random.randn(data.shape[1]))
    return np.array(centroids)

# Hiển thị và lưu biểu đồ các cụm vào thư mục 'graph'
def plot_clusters(data, clusters, centroids):
    global chart_counter

    plt.figure(figsize=(8, 6))
    # Số lượng màu sẽ phải bằng số lượng cụm
    colors = ['r', 'g', 'b', 'c', 'm']  # Thêm các màu khác nếu có nhiều hơn 3 cụm
    for i in range(len(centroids)):
        cluster_data = data[clusters == i]
        if i < len(colors):  # Kiểm tra xem chỉ số i có trong phạm vi của colors không
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f'Cluster {i}')
        else:
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')
        plt.scatter(centroids[i][0], centroids[i][1], c='black', marker='x', s=100, label=f'Centroid {i}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Clusters')
    plt.legend()

    save_path = 'graph'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    chart_counter += 1
    chart_filename = os.path.join(save_path, f'clusters_{chart_counter}.png')

    plt.savefig(chart_filename)
    print({chart_filename})

    plt.show()

# Sửa hàm kmeans để nhận thêm k
def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        distances = compute_distances(data, centroids)
        clusters = assign_clusters(distances)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Đọc dữ liệu từ file CSV
data = read_data('data/file_normalized.csv')

# Chọn chỉ 5 thuộc tính
data_selected = select_features(data)

# Chọn số cụm và chạy thuật toán KMeans
k = 5
clusters, centroids = kmeans(data_selected, k)

# In danh sách các tâm sau khi hoàn thành thuật toán
print("Centroids:", centroids)

# Vẽ biểu đồ các cụm
plot_clusters(data_selected, clusters, centroids)
