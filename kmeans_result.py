import csv
import numpy as np

# Đọc dữ liệu từ file CSV, bỏ qua hàng đầu tiên (tiêu đề)
def read_data(filename, columns):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Bỏ qua hàng đầu tiên (tiêu đề)
        for row in csv_reader:
            data.append([float(x) for x in row[:columns]])  # Lấy số lượng cột muốn lấy từ cột đầu tiên
    return np.array(data)

# Khởi tạo các điểm trung tâm từ input của người dùng
def initialize_centroids(k, data):
    if k == 3:
        centroids_indices = [1, 50, 150]
    elif k == 5:
        centroids_indices = [1, 50, 150, 200, 250]
    else:
        raise ValueError("Unsupported value of k")
    centroids = data[centroids_indices]
    return centroids

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

# phân cụm bằng kmeans
def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(k, data)
    for _ in range(max_iterations):
        distances = compute_distances(data, centroids)
        clusters = assign_clusters(distances)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters

# Ghi kết quả phân cụm vào file CSV
def write_result_to_csv(clusters, output_file):
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Cluster'])  # Tiêu đề của các cột
        for idx, cluster in enumerate(clusters):
            writer.writerow([idx, cluster])

# File input và output
input_file = 'data/file_normalized.csv'
output_file = 'result/output.csv'

# Số lượng cụm và số cột muốn lấy từ cột đầu tiên
k = 5
columns = 6

# Đọc dữ liệu từ file CSV
data = read_data(input_file, columns)

# Thực hiện phân cụm bằng thuật toán KMeans
clusters = kmeans(data, k)

# Ghi kết quả phân cụm vào file CSV
write_result_to_csv(clusters, output_file)

print("Clustering result has been saved to", output_file)
