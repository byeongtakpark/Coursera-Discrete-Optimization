import numpy as np
import torch

def nearest_neighbor_tsp(points):
    """
    Nearest Neighbor Heuristic to solve TSP inside a cluster.
    
    Parameters:
    - points: List of (x, y) coordinates

    Returns:
    - tour: List of indices representing the TSP path within the cluster
    """
    
    num_points = len(points)
    if num_points <= 1:
        return list(range(num_points))  # Return single point or empty

    visited = [False] * num_points
    tour = [0]  # Start from first node
    visited[0] = True

    for _ in range(num_points - 1):
        last = tour[-1]
        nearest = None
        min_dist = float("inf")
        for i in range(num_points):
            if not visited[i]:
                dist = np.linalg.norm(np.array(points[last]) - np.array(points[i]))
                if dist < min_dist:
                    min_dist = dist
                    nearest = i
        tour.append(nearest)
        visited[nearest] = True

    return tour

def kmeans_pytorch(X, num_clusters=10, batch_size=1000, max_iters=100, tol=1e-4, device=None):
    """
    Batch-based K-Means Clustering in PyTorch.
    
    Returns:
    - labels: Cluster assignments for each point
    - centroids: Cluster center coordinates
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(X, dtype=torch.float32, device=device)  # Move data to device
    N, D = X.shape  

    centroids = X[torch.randperm(N)[:num_clusters]].clone()
    prev_centroids = torch.zeros_like(centroids)
    labels = torch.zeros(N, dtype=torch.long, device=device)

    for i in range(max_iters):
        distances = torch.cdist(X, centroids)
        labels = torch.argmin(distances, dim=1)

        for k in range(num_clusters):
            mask = labels == k
            if mask.any():
                centroids[k] = X[mask].mean(dim=0)

        shift = torch.norm(centroids - prev_centroids, p=2, dim=1).mean().item()
        if shift < tol:
            break
        prev_centroids = centroids.clone()

    return labels.cpu().numpy(), centroids.cpu().numpy()

def generate_initial_solution(points, num_clusters=10):
    """
    Generates an initial TSP solution by solving TSP within each cluster 
    and connecting clusters in order of their centroids.

    Parameters:
    - points: List of Point objects (parsed from parse_input_data)
    - num_clusters: Number of clusters for K-Means

    Returns:
    - initial_solution: List of node indices in the TSP tour
    """
    
    # Convert points to numpy array
    coords = np.array([[point.x, point.y] for point in points], dtype=np.float32)

    # Run PyTorch K-Means
    labels, centroids = kmeans_pytorch(coords, num_clusters=num_clusters)

    # Group nodes by cluster
    cluster_dict = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(labels):
        cluster_dict[label].append(i)

    # Solve TSP inside each cluster
    cluster_tours = {}
    for cluster_id, node_indices in cluster_dict.items():
        cluster_points = [coords[i] for i in node_indices]
        local_tour = nearest_neighbor_tsp(cluster_points)
        cluster_tours[cluster_id] = [node_indices[i] for i in local_tour]

    # Order clusters by centroid positions (left to right, top to bottom)
    cluster_order = np.argsort(centroids[:, 0])  # Sort clusters by X-coordinate

    # Construct full tour: visit clusters in order and traverse each local TSP path
    initial_solution = []
    for cluster_id in cluster_order:
        initial_solution.extend(cluster_tours[cluster_id])

    return initial_solution
