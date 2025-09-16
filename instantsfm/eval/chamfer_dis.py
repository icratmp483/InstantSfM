from argparse import ArgumentParser
import os
import numpy as np
from scipy.spatial import KDTree
from instantsfm.utils.read_write_model import read_points3D_binary, read_points3D_text
# others: https://arxiv.org/pdf/2412.17951

def chamfer_distance_kdtree(point_cloud_a, point_cloud_b):
    A = np.asarray(point_cloud_a)
    B = np.asarray(point_cloud_b)
    
    # Build KD-Trees for efficient nearest-neighbor search
    tree_a = KDTree(A)
    tree_b = KDTree(B)
    
    # Query nearest neighbors
    dist_a_to_b, _ = tree_b.query(A)  # Distance from A to nearest in B
    dist_b_to_a, _ = tree_a.query(B)  # Distance from B to nearest in A
    
    return np.mean(dist_a_to_b) + np.mean(dist_b_to_a)

def load_point_cloud(file_path):
    if os.path.basename(file_path).endswith(".ply"):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)
    elif os.path.basename(file_path).endswith("points3D.txt"):
        points3D = read_points3D_text(file_path)
        return np.stack([point.xyz for point in points3D.values()])
    elif os.path.basename(file_path).endswith("points3D.bin"):
        points3D = read_points3D_binary(file_path)
        return np.stack([point.xyz for point in points3D.values()])

if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments: paths to the two point cloud files
    parser.add_argument("point_cloud_a", type=str, 
                        help="Path to the first point cloud file")
    parser.add_argument("point_cloud_b", type=str, 
                        help="Path to the second point cloud file")
    
    args = parser.parse_args()

    point_cloud_a = load_point_cloud(args.point_cloud_a)
    point_cloud_b = load_point_cloud(args.point_cloud_b)
    result = chamfer_distance_kdtree(point_cloud_a, point_cloud_b)
    print(f"Chamfer distance between the two point clouds: {result}")

