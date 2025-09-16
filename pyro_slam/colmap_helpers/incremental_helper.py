import argparse
import pathlib
import subprocess

import numpy as np
import pycolmap


def colmap_depth(image_dir, output_path, max_num_features):
    output_path: pathlib.Path = pathlib.Path(output_path)
    image_dir: pathlib.Path = pathlib.Path(image_dir)
    database_path = output_path / "database.db"
    output_path.mkdir(parents=True, exist_ok=True)
    sparse_output_path = output_path / "sparse"
    sparse_output_path.mkdir(parents=True, exist_ok=True)

    sift_extration_options = pycolmap.SiftExtractionOptions(
        estimate_affine_shape=True,
        domain_size_pooling=True,
        max_num_features=max_num_features, 
    )
    pycolmap.extract_features(database_path, image_dir, sift_options=sift_extration_options)
    # verification_options = pycolmap.TwoViewGeometryOptions
    sift_matching_options = pycolmap.SiftMatchingOptions(
        guided_matching=True, # disabled: 2500, enabled: 1500
    )

    pycolmap.match_exhaustive(database_path, sift_options=sift_matching_options)
    mapper_cmd = [
        "/home/csgrad/zzhan4/colmap/build/src/colmap/exe/colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_output_path),
        "--Mapper.ba_global_max_num_iterations", "0",
        "--Mapper.ba_local_max_num_iterations", "0",
        "--Mapper.ba_global_max_refinements", "0",
        "--Mapper.ba_local_max_refinements", "0",
        "--Mapper.ba_local_max_refinement_change", "0.0",
        "--Mapper.ba_global_max_refinement_change", "0.0",
    ]
    subprocess.run(mapper_cmd, check=True)
    

def main():
    parser = argparse.ArgumentParser(description="Run a series of COLMAP commands with specified parameters.")
    
    parser.add_argument("--image_path", type=str, required=True, help="Path to the images")
    parser.add_argument("--result_folder", type=str, required=True, help="Folder to store the final result")
    parser.add_argument("--max_num_features", type=int, default=1300, help="Maximum number of matches to keep")

    args = parser.parse_args()

    colmap_depth(
        args.image_path,
        args.result_folder,
        args.max_num_features,
    )

if __name__ == "__main__":
    main()


