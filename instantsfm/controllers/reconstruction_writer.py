import numpy as np
import os

from instantsfm.scene.reconstruction import Reconstruction, point3d

def ExportReconstruction(output_path, cameras, images, tracks, image_path, cluster_id=-1, include_image_points=False):
    # Add cameras
    reconstruction = Reconstruction(cameras=cameras)

    images_selected = {}
    for image in images:
        if cluster_id != -1 and image.cluster_id != cluster_id:
            continue
        if not image.is_registered:
            continue
        images_selected[image.id] = image
    images = images_selected

    # Prepare the 2d-3d correspondences
    image2point3d = {}
    if len(tracks) > 0 or include_image_points:
        for image_id, image in images.items():
            image2point3d[image_id] = -np.ones(len(image.features), dtype=int)

        if len(tracks) > 0:
            for track_id, track in tracks.items():
                if len(track.observations) < 3:
                    continue
                for obs in track.observations:
                    if obs[0] in image2point3d:
                        image2point3d[obs[0]][obs[1]] = track_id
    
    # Add points
    for track_id, track in tracks.items():
        point = point3d(xyz=track.xyz, color=track.color, error=0, track_elements=track.observations)
        if len(point.track_elements) >= 2:
            reconstruction.point3d[track_id] = point
    print(f'Exporting {len(reconstruction.point3d)} points')
    
    # Add images
    for image_id, image in images.items():
        image.point3d_ids = -np.ones(len(image.features), dtype=int)
        if image_id in image2point3d:
            track_ids = image2point3d[image_id]
            for idx, feat in enumerate(image.features):
                if track_ids[idx] != -1:
                    if image.point3d_ids[idx] == -1:
                        image.num_points3d += 1
                    image.point3d_ids[idx] = track_ids[idx]
    reconstruction.images = images

    if image_path != "":
        reconstruction.ExtractColorsForAllImages(image_path)
    
    cluster_path = os.path.join(output_path, '0' if cluster_id == -1 else str(cluster_id))
    os.makedirs(cluster_path, exist_ok=True)
    reconstruction.WriteBinary(cluster_path)

def WriteGlomapReconstruction(output_path, cameras, images, tracks, image_path):
    largest_component_num = -1
    for image in images:
        if hasattr(image, 'cluster_id') and image.cluster_id > largest_component_num:
            largest_component_num = image.cluster_id
    
    if largest_component_num == -1:
        ExportReconstruction(output_path, cameras, images, tracks, image_path)
    else:
        for i in range(largest_component_num):
            print(f'Exporting reconstruction {i+1} / {largest_component_num+1}')
            ExportReconstruction(output_path + f'_{i}', cameras, images, tracks, image_path, i)