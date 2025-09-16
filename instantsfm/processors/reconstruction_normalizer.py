import numpy as np

def NormalizeReconstruction(images, tracks, fixed_scale=False, extent=10., p0=0.1, p1=0.9):
    coords = np.array([image.center() for image in images])
    coords_sorted = np.sort(coords, axis=0)
    P0 = int(p0 * (coords.shape[0] - 1)) if coords.shape[0] > 3 else 0
    P1 = int(p1 * (coords.shape[0] - 1)) if coords.shape[0] > 3 else coords.shape[0] - 1
    bbox_min = coords_sorted[P0]
    bbox_max = coords_sorted[P1]

    mean_coord = np.mean(coords_sorted[P0:P1+1], axis=0)
    scale = 1.
    if not fixed_scale:
        old_extent = np.linalg.norm(bbox_max - bbox_min)
        if old_extent >= 1e-6:
            scale = extent / old_extent
    
    coords = (coords - mean_coord) * scale
    for idx, image in enumerate(images):
        image.world2cam[:3, 3] = -image.world2cam[:3, :3] @ coords[idx]
    for track in tracks.values():
        track.xyz = (track.xyz - mean_coord) * scale