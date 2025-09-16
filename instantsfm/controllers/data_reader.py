import numpy as np
import time
import os
import cv2
import glob

from instantsfm.scene.defs import Image, ImagePair, Camera, ConfigurationType, PairId2IdsInversed, CameraModelId, Ids2PairId, ViewGraph
from instantsfm.utils.database import COLMAPDatabase, blob_to_array

class PathInfo:
    def __init__(self):
        self.image_path = ""
        self.database_path = ""
        self.output_path = ""
        self.database_exists = False
        self.depth_path = ""

def ReadData(path) -> PathInfo:
    path_info = PathInfo()
    if os.path.exists(os.path.join(path, 'images')):
        # COLMAP format
        path_info.image_path = os.path.join(path, 'images')
    elif os.path.exists(os.path.join(path, 'color')):  
        # ScanNet format
        path_info.image_path = os.path.join(path, 'color')
    
    path_info.database_path = os.path.join(path, 'database.db')
    path_info.output_path = os.path.join(path, 'sparse')
    path_info.database_exists = os.path.exists(path_info.database_path)
    if os.path.exists(os.path.join(path, 'depth')):
        path_info.depth_path = os.path.join(path, 'depth')

    return path_info

def ReadColmapDatabase(path):
    start_time = time.time()
    view_graph = ViewGraph()
    db = COLMAPDatabase.connect(path)
    
    images = {id: Image(id=id, filename=filename, cam_id=cam_id) for id, filename, cam_id in db.execute("SELECT image_id, name, camera_id FROM images")}
    cameras = {id: Camera(id=id, model_id=CameraModelId(model_id), width=width, height=height, params=blob_to_array(params, np.float64),
                          has_prior_focal_length=prior_focal_length > 0)
                          for id, model_id, width, height, params, prior_focal_length in db.execute("SELECT * FROM cameras")}
    for cam in cameras.values():
        cam.set_params(cam.params)
    
    keypoints = [(image_id, blob_to_array(data, np.float32, (-1, cols)))
                 for image_id, cols, data in db.execute("SELECT image_id, cols, data FROM keypoints") if not data is None]
    for image_id, data in keypoints:
        images[image_id].features = data[:, :2]

    query = """
    SELECT m.pair_id, m.data, t.config, t.F, t.E, t.H
    FROM matches AS m
    INNER JOIN two_view_geometries AS t ON m.pair_id = t.pair_id
    """
    matches_and_geometries = db.execute(query)
    image_pairs = {}
    invalid_count = 0

    for group in matches_and_geometries:
        pair_id, data, config, F_blob, E_blob, H_blob = group
        if data is None:
            invalid_count += 1
            continue
        data = blob_to_array(data, np.uint32, (-1, 2))
        image_id1, image_id2 = PairId2IdsInversed(pair_id)
        image_pairs[pair_id] = ImagePair(image_id1=image_id1, image_id2=image_id2)
        keypoints1 = images[image_id1].features
        keypoints2 = images[image_id2].features
        idx1 = data[:, 0]
        idx2 = data[:, 1]
        valid_indices = (idx1 != -1) & (idx2 != -1) & (idx1 < len(keypoints1)) & (idx2 < len(keypoints2))
        valid_matches = data[valid_indices]
        image_pairs[pair_id].matches = valid_matches

        config = ConfigurationType(config)
        image_pairs[pair_id].config = config
        if config in [ConfigurationType.UNDEFINED, ConfigurationType.DEGENERATE, ConfigurationType.WATERMARK, ConfigurationType.MULTIPLE]:
            image_pairs[pair_id].is_valid = False
            invalid_count += 1
            continue

        F = blob_to_array(F_blob, np.float64).reshape(3, 3)
        E = blob_to_array(E_blob, np.float64).reshape(3, 3)
        H = blob_to_array(H_blob, np.float64).reshape(3, 3)
        image_pairs[pair_id].F = F
        image_pairs[pair_id].E = E
        image_pairs[pair_id].H = H
        image_pairs[pair_id].config = config

    view_graph.image_pairs = {pair_id: image_pair for pair_id, image_pair in image_pairs.items() if image_pair.is_valid}
    print(f'Pairs read done. {invalid_count} / {len(image_pairs)+invalid_count} are invalid')

    # We convert the storage type to list here. Images and Cameras are converted, while ViewGraph.image_pairs remains dict for its complexity
    cam_id2idx = {cam_id:idx for idx, cam_id in enumerate(cameras.keys())}
    cameras = [cam for cam in cameras.values()]
    img_id2idx = {img_id:idx for idx, img_id in enumerate(images.keys())}
    images = [image for image in images.values()]
    for cam in cameras:
        cam.id = cam_id2idx[cam.id]
    for image in images:
        image.id = img_id2idx[image.id]
        image.cam_id = cam_id2idx[image.cam_id]
    for pair in view_graph.image_pairs.values():
        pair.image_id1 = img_id2idx[pair.image_id1]
        pair.image_id2 = img_id2idx[pair.image_id2]
    view_graph.image_pairs = {Ids2PairId(pair.image_id1, pair.image_id2): pair for pair in view_graph.image_pairs.values()}
    print(f'Reading database took: {time.time() - start_time:.2f}')

    try:
        feature_name = db.execute("SELECT feature_name FROM feature_name").fetchone()[0]
    except:
        # if the database does not have feature_name, then assume it's originated from COLMAP-compatibale workflow
        feature_name = 'colmap'

    return view_graph, cameras, images, feature_name

def ReadDepths(path):
    depth_files = sorted(glob.glob(os.path.join(path, '*.png')))
    depths = []
    for depth_file in depth_files:
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        # ScanNet format: depth represents millimeters
        depth = depth.astype(np.float32) / 1000.0
        depths.append(depth)
    return np.array(depths, dtype=np.float32)