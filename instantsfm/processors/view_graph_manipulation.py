
from instantsfm.scene.defs import ConfigurationType, PairId2Ids, ViewGraph
from instantsfm.utils.two_view_geometry import F_from_motion_and_camera


def UpdateImagePairsConfig(view_graph:ViewGraph, cameras, images):
    camera_counter_total, camera_counter_calib = [0 for _ in cameras], [0 for _ in cameras]
    for pair_id, image_pair in view_graph.image_pairs.items():
        if not image_pair.is_valid:
            continue
        image_id1, image_id2 = PairId2Ids(pair_id)
        cam_id1 = images[image_id1].cam_id
        cam_id2 = images[image_id2].cam_id
        camera1 = cameras[cam_id1]
        camera2 = cameras[cam_id2]
        if not camera1.has_prior_focal_length or not camera2.has_prior_focal_length:
            continue
        if image_pair.config == ConfigurationType.CALIBRATED:
            camera_counter_total[cam_id1] += 1
            camera_counter_total[cam_id2] += 1
            camera_counter_calib[cam_id1] += 1
            camera_counter_calib[cam_id2] += 1
        elif image_pair.config == ConfigurationType.UNCALIBRATED:
            camera_counter_total[cam_id1] += 1
            camera_counter_total[cam_id2] += 1
    
    camera_validity = [True for _ in cameras]
    for cam_id in range(len(cameras)):
        first = camera_counter_total[cam_id]
        second = camera_counter_calib[cam_id]
        if first == 0:
            camera_validity[cam_id] = False
        elif second/first < 0.5:
            camera_validity[cam_id] = False
    
    for pair_id, image_pair in view_graph.image_pairs.items():
        if not image_pair.is_valid or not image_pair.config == ConfigurationType.UNCALIBRATED:
            continue
        image_id1, image_id2 = PairId2Ids(pair_id)
        cam_id1 = images[image_id1].cam_id
        cam_id2 = images[image_id2].cam_id
        if camera_validity[cam_id1] and camera_validity[cam_id2]:
            image_pair.config = ConfigurationType.CALIBRATED

def DecomposeRelPose(view_graph:ViewGraph, cameras, images):
    image_pair_ids = []
    for pair_id, image_pair in view_graph.image_pairs.items():
        if not image_pair.is_valid:
            continue
        image_id1, image_id2 = PairId2Ids(pair_id)
        cam_id1 = images[image_id1].cam_id
        cam_id2 = images[image_id2].cam_id
        if not cameras[cam_id1].has_prior_focal_length or not cameras[cam_id2].has_prior_focal_length:
            continue
        image_pair_ids.append(pair_id)
    
    print('Decomposing relative poses for', len(image_pair_ids), 'pairs')

    for pair_id in image_pair_ids:
        image_pair = view_graph.image_pairs[pair_id]
        image_id1, image_id2 = PairId2Ids(pair_id)
        cam_id1 = images[image_id1].cam_id
        cam_id2 = images[image_id2].cam_id
        camera1 = cameras[cam_id1]
        camera2 = cameras[cam_id2]
        if image_pair.config == ConfigurationType.PLANAR and camera1.has_prior_focal_length and camera2.has_prior_focal_length:
            image_pair.config = ConfigurationType.CALIBRATED

    counter = 0
    for pair_id in image_pair_ids:
        image_pair = view_graph.image_pairs[pair_id]
        if not image_pair.config in [ConfigurationType.CALIBRATED, ConfigurationType.PLANAR_OR_PANORAMIC]:
            counter += 1
    print(f'Decompose relative pose done. {counter} pairs are pure rotation.')