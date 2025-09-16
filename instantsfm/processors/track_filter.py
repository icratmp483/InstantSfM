import numpy as np

EPSILON = 1e-10

def FilterTracksByAngle(cameras, images, tracks, max_angle_error):
    counter = 0
    thres = np.cos(np.deg2rad(max_angle_error))
    for track in tracks.values():
        valid_idx = []
        for idx, (image_id, feature_id) in enumerate(track.observations):
            image = images[image_id]
            feature_undist = image.features_undist[feature_id]
            pt_calc = image.world2cam[:3, :3] @ track.xyz + image.world2cam[:3, 3]
            if pt_calc[2] < EPSILON:
                continue
            pt_calc = pt_calc / np.linalg.norm(pt_calc)
            if np.dot(pt_calc, feature_undist) > thres:
                valid_idx.append(idx)
        
        if len(valid_idx) != len(track.observations):
            counter += 1
            track.observations = track.observations[valid_idx]
    print(f'Filtered {counter} / {len(tracks)} tracks by angle error')
    return tracks

def FilterTracksByReprojectionNormalized(cameras, images, tracks, max_reprojection_error):
    counter = 0
    image_world2cams = np.array([image.world2cam for image in images]) # (M, 4, 4)
    track_id2idx = {track_id: idx for idx, track_id in enumerate(tracks.keys())}
    track_xyzs = np.hstack([np.array([track.xyz for track in tracks.values()]), np.ones((len(tracks), 1))]) # (N, 4)
    image_ids = []
    
    track_idx = []
    features_undist = []
    for track_id, track in tracks.items():
        track_idx += [track_id2idx[track_id]] * track.observations.shape[0]
        image_ids.append(track.observations[:, 0])
        for (image_id, feature_id) in track.observations:
            image = images[image_id]
            feature_undist = image.features_undist[feature_id]
            features_undist.append(feature_undist)

    image_ids = np.concatenate(image_ids) # (X)
    image_world2cams = image_world2cams[image_ids] # (X, 4, 4)
    track_idx = np.array(track_idx) # (X)
    track_xyzs = track_xyzs[track_idx] # (X, 4)
    features_undist = np.array(features_undist) # (X, 3)
    features_undist_reproj = features_undist[:, :2] / (features_undist[:, 2:] + EPSILON) # (X, 2)

    pts_calc = np.einsum('ijk,ik->ij', image_world2cams, track_xyzs) # (X, 4)
    pts_calc = pts_calc[:, :3] # (X, 3)
    valid_mask = pts_calc[:, 2] > EPSILON # (X)
    pts_reproj = pts_calc[:, :2] / (pts_calc[:, 2:] + EPSILON) # (X, 2)
    reprojection_errors = np.linalg.norm(pts_reproj - features_undist_reproj, axis=1) # (X)
    valid_mask = valid_mask & (reprojection_errors < max_reprojection_error) # (X)

    count = 0
    for track_id, track in tracks.items():
        obs_count = track.observations.shape[0]
        track.observations = track.observations[valid_mask[count:count + obs_count]]
        count += obs_count
        if not np.all(valid_mask[count:count + obs_count]):
            counter += 1

    print(f'Filtered {counter} / {len(tracks)} tracks by reprojection error')
    return counter

def FilterTracksByReprojection(cameras, images, tracks, max_reprojection_error):
    counter = 0
    image_world2cams = np.array([image.world2cam for image in images]) # (M, 4, 4)
    camera_indices = np.array([image.cam_id for image in images]) # (M)
    track_id2idx = {track_id: idx for idx, track_id in enumerate(tracks.keys())}
    track_xyzs = np.hstack([np.array([track.xyz for track in tracks.values()]), np.ones((len(tracks), 1))]) # (N, 4)
    image_ids = []
    
    track_idx = []
    features = []
    for track_id, track in tracks.items():
        track_idx += [track_id2idx[track_id]] * track.observations.shape[0]
        image_ids.append(track.observations[:, 0])
        for (image_id, feature_id) in track.observations:
            image = images[image_id]
            feature = image.features[feature_id]
            features.append(feature)

    image_ids = np.concatenate(image_ids) # (X)
    image_world2cams = image_world2cams[image_ids] # (X, 4, 4)
    camera_indices = camera_indices[image_ids] # (X)
    track_idx = np.array(track_idx) # (X)
    track_xyzs = track_xyzs[track_idx] # (X, 4)
    features = np.array(features) # (X, 2)

    pts_calc = np.einsum('ijk,ik->ij', image_world2cams, track_xyzs) # (X, 4)
    pts_calc = pts_calc[:, :3] # (X, 3)
    valid_mask = pts_calc[:, 2] > EPSILON # (X)
    pts_reproj = np.zeros((len(features), 2))
    for i in range(len(cameras)):
        camera_mask = camera_indices == i
        cam = cameras[i]
        pts_reproj[camera_mask] = cam.cam2img(pts_calc[camera_mask])

    reprojection_errors = np.linalg.norm(pts_reproj - features, axis=1) # (X)
    valid_mask = valid_mask & (reprojection_errors < max_reprojection_error) # (X)

    count = 0
    for track_id, track in tracks.items():
        obs_count = track.observations.shape[0]
        track.observations = track.observations[valid_mask[count:count + obs_count]]
        count += obs_count
        if not np.all(valid_mask[count:count + obs_count]):
            counter += 1

    print(f'Filtered {counter} / {len(tracks)} tracks by reprojection error')
    return counter

def FilterTracksTriangulationAngle(cameras, images, tracks, min_angle):
    counter = 0
    thres = np.cos(np.deg2rad(min_angle))
    image_centers = np.array([image.center() for image in images])

    for track_id in list(tracks.keys()):
        track = tracks[track_id]
        pts_calc = []
        
        unique_image_ids = np.unique(track.observations[:, 0])
        vectors = track.xyz - image_centers[unique_image_ids]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        pts_calc = vectors / (norms + EPSILON)
        result_matrix = pts_calc @ pts_calc.T

        # If the triangulation angle is too small, just remove it
        if np.all(result_matrix > thres):
            del tracks[track_id]
            counter += 1

    print(f'Filtered {counter} / {counter + len(tracks)} tracks by too small triangulation angle')
    return counter
