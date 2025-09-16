import numpy as np

def sample_depth_at_pixel(depth_map, pixel_coords, w, h, method='nearest'):
    """
    Args:
        depth_map: [H, W]
        pixel_coords: [2]
        w, h: width and height of original rgb image
        method: 'bilinear' or 'nearest'
    """
    H, W = depth_map.shape
    x, y = pixel_coords[0], pixel_coords[1]
    x_percent, y_percent = x / w, y / h
    if x_percent < 0 or x_percent > 1 or y_percent < 0 or y_percent > 1:
        # print(f"Warning: pixel coordinates {pixel_coords} are outside the range of the image with shape {w, h}")
        return 1.0, False
    # convert to size of depth map
    x_converted, y_converted = x_percent * W, y_percent * H

    if method == 'nearest':
        x_int = x_converted.astype(int)
        y_int = y_converted.astype(int)
        depth = depth_map[y_int, x_int]
    else:  # bilinear
        x0 = np.floor(x_converted).astype(int)
        x1 = (x0 + 1).clip(0, W-1)
        y0 = np.floor(y_converted).astype(int)
        y1 = (y0 + 1).clip(0, H-1)

        wx = x_converted - x0
        wy = y_converted - y0

        d00 = depth_map[y0, x0]
        d01 = depth_map[y1, x0]
        d10 = depth_map[y0, x1]
        d11 = depth_map[y1, x1]

        depth = (d00 * (1-wx) * (1-wy) + 
                 d10 * wx * (1-wy) + 
                 d01 * (1-wx) * wy + 
                 d11 * wx * wy)

    # if depth is unavailable, set it to 1.0
    depth_available = depth > 0.0
    depth = depth if depth_available else 1.0

    return depth, depth_available