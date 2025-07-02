import numpy as np
from .skeleton import extract_skeleton_and_normals

def compute_average_width_px(mask: np.ndarray) -> float:
    """
    计算裂缝的平均宽度（单位：像素）= 区域面积 / 骨架长度
    """
    area = np.sum(mask)
    _, skeleton_points, _ = extract_skeleton_and_normals(mask)

    length = len(skeleton_points)
    if length == 0:
        return 0.0

    avg_width = area / length
    return round(avg_width, 2)
