import numpy as np
from .skeleton import extract_skeleton_and_normals

def compute_crack_length_px(mask: np.ndarray) -> int:
    """
    计算裂缝长度（单位：像素），使用骨架点数量作为估计值。
    """
    _, skeleton_points, _ = extract_skeleton_and_normals(mask)
    return len(skeleton_points)
