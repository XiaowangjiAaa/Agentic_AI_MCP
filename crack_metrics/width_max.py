import numpy as np
import cv2
from scipy.spatial import cKDTree
from .skeleton import extract_skeleton_and_normals

def compute_max_width_px(mask: np.ndarray) -> float:
    """
    基于骨架点到轮廓的成对距离，计算裂缝最大宽度（单位：像素）。
    """
    _, skeleton_points, _ = extract_skeleton_and_normals(mask)
    if len(skeleton_points) == 0:
        return 0.0

    # 提取轮廓点
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0

    contour_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    tree = cKDTree(contour_pts)

    max_dist = 0.0
    for pt in skeleton_points:
        dists, idxs = tree.query(pt, k=2)
        if len(idxs) == 2:
            p1, p2 = contour_pts[idxs[0]], contour_pts[idxs[1]]
            dist = np.linalg.norm(p1 - p2)
            max_dist = max(max_dist, dist)

    return round(max_dist, 2)
