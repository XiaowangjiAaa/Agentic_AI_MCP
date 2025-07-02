import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.neighbors import KDTree
from typing import Tuple
from skimage.morphology import thin, remove_small_objects


def extract_skeleton_and_normals(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用简洁版骨架化方法（学习自你提供的脚本），结合 thin + remove_small_objects 清理。
    输入：
        mask: 二值掩膜图，背景为 0，裂缝区域为 1
    返回：
        skeleton_mask: 骨架二值图（0/1）
        skeleton_points: (N, 2) 骨架点坐标 [x, y]
        normals: 全为零数组（占位，为接口兼容性保留）
    """
    # Step 1: 骨架提取（你提供的方法）
    skeleton = skeletonize(mask > 0)

    # Step 2: 可选进一步细化（你参考中使用了 thin）
    skeleton = thin(skeleton)

    # Step 3: 去除小杂点
    skeleton = remove_small_objects(skeleton, min_size=1)

    # Step 4: 提取骨架点
    ys, xs = np.where(skeleton > 0)
    skeleton_points = np.stack([xs, ys], axis=1)

    # Step 5: 返回值（兼容原函数格式）
    skeleton_mask = skeleton.astype(np.uint8)
    normals = np.zeros_like(skeleton_points, dtype=np.float32)  # 占位
    return skeleton_mask, skeleton_points, normals