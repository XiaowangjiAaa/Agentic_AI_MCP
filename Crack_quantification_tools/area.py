import numpy as np

def compute_crack_area_px(mask: np.ndarray) -> int:
    """
    计算裂缝像素面积（白色像素数）
    支持 0/1 或 0/255 二值图。
    """
    binary = (mask > 0).astype(np.uint8)
    return int(np.sum(binary))
