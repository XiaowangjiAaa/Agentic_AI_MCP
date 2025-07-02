import cv2
import numpy as np

def binarize(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    将图像二值化为 0/1 掩膜。
    输入图像可以为彩色或灰度，输出为 0/1 类型的 uint8 数组。
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 自动适配：若图像为 float（0~1），先转换为 0~255
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)

    _, binary = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)
