import cv2
import numpy as np
import matplotlib.pyplot as plt
from Crack_quantification_tools.binarize import binarize
from Crack_quantification_tools.skeleton import extract_skeleton_and_normals
from tools.visualize import visualize_max_width

# 1. 读取图像并 binarize
mask_path = "outputs/masks/7_crack.png"
img_raw = cv2.imread(mask_path)
mask = binarize(img_raw)

# 2. 提取骨架点和法向
_, centers, normals = extract_skeleton_and_normals(mask)

# 3. 可视化骨架点（红色）
overlay = img_raw.copy()
for pt in centers:
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(overlay, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

# 4. 计算并可视化最大宽度线段
vis_width, max_width = visualize_max_width(img_raw)

# 5. 显示两个图像对比
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Skeleton Overlay")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Max Width Visualization (max = {max_width:.2f})")
plt.imshow(cv2.cvtColor(vis_width, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
