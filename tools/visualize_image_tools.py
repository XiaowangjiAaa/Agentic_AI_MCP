import os
import cv2
import matplotlib.pyplot as plt
from typing import Optional


def visualize_result(
    image_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    overlay: bool = True,
    max_width_path: Optional[str] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    灵活可扩展的可视化工具：支持原图、分割图、宽度图等的组合展示。
    
    参数：
        image_path: 原始图像路径
        mask_path: 分割掩膜路径
        overlay: 是否将掩膜叠加到原图
        max_width_path: 最大宽度可视化图路径（可选）
        save_path: 若提供则保存为图片，否则直接 plt.show()
        title: 图像标题
    """
    if not image_path and not mask_path and not max_width_path:
        raise ValueError("必须至少提供一个图像路径")

    visual_items = []

    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visual_items.append(("原始图像", image))

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if overlay and image_path:
            overlay_img = image.copy()
            overlay_img[mask > 0] = [255, 0, 0]  # 红色叠加
            visual_items.append(("原图 + 分割掩膜", overlay_img))
        elif not overlay:
            visual_items.append(("分割掩膜", mask))

    if max_width_path:
        width_vis = cv2.imread(max_width_path)
        width_vis = cv2.cvtColor(width_vis, cv2.COLOR_BGR2RGB)
        visual_items.append(("最大宽度图", width_vis))

    n = len(visual_items)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (sub_title, img) in zip(axes, visual_items):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(sub_title)
        ax.axis('off')

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ 可视化结果已保存: {save_path}")
    else:
        plt.show()

    plt.close(fig)