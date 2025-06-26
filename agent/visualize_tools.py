import os
from pathlib import Path
from tools.visualize_image_tools import visualize_result
from agent.memory import MemoryController


def visualize_crack_result(
    subject_name: str,
    memory: MemoryController,
    visual_types: list = ["original", "mask"],
    show: bool = True,
    save_path: str = None
):
    """
    可视化原图和分割掩膜图。

    参数：
        subject_name: 图像名（不含扩展名）
        memory: MemoryController 实例
        visual_types: ["original", "mask"]
        show: 是否显示（默认 True）
        save_path: 保存路径（可选）
    """

    image_path = Path(f"data/Test_images/{subject_name}.jpg").resolve()
    mask_path_raw = memory.get_mask_path(subject_name)
    mask_path = Path(mask_path_raw).resolve() if mask_path_raw else None

    img_exists = image_path.exists()
    mask_exists = mask_path and mask_path.exists()

    print(f"[DEBUG] 原图路径: {image_path} | 存在: {img_exists}")
    print(f"[DEBUG] 掩膜路径: {mask_path} | 存在: {mask_exists}")

    show_img = str(image_path) if "original" in visual_types and img_exists else None
    show_mask = str(mask_path) if "mask" in visual_types and mask_exists else None

    if not any([show_img, show_mask]):
        print("⚠️ 没有可视化内容可用")
        return

    visualize_result(
        image_path=show_img,
        mask_path=show_mask,
        overlay=False,
        max_width_path=None,
        save_path=save_path if not show else None,
        title=f"原图与分割图 - {subject_name}"
    )
