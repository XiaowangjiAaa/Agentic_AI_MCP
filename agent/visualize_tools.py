import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import cv2
from utils.visualize_image_tools import visualize_result
from agent.memory import MemoryController
from task_tools.quantify import quantify_crack_geometry


def visualize_crack_result(
    subject_name: str,
    memory: MemoryController,
    visual_types: list = ["original", "mask"],
    show: bool = True,
    save_dir: str = "outputs/visuals"
) -> dict:
    """
    可视化裂缝图像及其相关图层，若 memory 未命中则自动 fallback 调用 quantify 工具生成。
    """
    from pathlib import Path
    import shutil
    import matplotlib.pyplot as plt
    import cv2
    from utils.visualize_image_tools import visualize_result

    image_path = Path(f"data/Test_images/{subject_name}.jpg").resolve()
    mask_path_raw = memory.get_mask_path(subject_name)
    mask_path = Path(mask_path_raw).resolve() if mask_path_raw else None

    vis_paths = {}
    os.makedirs(save_dir, exist_ok=True)

    # 原图
    if "original" in visual_types and image_path.exists():
        save_path = os.path.join(save_dir, f"{subject_name}_original.png")
        shutil.copy(str(image_path), save_path)
        vis_paths["original"] = save_path

    # 掩膜图
    if "mask" in visual_types and mask_path and mask_path.exists():
        save_path = os.path.join(save_dir, f"{subject_name}_mask.png")
        shutil.copy(str(mask_path), save_path)
        vis_paths["mask"] = save_path

    # ✅ skeleton / max_width fallback 自动生成
    fallback_visuals = []
    for vt in visual_types:
        if vt in {"skeleton", "max_width", "normals"}:
            cached = memory.get_visualization_path(subject_name, vt)
            if cached and os.path.exists(cached):
                vis_paths[vt] = cached
            else:
                fallback_visuals.append(vt)

    if fallback_visuals:
        # 获取必要参数
        mask_path_str = str(mask_path) if mask_path else None
        pixel_size = memory.get_pixel_size(subject_name) or 0.5  # fallback 默认像素尺寸

        if mask_path_str and os.path.exists(mask_path_str):
            print(f"[fallback] 自动调用 quantify_crack_geometry 生成视觉图: {fallback_visuals}")
            result = quantify_crack_geometry(
                mask_path=mask_path_str,
                pixel_size_mm=pixel_size,
                metrics=[],  # 不生成指标
                visuals=fallback_visuals
            )
            if result.get("status") == "success":
                for k, v in result.get("visualizations", {}).items():
                    vis_paths[k] = v
                    memory.update_visualization_path(subject_name, k, v)
        else:
            print(f"[❌ fallback] 无掩膜路径，无法生成视觉图")

    # 可视化显示
    if show:
        if "original" in vis_paths and "mask" in vis_paths:
            visualize_result(
                image_path=vis_paths["original"],
                mask_path=vis_paths["mask"],
                overlay=False,
                max_width_path=None,
                save_path=None,
                title=f"原图与分割图 - {subject_name}"
            )
        elif "mask" in vis_paths:
            mask = cv2.imread(vis_paths["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                plt.imshow(mask, cmap='gray')
                plt.title(f"掩膜图 - {subject_name}")
                plt.axis("off")
                plt.show()

    return vis_paths
