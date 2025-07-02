import os
import cv2
import numpy as np
import traceback
from MCP.tool import tool

from Crack_quantification_tools.binarize import binarize
from Crack_quantification_tools.area import compute_crack_area_px
from Crack_quantification_tools.length import compute_crack_length_px
from Crack_quantification_tools.width_max import compute_max_width_px
from Crack_quantification_tools.width_avg import compute_average_width_px
from Crack_quantification_tools.skeleton import extract_skeleton_and_normals
from tools.visualize import visualize_max_width, save_visual
from tools.io_utils import append_to_csv


@tool(name="quantify_crack_geometry")
def quantify_crack_geometry(
    mask_path: str,
    pixel_size_mm: float,
    metrics: list = None,
    visuals: list = None
) -> dict:
    """
    裂缝图像量化工具，支持选择性计算和可视化，并将结果写入 CSV。
    """
    try:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # ✅ 原始图像（用于可视化）
        img_raw = cv2.imread(mask_path)
        if img_raw is None:
            raise ValueError(f"Invalid image format: {mask_path}")

        # ✅ 二值化图像（用于分析）
        mask = binarize(img_raw)

        # ✅ 量化指标计算
        all_metrics = {
            "Length (mm)": lambda: round(compute_crack_length_px(mask) * pixel_size_mm, 2),
            "Area (mm^2)": lambda: round(compute_crack_area_px(mask) * pixel_size_mm ** 2, 2),
            "Max Width (mm)": lambda: round(compute_max_width_px(mask) * pixel_size_mm, 2),
            "Avg Width (mm)": lambda: round(compute_average_width_px(mask) * pixel_size_mm, 2),
        }

        metric_alias_map = {
            "Length (mm)": "length",
            "Area (mm^2)": "area",
            "Max Width (mm)": "max_width",
            "Avg Width (mm)": "avg_width"
        }

        if not metrics:
            selected_metrics = list(all_metrics.keys())
        else:
            selected_metrics = []
            for m in metrics:
                for k in all_metrics:
                    if m.lower().replace(" ", "").replace("_", "") in k.lower().replace(" ", "").replace("_", ""):
                        selected_metrics.append(k)

        print(f"\U0001F4D0 最终启用指标: {selected_metrics}")

        results = {
            metric_alias_map[name]: all_metrics[name]() for name in selected_metrics
        }

        # ✅ 写入 CSV（保留原始带单位的列名）
        results_for_csv = {name: all_metrics[name]() for name in selected_metrics}
        image_name = os.path.splitext(os.path.basename(mask_path))[0]
        os.makedirs("outputs/csv", exist_ok=True)
        append_to_csv("outputs/csv/predicted_metrics.csv", image_name, results_for_csv)

        # ✅ 可视化结果
        vis_results = {}
        if visuals:
            image_base = os.path.splitext(os.path.basename(mask_path))[0]
            visual_dir = os.path.join("outputs", "visuals")
            os.makedirs(visual_dir, exist_ok=True)

            # ✅ 骨架与法向
            _, centers, normals = extract_skeleton_and_normals(mask)

            # 1. 骨架点（红色点）
            if "skeleton" in visuals or "all" in visuals:
                overlay = img_raw.copy()
                for pt in centers:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(overlay, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
                path = save_visual(overlay, os.path.join(visual_dir, f"{image_base}_skeleton.png"))
                vis_results["skeleton_overlay"] = path

            # 2. 法向箭头（绿色线段）
            if "normals" in visuals or "all" in visuals:
                normal_overlay = img_raw.copy()
                for pt, n in zip(centers, normals):
                    x, y = int(pt[0]), int(pt[1])
                    dx, dy = n[0], n[1]
                    pt2 = (int(x + dx * 10), int(y + dy * 10))
                    cv2.arrowedLine(normal_overlay, (x, y), pt2, (0, 255, 0), 1, tipLength=0.3)
                path = save_visual(normal_overlay, os.path.join(visual_dir, f"{image_base}_skeleton_normals.png"))
                vis_results["skeleton_normals"] = path

            # 3. 最大宽度线段（背景为原图）
            if "max_width" in visuals or "all" in visuals:
                vis_width, max_width = visualize_max_width(img_raw)
                path = save_visual(vis_width, os.path.join(visual_dir, f"{image_base}_max_width.png"))
                vis_results["max_width_overlay"] = path

        return {
            "status": "success",
            "summary": f"量化完成，计算 {len(results)} 项，可视化图 {len(vis_results)} 张",
            "outputs": results,
            "visualizations": vis_results,
            "error": None
        }

    except Exception as e:
        try:
            print("[❗] quantify_crack_geometry 捕获异常:", str(e))
            traceback.print_exc()
        except Exception as log_error:
            print("[❗] ERROR 打印 traceback 时出错:", str(log_error))
        return {
            "status": "error",
            "summary": "量化失败",
            "outputs": None,
            "visualizations": None,
            "error": str(e)
        }