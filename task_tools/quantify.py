import os
import cv2
import traceback
from task_tools.registry import tool

from crack_metrics.binarize import binarize
from crack_metrics.area import compute_crack_area_px
from crack_metrics.length import compute_crack_length_px
from crack_metrics.width_max import compute_max_width_px
from crack_metrics.width_avg import compute_average_width_px
from crack_metrics.skeleton import extract_skeleton_and_normals
from utils.visualize import visualize_max_width, save_visual
from utils.io_utils import append_to_csv


@tool(name="quantify_crack_metrics")
def quantify_crack_metrics(mask_path: str, pixel_size_mm: float, metrics: list | None = None) -> dict:
    """Compute crack geometry metrics from a mask image and append results to CSV."""
    try:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Invalid image format: {mask_path}")

        binary = binarize(mask)

        all_metrics = {
            "Length (mm)": lambda: round(compute_crack_length_px(binary) * pixel_size_mm, 2),
            "Area (mm^2)": lambda: round(compute_crack_area_px(binary) * pixel_size_mm ** 2, 2),
            "Max Width (mm)": lambda: round(compute_max_width_px(binary) * pixel_size_mm, 2),
            "Avg Width (mm)": lambda: round(compute_average_width_px(binary) * pixel_size_mm, 2),
        }
        alias = {
            "Length (mm)": "length",
            "Area (mm^2)": "area",
            "Max Width (mm)": "max_width",
            "Avg Width (mm)": "avg_width",
        }

        if not metrics:
            selected = list(all_metrics.keys())
        else:
            selected = []
            for m in metrics:
                for k in all_metrics:
                    if m.lower().replace(" ", "").replace("_", "") in k.lower().replace(" ", "").replace("_", ""):
                        selected.append(k)

        results = {alias[name]: all_metrics[name]() for name in selected}

        csv_values = {name: all_metrics[name]() for name in selected}
        image_name = os.path.splitext(os.path.basename(mask_path))[0]
        os.makedirs("outputs/csv", exist_ok=True)
        append_to_csv("outputs/csv/predicted_metrics.csv", image_name, csv_values)

        return {
            "status": "success",
            "summary": f"量化完成，共 {len(results)} 项",
            "outputs": results,
            "visualizations": None,
            "error": None,
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": "量化失败",
            "outputs": None,
            "visualizations": None,
            "error": str(e),
        }


@tool(name="generate_crack_visuals")
def generate_crack_visuals(mask_path: str, pixel_size_mm: float = 0.5, visuals: list | None = None) -> dict:
    """Generate visualization images for a crack mask (skeleton, normals, max width)."""
    try:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img_raw = cv2.imread(mask_path)
        if img_raw is None:
            raise ValueError(f"Invalid image format: {mask_path}")

        mask = binarize(img_raw)
        if visuals is None:
            visuals = ["skeleton", "max_width"]

        vis_results: dict[str, str] = {}
        image_base = os.path.splitext(os.path.basename(mask_path))[0]
        visual_dir = os.path.join("outputs", "visuals")
        os.makedirs(visual_dir, exist_ok=True)

        if any(v in visuals for v in ["skeleton", "normals"]):
            _, centers, normals = extract_skeleton_and_normals(mask)
        else:
            centers, normals = [], []

        if "skeleton" in visuals or "all" in visuals:
            overlay = img_raw.copy()
            for pt in centers:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)
            path = save_visual(overlay, os.path.join(visual_dir, f"{image_base}_skeleton.png"))
            vis_results["skeleton"] = path

        if "normals" in visuals or "all" in visuals:
            normal_overlay = img_raw.copy()
            for pt, n in zip(centers, normals):
                x, y = int(pt[0]), int(pt[1])
                dx, dy = n[0], n[1]
                pt2 = (int(x + dx * 10), int(y + dy * 10))
                cv2.arrowedLine(normal_overlay, (x, y), pt2, (0, 255, 0), 1, tipLength=0.3)
            path = save_visual(normal_overlay, os.path.join(visual_dir, f"{image_base}_normals.png"))
            vis_results["normals"] = path

        if "max_width" in visuals or "all" in visuals:
            vis_width, _ = visualize_max_width(img_raw)
            path = save_visual(vis_width, os.path.join(visual_dir, f"{image_base}_max_width.png"))
            vis_results["max_width_overlay"] = path

        return {
            "status": "success",
            "summary": f"生成 {len(vis_results)} 张图像",
            "outputs": {},
            "visualizations": vis_results,
            "error": None,
        }

    except Exception as e:
        return {
            "status": "error",
            "summary": "生成可视化失败",
            "outputs": None,
            "visualizations": None,
            "error": str(e),
        }
