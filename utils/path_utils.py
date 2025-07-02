from pathlib import Path
from typing import List, Tuple
import re

def list_image_paths(folder: str, suffixes: List[str] = [".jpg", ".png", ".jpeg"]) -> List[str]:
    """
    通用路径读取函数，返回指定文件夹下的图像文件路径列表（POSIX 相对路径）
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"❌ 文件夹不存在: {folder}")

    paths = []
    for suf in suffixes:
        paths.extend(folder_path.glob(f"*{suf}"))

    # 🔒 确保排序稳定（按文件名）
    return sorted([str(p.as_posix()) for p in paths], key=lambda x: Path(x).name)

def get_test_image_paths() -> List[str]:
    """
    返回用于分割的所有测试图像路径（来自 data/Test_images）
    """
    return list_image_paths("data/Test_images")

def get_test_image_by_index(index: int) -> str:
    """
    返回 Test_images 中按排序的第 index 张图像路径
    支持 index = -1 代表最后一张图像
    """
    images = get_test_image_paths()
    if index < 0:
        index = len(images) + index  # 支持 -1 表示最后一张
    if index < 0 or index >= len(images):
        raise IndexError(f"❌ 图像索引 {index} 越界，当前仅有 {len(images)} 张图像")
    return images[index]

def generate_segment_plan_from_paths(image_paths: list) -> list:
    return [
        {"tool": "segment_crack_image", "args": {"image_path": path}}
        for path in image_paths
    ]

def get_comparison_image_pairs() -> List[Tuple[str, str]]:
    """
    从 data/Test_images_GT 与 outputs/masks 中配对获取可比较的 GT vs 预测掩膜路径。
    返回：[(gt_path, pred_path), ...]
    """
    gt_paths = list_image_paths("data/Test_images_GT", suffixes=[".png"])
    pred_paths = list_image_paths("outputs/masks", suffixes=[".png"])

    # 按照文件名进行配对
    gt_dict = {Path(p).name: p for p in gt_paths}
    pred_dict = {Path(p).name: p for p in pred_paths}

    pairs = []
    for name in gt_dict:
        if name in pred_dict:
            pairs.append((gt_dict[name], pred_dict[name]))

    return pairs

def get_csv_paths(results_dir: str = "outputs/csv/") -> Tuple[str, str]:
    """
    返回 prediction.csv 与 ground_truth.csv 的路径，用于对比分析。
    默认目录为 outputs/results
    """
    base = Path(results_dir)
    pred = base / "prediction.csv"
    gt = base / "ground_truth.csv"
    if not pred.exists() or not gt.exists():
        raise FileNotFoundError("❌ prediction.csv 或 ground_truth.csv 不存在")

    return str(gt.as_posix()), str(pred.as_posix())

def extract_image_indices(text: str) -> List[int]:
    """
    从自然语言中提取“第1张”、“第2张”、“image 3”等，转为 index（从0开始）
    """
    text = text.lower()
    indices = []

    index_map = {
        "第一": 0, "第1": 0, "1st": 0, "image 1": 0, "first": 0,
        "第二": 1, "第2": 1, "2nd": 1, "image 2": 1, "second": 1,
        "第三": 2, "第3": 2, "3rd": 2, "image 3": 2, "third": 2,
    }
    for k, v in index_map.items():
        if k in text:
            indices.append(v)

    # 正则匹配：image N
    matches = re.findall(r'image\\s*(\\d+)', text)
    indices.extend([int(m) - 1 for m in matches])

    return sorted(set(indices))
