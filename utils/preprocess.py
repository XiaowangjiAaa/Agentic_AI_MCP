import os
from typing import Optional, Tuple, List, Dict
from PIL import Image
import torch
import torchvision.transforms as T


def is_valid_image(path: str) -> bool:
    return os.path.exists(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))


def load_image_as_tensor(image_path: str, size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    加载图像为 Tensor，默认归一化 [-1, 1]，供模型推理使用
    """
    if not is_valid_image(image_path):
        raise FileNotFoundError(f"❌ Invalid image path: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image)


def load_image_pair(image_path: str, gt_path: Optional[str] = None) -> Tuple[Image.Image, Optional[Image.Image]]:
    """
    加载 RGB 图像 + 可选 GT（灰度）图像为 PIL 对象
    """
    if not is_valid_image(image_path):
        raise FileNotFoundError(f"❌ Invalid image path: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    gt = Image.open(gt_path).convert("L") if gt_path and is_valid_image(gt_path) else None
    return image, gt


def resolve_output_path(image_path: str, suffix: str, output_dir: str = "outputs") -> str:
    """
    根据输入图像名和输出后缀生成输出路径（自动创建文件夹）
    """
    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{name}.png")


def list_image_pairs(images_dir: str, gt_dir: Optional[str] = None) -> List[Dict]:
    """
    扫描一个文件夹下所有图像，构建 image + gt 路径对列表
    """
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if is_valid_image(os.path.join(images_dir, f))
    ])
    
    results = []
    for fname in image_files:
        img_path = os.path.join(images_dir, fname)
        gt_path = os.path.join(gt_dir, fname) if gt_dir else None
        if gt_path and not os.path.exists(gt_path):
            gt_path = None
        results.append({
            "image_path": img_path,
            "gt_path": gt_path
        })
    return results
