import os
import torch
import numpy as np
import cv2
import traceback
from PIL import Image as PILImage
from torchvision import transforms
from MCP.tool import tool

from model.unet import UNet
from tools.preprocess import resolve_output_path

# 初始化模型（只加载一次）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

def load_model(checkpoint_path: str = "checkpoints/unet_best.pth") -> torch.nn.Module:
    global _model
    if _model is None:
        model = UNet(in_channels=3, num_classes=1)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        _model = model
    return _model


@tool(name="segment_crack_image")
def segment_crack_image(image_path: str, checkpoint_path: str = "checkpoints/unet_best.pth") -> dict:
    """
    图像分割工具：输入图像路径，输出掩膜图像路径（0/255 单通道图）
    """
    try:
        # 1. 读取原图（BGR）
        image_np = cv2.imread(image_path)
        if image_np is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 2. 转换为 PIL，RGB，并预处理
        pil_img = PILImage.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((896, 896)),
            transforms.ToTensor()
        ])
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # 3. 推理
        model = load_model(checkpoint_path)
        with torch.no_grad():
            pred = model(input_tensor)
            pred = torch.sigmoid(pred)
            pred_mask = pred[0, 0].cpu().numpy()
            binary_mask = (pred_mask > 0.9).astype(np.uint8)

        # 4. 保存掩膜图像
        output_path = resolve_output_path(image_path, suffix="mask", output_dir="outputs/masks")
        cv2.imwrite(output_path, binary_mask * 255)

        if not os.path.exists(output_path):
            raise RuntimeError(f"掩膜保存失败，未找到文件: {output_path}")

        return {
            "status": "success",
            "summary": "裂缝分割完成，掩膜图像已保存",
            "outputs": {
                "mask_path": output_path
            },
            "error": None
        }

    except Exception as e:
        print("[ERROR] segment_crack_image 异常:", str(e))
        traceback.print_exc()
        return {
            "status": "error",
            "summary": "分割失败",
            "outputs": None,
            "error": str(e)
        }
