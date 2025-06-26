from pathlib import Path
from tools.path_utils import get_test_image_paths

# === 图像索引与文件名映射工具 ===

def get_image_name_by_index(index: int) -> str:
    """
    根据图像索引返回对应图像的标准化名称（不含后缀）。
    例如 index=0 -> "0_crack"
    """
    paths = get_test_image_paths()
    if index < 0 or index >= len(paths):
        raise IndexError(f"图像索引 {index} 超出范围，共有 {len(paths)} 张图像")
    return Path(paths[index]).stem

def get_index_by_image_name(name: str) -> int:
    """
    通过图像名（不含扩展名）查找其在 Test_images 中的索引位置。
    """
    paths = get_test_image_paths()
    for i, path in enumerate(paths):
        if Path(path).stem == name:
            return i
    raise ValueError(f"找不到图像名: {name}")
