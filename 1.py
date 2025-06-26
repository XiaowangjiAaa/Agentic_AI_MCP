# test_normals_visual.py
import sys, os
sys.path.append(os.path.abspath("."))  # 加入当前目录为模块搜索路径

from MCP.tool import quantify_crack_geometry
import pprint

mask_path = "outputs/masks/3_crack.png"
pixel_size_mm = 0.05

result = quantify_crack_geometry(
    mask_path=mask_path,
    pixel_size_mm=pixel_size_mm,
    metrics=["all"],
    visuals=["skeleton", "normals", "max_width"]
)

print("=== 测试完成 ===")
pprint.pprint(result["summary"])
pprint.pprint(result["visualizations"])
