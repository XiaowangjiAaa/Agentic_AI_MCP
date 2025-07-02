from typing import List, Dict
from utils.path_utils import (
    get_test_image_by_index,
    get_test_image_paths,
    generate_segment_plan_from_paths,
    list_image_paths
)
from agent.nlp_parser import parse_image_indices_with_gpt

def generate_plan(user_input: str, memory: Dict = None) -> List[Dict]:
    """
    基于规则和自然语言分析生成工具调用计划。
    支持 segment、quantify、compare、plot 等任务识别，并解析图像索引（第几张图）。
    """
    user_input = user_input.lower()
    plan = []
    memory = memory or {}
    pixel_size = memory.get("pixel_size_mm", 0.5)

    # === SEGMENT ===
    if "segment" in user_input or "detect" in user_input:
        if "all" in user_input:
            print("🧪 触发 segment all")
            image_paths = get_test_image_paths()
            plan += generate_segment_plan_from_paths(image_paths)
        else:
            indices = parse_image_indices_with_gpt(user_input)
            print("🧪 提取到图像索引:", indices)
            if indices:
                for idx in indices:
                    try:
                        path = get_test_image_by_index(idx)
                        print("✅ 生成路径:", path)
                        plan.append({"tool": "segment_crack_image", "args": {"image_path": path}})
                    except IndexError:
                        print(f"⚠️ 第 {idx+1} 张图像不存在，跳过")
            else:
                image_paths = get_test_image_paths()
                plan += generate_segment_plan_from_paths(image_paths)

    # === QUANTIFY ===
    if "quantify" in user_input or "geometry" in user_input:
        mask_dir = "outputs/masks"
        mask_paths = []

        if "all" in user_input:
            mask_paths = list_image_paths(mask_dir, suffixes=[".png"])
        else:
            indices = parse_image_indices_with_gpt(user_input)
            all_masks = list_image_paths(mask_dir, suffixes=[".png"])
            for idx in indices:
                if idx < len(all_masks):
                    mask_paths.append(all_masks[idx])

        for path in mask_paths:
            step = {
                "tool": "quantify_crack_metrics",
                "args": {
                    "mask_path": path,
                    "pixel_size_mm": pixel_size,
                    "metrics": ["length", "area", "max_width", "avg_width"]
                }
            }

            # ✅ 仅在用户明确请求时才添加可视化
            if any(word in user_input.lower() for word in ["skeleton", "width map", "max width image", "visualize"]):
                step["args"]["visuals"] = ["skeleton", "max_width"]

            plan.append(step)

    # === COMPARE ===
    if "compare" in user_input or "ground truth" in user_input or "gt" in user_input:
        plan.append({
            "tool": "compare_results_csv",
            "args": {
                "gt_csv_path": "outputs/results/ground_truth.csv",
                "pred_csv_path": "outputs/results/prediction.csv"
            }
        })

    # === PLOT ===
    if "plot" in user_input or "draw" in user_input or "graph" in user_input:
        plan.append({
            "tool": "plot_comparison_graphs",
            "args": {
                "gt_csv_path": "outputs/results/ground_truth.csv",
                "pred_csv_path": "outputs/results/prediction.csv"
            }
        })

    # === ADVICE ===
    if "advice" in user_input or "summary" in user_input or "suggestion" in user_input:
        plan.append({
            "tool": "summarize_and_advice",
            "args": {}
        })

    return plan

# === 示例测试入口 ===
if __name__ == "__main__":
    examples = [
        "segment the first image",
        "segment the second and third crack image",
        "segment all crack images",
        "quantify all crack images",
        "quantify the first and second image",
        "compare predicted and GT and plot graphs",
        "give me a summary advice"
    ]

    for i, prompt in enumerate(examples):
        print(f"\n🧠 Test {i+1}: {prompt}")
        p = generate_plan(prompt, memory={"pixel_size_mm": 0.5})
        for step in p:
            print(f"→ Tool: {step['tool']} | Args: {step['args']}")
