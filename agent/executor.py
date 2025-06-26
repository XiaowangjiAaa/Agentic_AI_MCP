from typing import List, Dict, Any
import os
import traceback
from MCP.tool import tool_registry
from agent.object_memory_manager import ObjectMemoryManager

object_store = ObjectMemoryManager()  # 可改为外部注入

def patch_image_paths(plan: list, base_folder: str = "data") -> list:
    for step in plan:
        args = step.get("args", {})
        for key in args:
            if "path" in key and isinstance(args[key], str):
                path = args[key]
                if not os.path.exists(path) and not os.path.dirname(path):
                    args[key] = os.path.join(base_folder, path)
    return plan

def execute_plan(plan: List[Dict[str, Any]], memory=None) -> List[Dict[str, Any]]:
    results = []

    for step in plan:
        tool_name = step.get("tool")
        args = step.get("args", {})
        tool_fn = tool_registry.get(tool_name)

        print(f"[DEBUG] 调用工具 {tool_name} -> {tool_fn}")
        if not callable(tool_fn):
            results.append({
                "tool": tool_name,
                "status": "error",
                "summary": f"工具未注册: {tool_name}",
                "outputs": None,
                "visualizations": None,
                "error": "Tool not found in registry",
                "args": args
            })
            continue

        try:
            result = tool_fn(**args)
            print(f"[DEBUG] 工具 {tool_name} 返回结果:", result)

            # 自动写入 memory
            if memory and result.get("status") == "success":
                outputs = result.get("outputs", {})
                if tool_name == "segment_crack_image":
                    image_path = args.get("image_path", "")
                    subject = os.path.splitext(os.path.basename(image_path))[0]
                    if "mask_path" in outputs:
                        memory.save_mask_path(subject, outputs["mask_path"])

                elif tool_name == "quantify_crack_geometry":
                    mask_path = args.get("mask_path", "")
                    pixel_size = args.get("pixel_size_mm", 0.5)
                    subject = step.get("subject") or os.path.splitext(os.path.basename(mask_path))[0]
                    if outputs:
                        memory.save_metrics(subject, pixel_size, outputs)

            # 更新 object memory（原逻辑保留）
            if tool_name == "segment_crack_image" and result.get("status") == "success":
                image_path = args.get("image_path", "")
                object_id = object_store.find_id_by_image_path(image_path)
                mask_path = result["outputs"].get("mask_path")
                if object_id and mask_path:
                    object_store.update(object_id, "segmentation_path", mask_path)
                    object_store.add_status(object_id, "segmented")

            elif tool_name == "quantify_crack_geometry" and result.get("status") == "success":
                mask_path = args.get("mask_path", "")
                object_id = object_store.find_id_by_mask_path(mask_path)
                vis_path = result.get("visualizations", {}).get("max_width_overlay")
                if object_id:
                    if vis_path:
                        object_store.update(object_id, "visualization_path", vis_path)
                    object_store.add_status(object_id, "quantified")

            results.append({
                "tool": tool_name,
                "status": result.get("status", "unknown"),
                "summary": result.get("summary", ""),
                "outputs": result.get("outputs", {}),
                "visualizations": result.get("visualizations", None),
                "error": result.get("error", None),
                "args": args
            })

        except Exception as e:
            print(f"[❌ ERROR] 工具 {tool_name} 执行时出错: {e}")
            traceback.print_exc()
            results.append({
                "tool": tool_name,
                "status": "error",
                "summary": f"执行 {tool_name} 失败",
                "outputs": None,
                "visualizations": None,
                "error": traceback.format_exc(),
                "args": args
            })

    return results
