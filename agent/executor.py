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
        subject = step.get("subject", "")  # ✅ 提前获取 subject
        tool_fn = tool_registry.get(tool_name)

        # ✅ Patch: 移除空的 visuals 参数（兜底）
        if tool_name == "quantify_crack_geometry":
            visuals = args.get("visuals", None)
            if visuals is not None and len(visuals) == 0:
                del args["visuals"]

        if not callable(tool_fn):
            results.append({
                "tool": tool_name,
                "status": "error",
                "summary": f"工具未注册: {tool_name}",
                "outputs": None,
                "visualizations": None,
                "error": "Tool not found in registry",
                "args": args,
                "subject": subject  # ✅ 加入 subject
            })
            continue

        try:
            result = tool_fn(**args)

            # ✅ 更新 object memory（原逻辑保留）
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
                "args": args,
                "subject": subject  # ✅ 加入 subject 字段（确保 memory 使用）
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
                "args": args,
                "subject": subject  # ✅ 即使失败也记录 subject
            })

    return results
