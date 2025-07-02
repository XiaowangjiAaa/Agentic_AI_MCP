import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from agent.executor import execute_plan
from tools.path_utils import get_test_image_paths, get_test_image_by_index
from agent.visualize_tools import visualize_crack_result
from agent.gpt_intent_parser import generate_composite_plan
from agent.object_memory_manager import ObjectMemoryManager
from agent.session_manager import SessionManager

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

session = SessionManager()
logger = session.get_logger()
memory = session.get_memory()
object_store = ObjectMemoryManager()

DEFAULT_METRICS = ["max_width", "avg_width", "length", "area"]


def generate_agent_reply(user_input: str, plan: dict, results: list) -> str:
    prompt = (
        "你是一个专注于裂缝图像分析的 AI 助手\n"
        "请根据以下信息，用自然语言总结任务完成情况\n"
        "要求语言简洁清晰，尽可能提及图像编号、处理内容和关键结果：\n\n"
        f"🖍️ 用户请求：{user_input}\n"
        f"📋 执行计划：{json.dumps(plan, ensure_ascii=False)}\n"
        f"✅ 执行结果：{json.dumps(results, ensure_ascii=False)}\n\n"
        "请直接生成一段回复，不要附加解释或格式标签。"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是一个专业裂缝图像分析 AI，负责生成自然语言分析答复。"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def chat_fallback(user_input: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你是一个专注于裂缝图像分析的 AI Agent。"},
            {"role": "user", "content": user_input}
        ]
    )
    reply = response.choices[0].message.content.strip()
    logger.log_agent(reply)
    return reply


def normalize(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "")


def match_metric_key(requested: str, candidate: str) -> bool:
    return normalize(requested) in normalize(candidate)


if __name__ == "__main__":
    while True:
        user_input = input("\n🧠 请输入自然语言指令（或 exit）: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            session.export_memory_snapshot()
            session.print_summary()
            break

        logger.log_user(user_input)
        print("🗽 正在理解意图...")

        plan = generate_composite_plan(user_input)
        steps = plan.get("steps", [])

        if all(step["action"] == "chat" for step in steps):
            print("💬", chat_fallback(user_input))
            continue

        if not steps:
            print("💬", chat_fallback(user_input))
            continue

        print("\n📋 任务计划:")
        for i, step in enumerate(steps):
            print(f"{i+1}. {step['action']} → index={step.get('target_indices')}")

        tool_plan = []
        results = []
        all_images = get_test_image_paths()
        index_to_image_name = {
            i: os.path.basename(p).replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
            for i, p in enumerate(all_images)
        }

        for step in steps:
            action = step["action"]
            indices = step.get("target_indices", [])

            if isinstance(indices, list) and "all" in indices:
                indices = list(range(len(all_images)))
            elif isinstance(indices, str) and indices == "all":
                indices = list(range(len(all_images)))

            step["target_indices"] = indices
            pixel_size = step.get("pixel_size_mm", 0.5)
            metrics = step.get("metrics", [])
            visual_types = step.get("visual_types", [])

            if action == "quantify" and not metrics:
                metrics = DEFAULT_METRICS
                step["metrics"] = metrics
            if action == "visualize" and not visual_types:
                visual_types = ["original", "mask"]
                step["visual_types"] = visual_types

            if not indices:
                print(f"⚠️ 步骤 [{action}] 缺少图像索引")
                continue

            for i in indices:
                img_path = get_test_image_by_index(i)
                name = index_to_image_name[i]
                object_store.register_image(img_path)
                mask_name = os.path.basename(img_path).replace(".jpg", ".png").replace(".jpeg", ".png")
                mask_path = os.path.join("outputs/masks", mask_name)

                if action == "segment":
                    if os.path.exists(mask_path):
                        print(f"♻️ 掩膜已存在，跳过 segment: {mask_path}")
                        results.append({
                            "tool": "segment_crack_image",
                            "status": "success",
                            "summary": "掩膜已存在于磁盘，跳过执行",
                            "outputs": {"mask_path": mask_path},
                            "visualizations": None,
                            "args": {"image_path": img_path},
                            "subject": name
                        })
                        continue
                    tool_plan.append({
                        "tool": "segment_crack_image",
                        "args": {"image_path": img_path},
                        "subject": name
                    })

                elif action == "quantify":
                    if memory.has_metrics(name, metrics, pixel_size):
                        print(f"✅ 图像 {name} 的指标已存在于 memory，使用缓存结果。")
                        metric_values = memory.get_metrics_by_name(name, pixel_size)
                        outputs = {k: v for k, v in metric_values.items() if any(map(lambda m: m.lower() in k.lower(), metrics))}
                        results.append({
                            "tool": "quantify_crack_geometry",
                            "status": "success",
                            "summary": f"读取自 memory，包含 {len(outputs)} 项指标",
                            "outputs": outputs,
                            "visualizations": None,
                            "args": {
                                "mask_path": mask_path,
                                "pixel_size_mm": pixel_size,
                                "metrics": metrics,
                                "visuals": []
                            },
                            "subject": name
                        })
                        continue

                    args = {
                        "mask_path": mask_path,
                        "pixel_size_mm": pixel_size,
                        "metrics": metrics
                    }
                    if "visual_types" in step:
                        args["visuals"] = step["visual_types"]

                    tool_plan.append({
                        "tool": "quantify_crack_geometry",
                        "args": args,
                        "subject": name
                    })

                elif action == "generate":
                    args = {
                        "mask_path": mask_path,
                        "pixel_size_mm": pixel_size,
                        "metrics": [],
                        "visuals": visual_types
                    }
                    tool_plan.append({
                        "tool": "quantify_crack_geometry",
                        "args": args,
                        "subject": name
                    })

                elif action == "visualize":
                    vis_paths = visualize_crack_result(subject_name=name, memory=memory, visual_types=visual_types)
                    results.append({
                        "tool": "visualize_crack_result",
                        "status": "success" if vis_paths else "no_output",
                        "summary": f"生成了 {len(vis_paths)} 张可视化图" if vis_paths else "未生成可视化图像",
                        "outputs": {},
                        "visualizations": vis_paths,
                        "args": {"subject_name": name, "visual_types": visual_types},
                        "subject": name
                    })

        if not tool_plan and results:
            print("♻️ 所有步骤已由 memory 命中，无需执行工具链。")
        else:
            print("\n🚀 正在执行:")
            for step in tool_plan:
                print(f"→ {step['tool']}({step['args']})")
            exec_results = execute_plan(tool_plan, memory=memory)
            results += exec_results

            # ✅ 新增：保存 generate 可视化图到 memory
            for r in exec_results:
                if r['tool'] == "quantify_crack_geometry" and r['status'] == "success":
                    subject = r["subject"]
                    visual_paths = r.get("visualizations", {})
                    if visual_paths:
                        memory.save_visualizations(subject, pixel_size, visual_paths)

        for r in results:
            print(f"[{r['tool']}] → {r['status']}")
            if r['tool'] == "quantify_crack_geometry" and r['status'] == "success":
                print("📊 请求指标:")
                for k, v in r.get("outputs", {}).items():
                    print(f"  {k}: {v}")

        memory.update_context(
            intent="multi_step",
            indices=[i for step in steps for i in step.get("target_indices", [])],
            pixel_size=pixel_size,
            results=results,
            plan=tool_plan
        )

        logger.log_agent_structured({
            "intent": "multi_step",
            "user_input": user_input,
            "steps": steps,
            "tool_plan": tool_plan,
            "result": results,
            "message": "✅ 多步骤任务完成"
        })

        reply = generate_agent_reply(user_input, tool_plan, results)
        print("\n💬 AI 总结答复:")
        print(reply)
        logger.log_agent(reply)
