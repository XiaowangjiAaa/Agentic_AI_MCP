from openai import OpenAI
from dotenv import load_dotenv
from agent.executor import execute_plan
from tools.path_utils import (
    get_test_image_paths,
    generate_segment_plan_from_paths,
    get_test_image_by_index,
    list_image_paths
)
from agent.visualize_tools import visualize_crack_result
from agent.gpt_intent_parser import classify_intent
from agent.object_memory_manager import ObjectMemoryManager
from agent.session_manager import SessionManager
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 初始化 Session 管理器 ===
session = SessionManager()
logger = session.get_logger()
memory = session.get_memory()
object_store = ObjectMemoryManager()

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

METRIC_ALIASES = {
    "maxwidth": "max_width",
    "maximumwidth": "max_width",
    "avgwidth": "avg_width",
    "averagewidth": "avg_width",
    "meanwidth": "avg_width",
    "length": "length",
    "area": "area"
}

def map_to_standard_metric(name: str) -> str:
    key = normalize(name)
    return METRIC_ALIASES.get(key, key)

if __name__ == "__main__":
    while True:
        user_input = input("\n🧠 请输入自然语言指令（或 exit）: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            session.export_memory_snapshot()
            session.print_summary()
            break

        logger.log_user(user_input)
        print("🧭 正在理解意图...")
        intent_info = classify_intent(user_input)
        intent = intent_info["intent"]
        indices = intent_info.get("target_indices", [])
        pixel_size = intent_info.get("pixel_size_mm", 0.5)

        raw_metrics = intent_info.get("metrics", [])
        metrics = [map_to_standard_metric(m) for m in raw_metrics]
        if not metrics:
            metrics = ["length", "area", "max_width", "avg_width"]

        visual_types = intent_info.get("visual_types") or ["mask"]
        print(f"[DEBUG] 使用的 visual_types: {visual_types}")

        if intent == "chat":
            reply = chat_fallback(user_input)
            print("💬", reply)
            continue

        if intent == "visualize":
            if not indices:
                print("⚠️ 未指定可视化图像索引")
                continue
            for i in indices:
                name = os.path.basename(get_test_image_by_index(i)).replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
                visualize_crack_result(subject_name=name, memory=memory, visual_types=visual_types)
            continue

        print(f"🧭 识别到意图: {intent} | 图像索引: {indices} | 像素尺寸: {pixel_size} mm/pixel | 指标: {metrics}")
        logger.log_agent(f"识别到意图: {intent} | 图像索引: {indices} | 像素尺寸: {pixel_size} mm/pixel | 指标: {metrics}")

        plan = []
        all_images = get_test_image_paths()
        index_to_image_name = {
            i: os.path.basename(p).replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
            for i, p in enumerate(all_images)
        }

        if intent == "segment":
            image_paths = all_images if not indices else [get_test_image_by_index(i) for i in indices]
            for img_path in image_paths:
                object_store.register_image(img_path)
            plan = generate_segment_plan_from_paths(image_paths)

        elif intent == "quantify":
            image_paths = all_images if not indices else [get_test_image_by_index(i) for i in indices]
            if not indices:
                indices = list(range(len(image_paths)))

            all_satisfied = True
            for i in indices:
                name = index_to_image_name[i]
                if not memory.has_metrics(name, metrics, pixel_size):
                    all_satisfied = False
                    break

            if all_satisfied:
                log_msg = []
                for i in indices:
                    name = index_to_image_name[i]
                    stored = memory.get_metrics_by_name(name, pixel_size)
                    result_lines = [f"  🔹 {name}"]
                    for m in metrics:
                        found = False
                        for k, v in stored.items():
                            if match_metric_key(m, k):
                                result_lines.append(f"     {k}: {v}")
                                found = True
                        if not found:
                            result_lines.append(f"     ⚠️ 未找到指标: {m}")
                    log_msg.extend(result_lines)
                agent_reply = "\n".join(["📋 所请求图像指标已在记忆中，直接读取:"] + log_msg)
                print(agent_reply)
                logger.log_agent(agent_reply)
                continue

            for i, img_path in zip(indices, image_paths):
                name = index_to_image_name[i]
                object_store.register_image(img_path)
                mask_name = os.path.basename(img_path).replace(".jpg", ".png").replace(".jpeg", ".png")
                mask_path = os.path.join("outputs/masks", mask_name)
                if not os.path.exists(mask_path):
                    print(f"🔁 掩膜不存在，插入 segment 任务: {mask_path}")
                    plan.append({"tool": "segment_crack_image", "args": {"image_path": img_path}})
                plan.append({
                    "tool": "quantify_crack_geometry",
                    "args": {
                        "mask_path": mask_path,
                        "pixel_size_mm": pixel_size,
                        "metrics": metrics,
                        "visuals": ["skeleton", "max_width"]
                    },
                    "subject": name
                })

        elif intent == "compare":
            plan = [{
                "tool": "compare_results_csv",
                "args": {
                    "gt_csv_path": "outputs/results/ground_truth.csv",
                    "pred_csv_path": "outputs/results/prediction.csv"
                }
            }]

        elif intent == "plot":
            plan = [{
                "tool": "plot_comparison_graphs",
                "args": {
                    "gt_csv_path": "outputs/results/ground_truth.csv",
                    "pred_csv_path": "outputs/results/prediction.csv"
                }
            }]

        if not plan:
            print("⚠️ 无法生成工具链计划")
            logger.log_agent("⚠️ 无法生成工具链计划")
            continue

        print("\n📋 工具链执行计划:")
        for step in plan:
            print(f"→ {step['tool']}({step['args']})")

        print("\n🚀 执行中...")
        results = execute_plan(plan, memory=memory)

        print("\n✅ 执行结果:")
        quantify_steps = [step for step in plan if step["tool"] == "quantify_crack_geometry"]
        for i, r in enumerate(results):
            tool = r['tool']
            status = r['status']
            summary = r['summary']
            print(f"[{tool}] → {status}")

            if tool == "quantify_crack_geometry":
                outputs = r.get("outputs", {})
                if outputs and len(quantify_steps) == 1:
                    print("📊 请求指标:")
                    for m in metrics:
                        matched = False
                        for k, v in outputs.items():
                            if match_metric_key(m, k):
                                print(f"  {k}: {v}")
                                matched = True
                        if not matched:
                            print(f"  ⚠️ 未匹配到指标: {m}")
                elif len(quantify_steps) == 1:
                    print(f"📊 {summary}")

        for i, step in enumerate(plan):
            if step["tool"] == "quantify_crack_geometry":
                subject = step.get("subject", f"image_{i}")
                results[i]["subject"] = subject

        memory.update_context(intent, indices, pixel_size, results, plan)
        logger.log_agent_structured({
            "intent": intent,
            "images": [index_to_image_name[i] for i in indices],
            "pixel_size": pixel_size,
            "metrics": metrics,
            "plan": plan,
            "result": results,
            "message": "✅ 执行完成。"
        })
