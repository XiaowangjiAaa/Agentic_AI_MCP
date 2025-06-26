# agent/gpt_intent_parser.py

from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_intent(user_input: str) -> dict:
    """
    使用 GPT 提取意图、图像索引、像素物理比例（mm/pixel）、指定量化指标、可视化类型。
    GPT 输出的 metrics 应尽可能标准化（如 max_width, avg_width, length, area）。
    visual_types 仅包含用户明确表达的内容（不自动默认添加）。
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个用于裂缝图像分析的智能意图解析器。\n"
                    "请根据用户输入提取以下字段：\n"
                    "1. intent（segment, quantify, compare, plot, visualize, chat）\n"
                    "2. target_indices（如“第一张”→0）\n"
                    "3. pixel_size_mm（如 spatial ratio = 0.2mm/pixel）\n"
                    "4. metrics（请求的几何指标，必须使用以下标准字段之一：length, area, max_width, avg_width）\n"
                    "5. visual_types（若 intent=visualize，仅包含用户明确表达的类型，可选值有：mask, overlay, original, max_width）"
                )
            },
            {"role": "user", "content": user_input}
        ],
        functions=[
            {
                "name": "classify_intent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": ["segment", "quantify", "compare", "plot", "visualize", "chat"]
                        },
                        "target_indices": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "pixel_size_mm": {
                            "type": "number",
                            "description": "optional pixel size in millimeters"
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "必须为标准字段之一: length, area, max_width, avg_width"
                        },
                        "visual_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["original", "mask", "overlay", "max_width"]
                            },
                            "description": "若 intent 为 visualize，仅包含用户明确要求展示的图像类型"
                        }
                    },
                    "required": ["intent"]
                }
            }
        ],
        function_call={"name": "classify_intent"}
    )

    result = json.loads(response.choices[0].message.function_call.arguments)

    # ✅ 后处理规则优化：简单语义修复
    if result.get("intent") == "visualize":
        vis_text = user_input.lower()
        vt = result.get("visual_types", [])

        # 自动判断语义意图（仅当 GPT 未提取 visual_types 时）
        if not vt:
            if "segmentation" in vis_text or "mask" in vis_text:
                result["visual_types"] = ["mask"]
            elif "image" in vis_text or "original" in vis_text:
                result["visual_types"] = ["original"]

    return result
