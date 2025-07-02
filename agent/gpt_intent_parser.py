import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "你是一个裂缝图像分析任务的规划器。\n"
    "你的任务是将用户的自然语言请求转换为一个多步骤的结构化任务计划。\n"
    "最终返回一个 JSON 对象，包含一个名为 steps 的数组，每个 step 是一个任务步骤，格式如下：\n"
    "\n"
    "- action: 必选，取值为以下之一：segment / quantify / visualize / compare / chat / generate\n"
    "- target_indices: 必选，数组，如 [0] 表示第一张图像，支持 'all'\n"
    "- pixel_size_mm: 可选，仅在 quantify 和 generate 中使用，单位为毫米\n"
    "- metrics: 可选，仅 quantify 使用，值为以下之一：max_width, avg_width, length, area\n"
    "- visual_types: 可选，仅当用户请求生成图像时使用，可选值包括：original, mask, overlay, max_width, skeleton, normals\n"
    "\n"
    "🎯 编排策略：\n"
    "1. 如果用户提到“计算、量化、测量、how long、how wide、how big”等，理解为需要计算几何指标，action 为 quantify，填写 metrics。\n"
    "2. 如果用户提到“展示、可视化、生成图像、visualize、show、plot”等，理解为只需展示已生成的图像，action 为 visualize。\n"
    "3. 如果用户提到“保存、生成、输出图像、save、generate”，特别是要求生成 skeleton、max width、overlay、normals 等视觉图，action 为 generate，填写 visual_types，不填写 metrics。\n"
    "4. 若用户只提到图像保存或生成视觉图（如“save the skeleton of this crack image”），请输出 generate step，填写 visual_types。\n"
    "5. 不要自动补全 metrics 或 visual_types 字段，除非用户明确请求。\n"
    "6. 如用户表达多个意图，请输出多个 step。\n"
    "\n"
    "✅ 示例：\n"
    "用户说：“save the skeleton and max width of this crack image”\n"
    "应输出：\n"
    "{\n"
    "  \"action\": \"generate\",\n"
    "  \"target_indices\": [0],\n"
    "  \"pixel_size_mm\": 0.5,\n"
    "  \"visual_types\": [\"skeleton\", \"max_width\"]\n"
    "}\n"
    "\n"
    "保持输出结构稳定，不添加多余信息。"
)

FUNCTION_SCHEMA = [
    {
        "name": "generate_composite_plan",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["segment", "quantify", "visualize", "compare", "chat", "generate"]
                            },
                            "target_indices": {
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {"type": "integer"},
                                        {"type": "string", "enum": ["all"]}
                                    ]
                                }
                            },
                            "pixel_size_mm": {"type": "number"},
                            "metrics": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["max_width", "avg_width", "length", "area"]
                                }
                            },
                            "visual_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["original", "mask", "overlay", "max_width", "skeleton", "normals"]
                                }
                            }
                        },
                        "required": ["action", "target_indices"]
                    }
                }
            },
            "required": ["steps"]
        }
    }
]

def generate_composite_plan(user_input: str) -> dict:
    """
    使用 GPT function calling 输出多步骤分析计划。
    每个 step 是一个结构体，字段受控，提升稳定性。
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        functions=FUNCTION_SCHEMA,
        function_call={"name": "generate_composite_plan"}
    )

    try:
        result = json.loads(response.choices[0].message.function_call.arguments)
        return result
    except Exception as e:
        print("❌ 解析计划失败:", e)
        print("返回内容:", response.choices[0].message.content)
        return {"steps": []}
