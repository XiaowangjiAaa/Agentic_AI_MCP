# MCP/tool.py

tool_registry = {}

def tool(name=None):
    def decorator(fn):
        fn._tool_name = name or fn.__name__
        tool_registry[fn._tool_name] = fn  # ✅ 注册工具
        return fn
    return decorator