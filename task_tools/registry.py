# task_tools/registry.py

"""Tool registry and decorator for registering executable tools."""

# mapping of tool name to callable
tool_registry = {}


def tool(name=None):
    """Decorator to register a function as a tool."""
    def decorator(fn):
        fn._tool_name = name or fn.__name__
        tool_registry[fn._tool_name] = fn
        return fn
    return decorator


