import json
from datetime import datetime
from pathlib import Path


class ChatLogger:
    def __init__(self, log_path="chat_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_entry(self, role: str, message: str):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "role": role,
            "message": message.strip()
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_user(self, message: str):
        self._write_entry("user", message)

    def log_agent(self, message: str):
        self._write_entry("agent", message)

    def log_agent_structured(self, data: dict):
        """
        结构化记录 Agent 行为，包括意图、图像、计划、结果等
        """
        data["timestamp"] = datetime.utcnow().isoformat()
        data["role"] = "agent"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
