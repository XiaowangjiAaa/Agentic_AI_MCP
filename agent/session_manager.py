import os
from datetime import datetime
from pathlib import Path
from agent.chat_logger import ChatLogger
from agent.memory import MemoryController


class SessionManager:
    def __init__(self, base_dir="logs"):
        self.session_id = datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
        self.session_dir = Path(base_dir) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # 初始化各类模块路径
        self.chat_log_path = self.session_dir / "chat_log.jsonl"
        self.memory_path = self.session_dir / "memory_store.jsonl"
        self.memory_summary_path = self.session_dir / "summary.json"

        # 初始化模块
        self.logger = ChatLogger(log_path=self.chat_log_path)
        self.memory = MemoryController(filepath=self.memory_path)

    def get_logger(self) -> ChatLogger:
        return self.logger

    def get_memory(self) -> MemoryController:
        return self.memory

    def get_session_dir(self) -> Path:
        return self.session_dir

    def export_memory_snapshot(self):
        if hasattr(self.memory, "export_latest_snapshot"):
            self.memory.export_latest_snapshot(str(self.memory_summary_path))

    def print_summary(self):
        print(f"\n🧾 当前 Session ID: {self.session_id}")
        print(f"📁 日志路径: {self.session_dir.resolve()}")
