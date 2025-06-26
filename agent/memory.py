import os
import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime


class MemoryController:
    def __init__(self, filepath: str = "memory_store.jsonl"):
        self.filepath = Path(filepath)
        self.records: List[Dict[str, Any]] = []
        self.alias_map = {
            "最大宽度": "max_width",
            "平均宽度": "avg_width",
            "最大裂缝宽度": "max_width",
            "宽度最大值": "max_width",
            "avgwidth": "avg_width",
            "最大宽": "max_width"
        }
        self._load_memory()

    def _load_memory(self):
        if not self.filepath.exists():
            return
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    self.records.append(obj)
                except Exception:
                    continue

    def _save_record(self, record: Dict):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def normalize(self, s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "")

    def to_standard_metric(self, name: str) -> str:
        name = self.normalize(name)
        for alias, standard in self.alias_map.items():
            if self.normalize(alias) == name:
                return standard
        return name

    def update_context(self, intent: str, indices: List[int], pixel_size: float, results: List[Dict], plan: List[Dict] = None):
        for r in results:
            tool = r.get("tool")
            status = r.get("status")
            if tool not in {"quantify_crack_geometry", "segment_crack_image"} or status != "success":
                continue

            args = r.get("args", {})
            if not args and plan:
                for step in plan:
                    if step["tool"] == tool:
                        args = step["args"]
                        break

            image_path = args.get("image_path") or args.get("mask_path")
            if not image_path:
                print("⚠️ 无法提取 image/mask path，跳过记录")
                continue

            image_name = Path(image_path).stem
            record = {
                "subject": image_name,
                "context": {
                    "task": intent,
                    "pixel_size_mm": pixel_size,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

            outputs = r.get("outputs", {})
            if outputs:
                record["observation"] = outputs
            elif tool == "segment_crack_image" and "mask_path" in args:
                record["observation"] = {"mask_path": args["mask_path"]}

            self.records.append(record)
            self._save_record(record)

    def save_mask_path(self, subject_name: str, mask_path: str):
        """
        显式保存某图像的掩膜路径（供 executor 使用）。
        """
        record = {
            "subject": subject_name,
            "context": {
                "task": "segment",
                "timestamp": datetime.utcnow().isoformat()
            },
            "observation": {
                "mask_path": mask_path
            }
        }
        self.records.append(record)
        self._save_record(record)

    def save_metrics(self, subject_name: str, pixel_size: float, metrics: Dict[str, Any]):
        """
        显式保存某图像的量化指标（供 executor 使用）。
        """
        record = {
            "subject": subject_name,
            "context": {
                "task": "quantify",
                "pixel_size_mm": pixel_size,
                "timestamp": datetime.utcnow().isoformat()
            },
            "observation": metrics
        }
        self.records.append(record)
        self._save_record(record)

    def get_metrics_by_name(self, name: str, pixel_size: float = None) -> Dict[str, Any]:
        matches = [r for r in self.records if r.get("subject") == name]
        if pixel_size is not None:
            matches = [r for r in matches if abs(r["context"].get("pixel_size_mm", 0) - pixel_size) < 1e-6]
        if not matches:
            return {}
        return matches[-1].get("observation", {})

    def get_mask_path(self, name: str) -> str:
        for r in reversed(self.records):
            if r.get("subject") == name:
                obs = r.get("observation", {})
                if isinstance(obs, dict) and "mask_path" in obs:
                    return obs["mask_path"]
        return ""

    def get_last_metrics(self, count: int = 5) -> Dict[str, Dict[str, Any]]:
        latest = self.records[-count:]
        return {r["subject"]: r["observation"] for r in latest if "observation" in r}

    def has_metrics(self, name: str, requested_metrics: List[str], pixel_size: float = None) -> bool:
        existing = self.get_metrics_by_name(name, pixel_size)
        for m in requested_metrics:
            found = False
            norm_m = self.normalize(self.to_standard_metric(m))
            for k in existing:
                if norm_m in self.normalize(k):
                    found = True
                    break
            if not found:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_task": self.records[-1]["context"]["task"] if self.records else None,
            "known_subjects": list({r["subject"] for r in self.records}),
            "recent_metrics": self.get_last_metrics()
        }

    def export_latest_snapshot(self, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def clear(self):
        self.records = []
        if self.filepath.exists():
            self.filepath.unlink()
