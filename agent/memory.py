import os
import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from datetime import datetime, timezone

datetime.now(timezone.utc).isoformat()



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
            if status != "success":
                continue

            args = r.get("args", {})
            subject = Path(args.get("image_path") or args.get("mask_path", "")).stem

            if tool == "segment_crack_image":
                record = {
                    "subject": subject,
                    "context": {
                        "task": "segment",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "observation": {
                        "mask_path": r.get("outputs", {}).get("mask_path", "")
                    }
                }
            
            elif tool in {"quantify_crack_geometry", "quantify_crack_metrics"}:
                pixel_size = args.get("pixel_size_mm", 0.5)
                outputs = r.get("outputs", {})
                visuals = r.get("visualizations", {})

                if outputs and not visuals:
                    # 仅保存指标 → task 为 quantify
                    record = {
                        "subject": subject,
                        "context": {
                            "task": "quantify",
                            "pixel_size_mm": pixel_size,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        "observation": outputs
                    }
                elif visuals and not outputs:
                    # 仅保存视觉图 → task 为 generate
                    record = {
                        "subject": subject,
                        "context": {
                            "task": "generate",
                            "pixel_size_mm": pixel_size,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        "observation": visuals
                    }
                elif outputs and visuals:
                    # 若两者都有，仍归为 quantify
                    record = {
                        "subject": subject,
                        "context": {
                            "task": "quantify",
                            "pixel_size_mm": pixel_size,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        "observation": {**outputs, **visuals}
                    }
                else:
                    continue  # 没有任何 observation，跳过

            elif tool == "generate_crack_visuals":
                pixel_size = args.get("pixel_size_mm", 0.5)
                visuals = r.get("visualizations", {})
                if not visuals:
                    continue
                record = {
                    "subject": subject,
                    "context": {
                        "task": "generate",
                        "pixel_size_mm": pixel_size,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "observation": visuals
                }

            already_exists = any(
                r["subject"] == record["subject"] and
                r.get("context", {}).get("task") == record["context"]["task"]
                for r in self.records
            )

            if not already_exists:
                self.records.append(record)
                self._save_record(record)
            else:
                print(f"[⏩] 跳过重复记录: {record['subject']} ({record['context']['task']})")

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
        matches = [
            r for r in self.records
            if r.get("subject") == name
            and r.get("context", {}).get("task") == "quantify"
        ]
        if pixel_size is not None:
            matches = [
                r for r in matches
                if abs(r["context"].get("pixel_size_mm", 0) - pixel_size) < 1e-6
            ]
        print(f"[DEBUG] matches count: {len(matches)} → {[r.get('context', {}).get('task') for r in matches]}")
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
    
    def get_pixel_size(self, subject_name: str) -> float:
        """
        返回某图像最近一次 quantify 时使用的像素物理尺寸（单位 mm）。
        若未找到则返回 None。
        """
        for r in reversed(self.records):
            if r.get("subject") == subject_name and r.get("context", {}).get("task") == "quantify":
                return r.get("context", {}).get("pixel_size_mm")
        return None


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


    def get_visualization_path(self, subject_name: str, visual_type: str) -> str:
        """
        获取某图像在记录中对应的可视化图路径（如 max_width、skeleton）
        """
        for r in reversed(self.records):
            if r.get("subject") == subject_name:
                vis = r.get("observation", {})
                if isinstance(vis, dict) and visual_type in vis:
                    return vis[visual_type]
        return ""

    def update_visualization_path(self, subject_name: str, visual_type: str, path: str):
        """
        在 memory 中更新某图像的特定类型的可视化图路径。
        如果存在已有记录则追加到 observation，否则新建一条记录。
        """
        for r in reversed(self.records):
            if r.get("subject") == subject_name and r.get("context", {}).get("task") == "quantify":
                if "observation" not in r or not isinstance(r["observation"], dict):
                    r["observation"] = {}
                r["observation"][visual_type] = path
                self._save_record(r)
                return

        # fallback: 没找到则添加一条新的记录
        new_record = {
            "subject": subject_name,
            "context": {
                "task": "quantify",
                "pixel_size_mm": 0.5,
                "timestamp": datetime.utcnow().isoformat()
            },
            "observation": {
                visual_type: path
            }
        }
        self.records.append(new_record)
        self._save_record(new_record)

    def save_visualizations(self, subject_name: str, pixel_size_mm: float, visual_paths: Dict[str, str]):
        """
        批量保存某图像对应的可视化图路径，如 skeleton、max_width 等。
        会追加到该图像最近的 quantify 记录中，如无记录则新建。
        """
        for r in reversed(self.records):
            if r.get("subject") == subject_name and r.get("context", {}).get("task") == "quantify":
                if "observation" not in r or not isinstance(r["observation"], dict):
                    r["observation"] = {}
                r["observation"].update(visual_paths)
                self._save_record(r)
                return

        # fallback: 没找到旧记录则添加新记录
        new_record = {
            "subject": subject_name,
            "context": {
                "task": "quantify",
                "pixel_size_mm": pixel_size_mm,
                "timestamp": datetime.utcnow().isoformat()
            },
            "observation": visual_paths
        }
        self.records.append(new_record)
        self._save_record(new_record)

