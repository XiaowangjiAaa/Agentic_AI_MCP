# agent/object_memory_manager.py

from typing import Dict
import os

class ObjectMemoryManager:
    def __init__(self):
        self.objects: Dict[str, Dict] = {}
        self.counter = 1

    def register_image(self, image_path: str) -> str:
        object_id = f"image_{self.counter:03d}"
        self.objects[object_id] = {
            "original_path": image_path,
            "segmentation_path": None,
            "skeleton_path": None,
            "visualization_path": None,
            "status": []
        }
        self.counter += 1
        return object_id

    def update(self, object_id: str, field: str, value):
        if object_id in self.objects:
            self.objects[object_id][field] = value

    def add_status(self, object_id: str, status: str):
        if object_id in self.objects:
            self.objects[object_id]["status"].append(status)

    def get(self, object_id: str):
        return self.objects.get(object_id, {})

    def find_by_status(self, status: str):
        return [oid for oid, obj in self.objects.items() if status in obj["status"]]

    def find_id_by_image_path(self, path: str) -> str:
        for oid, obj in self.objects.items():
            if obj["original_path"] == path:
                return oid
        return None

    def find_id_by_mask_path(self, path: str) -> str:
        mask_name = os.path.basename(path)
        for oid, obj in self.objects.items():
            if obj.get("segmentation_path", "").endswith(mask_name):
                return oid
        return None
