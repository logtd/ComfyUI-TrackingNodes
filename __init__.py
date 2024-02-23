from .nodes.yolo_tracker_node import YOLOTrackerNode


NODE_CLASS_MAPPINGS = {
    "YOLOTrackerNode": YOLOTrackerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLOTrackerNode": "YOLO BoundingBox Tracker",
}
