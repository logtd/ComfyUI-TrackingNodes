from .nodes.yolo_tracker_node import YOLOTrackerNode
from .nodes.openpose_tracker_node import OpenPoseTrackerNode

NODE_CLASS_MAPPINGS = {
    "YOLOTrackerNode": YOLOTrackerNode,
    "OpenPoseTrackerNode": OpenPoseTrackerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLOTrackerNode": "YOLO BoundingBox Tracker",
    "OpenPoseTrackerNode": "OpenPose BoundingBox Tracker"
}
