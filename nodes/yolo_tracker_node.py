import os
import folder_paths
import numpy as np
import torch
from PIL import Image
import supervision as sv
from ultralytics import YOLO

folder_paths.folder_names_and_paths["yolov8"] = ([os.path.join(
    folder_paths.models_dir, "yolov8")], folder_paths.supported_pt_extensions)


def get_model_dir():
    root_path = folder_paths.folder_names_and_paths['custom_nodes'][0][0]
    path = os.path.join(root_path, 'ComfyUI-TrackingNodes', 'models')
    return path


def get_model_list() -> list[str]:
    path = get_model_dir()
    return os.listdir(path)


class YOLOTrackerNode:
    def __init__(self) -> None:
        ...

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": (get_model_list(), ),
            },
        }

    RETURN_TYPES = ("IMAGE", "TRACKING")
    FUNCTION = "track"
    CATEGORY = "tracking"

    def track(self, images, model_name):
        B, H, W, C = images.shape
        # load a custom model
        model = YOLO(
            f'{os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')
        tracker = sv.ByteTrack()
        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated = []

        tracked = {}

        class_tracker_id_map = {}

        for idx, image in enumerate(images):
            image = (image.cpu().numpy() * 255).astype(np.uint8)
            # bug with opencv2 and need to convert to pillow first
            results = model(Image.fromarray(image))[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            for i in range(len(detections)):
                detection = detections[i]
                class_id = model.names[detection.class_id[0]].lower()
                tracker_id = detection.tracker_id[0]
                if class_id not in class_tracker_id_map:
                    class_tracker_id_map[class_id] = {}

                if tracker_id not in class_tracker_id_map[class_id]:
                    next_id = len(class_tracker_id_map[class_id])
                    class_tracker_id_map[class_id][tracker_id] = next_id

                if class_id not in tracked:
                    tracked[class_id] = {}
                if tracker_id not in tracked[class_id]:
                    tracked[class_id][tracker_id] = [None] * len(images)

                tracked[class_id][tracker_id][idx] = list(
                    map(lambda x: int(x), detection.xyxy[0])) + [W, H]

            labels = [
                f"#{tracker_id}.{results.names[class_id].lower()}"
                for class_id, tracker_id
                in zip(detections.class_id, detections.tracker_id)
            ]

            annotated_frame = box_annotator.annotate(
                image.copy(), detections=detections)

            annotated.append(torch.FloatTensor(label_annotator.annotate(
                annotated_frame, detections=detections, labels=labels))/255.)

        annotated = torch.stack(annotated)
        del model
        return (annotated, tracked)
