import os
import folder_paths
import numpy as np
import supervision as sv
import torch


folder_paths.folder_names_and_paths["yolov8"] = ([os.path.join(folder_paths.models_dir, "yolov8")], folder_paths.supported_pt_extensions)

keypoint_to_class_id_map = {
    'pose_keypoints_2d': 0,
    'face_keypoints_2d': 1,
    'hand_left_keypoints_2d': 2,
    'hand_right_keypoints_2d': 3
}

class_ids_to_names_map = {
    0: 'body',
    1: 'face',
    2: 'hand_left',
    3: 'hand_right'
}

buffers = {
    'pose_keypoints_2d': { 'x': 0, 'y': 0},
    'face_keypoints_2d': { 'x': 20, 'y': 30},
    'hand_left_keypoints_2d': { 'x': 20, 'y': 20},
    'hand_right_keypoints_2d': { 'x': 20, 'y': 20},
}


class OpenPoseTrackerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",), 
                "pose_keypoints": ("POSE_KEYPOINT",),
            },
        }

    RETURN_TYPES = ("IMAGE", "TRACKING")
    FUNCTION = "track"
    CATEGORY = "tracking"

    # class_id.tracker_id.frame = [x0, y0, x1, y1, w,]
    # tracker.update_with_detections(sv.Detections(xyxy=empty_array, class_id=np.array([2]), confidence=np.array([1])))
    def _get_bounding_box(self, points, W, H, keypoint_name):
        x0, y0, x1, y1 = W, H, 0, 0
        confidence = 0
        for i in range(0, len(points), 3):
            # Extract groups of three
            x_point, y_point, c_point = points[i:i+3]
            if c_point < 0.1:
                return None
            x0 = min(x_point*W, x0)
            x1 = max(x_point*W, x1)
            y0 = min(y_point*H, y0)
            y1 = max(y_point*H, y1)
            confidence += c_point
        
        buffer = buffers[keypoint_name]
        x0 = max(0, x0-buffer['x'])
        x1 = min(W, x1+buffer['x'])
        y0 = max(0, y0-buffer['y'])
        y1 = min(H, y1+buffer['y'])
        return [x0, y0, x1, y1, W, H, 1] #confidence/len(points)]
    
    def _get_detections_from_person(self, person, W, H):
        detections = []
        for key in keypoint_to_class_id_map:
            if key not in person or person[key] is None:
                continue

            bb = self._get_bounding_box(person[key], W, H, key)
            if bb is None:
                continue
            xyxy = np.array([[bb[0], bb[1], bb[2], bb[3]]])
            confidence = bb[-1]
            if confidence < 0.1:
                continue
            detection = sv.Detections(xyxy=xyxy, class_id=np.array([keypoint_to_class_id_map[key]]), confidence=np.array([confidence]))
            detections.append(detection)

        return detections
    
    def _get_detections_for_frame(self, keypoint, W, H):
        frame_detections = []
        for person in keypoint['people']:
            detections = self._get_detections_from_person(person, W, H)
            frame_detections.extend(detections)
        
        return sv.Detections.merge(detections)
        

    def track(self, images, pose_keypoints):
        B, H, W, C = images.shape
        tracker = sv.ByteTrack()
        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated = []
        tracked = {}
        class_tracker_id_map = {}
        for idx, keypoint in enumerate(pose_keypoints):
            detections = self._get_detections_for_frame(keypoint, W, H)
            detections = tracker.update_with_detections(detections)

            for i in range(len(detections)):
                detection = detections[i]
                class_id = class_ids_to_names_map[detection.class_id[0]].lower()
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
                f"#{tracker_id}.{class_ids_to_names_map[class_id]}"
                for class_id, tracker_id
                in zip(detections.class_id, detections.tracker_id)
            ]

            image = (images[idx].cpu().numpy() * 255).astype(np.uint8)
            annotated_frame = box_annotator.annotate(
                image.copy(), detections=detections)

            annotated.append(torch.FloatTensor(label_annotator.annotate(
                annotated_frame, detections=detections, labels=labels))/255.)
            
        annotated = torch.stack(annotated)
        return (annotated, tracked)
