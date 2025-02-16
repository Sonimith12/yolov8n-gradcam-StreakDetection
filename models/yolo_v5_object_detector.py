import numpy as np
import torch
import cv2
from ultralytics import YOLO

class YOLOV8TorchObjectDetector:
    def __init__(self, 
                 model_weight,
                 device='cpu',
                 img_size=(640, 640),
                 confidence=0.4,
                 iou_thresh=0.45,
                 names=None):
        """
        YOLOv8 Object Detector using Ultralytics' official YOLO class.
        
        :param model_weight: Path to the YOLOv8 .pt weights (e.g., 'yolov8s.pt')
        :param device: 'cpu' or 'cuda'
        :param img_size: Tuple (width, height) for inference (not always necessary in YOLOv8)
        :param confidence: Confidence threshold (0.0 - 1.0)
        :param iou_thresh: IoU threshold for NMS
        :param names: (Optional) List of class names if you want to override the model's defaults
        """
        self.device = device
        self.img_size = img_size
        self.confidence = confidence
        self.iou_thresh = iou_thresh

        # Load YOLOv8 model
        self.model = YOLO(model_weight)
        self.model.to(device)
        print("[INFO] YOLOv8 model loaded successfully")

        # If user provided custom class names, use them; else use the model's names
        if names is not None:
            # If names is a comma-separated string (e.g., 'dog,cat'), split it:
            if isinstance(names, str):
                names = [n.strip() for n in names.split(',')]
            self.names = names
        else:
            # Access the default names from the model
            # Ultralytics YOLO typically stores them in self.model.model.names
            self.names = self.model.model.names
        print("[INFO] Class names:", self.names)

    def forward(self, img):
        """
        Run YOLOv8 inference and return bounding boxes, class IDs, and confidence scores.
        
        :param img: Can be a NumPy array (H, W, 3) or a file path
        :return: 
          - boxes: NumPy array of shape (N, 4) in xyxy format
          - class_ids: NumPy array of shape (N,) with integer class IDs
          - confidences: NumPy array of shape (N,) with confidence scores
        """

        # If img is a NumPy array in BGR, convert to RGB (YOLOv8 expects RGB)
        if isinstance(img, np.ndarray) and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 inference
        results = self.model.predict(
            source=img,
            conf=self.confidence,
            iou=self.iou_thresh,
            device=self.device
        )

        # results is a list of 'ultralytics.engine.results.Results' (one per image)
        detections = results[0].boxes  # first image's detections

        # Extract detection info
        boxes = detections.xyxy.cpu().numpy()      # (N, 4) in xyxy format
        confidences = detections.conf.cpu().numpy()  # (N,)
        class_ids = detections.cls.cpu().numpy().astype(int)  # (N,)

        return boxes, class_ids, confidences

    def preprocessing(self, img):
        # Resize, normalize, and transpose to (C, H, W)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # Transpose to (C, H, W)
        img = torch.from_numpy(img)
        return img.unsqueeze(0)  # Add batch dimension


if __name__ == '__main__':
    # Example usage:
    model_path = 'best.pt'  # or your custom path
    detector = YOLOV8TorchObjectDetector(
        model_weight=model_path, 
        device='cpu', 
        img_size=(640, 640), 
        confidence=0.4, 
        iou_thresh=0.45,
        names=None  # or 'dog,cat' for custom names
    )

    # Load an image with OpenCV (BGR format)
    img_path = 'images/cat-dog.jpg'
    orig_img = cv2.imread(img_path)

    # Optional: Preprocess (YOLOv8 can handle raw images, but here's how to do it)
    preprocessed = detector.preprocessing(orig_img)

    # Forward pass
    boxes, class_ids, confidences = detector.forward(orig_img)
    # or use preprocessed: boxes, class_ids, confidences = detector.forward(preprocessed)

    print("Detections:")
    for box, cid, conf in zip(boxes, class_ids, confidences):
        class_name = detector.names[cid] if cid < len(detector.names) else f"ID{cid}"
        print(f" - Class: {class_name}, Box: {box}, Confidence: {conf:.2f}")
