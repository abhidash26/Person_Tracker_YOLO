import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_weight='yolov8n.pt', conf_thresh=0.3):
        """
        Initializes the YOLOv8 detector.
        """
        # Load the pretrained model (will automatically download yolov8n.pt if not found)
        self.model = YOLO(model_weight)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        """
        Runs the detector on a single frame.
        
        Args:
            frame: A numpy array representing the image (e.g., from cv2.imread).
            
        Returns:
            dets: A numpy array of shape (N, 5) where each row is [x1, y1, x2, y2, conf]
                  for detected persons. If no persons are detected, returns an empty array.
        """
        # Run inference. classes=[0] filters for 'person' class in COCO.
        # verbose=False reduces terminal spam.
        results = self.model(frame, classes=[0], conf=self.conf_thresh, verbose=False)
        
        # Results is a list (one for each frame in the batch, we only passed one).
        result = results[0]
        boxes = result.boxes
        
        if len(boxes) == 0:
            return np.empty((0, 5))
            
        # Extract coordinates and confidence
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        
        # Format for SORT: [x1, y1, x2, y2, conf]
        dets = np.concatenate((xyxy, conf), axis=1)
        
        return dets
