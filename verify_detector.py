import cv2
import urllib.request
import os
from src.detector import YoloDetector

def verify():
    print("Downloading sample image...")
    img_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
    img_path = "sample_image.jpg"
    urllib.request.urlretrieve(img_url, img_path)

    print("Loading image...")
    frame = cv2.imread(img_path)
    
    if frame is None:
        print("Error: Could not load image.")
        return

    print("Initializing YOLOv8n detector...")
    detector = YoloDetector(model_weight='yolov8n.pt', conf_thresh=0.3)
    
    print("Running detection...")
    dets = detector.detect(frame)
    
    print(f"Detection completed. Found {len(dets)} person(s).")
    for i, d in enumerate(dets):
        print(f"Person {i+1}: Box [x1={d[0]:.1f}, y1={d[1]:.1f}, x2={d[2]:.1f}, y2={d[3]:.1f}], Confidence: {d[4]:.2f}")

    # Clean up
    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists('yolov8n.pt'):
        os.remove('yolov8n.pt')
        
if __name__ == "__main__":
    verify()
