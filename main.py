import cv2
from ultralytics import YOLO

# 1. Load your custom model
# Ensure 'best.pt' is the correct path to your custom weights
model = YOLO('best.pt', task='detect')  # Specify task if needed (e.g., 'detect', 'segment', etc.)

# 2. Run inference on a video file
# source: path to video file, URL, or '0' for webcam
# stream=True: memory-efficient, returns a generator for long videos
results = model.predict(
    source="C:\\Users\\finet\\Downloads\\dtinnerspotweld_conv.mp4", 
    conf=0.45,
    imgsz=640,     # resize to 640x640 for inference      
    save=True,      # save annotated video to 'runs/detect/predict'
    show=True,      # display video in a window while processing
    stream=True     # use a generator to save memory
)

# 3. (Optional) Process individual frames manually
for result in results:
    # 'result' contains boxes, masks, keypoints, etc.
    # boxes = result.boxes  # Bounding boxes object
    # names = result.names  # Class names mapping
    
    # If you want to stop the loop manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()