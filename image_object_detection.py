import cv2
from imread_from_url import imread_from_url

from yolov8onnx import YOLOv8

# Initialize yolov8 object detector
model_path = "models/rock.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Read imaged

img = cv2.imread('image.jpg')
frame_height, frame_width = img.shape[:2]
print(frame_height,frame_width)

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)
for i,box in enumerate(boxes):
    print(f"{i+1}: {box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}")


# Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
