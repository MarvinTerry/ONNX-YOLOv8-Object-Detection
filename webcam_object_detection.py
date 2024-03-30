import cv2
from cv2 import getTickCount, getTickFrequency

from yolov8onnx import YOLOv8

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv7 object detector
model_path = "models/rock.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():
    loop_start = getTickCount()

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)
    combined_img = yolov8_detector.draw_detections(frame)

    for box in boxes:
        print(box)


    # 中间放自己的显示程序
    loop_time = getTickCount() - loop_start
    total_time = loop_time / (getTickFrequency())
    FPS = int(1 / total_time)
    # 在图像左上角添加FPS文本
    fps_text = f"FPS: {FPS:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)  # 红色
    text_position = (10, 30)  # 左上角位置

    cv2.putText(combined_img, fps_text, text_position, font, font_scale, text_color, font_thickness)
    
    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
