import cv2
from cv2 import getTickCount, getTickFrequency
import numpy as np
import math
import rospy
from geometry_msgs.msg import Point

from yolov8onnx import YOLOv8

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width=1920
cap.set(4, 720)  # height=1080

# Initialize YOLOv8 object detector
model_path = "models/rock_480px.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.3)
class_names = ['rock','paper','scissors']
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]
print(frame_height,frame_width)
frame_center = (frame_width/2,frame_height/2)

# init ros
rospy.init_node('yolo', anonymous=True)
pub1 = rospy.Publisher('/target',Point,queue_size=10)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():
    # fps_counter
    loop_start = getTickCount()

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)
    combined_img = yolov8_detector.draw_detections(frame)
    # combined_img = np.zeros((480,640,3))

    for i,box in enumerate(boxes):
        # print(f"{i+1}: {class_names[class_ids[i]]} {box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}")
        box_center = ((box[0]+box[2])/2,(box[1]+box[3])/2)
        pub1.publish(Point(x=box_center[0],y=box_center[1]))
        distance = math.sqrt((box_center[0]-frame_center[0])**2 + (box_center[1]-frame_center[1])**2)
        print(f"***********{distance}")



    # print(yolov8_detector.input_width,yolov8_detector.input_height)

    # fps_counter
    loop_time = getTickCount() - loop_start
    total_time = loop_time / (getTickFrequency())
    FPS = int(1 / total_time)
    fps_text = f"FPS: {FPS:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255) 
    text_position = (10, 30) 
    cv2.putText(combined_img, fps_text, text_position, font, font_scale, text_color, font_thickness)
    

    cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
