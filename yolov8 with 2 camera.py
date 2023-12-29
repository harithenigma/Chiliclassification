import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO('/home/lattepanda/Desktop/Chiliclassification-main/best.pt')

video_capture_0 = cv2.VideoCapture(0)
video_capture_1 = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()

    if ret0:
        # Display the resulting frame
        result = model.predict(source=frame0, conf=0.5)
        annotedframe0 = result[0].plot()
        cv2.imshow('Cam 0', frame0)

    if ret1:
        # Display the resulting frame
        result = model.predict(source=frame1, conf=0.5)
        annotedframe1 = result[0].plot()
        cv2.imshow('Cam 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()
