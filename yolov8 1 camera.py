import cv2
from ultralytics import YOLO

model = YOLO('/home/lattepanda/Desktop/Chiliclassification-main/best.pt')

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    ret, frame = cap1.read()
    ret, frame = cap2.read()

    result = model.predict(source = frame, conf=0.5)

    annotedFrame = result[0].plot()

    cv2.imshow('Chili Detection', annotedFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
