import cv2
from ultralytics import YOLO

model = YOLO('C:/Users/harit/Desktop/Chili/content/runs/detect/train3/weights/best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    result = model.predict(source = frame, conf=0.88)

    annotedFrame = result[0].plot()

    cv2.imshow('Chili Detection', annotedFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()