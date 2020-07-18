import os
import cv2
from random import randrange


model = cv2.CascadeClassifier(
    os.path.join("models", "haarcascade_frontalface_default.xml")
)

# Read from webcam or default camera in the system, if video path is provided
# would read that
vid = cv2.VideoCapture(0)

while True:
    stub, frame = vid.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coords = model.detectMultiScale(gray_img)
    for (x, y, w, h) in coords:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (randrange(256), randrange(256), randrange(256)),
            5
        )

    cv2.imshow("Live Stream", frame)
    # in milliseconds, if time is specified it auto updates instead
    # of waiting for key press
    key = cv2.waitKey(1)

    # ASCII code for capital and small Q
    if key == 81 or key == 113:
        break

vid.release()
