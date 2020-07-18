import os
import cv2
from random import randrange


model = cv2.CascadeClassifier(
    os.path.join("models", "haarcascade_frontalface_default.xml")
)
img = cv2.imread(os.path.join("images", "Oval.png"))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
coords = model.detectMultiScale(gray_img)
# Draw a rectangle
# coords would be list of (x, y, w, h) with x and y coords for a 2-d pic with
# width and height identified by the trained model

for (x, y, w, h) in coords:
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (randrange(256), randrange(256), randrange(256)),
        5
    )

cv2.imshow("Random Image", img)

cv2.waitKey()
