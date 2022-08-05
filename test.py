import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

try:
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands = 1)
    classifier = Classifier("keras_model.h5", "labels.txt")

    # Define Constants
    offset = 20
    imgSize = 300
    counter = 0
    labels = ["Hello", "Yes", "No", "I Love You", "Sorry"]
    key = 0

    # Press Escape to quit
    while (key != 27):
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[(y - offset) : (y + h + offset), (x - offset) : (x + w + offset)]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap : wCal + wGap] = imgResize

                # Making predictions
                prediction, index = classifier.getPrediction(imgWhite, draw = False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap : hCal + hGap, :] = imgResize

                # Making predictions
                prediction, index = classifier.getPrediction(imgWhite, draw = False)

            cv2.putText(imgOutput, labels[index], (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,0), 2)

            cv2.rectangle(imgOutput, (x - offset, y - offset),(x + w + offset, y + h + offset), (51, 255, 51), 4)

        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)

    source.release()

except:
    print("You are moving out of window")