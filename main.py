#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import sys

import cv2
from cv2 import dnn

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
threshold = 0.2

classNames = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
              'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor')

cap = cv2.VideoCapture("video.mp4")
net = dnn.readNetFromCaffe("MobileNetSSD_300x300.prototxt",
                           "MobileNetSSD_train.caffemodel")

if cap.isOpened() == False:
    print("Error opening video file")

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        break

    blob = dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight),
                             meanVal)
    net.setInput(blob)
    detections = net.forward()

    rows = frame.shape[0]
    cols = frame.shape[1]

    if cols / float(rows) > WHRatio:
        cropSize = (int(rows * WHRatio), rows)
    else:
        cropSize = (cols, int(cols * WHRatio))

    y1 = (rows - cropSize[1]) // 2
    y2 = y1 + cropSize[1]
    x1 = (cols - cropSize[0]) // 2
    x2 = x1 + cropSize[0]
    frame = frame[y1:y2, x1:x2]

    rows = frame.shape[0]
    cols = frame.shape[1]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom),
                          (xRightTop, yRightTop), (0, 255, 0))
            label = classNames[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                          (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
