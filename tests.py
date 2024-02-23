import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


cap = cv2.VideoCapture(0)                                                                           #id num for webcam = 0
detector = HandDetector(maxHands=1)                                                                 #one hand
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")


offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:                                                                                      #if there's something in the hands
        hand = hands[0]                                                                            #one hand
        x, y, w, h = hand['bbox']                                                                  #gives values of x, y, width, height

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255                              #white backround
        cv2.imshow("ImageWhite", imgWhite)

        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]                                #gives bounding box that we require


        imgCropShape = imgCrop.shape                                                               #matrix of 3 channels: height, width, and channels


        aspectRatio = h/w

        if aspectRatio > 1:                                                                        #if h is bigger than w
            k = imgSize/h                                                                          #konstant = stretching height
            wCal = math.ceil(k*w)                                                                  #if it's 3.5 or 3.2, it'll go to 4  wCal = width calculated
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)                                                     #centers the image on the white space
            imgWhite[:, wGap:wCal+wGap] = imgResize                                                #put image crop matrix inside the image white matrix at these values  and centers image on the white space
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)


        else:                                                                                      #if h is bigger than w
            k = imgSize/w                                                                          #konstant = stretching width
            hCal = math.ceil(k*h)                                                                  #if it's 3.5 or 3.2, it'll go to 4  hCal = hight calculated
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)                                                     #centers the image on the white space
            imgWhite[hGap:hCal+hGap, :] = imgResize                                                # put image crop matrix inside the image white matrix at these values  and centers image on the white space
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255), cv2.FILLED)      #Rectangle on Letter


        cv2.putText(imgOutput, labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)

        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)                        #Rectangle on Hand



        cv2.imshow("ImageCrop", imgCrop)                                                #gives another image that's cropped
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
