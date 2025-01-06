import cv2
import mediapipe
import os
import HandDetectModule as hdm

wCam, hcam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4, hcam)

detector = hdm.handDetector()
myImgs = os.listdir("numbers")
overlayList= []
fPath = "numbers"
for im in myImgs:
    image = cv2.imread(f'{fPath}/{im}')
    overlayList.append(image)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, drawBool=False)
    
    if(len(landmarkList)!=0):
        numFingers = 0 

        #thumb
        
        thumb2 = detector.getXVal(landmarkList, 2)
        thumb4 = detector.getXVal(landmarkList, 4)
        
        if(thumb4 > thumb2):
            numFingers = numFingers + 1
        
        #index
        index6 = detector.getYVal(landmarkList, 6)
        index8 = detector.getYVal(landmarkList, 8)
    
        if(index8 < index6):
            numFingers = numFingers + 1
    
        #middle
        
        middle10 = detector.getYVal(landmarkList, 10)
        middle12 = detector.getYVal(landmarkList, 12)

        if(middle12 < middle10):
            numFingers = numFingers + 1
    
        #ring
        ring14 = detector.getYVal(landmarkList, 14)
        ring16 = detector.getYVal(landmarkList, 16)

        if(ring16 < ring14):
            numFingers = numFingers + 1
    
        #pinky
        pinky18 = detector.getYVal(landmarkList, 18)
        pinky20 = detector.getYVal(landmarkList, 20)

        if(pinky20 < pinky18):
            numFingers = numFingers + 1
        
        if(numFingers!=0):
            img[0:200, 0:200] = overlayList[numFingers-1]

        print(numFingers)

    cv2.imshow("Image", img)
    cv2.waitKey(1)