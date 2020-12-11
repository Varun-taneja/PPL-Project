import numpy as np
import cv2
import imutils

gun_cascade = cv2.CascadeClassifier('guns.xml')
cap = cv2.VideoCapture(0)

#Set default values
firstFrame = None
gun_exist = False

while True:
    #Read Frames
    ret, frame = cap.read()
    #Resize and Grayscale frames
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detect Guns in Frames
    gun = gun_cascade.detectMultiScale(gray,1.3, 5, minSize=(100, 100))

    if len(gun) > 0:
        gun_exist = True
    #Mark Guns Detected using Contours
    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame,
                              (x, y),
                              (x + w, y + h),
                              (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(gun, 'Gun Detected', (x + 6, y - 6), font, 0.5, (0, 0, 255), 1)

    if firstFrame is None:
        firstFrame = gray
        continue
    #Display Live Feed
    cv2.imshow("Security Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

if gun_exist:
    print("guns detected")
else:
    print("guns NOT detected")

cap.release()
cv2.destroyAllWindows()
