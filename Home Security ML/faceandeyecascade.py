import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
cap= cv2.VideoCapture(0)

while cap.isOpened():
    #reading frames of video
    _, img = cap.read()
    #converting frames to grayscale
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #detecting faces using cascade
    faces = face_cascade.detectMultiScale(gr, 1.1, 4)

    #looping over detected faces to mark them in video
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        roi_gray = gr[y:y+h, x:x+w]
        roi_color = img[y:y + h, x:x + w]
        #detecting eyes using cascade
        eyes = eye_cascade.detectMultiScale(roi_gray)
        #looping over detected eyes to mark them in respective detected faces
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
        cv2.imshow("img", img)
        if cv2.waitKey(1)==27:
            break
cap.release()
cv2.destroyAllWindows()
