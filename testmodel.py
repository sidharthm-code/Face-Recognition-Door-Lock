# pip install opencv-python==4.5.2

import cv2 
# from controller import doorAutomate
def doorAutomate(val):
    if val == 0:
        print("Door would OPEN")
    else:
        print("Door would CLOSE")
import time

video=cv2.VideoCapture(0)


facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["", "Sidharth", "Person2"]

imgBackground = cv2.imread("background.png")

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf<50:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, name_list[serial], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, "Unknown", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    frame=cv2.resize(frame, (730, 485))
    imgBackground[150:150 + 485, 55:55 + 730] = frame
    cv2.imshow("Frame",imgBackground)
    
    k=cv2.waitKey(1)
    
    if k==ord('o') and conf>50:
        doorAutomate(0)
        time.sleep(10)
        doorAutomate(1)
    if k==ord("q"):
        break

video.release()
cv2.destroyAllWindows()
