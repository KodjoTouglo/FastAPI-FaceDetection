import cv2


def detect_smile(img):
    face_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")
    face_rect = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    
    for (x,y,w,h) in face_rect:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,0,0),2)
    return face_img