import cv2


def detect_plate(img):
    plate_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_russian_plate_number.xml")
    plate_ret = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3,minNeighbors=3)

    for (x,y,w,h) in plate_ret:
        cv2.rectangle(plate_img,(x,y),(x+w,y+h),(255,0,0),2)
    return plate_img