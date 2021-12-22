import cv2



def blur_face(img):
    face_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    roi = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    plate_ret = plate_cascade.detectMultiScale(face_img,scaleFactor=1.3,minNeighbors=3)

    for (x,y,w,h) in plate_ret:
        roi = roi[y:y+h,x:x+w]
        blurred_roi = cv2.blur(roi,(50,50))
        face_img[y:y+h,x:x+w] = blurred_roi
    return face_img