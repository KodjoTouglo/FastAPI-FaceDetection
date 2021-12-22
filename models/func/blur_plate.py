import cv2


def blur_plate(img):
    plate_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    roi = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_russian_plate_number.xml")
    plate_ret = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3,minNeighbors=3)

    for (x,y,w,h) in plate_ret:
        roi = roi[y:y+h,x:x+w]
        blurred_roi = cv2.medianBlur(roi,7)
        plate_img[y:y+h,x:x+w] = blurred_roi
    return plate_img