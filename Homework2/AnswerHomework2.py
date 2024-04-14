#opencv içe aktarma
import cv2
#numpy içe aktar
import numpy as np

img=cv2.imread("yaya.jpg")
#cv2.imshow('resim',img)

#resim üzerinde bulunan kenarlari tespit etme
edge=cv2.Canny(image=img,threshold1=200,threshold2=255)
cv2.imshow(edge)

#yüz tespiti için gerekli  haar cascade'i içe aktar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #face_cascade=cv2.CascadeClassifier(".xml")

#yüz tespiti yapip sonuçlari göresellestir ,burada parametreler ile ne kadar oynarsak yüzü tespit etme ihitmalimiz daha da artar
face_rect=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=10,minSize=(30,30))

for (x,y,w,h) in face_rect:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
cv2.imshow("yüz tespiti ",img)

#Hog ile insan tesiti algoritmasini çagiralim ve svm e set edelim

hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#resme insan tespiti algoritmasi uygulayalim ve göreselleştirelim
rects, _=hog.detectMultiScale(img,padding=(8,8),scale=1.05)  #(rects,weights)=hog.detectMultiScale(img,padding=(8,8),scale=1.05)

for (xA,yA,xB,yB) in rects:
    cv2.rectangle(img,(xA,yA),(xB,yB),(0,0,255),2)

cv2.imshow("insan tespiti",img)

cv2.waitKey(0)
cv2.destroyAllWindows()

