#Haar özellikler terimsel olrak çok geçti
#githubeden bir yüz tanima şeysi aldik

import cv2
import matplotlib.pyplot as plt

#içe aktar
einstain=cv2.imread("einstainResim.jpg",0)
plt.figure(),plt.imshow(einstain,cmap="gray"),plt.axis("off")
#einstani taniöıyor yüz var diyor siniflandirici
face_cascade=cv2.CascadeClassifier("indirdiğimiz.xml")

face_rect=face_cascade.detectMultiScale(einstain)
for (x,y,w,h) in face_rect:
    cv2.rectangle(einstain,(x,y),(x+w,y+h),(255,255,255),10)
plt.figure(),plt.imshow(einstain,cmap="gray"),plt.axis("off")

#şimdi de başka bir çoklu resim deniyoruz
#içe aktar
barce=cv2.imread("barcelona.jpg",0)
plt.figure(),plt.imshow(barce,cmap="gray"),plt.axis("off")

face_rect=face_cascade.detectMultiScale(barce,minNeighbors=3)#bazen hatalar olabilr bunun için yeni özellik eklebilir mesela minNeighbos 

for (x,y,w,h) in face_rect:
    cv2.rectangle(barce,(x,y),(x+w,y+h),(255,255,255),10)
plt.figure(),plt.imshow(barce,cmap="gray"),plt.axis("off")
 

#video
cap=cv2.VideoCapture(0)#kendi kameramiza baglanti oluşturduk
while True:
    ret,frame=cap.read()
    if ret:
        face_rect=face_cascade.detectMultiScale(frame,minNeighbors=7)

        for (x,y,w,h) in face_rect:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),10)
        cv2.imshow("face detect",frame)
    if cv2.waitKey(1)& 0xFF== ord("q"):break
cap.release()
cv2.destroyAllWindows()