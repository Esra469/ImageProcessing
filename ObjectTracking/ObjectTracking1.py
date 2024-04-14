#opencv ile nesne takibi
"""
opencv de nesne takipi algortimalari
ortalama kayma algoritmasi
takip agoritmasi
çoklu nesne takibi

1)MeanShift (ortalama kayma) noktalari moda doğru kaydirarak veri noktalarini kümelere yinelemeli olarak atayan bir kümeleyen bir algoirtma
mod en yüksek veri noktasi yoğunlugudur 
"""
#şidiki poroblemde bir yüz taniyacagiz daha sonra kameradan video alacagiz ve o yüzü tanimasini saglamaliyiz

import cv2
import numpy as np

#kamera aç
cap=cv2.VideoCapture(0)


#bir tane frame oku
ret,frame=cap.read()

if ret==False:
    print("uyari kamera açilmadi ")

#detection
#face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_rects=face_cascade.detectMultiScale(frame)

(face_x,face_y,w,h)=tuple(face_rects[0])

track_window=(face_x,face_y,w,h)#meanshift algoritmasi girdisi

#region of interested
roi=frame[face_y:face_y+h,face_x:face_x+w]#roi=face

hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)


roi_hist=cv2.calcHist([hsv_roi],[0],None,[180],[0,180])#takip için histogram gereklidir

cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)#yani 0 ve 255 arasinda sıkıştır demek
#takip için gerekli durma kriterleri
#count=hesaplanacak maksimum oge sayisi
#eps=degisiklilik

term_crit=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
while True:
    ret,frame=cap.read()
    if ret:
        hsv=cv2.Color(frame,cv2.COLOR_BGR2HSV)
        #histogrami bir goruntude bulmak için kullaniyoruz
        #piksel karsilastirma
        dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        ret,track_window=cv2.meanShift(dst,track_window,term_crit)

        x,y,w,h=track_window

        img2=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)
        cv2.imshow("takip",img2)

        if cv2.waitKey(1) & 0xFF==ord("q"):break

cap.release()
cv2.destroyAllWindows()






