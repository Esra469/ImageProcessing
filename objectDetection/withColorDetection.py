#opencv de kameradan gelen rgb veya bgr olarak gelen kamera verileirini hsv ye dönüştürüyoruz 
#h=renk özü s=doygunluk v=parlaklik

#bu cözümde kontur kisminda hata veriiyor oan bak ve anla
import cv2
import numpy as np
from collections import deque #ihtiyacimiz olan merkezleri depolamamizi saglar
#nesne merkezini depolayan veri tipi
buffer_size=16
pts=deque(maxlen=buffer_size)

#mavi renk araliği HSV
blueLower=(84,98,0)
blueUpper=(179,255,255)

#capture
cap=cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    success,imgOriginal=cap.read()#kameradan a-kaynakli sorun olunca opencv hata vermiyor hatayi bizim yazmamiz lazim

    if success:
        blurred=cv2.GaussianBlur(imgOriginal,(11,11),0)

        #hsv
        hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HLS)
        cv2.imshow("HSV gortuntu:",hsv)
        
        #mavi için maske oluşturma

        mask=cv2.inRange(hsv,blueLower,blueUpper)
        cv2.imshow("mask goruntu",mask)

        #maskenin etrafinda kalan gürültüleri sil
        mask=cv2.erode(mask,None,iterations=2)
        mask=cv2.dilate(mask,None,iterations=2)
        cv2.imshow("mask +erozyon ve genisleme ",mask)

        #kontur
        (_, contours,_)=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center=None

        if len(contours)>0:
            #en buyuk kontoru al
            c=max(contours,key=cv2.contourArea)

            #dikdörtgene çevir
            rect=cv2.minAreaRect(c)
            ((x,y),(width,height),rotation)=rect
            s="x: {},y: {},weight: {},height: {}".format(np.round(x),np.round(y),np.round(height),np.round(width),np.round(rotation))
            print(s)
            #kutucuk
            box=cv2.boxPoints(rect)
            box=np.int64(box)
            #moment
            M=cv2.moments(c)
            center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))

            #kontor cizdir
            cv2.drawContours(imgOriginal,[box],0,(0,255,255),2)

            #merkeze bir tane nokta çizelim:pembe
            cv2.circle(imgOriginal,center,5,(255,0,255),-1)

            #bilgileri ekrana yazdiralim
            cv2.putText(imgOriginal,s,(25,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
        #deque
        pts.appendleft(center)
        for i in range(1,len(pts)):
            if pts[i-1] is None:continue
            cv2.line(imgOriginal,pts[i-1],pts[i],(0,255,0),3)

            
        cv2.imshow("orijinal tespit",imgOriginal)

    if cv2.waitKey(1)& 0xFF==ord("q"):break


