#kayan pencere kullanmak yerine goruntuyu ızgara kullanarak böler
#ve her bir ızgara hücresinin goruntunun o bolgesindeki nesneleri tespit etmekten sorumlu olmasını saglar
#nesneleri algilamak, o bolgedeki bir nesnenin sinifini ve konumunu tahmn etmke anlamina gelir

import numpy as np
import os
import cv2

CLASSES=["background","aeroplane","biciycle","bird","boat",
         "bottle","bus","car","cat","chair","cow","diningtable",
         "dog","horse","motorbike","person","pottedplant","sheep",
         "sofa","train","tvmonitor"]

COLOR=np.random.uniform(0,255,size=(len(CLASSES),3))#her classa ait bir tane renk olustu

net=cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel")#hazir bir sey kullaniyoruz dirkt bir sey egitmiyoruz

#eger video yakalamak istersek yani reel time calsmak isteersek eger
# vc=cv2.VideoCapture(0)
# vc.set(3,800)
# vc.set(4,600) boyle ve birkac sey degisecek

files=os.listdir()
img_path_list=[]
for f in files:
    if f.endswith(".jpg"):#jpg ile bitenleri ekle
        img_path_list.append(f)

for i in img_path_list:
    image=cv2.imread(i)
    (h,w)=image.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007843,(300,300),127.5)

    net.setInput(blob)
    detection=net.forward()

    for j in np.arange(0,detection.shape[2]):
        confidence=detection[0,0,j,2]
        if confidence>0.3:#buradaki degerin artip azalmasina bagli olarak sonuc degisir
            idx=int(detection[0,0,j,1])
            box=detection[0,0,j,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")

            label="{}:{}".format(CLASSES[idx],confidence)
            cv2.rectangle(image,(startX,startY),(endX,endY),COLOR[idx],2)
            y=startY-16 if startY-16>15 else startY+16
            cv2.putText(image,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLOR[idx],2)
        
        cv2.imshow("ssd",image)
        if cv2.waitKey(0) & 0xFF==ord("q"):continue