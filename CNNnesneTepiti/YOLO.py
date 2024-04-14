#yolo yu bir nesne bulucu ve taniyici olarak tanimlayabilirz
#nesne algilamak icin evrisimsel sinir agini kullanir.Bu calismada bize verilen yolo kutuphansini kullanarak nesne tespiti yapacagiz

import cv2
import numpy as np
from yolo_model import YOLO

yolo=YOLO(0.6,0.5)
file="data/coco_classes.txt"#belli dosyalardan birini cektik

with open(file) as f:
    class_name=f.readline()#burada acilan classlarin ne oldugunu gorduk

all_classes=[c.strip() for c in class_name]#numpy ın bosluk silmek icin bir ozellesmis fonksiyonu strip

f="dog_cat.jpg"
path="images/"+f

image=cv2.imread(path)
cv2.imshow("image",image)

#preprocess yapıyoruz
pimage=cv2.resize(image,(416,416))
pimage=np.array(pimage,dtype="float32")
pimage /=255.0 #resmi normalize etmek icin yapiyoruz
pimage=np.expand_dims(pimage,axis=0)#yoo için gerekli olan bir sey bu sonrada bir parranetez genisletiyor

#yolo
boxes,classes,scores=yolo.predict(pimage,image.shape)

for box,score,c1 in zip(boxes,scores,classes):
    x,y,w,h=box

    top=max(0,np.floor(x+0.5).astype(int))#buradaki floor ondalikli sayiyi tabana yuvarlar o sayiyi alir
    left=max(0,np.floor(y+0.5).astype(int))
    right=max(0,np.floor(x+w+0.5).astype(int))
    bottom=max(0,np.floor(y+h+0.5).astype(int))

    cv2.rectangle(image,(top,left),(right,bottom),(255,0,0),2)
    cv2.putText(image,"{} {}".format(all_classes[c1].score),(top,left-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)

cv2.imshow("yolo",image)