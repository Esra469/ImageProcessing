#kontur algilama, ayni renk ve yogunluga sahip tüm kessintisiz noktarli birleştirmeyei amaçlayan bir yöntemdir genelde şekil analizide kullanılır

import cv2 
import matplotlib.pyplot as plt
import numpy as np

#resmi içe aktar
img=cv2.imread("C:/Users/ASUS/Pictures/sekiller.jpg",0)
plt.figure(),plt.imshow(img,cmap="gray"),plt.axis("off")
image,contours,hierarch= cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)#Kontur olarak 4 kenar çizmeye yarar->find contour ile hem iç hem diş kontur alinir buna da hierarch değişkeni ile yapacagiz

external_contour=np.zero(img.shape)
internal_contour=np.zeros(img.shape)

for i in range(len(contours)):
    #extrenal
    if hierarch[0][i][3]==-1:
        cv2.drawContours(external_contour,contours,i,255,-1)#255=renk -1 =kalinlik saglar
    else:
        cv2.drawContours(internal_contour,contours,i,255,-1)
plt.figure(),plt.imshow(external_contour,cmap="gray"),plt.axis("off")#burada distan işlem yapiliyor distai kenarara işlme yapar
plt.figure(),plt.imshow(internal_contour,cmap="gray"),plt.axis("off")








