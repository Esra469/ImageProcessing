#gorüntü esikleme
import cv2
import matplotlib.pyplot as plt

#resmi içe aktar
img=cv2.imread("resim adi")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(img,cmap="gray")
plt.axis("off")#eksenleri kapa
plt.show()

#eşikleme(treshold bak)
_,thresh_img=cv2.threshold(img,thresh=60,maxval=255,type=cv2.THRESH_BINARY)#buraya TRASH_BINARY_INV yazsam bu sefer beyaz yapmak yerine koyu yapar
plt.figure()
plt.imshow(thresh_img,cmap="gray")
plt.axis("off")
plt.show()


#uyarlamalı eşik değer(burada öneli bölgeler belirlenmiş oluyor)
thresh_img2=cv2.adaptiveTreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
plt.figure()
plt.imshow(thresh_img2,cmap="gray")
plt.axis("off")
plt.show()


