#morfolojik işlemler erezyon, genişleme ,açma ,kapama,morfolojik gradyan
"""
Erezyon=çn plandaki nesennin sinirlarini söndürür
genişleme=beyaz bölgeyi artitti erezyon tam tersi

açma=erezyon+genişleme
kaptma=genişleme+erzyon
morfolojik gradyan->bir görüntünün genişlemesi ve erozyonu arasindaki farktir

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
#resmi içe aktarma
img=cv2.imread("resim adi",0)
plt.figure(),plt.imshow(img,cmap="gray"),plt.axis("off"),plt.title("orijinal resim")

#erezyon
kernel=np.ones((5,5),dtype=np.uint8)
result=cv2.erode(img,kernel,iteration=1)#iteration ne kadar fazlaysa o kadar fazal erozyon olur 
plt.figure(),plt.imshow(result,cmap="gray"),plt.axis("off"),plt.title("erozyon")

#genişleme(dilation)
result2=cv2.dilate(img,kernel,iteration=1)
plt.figure(),plt.imshow(result2,cmap="gray"),plt.axis("off"),plt.title("genişleme")

#açilma yöntemi(beyaz gürültüyü azaltmak için)

#white noise
whiteNoise=np.random.randit(0,2,size=img.shape[:2])
whiteNoise=whiteNoise*255
plt.figure(),plt.imshow(whiteNoise,cmap="gray"),plt.axis("off"),plt.title("white noise")
noise_img=whiteNoise+img
plt.figure(),plt.imshow(noise_img,cmap="gray"),plt.axis("off"),plt.title("img with white noise")

#açilma
opening=cv2.morphology(noise_img.astype(np.float32),cv2.MORPH_OPEN,kernel)
plt.figure(),plt.imshow(opening,cmap="gray"),plt.axis("off"),plt.title("acilma")
#gürültü oluşturup yok etti sonra

#black noise yap
blackNoise=np.random.randit(0,2,size=img.shape[:2])
blackNoise=blackNoise*-255
plt.figure(),plt.imshow(blackNoise,cmap="gray"),plt.axis("off"),plt.title("white noise")

blackNoise_img=blackNoise+img
blackNoise_img[blackNoise_img<=-245]=0
plt.figure(),plt.imshow(blackNoise_img,cmap="gray"),plt.axis("off"),plt.title("whith black img ")

#kapatma
closing=cv2.morphologyEx(blackNoise_img.astype(np.float32),cv2.MORPH_CLOSE,kernel)
plt.figure(),plt.imshow(closing,cmap="gray"),plt.axis("off"),plt.title("kapama")

#gradient
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
plt.figure(),plt.imshow(gradient,cmap="gray"),plt.axis("off"),plt.title("gradyan")



#gradyanlar görüntüdeki yoğunluk veya renkteki yönlü değişiklikler(kenar tespitlerde kullanılır)

import cv2
import matplotlib.pyplot as plt

img=cv2.imread("resim adi",0)
plt.figure(),plt.imshow(img,cmap="gray"),plt.axis("off"),plt.title("orijinal resim")

#x gradyan
sobelx=cv2.Sobel(img,ddepth=cv2.CV_165,dx=1,dy=1,ksize=5)
plt.figure(),plt.imshow(sobelx,cmap="gray"),plt.axis("off"),plt.title("sobel X")
#y gardyan
sobely=cv2.Sobel(img,ddepth=cv2.CV_165,dx=0,dy=1,ksize=5)
plt.figure(),plt.imshow(sobely,cmap="gray"),plt.axis("off"),plt.title("sobel y")

#laplace gradyan
laplace=cv2.Laplacian(img,ddepth=cv2.CV_165)
plt.figure(),plt.imshow(laplace,cmap="gray"),plt.axis("off"),plt.title("laplacian")

