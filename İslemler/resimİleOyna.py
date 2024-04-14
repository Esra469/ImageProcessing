import cv2
#RGB->bizim günlük hayatta kullandigimiz renkler
img=cv2.imread("resmin yolu",0)#0 yazmak siyyah beyaz olarak ayrlamayya yarar renkli bakmak istemiyorsan 0 verme hiç
print("resim boyutu",img.shape)#piksel sayısını verir
cv2.imshow("orijinal ",img)

#resize
imgResize=cv2.resize(img,(800,800))#yeniden boyutllandırma
print("yendenn boyut",imgResize.shape)
cv2.imshow("img resize",imgResize)

#kırp
imgCropped=img[:200,0:300]#x eksenşnde ilk 200 piksek y eksenşnden 0 dan 300 e kadar olan piksel önce yükseklik sonra genişlik
cv2.imshow("kirpimis resim",imgCropped)


#bir metin üzerine metin ekleme opencv renk kodlarını rgb olarak değil de bgr olarak alıyor buna örnek olarak normlade kırmızı renk=(255,0,0) ama opencv de (0,0,255)dir
import cv2 
import numpy as np

#resim oluştur
img=np.zero((512,512,3),np.uint8)#siyah bir resim çizer
print(img.shape)
cv2.imshow("siyah ",img)

#çizgi ->(resim başlamgiç noktasi,resim bitiş noktasi,renk,kalinlik)
cv2.line(img,(100,100),(100,300),(0,255,0),3)
cv2.imshow("çizgi",img)

#dikdörtgen
#(resim,başlangiç, bitiş ,renk,doldurma)
cv2.rectangle(img,(0,0),(256,256),(255,0,0),cv2.FILLED)
cv2.imshow("dikdörtgen",img)

#çember
cv2.circle(img,(300,300),45,(0,0,255),cv2.FILLED)
cv2.imshow("çember",img)

#metin ekleme
#resim başlangiç noktasi ,fontkalinlği,renk
cv2.putText(img,"metin",(350,350),cv2.FONT_HERSHEY_COMLEX,1,(255,255,255))
cv2.imshow("metin",img)


#birden fazla görüntünün birleştirilmesi
import cv2
import numpy as np

#resmi içe aktarma
img=cv2.imread("resima adi")
cv2.imshow("orijianl",img)
#yatay
hor=np.hstack((img,img))
cv2.imshow("yatay(horizonal)",hor)
#dikey
ver=np.vstack((img,img))
cv2.imshow("dikey",ver)

#perspektif çarpitma
import cv2 
import numpy as np
#resmi içe aktar
img=cv2.imread("resim adi")
cv2.imshow("orijinal ",img)

width=400
height=500

pts1=np.float([[230,1],[1,472],[540,150],[338,617]])
pts2=np.float([[0,0],[0,height],[width,0],[width,height]])

matrix=cv2.getPerspectiveTransform(pts1,pts2)
print(matrix)
#nihai dönüştürülmüş resim
imgOutput=cv2.warperspective(img,matrix,(width,height))
cv2.imshow("nihai resim",imgOutput)


#görüntüleri kariştirma
import cv2
import matplotlib.pyplot as plt

#kariştirma
#opencv resimleri yüklerken bgr ye göre girdiğini söylemiştik bunu düzeltmenşn yolları da var aslinda 
img1=cv2.imread("resim1")
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)#bu sayede sorun çözülür
img2=cv2.imread("resim2")
img2=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

print(img1.shape)
print(img2.shape)

img1=cv2.resize(img1,(600,600))
print(img1.shape)

img2=cv2.resize(img2,(600,600))
print(img2.shape)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

#kariştirilmiş resim=alpha+img1+beta*img2
blended=cv2.addWeigted(src1=img1,alpha=0.5,src2=img2,beta=0.5,gamma=0)
plt.figure()
plt.imshow(blended)













