#nesne tespiti siniflandirma ile alakasi yoktur tespite o nesne var mi yok mu bakilir

"""
Kenar algilama(edge detection)
önce resmi siyah beyaz yap sonra kenar algila

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Burada içe aktarilmasi gerekiyor ama aktarilmiyor. neden diye sor
img=cv2.imread("C:/Users/ASUS/Pictures/kusResmi.webp")
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# plt.figure()
# plt.imshow(img_rgb)
# plt.axis("off")

edge=cv2.Canny(image=img,threshold1=0,threshold2=255)
plt.figure(),plt.imshow(edge,cmap="gray"),plt.axis("off")

#saçma bir şekilde kenarlari bize vermemesi için thresold değerini veya başka değişimler yapabilirz
#medyan değiştirme gibi

#buradaki amaç yaniş kenarlari almamak bunun için önce medyan uyguladik threshlerini ayarladik

med_val=np.median(img)
print(med_val)

low=int(max(0,(1-0.33)*med_val))
hight=int(min(255,(1+0.33)*med_val))
print(low)
print(hight)

#şimdi de tüm resme blur uygulayacagiz

blured_img=cv2.blur(img,ksize=(3,3))
plt.figure(),plt.imshow(blured_img,cmap="gray"),plt.axis("off")


med_val=np.median(blured_img)
print(med_val)
#resmi blurladigimizda resim bulaniklasiyor ama amacimiz kenar tespiti oldugu için çok sorun olmuyor



#Köse algilama
#formülde mantik söyle işliyor belirlediğimiz x y koordinatindaki bölge u kadar kayinca bir yoğunluk farki oluyorsa demek ki bir köseden geçmişiz diyoruz

img=cv2.imread("C:/Users/ASUS/Pictures/sudoku.png",0)
img=np.float32(img)
print(img.shape)
plt.figure(),plt.imshow(img,cmap="gray"),plt.axis("off")

#harris corner detection
dst=cv2.cornerHarris(img,blockSize=2,ksize=3,k=0.04) #blockSize->komsuluk boyutudur ne kadar komsusuna bakacagimizi belirliyor ksize=kutucuk boyut k=harrisdekş filli parametrelerden biri
plt.figure(),plt.imshow(dst,cmap="gray"),plt.axis("off")

#resmi genişlettik
dst=cv2.dilate(dst,None)
img[dst>0.2*dst.max()]=1
plt.figure(),plt.imshow(dst,cmap="gray"),plt.axis("off")

#kenar algilama2 (shi tomasi detection) bunlari uygulamak için kessinlikle önceden siyah beyaz yapilmali
img=cv2.imread("C:/Users/ASUS/Pictures/sudoku.png",0)
img=np.float32(img)
corners=cv2.goodFeaturesToTrack(img,100,0.01,10) #buradaki 100 kaç tane köse tespit lazim ona bakiyor 10 buradaki minimun distance(mesafe)
corners=np.int64(corners)

for i in corners:
    x,y=i.ravel()
    cv2.circle(img,(x,y),3,(125,125,125),cv2.FILLED)
plt.imshow(img)
plt.axis("off")
