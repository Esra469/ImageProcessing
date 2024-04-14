#görüntü histogrami, dijital görüntüdeki ton dağiliminin grafiksel bir temsili olarak işlev gören bir histogram türüdür.
#her bir ton değeri için piksel sayisi içerir
import cv2
import matplotlib.pyplot as plt
import numpy as np

#içe aktarma
img=cv2.imread("dosya adi")
img_vis=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#burada dönüştürme işlemi oldu
plt.figure(),plt.imshow(img_vis)

print(img.shape)

img_hist=cv2.calHist([img],channels=[0],mask=None,histSize=[256],ranges=[0,256])
print(img_hist.shape)
plt.figure(),plt.plot(img_hist)

color=("b","g","r")
plt.figure()
for i,c in enumerate(color):
    hist=cv2.calHist([img],channels=[i],mask=None,histSize=[256],ranges=[0,256])#burada chnnels dediği renk kanalları kaç tane renk kullanilacaği aslinda
    plt.plot(hist,color=c)


golden_gate=cv2.imread("resim adi")
golden_gate_vis=cv2.cvtColor(golden_gate,cv2.COLOR_BGR2RGB)
plt.figure(),plt.imshow(golden_gate_vis)

print(golden_gate.shape)

mask=np.zero(golden_gate.shape[:2],np.uint8)
plt.figure(),plt.imshow(mask,cmap="gray")

masked_img_vis=cv2.bitwise_and(golden_gate,golden_gate_vis,mask=mask)
plt.figure(),plt.imshow(masked_img_vis,cmap="gray")

masked_img=cv2.bitwise_and(golden_gate_vis,golden_gate,mask=mask)
masked_img_hist=cv2.calcHist([golden_gate],channel=[0],mask=mask,histSize=[256],ranges=[0,256])
plt.figure(),plt.imshow(masked_img_hist)

#histogram eşitleme
#karşitlik artirma

img=cv2.imread("dosya adi",0)
plt.figure(),plt.imshow(img,cmap="gray")

img_hist=cv2.calcHist([img],channel=[0],mask=None,histSize=[256],ranges=[256])
plt.figure(),plt.imshow(img_hist)

eq_hist=cv2.equalizeHist(img)
plt.figure(),plt.imshow(eq_hist,cmap="gray")