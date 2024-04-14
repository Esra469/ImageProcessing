#hazır haar kullanmayacagiz biz haar ve veri seti oluşturup nesneleri tespit edecegiz
"""
1)veri seti oluştur
negatif pozitif resim(n,p)
cascade programi indir
cascade kullanarak tespit algoritmasi yaz

"""
import cv2
import os
#resim depo klasörü
path="images"

#resim boyutu
imgwidth=180
imgHeight=120

#video Capture
cap=cv2.VideoCapture(0)#default kamera
cap.set(3,640)#kamera genişliği
cap.set(4,480)#kamera yüksekliği
cap.set(10,180)#kamera parlakligi

#kalsör olusuturma (Burada klasörler olustur işe yararmazasa sonra yolunu bulup silebilirsin)
global countFolder
def saveDataFunc():
    global countFolder
    countFolder=0
    while os.path.exists(path+str(countFolder)):
        countFolder +=1
    os.makedirs(path+str(countFolder))
saveDataFunc()

count=0
conutSave=0

while True:
    success,img=cap.read()
    
    if success:
        img=cv2.resize(img,(imgwidth,imgHeight))

        if count%5==0:
            cv2.imwrite(path+str(countFolder)+"/I+str(countSave)"+".png",img)
            conutSave+=1
            print(conutSave)
        count+=1

        cv2.imshow("image",img)
    if cv2.waitKey(1)& 0xFF==ord("q"):break
cap.release()
cv2.destroyAllWindows()