import cv2
import os

files=os.listdir()#bu şekilde bütün resimlerimz files e gelecektir
img_path_list=[] #resmimleri depolamak için

for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)#eğer dosya sonu jpg ile bitiyorsa bunları img_path_list içerisine at
print(img_path_list)

#hog tanimlayicisi (bir tespit algoritmasi)
hog=cv2.HOGDescriptor()

#tanimlayiciya SVM ekle (siniflandirma için lazim)
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopeDetector())

for imagepath in img_path_list:
    print(imagepath)
    
    image=cv2.imread(imagepath)
    (rects,weights)=hog.detectMultiScale(image,padding=(8,8),scale=1.05)#padding resmin etrafında boşluklar oluşturuyor bu şekilde boyut kaybi da yasanmiyor ->Bu bize tuple içerisinde rectange ve genişlik döndürüyor
    
    for(x,y,w,h) in rects:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)#Burada rectangleleri aliyoruz (x,y) den başla yükseklik ve genişlik al diyoruz bunalri da kirmizi ve kalinliği 2 ayarlayacak şekilde yapiyoruz

    cv2.imshow("yaya:",image)
    if cv2.waitKey(0) & 0xFF==ord("q"):continue #Burada demek istenen eğer q ya basilirsa devam et yani diğer resme geçiş yap demek oluyor
