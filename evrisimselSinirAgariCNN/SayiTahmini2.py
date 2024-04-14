#sayi okuma sonuç değerlendirme
import cv2
import pickle #pickle diye kaydetmiştik
import numpy as np #bunu grafik içn kullaniyoruz

def preProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#burada bgr den gray a dönüştürüyoruz
    img=cv2.equalizeHist(img)#daha sonra equalization fonksiyonu ile 0-255 arasinda dağılım uyguluyorz
    img=img/255.0#sonra da 255 e bölerek normlaize etme işlemi uyguluoyroz

    return img#video ile aldiğin görüntüleri preprocess sokmadan noural networda çaliştiramazsin
cap=cv2.VideoCapture(0)#video kameramiza bağlandik
cap.set(3,480)#set metodu ile genişlik ve yukseklik ayarliyoruz
cap.set(4,480)

pickle_in=open("model_trained_v4.p","rb")#eğitmiş oldugumuz modeli içeri aliyoruz
model=pickle.load(pickle_in)

while True:
    success,frame=cap.read()
    img=np.asarray(frame)#burada frame mizi bir array e dönüştürüyoruz
    img=cv2.resize(img,(32,32))#yeniden noural network yaparken shape yi 32,32 ayarladigimiz için bunu da düzenlememiz lazim
    img=preProcess(img)#sonra img preprcess e sokmak lazim hazir fonksiyon oldugu için fonsiyondan aldim

    img=img.reshape(1,32,32,1)#bir görüntü + boyut+channel

    #predict
    classIndex=int(model.predict_classes(img))

    predictions=model.predict(img)
    probVal=np.amax(predictions)
    print(classIndex,probVal)#mesela 5 olma ihtmalinin yüzdesini yazdiracak vs
      #burada gösrelleştirdik
    if probVal>0.7:
        cv2.putText(frame,str(classIndex)+"  "+str(probVal),(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),1)

    cv2.imshow("rakam siniflandirma",frame)
    
    if cv2.waitKey(1) & 0xFF==ord("q"):break