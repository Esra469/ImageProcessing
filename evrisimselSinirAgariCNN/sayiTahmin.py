import numpy as np
import cv2
import os#bu kütüphane ile verimizi içe aktaracağiz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns#görselleştirmek için seaborn kullanilir
import matplotlib.pyplot as plt#göreselleştirmek için matplotlib de kullanilir
from keras.models import Sequential#bizim tabanimiz bu tabanin içine layerlerimizi ekliyoruz
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator#farkli resimler generate ediyoruz
import pickle#modeli yüklemek ve kaydetmke için kullanilir

#myData denen veri klasörünü projede bulmamiz lazim
path="myData"#buradaki resmleri aliyoruz burada 1 den 9 a kadar sayilarin farklı yazılış şekli var

myList=os.listdir(path)#path içinde bulunan bütün verileri myList içerisine at
noOfClasses=len(myList)#myList uzanligini belirler

print("Label(sinif) sayisi:",noOfClasses)

images=[]
ClassNo=[]

for i in range(noOfClasses):
    myImageList=os.listdir(path+"\\"+str(i))#Burada klasörde bulunan dosyalari ayristiriyoruz myData\2 gibi dosya ismi olacak
    for j in myList:
        img=cv2.imread(path+"\\"+str(i)+"\\"+j)#tüm path birlerştirlir 2 nin içindeki hernegi bir 2 gibi
        img=cv2.resize(img,(32,32))#neural network girdisi 32,32 oldugu için bunu verdik
        images.append(img)#resimlerimizi image içerisine atiyoruz
        ClassNo.append(i)
print(len(images))#resim sayisini verir bize
print(len(ClassNo))

images=np.array(images)#burada frame leri arraylere dönüştürdük
ClassNo=np.array(ClassNo)

print(images.shape)#burada boyutlara bakiyoruz
print(ClassNo.shape)

#veriyi ayirma veriyi xtrain ve x_validation olarak eğitecegiz en son da test verisi ile dogruluguna bakacagiz
x_train,x_test,y_train,y_test=train_test_split(images,ClassNo,test_size=0.5,random_state=42)
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=0.2,random_state=42)
#bu şekilde egitim,test ve dogrulama kullanilir 
print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

#vis ->burasi görselleştirme kismi
# fig,axes=plt.subplots(3,1,figsize=(7,7))
# fig.subplots_adjust(hspace=0.5)#3 tane satir arasinda boşluk birakma
# sns.countplot(y_train,ax=axes[0])
# axes[0].set_title("y_train")
 #Bu adimlarad aslinda test validation ve dogruluk için ayrilmiş olan verisetini görselleştirip bölüyoruz
# sns.countplot(y_test,ax=axes[1])
# axes[1].set_title("y_test")
# sns.countplot(y_validation,ax=axes[2])
# axes[2].set_title("y_validation")

#preprocess
def proProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)#histogrami 0,255 araliginda kullanabilmek için
    img=img/255

    return img
# idx=50
# img=proProcess(x_train[idx])#burada x_trein in 50.indeksinen bakiliyor
# img=cv2.resize(img,(300,300))
# cv2.imshow("proprocess",img)

#bu preprocess işlemini tüm veriye uygulamak için kullanilir
x_train=np.array(list(map(proProcess,x_train)))#map şöyle işler 1.parametresi fonksiyondur 2.parametresi bir değerdir işleyiş 1.fonksiyonu 2. parametre üzerinde uygula demek oluyor
x_test=np.array(list(map(proProcess,x_test)))
x_validation=np.array(list(map(proProcess,x_validation)))

#bunlarin hepsi verimizi egitime hazir hale getirmek için yapiliyor
x_train=x_train.reshape(-1,32,32,1)#buradaki -1 x_train neyse ona göre boyut ayarla demek anlamina geliyor
print(x_train.shape)
x_test=x_test.reshape(-1,32,32,1)
x_validation=x_validation.reshape(-0,32,32,1)

#Data generate
data_Gen=ImageDataGenerator(width_shift_range=0.1,#0.1 oraninda kayma oluyor
                            height_shift_range=0.1,#0.1 oraninda yükseklikte kayiyor
                            zoom_range=0.1,
                            rotation_range=10)
data_Gen.fit(x_train)#x_trein kullanarak yeni resimler üretiyoruz

y_train=to_categorical(y_train,noOfClasses)#caategorikal onehotencoder ile ayni islevi goruyor
y_test=to_categorical(y_test,noOfClasses)#keras kütüphanesi için bunu yapmak gerekiyor sne yine de bir arastir
y_validation=to_categorical(y_validation,noOfClasses)

model=Sequential()#sequential bir temel oluşturuyoruz daha sonra ona eklemeler yapiyoruz
model.add(Conv2D(input_shape=(32,32,1),filter=8,kernel_size=(5,5),activation="relu",padding="same"))#burayi araştir parametreler tam olarak ne işe yarar diye
model.add(MaxPooling2D(pool_size=(2,2)))#piksel ekleme işlemi

model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))#yeni veri ekledik ezberlemeyi engellemk için dropuot ekliyoruz
model.add(Flatten())#düzlestirme işlemi yapilir
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=noOfClasses,activation="softmax"))#softmax neden kullanmak zorundasin bak adam öyle dedi
model.compile(loss="categorical_crossentropy",optimizer=("Adam"),metrics=["accuracy"])

batch_size=250

#history diye bir parametre ekliyoruz bu sayede modeli görselleştirebilrcegiz
hist=model.fit_generator(data_Gen.flow(x_train,y_train,batch_size=batch_size),
                         validation_data=(x_validation,y_validation),
                         epochs=15,steps_per_epoch=x_train.shape[0]//batch_size,shuffle=1)#// buşekilde kalansiz bölümüne bakilir

pickle_out=open("model_trained_new.p","wb")#burada modeli depolayacagiz
pickle.dump(model,pickle_out)#modeli sonradan tekrardan çagirmamiz gerektigi için bu yolu oluşturmak gerekiyor
pickle_out.close()


#  degerlendirme

hist.history.keys()

plt.figure()
plt.plot(hist.history["loss"],label="eğitim loss")
plt.plot(hist.history["val_accuracy"],label="val_accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"],label="Eğitim accuracy")
plt.plot(hist.history["val_accuracy"],label="val_accuracy")
plt.legend()
plt.show()

score=model.evaluate(x_test,y_test,verbose=1)
print("Test loss:",score[0])
print("Test accuracy:",score[1])

y_pred=model.predict(x_validation)
y_pred_class=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_validation,axis=1)

cm=confusion_matrix(y_true,y_pred_class)

f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm,annot="True",linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("cm")
plt.show()