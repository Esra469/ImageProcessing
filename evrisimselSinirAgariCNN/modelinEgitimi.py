import glob
import os #bunlar sayesinde resimlere ve klasörlere erişim saglanir
import numpy as np
from keras.models import Sequential #keras kullanarak egitim verisini egitmeyi görecegiz
from keras.layers import Dense ,Dropout,Flatten,Conv2D,MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

import warnings 
warnings.filterwarnings("ignore")#uyarilari göstrmek için bir kütüphane bu şekilde uyarilari kapatiyoruz

#bu yazdigimiz kod sayesnde bütün rsim dosyalarini okuyacak
imgs=glob.glob("./img_nihai/*.png")

width=125
height=50

x=[]
y=[]
for img in imgs:
    filename=os.path.basename(img)
    label=filename.split("_")[0]#bunun anlami dosya isminin ilk adini split olarak ayir ve al
    im=np.array(Image.open(img).convert("L").resize(width,height))
    im=im/255
    x.append(im)
    y.append(label)

x=np.array(x)
x=x.reshape(x.shape[0],width,height,1)#buradaki 1 degeri channel degeri olarak geciyor

#sns.countplot(y)#bu şekilde y görsellestirilip grafige erişilebilir

def onehot_label(value):
    label_encoder=LabelEncoder()
    integer_encoded=label_encoder.fit_transform(value)#hocani yaptiğinda values yaziyordu ama sende onun için hata verdir
    onehot_encoder=OneHotEncoder(sparse=False)
    integer_encoded=integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoder=onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoder
y=onehot_label(y)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=2)#x ler resimler y ler ise bizim etiketlerimizdi

#cnn model
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(width,height,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu",))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(3,activation="softmax"))

# if os.path.exists("./trex_weight.h5"):
#     model.load_weights("trex.weight.h5")
#     print("weight yuklendi") yükleme yapmak için yapilir

#burada geriye dogru turev alma islemi uyguluyoruz
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])#en son nihai olarak hatalarimizi gormemizi saglayan fonksiyon
model.fit(train_x,train_y,epochs=35,batch_size=64)#epochs=ne kadar egitim gerecklesecegini belirtiyor batch_size=kac grup seklinde bir iterasyona girecegini belirtir

score_train=model.evaluate(train_x,train_y)
print("egitim dogrulugu: %",score_train[1]*100)
score_test=model.evaluate(test_x,test_y)
print("test dogrulugu: %",score_test[1]*100)

open("model.json","w").write(model.to_json())
model.save_weights("trex_weight_new.h5")

#bu şekilde modeli egittik