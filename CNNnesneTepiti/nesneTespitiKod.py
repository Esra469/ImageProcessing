from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2


from sliding_windows import sliding_windows
from image_pyramid  import image_pyramid
from non_max_supression import non_max_supression

#ilklendirme parametreleri
WIDTH=600
HEIGHT=600
PYR_SCALE=1.5
WIN_STEP=16
ROI_SIZE=(200,150)
INPUT_SIZE=(224,224)#resnet in input size si bu 224 olmak zorunda

print("Resnet yükleniyor")
model=ResNet50(weights="imagenet",include_top=True)#imagenet resnete ait bir veri setidir

orig=cv2.imread("husky.jpg")
orig=cv2.resize(orig,dsize=(WIDTH,HEIGHT))#resmi yeniden boyutlandirdik 600,600 olacak sekilde
cv2.imshow("husky",orig)
(H,W)=orig.shape[:2]#shape nin ilk ikisini alıyor çunku bu bana 600,600 veriyor

#image pyramid
pyramid=image_pyramid(orig,PYR_SCALE,ROI_SIZE)#resmi generator a donusturmemiz lazim resim olarak donmuyor
#aksi durumda resimlerin boyutu atrigi icin hadizada dolma olur.Bu degerler donereken bize roi sonucunu verecek siniflandirma sonucunda
#
rois=[]#olusan roi ler
locs=[]

for image in pyramid:
    scale=W/float(image.shape[1])

    for(x,y,roiOrig) in sliding_windows(image,WIN_STEP,ROI_SIZE):
        
        x=int(x*scale)
        y=int(y*scale)
        w=int(ROI_SIZE[0]*scale)#0.index 200
        h=int(ROI_SIZE[1]*scale)#burdan da 1. index deki 150 alnmis olur
#simdi oo verileri alip siniflandirma yapmak gerekiyor yani preprocess islemini
        roi=cv2.resize(roiOrig,INPUT_SIZE)
        roi=img_to_array(roi)
        roi=preprocess_input(roi)#siniflandirip hazir preprocess uyguladik

        rois.append(roi)
        locs.append((x,y,x+w,y+h))#burada tuple kullaniyoruz 1 tane sey ekleyebilecegimizden dolayi ,2 tane parantez kullandik

rois=np.array(rois,dtype="float32")#roi leri arraye donusturduk float bir tipte

print("siniflandirma islemi")
preds=model.predict(rois)

preds=imagenet_utils.decode_predictions(preds,top=1)
labels={}#label diye bir dictionary olusturduk
min_conf=0.9 #min threshold degeri

for (i,p) in enumerate(preds):#(i,p) olusturduk bir tane tuple dondurecegz
    (_, label, prob)=p[0]
    if prob>=min_conf:
        box=locs[i]

        L=labels.get(label,[])
        L.append((box,prob))#append bir tane parametre dondurebildigi icin bu sekilde kullaniyoruz
        labels[label]=L
        #yukarıda label{} den sonra yaptigimiz islemler verilerin %90 ustu olanlarinin alinmasi

for label in labels.keys():
    clone=orig.copy()
    #kutucuk cizdirme
    for(box,prob) in labels[label]:
        (startX,startY,endX,endY)=box
        cv2.rectangle(clone,(startX,startY),(endX,endY),(0,255,0),2)

    cv2.imshow("ilk",clone)

    #kopyasini olusturduk dah sonra bunu supression yaptimak icin
    clone=orig.copy()

    #non-maxima
    boxes=np.array([p[0] for p in label[label]])
    proba=np.array([p[1] for p in labels[label]])

    boxes=non_max_supression(boxes,proba)#diger olustudugumuz non_max_supressiondan aldik

    for (startX,startY,endX,endY) in boxes:
        cv2.rectangle(clone,(startX,startY),(endX,endY),(0,255,0),2)
        y=startY-10 if startY-10>10 else startY+10
        cv2.putText(clone,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)

    cv2.imshow("Maxima",clone)

    if cv2.waitKey(1) & 0xFF==ord("q"):break