#kesişim/birleşim gibi oluyor(intersection over union),belirli bir eşik degerin altinda kalan alanalr eleniyor

import numpy as np
import cv2

def non_max_supression(boxes,probs=None,overlapThresh=0.3):#burada kessisimlere bir threshold uygulayacagız
    if len(boxes)==0:
        return[]#eger gelen kutularimiz bos olursa diye
    if boxes.dtype.kind=="i":#gelen kutularin type si int ise
        boxes=boxes.astype("float")#float yap
    x1=boxes[:,0]#bu seikilde aslinda kose degerlerimizi aldik
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]

    #alani bulalim 
    area=(x2-x1+1)*(y2-y1+1)
    idxs=y2

    #olasilik degerleri (probability)
    if probs is not None:
        idxs=probs#eger bos degilse olasiliklari idxs yap

    #indeksi sirala
    idxs=np.argsort(idxs)#parametre olarak aldigi idxs i sortlayacak indexleri verir
    pick=[]#secilen kutular

    while len(idxs)>0:

        last=len(idxs)-1
        i=idxs[last]
        pick.append(i)

        #en büyük ve en küçük x ve y
        xx1=np.maximum(x1[i],x1[idxs[:last]])
        yy1=np.maximum(y1[i],y1[idxs[:last]])
        xx2=np.minimum(x2[i],x2[idxs[:last]])
        yy2=np.minimum(y2[i],y2[idxs[:last]])

        #w,h bul
        w=np.maximum(0,xx2-xx1+1)
        h=np.maximum(0,yy2-yy1+1)

        #overlap(IoU olarak hesapladigimiz sey ) kesişim/birlesim dye
        overlap=(w*h)/area[idxs[:last]]
        #indexlerin bazılarini silmemiz lazim bu durumda thresholdlarin altinda olanlari siliyoruz
        idxs=np.delete(idxs,np.concatenate(([last],np.where(overlap>overlapThresh)[0])))#np.where istedigimiz degerin hangi indexte oldugunu bulur
    
    return boxes[pick].astype("int")#boxes içindeki int degerler return ediliyor 

#bu işlemler nesne tespitinden sonra genelde yapiliyor