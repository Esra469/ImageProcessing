#resim işlmeleri
"""
#resimsel işlemler
import cv2
img=cv2.imread("resim dosyasinin adi",0)->içe aktarma 
cv2.imshow("ilk resim",img)->görselleştirme

k=cv2.waitKey(0) &0xFF ->bekletir

if k=27: ->ESC tuşunun pc dili
    cv2.destroyAllWindows()
elifk ==ord('s'):->buradaki tüşü sen belirliyorsun
    cv2.imwrite("kaydedilen resmin ismi",img)
    cv2.destroyAllWindows()

#videosal işlemler
-video okuma, video boyutu ayarlama, video gösterme
import cv2 
import time
video_name="video ismi girilir")->video ismi
cap=cv2.VideoCapture(video_name)->videoyu içe aktarma

print("genişlik",cap.get(3))
print("yükseklik",cap.get(4))
if cap.isOpened()==False:
    print("hata)
while(True):
    ret,frame=cap.read()
    if ret==True:
        time.sleep(0.01)#uyari:kullanmazsak çok hizli akar
        cv2.imshow("video",frame)
    else: break ->video yu okumadiği zman dişari atar 
    
    if cv2.waitKey(1) & 0xFF==ord("q"):
    break ->burada da biz istersek dişari atar
cap.release()->stop capture
cv2.destroyAllWindows()
"""
#Kamera açma ve video kaydetme
"""
import cv2
cap=cv2.VideoCapture(0)->Burada sifir değerinin olmasi bizim kameraye bağlanacağini gösterir. başka durumda başka kameraya bağlanir
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))->video genişliğini verir
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))->video yüksekliğini verir
print(width,height)

#video kaydet
fourcc->çerçeveleri sikiştirmak için kullanilan 4 karakterli codec kodu
writer=cv2.VideoWriter("video.kaydi.mp",cv2.VideoWriter_fourcc(*"DIVX"),20,(width,height))->frame for second (20) bu yüzden 20
while True:
    ret,frame=cap.read()
    cv2.imshow("video",frame)
    #save
    writer.write(frame)
    if cv2.waitKey(1) &0xFF==ord("q"):break
cap.release()
writer.release()
cv2.destroyAllWindows()

"""
#yeniden boyutlandırma
"""
bir resmin yeniden boyutlandilmasi ve kirpilmasi
import cv2
cap=cv2.VideoCaptur("video adi)

"""
