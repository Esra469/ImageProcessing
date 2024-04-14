#secmeli aramada belli benzerliklerden yola cikarak yapariz renk,doku,boyut,sekil,benzerliklerin dogrusal kombinasyon gibi
#secmeli arama sinif etiketleri degil bolgeleri olusturur.Burada bir nesne olabilir diye olasilik veriyor
import cv2
import random

image=cv2.imread("pyramid.jpg")
image=cv2.resize(image,dsize=(600,600))
cv2.imshow("image",image)

#ilklendir ss selective serach algoirmasini ice aktariyoruz
ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()#bu bir segmantasyon algoritmasidir
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()

print("start")
rects=ss.process()

output=image.copy()

#dikdortgenlerden ilk 50 tanesini siniflandiriyoruz
for (x,y,w,h) in rects[:50]:
    color=[random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(output,(x,y),(x+w,y+h),color,2)
cv2.imshow("output",output)
#selectiveSearch secmeli arama yaptigimiz 2 yonteme gore daha etkili bir algoritmadir
