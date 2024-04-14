"""
Multiple Object Tracking Benchmark ->Bu bir veri seti sağlar bize hazir veriseti paketi de sayilir
-veri setini incele
-veri setini indir(MOT17) burada indirme işlemini yaptik indirdikten sonra da işlem yapacakalrimizi işlem yaptiğimiz dizine almamiz gerekir
-resim2video
-eda->gt

"""
import cv2
import os
from os.path import isfile,join

import matplotlib.pyplot as plt

pathIn=r"img1"
pathOut="deneme.mp4"

files=[f for f in os.listdir(pathIn) if isfile(join(pathIn,f))] #bu şekilde dosyalari çekiyoruz iyi incele

# #burada resimleri aldik
# img=cv2.imread(pathIn+"\\"+files[44])#burada indirdigimiz dosyanin 44. resmini bize verir
# ig=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.imshow(img)

#şimdi videoya dönüştürelim(resimleri videoya dönüştürme)
#burada resimlerin hepsi çaliştigimiz dosyaya gelir ve çaliştirildiginda video şeklinde oynar
fps=25
size=(1920,1080)
out=cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(+"MP4V"),fps,size,True)

for i in files:
    print(i)

    filename=pathIn+"\\"+i
    img=cv2.imread(filename)

    out.write(img)
out.release()
