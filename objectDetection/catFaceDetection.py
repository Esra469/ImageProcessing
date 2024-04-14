import cv2
import os
#Burada yeni yollar kullanarak bir kedi yüzü tanıma prjesi yaptik 
files=os.listdir()
print(files)
img_path_list=[]

for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)
print(img_path_list)

for j in files:
    print(j)
    image=cv2.imread(j)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    detector=cv2.CascadeClassifier("indirdigimiz.xml")
    #bu kodda düzenleme yaparak kedileri daha hassas görebilir yani parametreler ile oyna
    rects=detector.detectMultiScale(gray,scaleFactor=1.045,minNeighbors=2)

    for (i,(x,y,w,h)) in enumerate(rects):
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(image,"kedi {}".format(i+1),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
    cv2.imshow(j,image)
    if cv2.waitKey(0)& 0xFF ==ord("q"):continue
    