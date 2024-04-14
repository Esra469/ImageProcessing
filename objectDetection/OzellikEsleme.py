#özellik eşleme karmaşik bir sahnede bir resmi belirtmek için (belli bir resmi  tanı)(Brude-Force)
import cv2
import matplotlib.pyplot as plt
#2 tane resmimiz var bir tespit edilmek istenen resim bir de üzerinde tespit yapilan ana resim
chos=cv2.imread("çikolata",0)
plt.figure(),plt.imshow(chos,cmap="gray"),plt.axis("off")

#arranacak olana goruntu
cho=cv2.imread("gortuntu2",0)
plt.figure(),plt.imshow(cho,cmap="gray"),plt.axis("off")

#orb tanimlayici
#kose kenar gibi nesneye ait özellikler
orb=cv2.ORB_create()

#anahtar nokta tespiti
kp1, des1=orb.detectAndCompute(cho,None)#maskeleme yok
kp2, des2=orb.detectAndCompute(chos,None)

#eslme kısmı bf matcher

bf=cv2.BFMatcher(cv2.NORM_HAMMING2)

#noktalari eşleştir
matches=bf.match(des1,des2)

#mesafeye göre sirala
matches=sorted(matches,key=lambda x:x.distance)

#eslesen resimleri gorselleştir
plt.figure()
img_match=cv2.drawMatches(cho,kp1,chos,kp2,matches[:20],None,flags=2)
plt.imshow(img_match),plt.axis("off"),plt.title("orb")

#Buraya kadar tamam oldu ama yetersiz kaldi baska yontemler de deneyecegiz

#sift opencv ye dişardan eklenmiş bir küütphanane yoksa eklemen gerkiyr bu şekilde (pip install opencv-contrib-python --user)

sift=cv2.xfeatures2d.SIFT_create()

#bf
bf=cv2.BFMatcher()
#anahtar nokta tespiti sift ile
kp1,des1=sift.detectAndCompute(cho,None)
kp2,des2=sift.detectAndCompute(chos,None)

matches=bf.knnMatch(des1,des2,k=2)

guzel_eslesme=[]

for match1,match2 in matches:
    if match1.distance<0.75*match2.distance:
        guzel_eslesme.append([match1])

plt.figure()
sift_matches=cv2.drawMatchesKnn(cho,kp1,chos,kp2,guzel_eslesme,None,flags=2)

plt.imshow(sift_matches),plt.axis("off"),plt.title("sift")
