import cv2
import numpy as np #gürültü olmasi için
import matplotlib.pyplot as plt #resim oluşturmak için
import warnings #uyari oluştrmak için

warnings.filterwarnings("ignore")

#blurlama(bulaniklaştirma) detaylar azaltilir ,gürültü engeller
"""
ortalama bulaniklaştirma->çekirdek alan altindaki tüm pikselleri ortalamasini alir ve bu ortalama bulaniklaştirma yapar(kutu mantiği)
gauss bulaniklaştirma-> pozitfi ve tek olamasi gerekn çekirdeğin genişliğini ve yüksekliğini belirtir .sigmaX ve sigmaY ,x ve y yönlerindeki sapmayi belirmeliyiz
medyan bulaniklaştirma->çekirdek alani altindaki tüm piksellerin medyanini alir ve merrkez öğe bu medyan değerinş değiştirir

"""
img=cv2.imread("resim yolu")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(),plt.imshow(img),plt.axis("off"),plt.title("orijinal"),plt.show()

#ortalamam bulaniklaştirma yöntemi
dst2=cv2.blur(img,ksize=(3,3))
plt.figure(),plt.imshow(dst2),plt.axis("off"),plt.title("ortalama blur"),plt.show()

#gaussian blur
gb=cv2.GaussianBlur(img,ksize=(3,3),sigmax=7)
plt.figure(),plt.imshow(gb),plt.axis("off"),plt.title("gauss blur "),plt.show()

#medyan blur
mb=cv2.medianBlur(img,ksize=3)
plt.figure(),plt.imshow(mb),plt.axis("off"),plt.title("medyan blur"),plt.show()

#resim üstüne gürültü koyup engelle

def gaussianNoise(image):
    row, col, ch=image.shape #ch->resmin rgb mi bgr mi olduğunu bulacak shape bize resim büyüklüğünü gösteriri ve kaç kanal kullandigini belirtir
    mean=0
    var=0.05
    sigma=var**0.5
    
    gauss=np.random.normal(mean,sigma,(row,col,ch))
    gauss=gauss.reshape(row,col,ch)

    noisy=image+gauss
    return noisy

#içe aktar normalize et 
img=cv2.imread("resim yolu")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
plt.figure(),plt.imshow(img),plt.axis("off"),plt.title("orijinal"),plt.show()

gaussianNoiseImage=gaussianNoise(img)
plt.figure(),plt.imshow(gaussianNoiseImage),plt.axis("off"),plt.title("gauss noise"),plt.show()


#gauss blur
gb2=cv2.GaussianBlur(gaussianNoiseImage,ksize=(3,3),sigmaX=7)
plt.figure(),plt.imshow(gb2),plt.axis("off"),plt.title("with gausisan blur")




#tuz karabiber gürültüsünü oluşturarak medyan ile ortadan kaldiralim
def saltPapperNoise(image):
    row,col,ch=image.shape
    s_vs_p=0.5
    amount=0.004
    noisy=np.copy(image)
    #salt(beyaz noktacik)
    num_salt=np.ceil(amount*image.size*s_vs_p) #ceil ondalikli sayi yuvarliyor 1.1=1 1.9=2 olarak ayarlar
    coords=[np.random.randit(0,i-1,int(num_salt)) for i in image.shape()]
    noisy[coords]=1

    #papper siyah
    num_papper=np.ceil(amount*image.size* s_vs_p) #ceil ondalikli sayi yuvarliyor 1.1=1 1.9=2 olarak ayarlar
    coords=[np.random.randit(0,i-1,int(num_papper)) for i in image.shape()]
    noisy[coords]=0

    return noisy

spImage=saltPapperNoise(img)
plt.figure(),plt.imshow(spImage),plt.axis("0ff"),plt.title("sp image")

#şimdi de oluşturduğumuz gürültüden kurtuluyoruz

mb2=cv2.medianBlur(spImage.astype(np.float32),ksize=3)#opencv ondalik 64 olarak istemiyor 32 olarak ister
plt.figure(),plt.imshow(mb2),plt.axis("off"),plt.title("with medyan blur")
