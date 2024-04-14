#görüntü boyunca kayan sabir genişlik ve yükseklikte dikdörtgen bir bölgedir
#görüntünün belli bir kismini aldik burda
import cv2
import matplotlib.pyplot as plt
def sliding_windows(image,step,ws):#step dikdortgenin resim uzerinde kaç piksel dolasacagini gosterir,ws ise rectangele nin size sidir
     for y in range(0,image.shape[0]-ws[1],step):
          for x in range(0,image.shape[1]-ws[0],step):
               yield(x,y,image[y:y+ws[1],x:x+ws[0]])#image icerisinde y,x sekline x,y degil

# burada ayni görselleştirme yapiyoruz
# img=cv2.imread("husky.jpg")
# im=sliding_windows(img,5,(200,150))
 
# for i,image in enumerate(im):
#      print(i)
#      if i==14125:#yukarida olustudugumuza gore resim olusturuluyor biz de ona gore bir resme baktik
#           print(image[0],image[1])
#           plt.imshow(image[2])
