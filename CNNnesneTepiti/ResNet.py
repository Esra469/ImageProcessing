#ResNet siniflandiricisi ile  nesne tespiti
"""
piramit gösterimi ,kayan pencere ,ImageNet veri seti resnett50 kullanacagiz 
maksimum olmayan bastirma

Piramit Yöntemi:
görüntünün çok ölçekli temsilidir
bir nesnenin farkli ölçeklerindeki görüntüleri bulmamizi saglar
piramidin altinda, orijinal boyutunda(genislik,yükseklik açisindan) orijinal görüntü
var.ve sonraki her katmanda, goruntu yeniden boyutlandirilir.(alt örneklenir) ve istege bagli 
olarak düzletştirilir.(genellikle gauss bulaniklastirmasi kullanilir)
"""

import cv2
import matplotlib.pyplot as plt
def image_pyramid(image,scale=1.5,minSize=(224,224)):#burada minimum sizeyi 2 boyutlu aldik resmin yuksekllik ve genisligi icin
    yield image #yield ile resimler generate edilir.yield anahtar kelimesi sadece bir fonksiyon veya bir döngü içinde kullanılabilir,sadece bir deger dondurebilir

    while True:
        w=int(image.shape[1]/scale)
        image=cv2.resize(image,dsize=(w,w))
        #bu sekilde resimleri küculttuk belli bir degere kadar
        if image.shape[0]<minSize[1] or image.shape[1]<minSize:
            break
        yield image#iamge miz generator icerisinde donuyor
# bunu resmi görselleştirmek için kullanacagiz
# img=cv2.imread("husky.jpg (dosya ismi yani)")
# im=image_pyramid(img,1.5,(10,10))
# for i,image in enumerate(im):
#     print(i)
#     if i==10:#Burada pikselller azalıyor deger arttıkça burasi ne kadar küçük olursa o kadar net bir foto elde ederiz
#         plt.imshow(image)
