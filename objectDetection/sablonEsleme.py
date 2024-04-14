#kaydirarak, sablonu bir seferde bir piksel hareket ettirmeyi kastediyoruz
#template matching
#ana resimden parça kirparak biz de sablonoluşturabilirz

import cv2

import matplotlib.pyplot as plt

#template matching(şablon eşleme)
img=cv2.imread("anaResim",0)
print(img.shape)
template=cv2.imread("resim sablonnu",0)
print(template.shape)
h,w=template.shape

methods=['cv2_TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR','cv2.TM_CCORR_NORMED','cv2._TM_SQDIFF','cv2.TM_SQDIFF_NORMED']

for meth in methods:
    method=eval(meth)#even burada yukarida olan fonksiyonlari alir eski haline dönüştürür  
    res=cv2.matchTemplate(img,template,method)
    print(res.shape)
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF_NORMED,cv2.TM_SQDIFF]:
        top_left=min_loc
    else:
        top_left=max_loc
    bottom_right=(top_left[0]+w,top_left[1]+h)
    cv2.rectangle(img,top_left,bottom_right,255,2)

    plt.figure(),plt.subplot(121),plt.imshow(res,cmap="gray")
    plt.title("eşleşen sonuç"),plt.axis("off")
    plt.figure(),plt.subplot(121),plt.imshow(res,cmap="gray")
    plt.title("tespit edilen sonuç"),plt.axis("off")
    plt.suptitle(meth)

    plt.show()
