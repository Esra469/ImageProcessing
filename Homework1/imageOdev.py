import cv2
import matplotlib.pyplot as plt


img=cv2.imread("C:/Users/ASUS/Pictures/kusResmi.webp",cv2.IMREAD_GRAYSCALE)


new_weight=480
new_height=280

resized_img=cv2.resize(img,(new_weight,new_height))

# cv2.imshow("yeniden boyutlandirilmis resim ",resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

text="Resmi yazi ekliyorum"
#metin için alan
position=(50,50)
font=cv2.FONT_HERSHEY_SIMPLEX #yazi tipi
font_scale=1
font_color=(255,255,255)
thickness=2 #metin kalinligi
#metin ekleme
cv2.putText(img,text,position,font,font_scale,font_color,thickness)

#metin eklenmiş görüntü göster
cv2.imshow("metin eklenmiş goruntu ",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, binary_threshold = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
cv2.imshow("ikili thresold goruntu ",binary_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.putText(img,"Esra",(375,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0))
cv2.imshow("Esra",img)





