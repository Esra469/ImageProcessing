import keyboard #bu kütüphan esayesinde klavyeden veri alabiliyoruz
import uuid #bu kütüphane sayesinde ekrandan görüntü alabiliryoruz
import time
from PIL import Image
from mss import mss
"""
normalde dosya yolu var ama 
buradaki dosya dan belli veriler aldik bu verileri paintte kestik düzelttik
"""
mon={"top":300,"left":770,"width":250,"height":100}
sct=mss()#mss kütüphanesi aldigimiz ilgili alani alip frame haline dönüştüren kütüphanedir

i=0
def record_screen(record_id,key):
    global i
    i+=1
    print("{}:{}".format(key,i))#buradaki key değeri hangi anahtara bastiğinizi i ise kac defa bastigimizi gösterir
    img=sct.grab(mon)
    im=Image.frombytes("RGB",img.size,img.rgb)
    im.save("./img/{}_{}_{}.png".format(key,record_id,i))#burada aslinda im diye bir klasöre resimler kayitediliyor bu yüzden im dye bir klasor oluşturmamizlazım
is_exit=False

def exit():
    global is_exit
    is_exit=True

keyboard.add_hotkey("esc",exit)

record_id=uuid.uuid4()

while True:
    if is_exit:break
    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id,"up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id,"down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"):
            record_screen(record_id,"right")
            time.sleep(0.1)
    except RuntimeError:continue
#Bu kisimda verii almayi ögrendik diger derste ise egitim ve test veriri oluşturmayi ogrenecegiz