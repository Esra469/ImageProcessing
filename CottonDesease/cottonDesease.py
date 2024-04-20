#resnet 50 modeli genellikle kullanılır kullanılan bu resnet 50 de
# resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
#sonra false olması gpu kullanımını azaltır ve çıkarım yapmaya kolaylık sağlar fazla bellek kullanmayı da önler

#for layer in resnet.layers: layer.trainable = False kod bloğu, ResNet50 modelindeki tüm katmanların eğitilemez (trainable=False) olarak işaretlenmesini sağlar. 
#Bu, ağın önceden eğitilmiş ağırlıklarını korur ve modelin bu ağırlıkları güncellemesini engeller. Resnette bu önemlidir hazır aldığım bir veriyi değişirmemk için bu kod kullanıır

#dosya yollarını dönürmek için-> folders = glob('Datasets/train/*')

#faltten gerlen görselleri düzleştirilmesine olanak tanır x = Flatten()(resnet.output) bu kod da resnet 50 denn gelen çıktıları düzleştirmeey yarar 2 boyut ise tek boyuta döndürür

#activation='softmax' parametresi, genellikle çok sınıflı sınıflandırma problemleri için son katmanda kullanılır.
#Softmax aktivasyon fonksiyonu, modelin çıkışlarını olasılık dağılımına dönüştürerek, her sınıf için bir olasılık değeri sağlar. Bu şekilde, modelin verilen bir girişi hangi sınıfa ait olduğunu tahmin etmesi daha kolay olur

#Dense katmanı, tam bağlantı (fully connected) bir katmandır ve bir yapay sinir ağı modelindeki her bir nöronun önceki katmandaki tüm nöronlarla bağlantılı olduğu bir yapı oluşturur.
#Bu katman, giriş verisinin (özellik vektörünün) her bir özelliğini alır ve bunları ağırlık matrisi ile çarparak bir çıktı vektörü üretir. 
#Ardından, bir aktivasyon fonksiyonu genellikle bu çıktı vektörüne uygulanır. Bu katmanın çıkışı, genellikle bir sonraki katmana (örneğin, başka bir Dense katmanına veya çıkış katmanına) gider.

#daha sonra kayıp,optimizer ve metrics  model.compile ile ayarlanır

#Daha sonra train dataseti oluşturukur data generator ile
#image data generator ->veri arttırmak için kullaınılır

#daha sonra model fit_generatore girmelidir

#burada aldığın doğruluk değerinden yola çıkarak eğitilmiş verini değiştirebilirsin mesela inceptionV3 resnet152V2 gibi seçerken paramerdeler ve doğruluk değerleri dikkate alınmalıdır.

#rapids incele
