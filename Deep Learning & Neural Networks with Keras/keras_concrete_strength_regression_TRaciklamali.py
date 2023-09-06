'''
Build a Regression Model in Keras

'''

# baseline modeli oluşturalım

# izleyeceğimiz adımlar:
# veri yükleme -> veri önişleme -> model oluşturma -> modeli derleme -> model eğitme -> modeli değerlendirme 
'''
Baseline modelde istenilen özellikler:

A. Build a baseline model
Use the Keras library to build a neural network with the following:
- One hidden layer of 10 nodes, and a ReLU activation function
- Use the adam optimizer and the mean squared error as the loss function.

1. Randomly split the data into a training and test sets by holding 30% of the data for testing. You can use the train
test_split  helper function from Scikit-learn.

2. Train the model on the training data using 50 epochs.

3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete
strength. You can use the mean_squared_error function from Scikit-learn.

4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

5. Report the mean and the standard deviation of the mean squared errors.

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# VERİ YÜKLEME
# Örnek olarak veri setini bir değişkene yüklüyoruz. Siz veriyi indirdikten sonra gerçek dosya yolu ile yüklemelisiniz.
data_url = "https://cocl.us/concrete_data"
data = pd.read_csv(data_url)

# Sütun adlarını yazdırma
print(data.columns)

# VERİ ÖNİŞLEME
# Hedef değişken ve özellikleri ayırma
X = data.drop('Strength', axis=1)  # Sütun adını 'Strength' olarak değiştirdik
y = data['Strength']
# Veriyi eğitim ve test kümelerine bölmek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MODEL OLUŞTURMA
model = Sequential()
# Sequential model, sıralı bir şekilde katmanları ekleyerek basit yapılarda sinir ağı modelleri oluşturmak için kullanılır.
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
# Bu katmanda, 10 adet nöron (node) bulunur ve ReLU (Rectified Linear Unit) aktivasyon fonksiyonu kullanılır.
model.add(Dense(1))
# Modelde yalnızca bir giriş katmanı ve bir çıkış katmanı bulunması, modeli basit bir şekilde ifade eder.
# Bu yüzden tek katmanlı sinir ağıdır diyebiliriz.

# MODELİ DERLEME
model.compile(optimizer='adam', loss='mean_squared_error')

# MODELİ EĞİTME
epochs = 50
model.fit(X_train, y_train, epochs=epochs, verbose=1)

# MODELİ DEĞERLENDİRME
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)




# veriyi normalize edelim.

'''
Veriyi normalize ederken istenilen özellikler:
B. Normalize the data

Repeat Part A but use a normalized version of the data. Recall that one way to normalize the data is by
subtracting the mean from the individual predictors and dividing by the standard deviation.

❓How does the mean of the mean squared errors compare to that from Step A?

Cevap: B kısmında veriyi normalize ettikten sonra elde ettiğimiz ortalama karesel hata (mean squared error)
A kısmındakinden farklı olacaktır. Normalizasyon, verileri benzer bir aralığa getirdiği için modelin daha
iyi performans göstermesini sağlayabilir. Bu nedenle, B kısmında elde edilen ortalama karesel hata muhtemelen
A kısmındakinden daha düşük olacaktır. Ancak, bunu kesin olarak belirlemek için 50 kez tekrarladığımız
ortalama karesel hataların ortalamasını ve standart sapmasını hesaplayarak B kısmının performansını net bir şekilde
değerlendirebiliriz.


Bilgilendirme:

Veriyi normalize etmek, verilerin aralığını ve ölçeğini değiştirerek, hepsini benzer bir aralığa getirme işlemidir.
Bu işlem, veri önişleme sürecinde verilerin model eğitimine daha uygun hale getirilmesini sağlar ve modelin performansını
artırabilir. Normalizasyon, özelliklerin büyüklüğündeki farklılıkları azaltarak, bazı özelliklerin diğerlerinden daha büyük
bir etkiye sahip olmasının önüne geçer.

Veriyi normalize etmek için genellikle iki yaygın yöntem kullanılır:
Min-Max Normalizasyon : normalized_value = (value - min_value) / (max_value - min_value)

Z-Score (Standard Score) Normalizasyon : normalized_value = (value - mean) / std

'''

# VERİYİ NORMALİZE ETME
X_mean = X_train.mean()
X_std = X_train.std()
X_train_normalized = (X_train - X_mean) / X_std
X_test_normalized = (X_test - X_mean) / X_std

# MODEL OLUŞTURMA
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train_normalized.shape[1],)))
# Bu sefer ise normalize ettiğimiz veriden bir sinir ağı oluşturuyoruz.
model.add(Dense(1))

# MODELİ DERLEME
model.compile(optimizer='adam', loss='mean_squared_error')

# MODELİ EĞİTME
epochs = 50
model.fit(X_train_normalized, y_train, epochs=epochs, verbose=1)

# MODELİ DEĞERLENDİRME
y_pred_normalized = model.predict(X_test_normalized)
mse_normalized = mean_squared_error(y_test, y_pred_normalized)
print("Mean Squared Error (Normalized):", mse_normalized)




# epoch sayısını arttıralım

'''
Epoch sayısını arttırırken istenilen özellikler:
C. Increase the number of epochs

Repeat Part B but this time use 100 epochs for training.

❓How does the mean of the mean squared errors compare to that from Step B?

Cevap:B kısmında elde edilen ortalama karesel hata değeri ile C kısmında elde edilen ortalama karesel hata değeri
arasındaki farkı incelediğimizde, C kısmındaki ortalama karesel hata değerinin B kısmındakinden düşük olması beklenir.
Çünkü C kısmında modeli daha fazla epoch ile eğitmiş olduk, bu da modelin verileri daha iyi öğrenmesini sağlamış olabilir
ve dolayısıyla daha iyi tahminler yapmasını bekleriz. C kısmındaki sonuçları değerlendirerek modelin epoch sayısının
optimum değerini belirleyebiliriz.


BİLGİLENDİRME:
C kısmında ise B kısmının aynısını yaptık (veriyi normalize edip modeli eğittik),
ancak bu kez modeli daha fazla epoch (yinelemeli eğitim adımı) ile eğittik. Epoch, modelin tüm veri kümesini kaç kez
gözden geçirdiğini ifade eder. Örneğin, 100 epoch ile eğitim yaparsak, model veri kümesini 100 kez tamamen geçer.

Şimdi C kısmında yaptığımız işlemle B kısmında yaptığımız işlem arasındaki farkı netleştirelim:
C kısmında modeli daha uzun süre (daha fazla epoch) eğittiğimiz için, modelin daha iyi öğrenme fırsatı oldu.
Dolayısıyla, modelin B kısmına göre daha iyi bir performans göstermesi beklenir. Yani, C kısmında elde ettiğimiz
ortalama karesel hata değeri, B kısmında elde ettiğimiz ortalama karesel hata değerinden daha düşük olmalıdır.

'''
# VERİYİ NORMALİZE ETME (Z-Score Normalizasyon)
X_mean = X_train.mean()
X_std = X_train.std()
X_train_normalized = (X_train - X_mean) / X_std
X_test_normalized = (X_test - X_mean) / X_std

# MODEL OLUŞTURMA
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train_normalized.shape[1],)))
model.add(Dense(1))

# MODELİ DERLEME
model.compile(optimizer='adam', loss='mean_squared_error')

# MODELİ EĞİTME (100 Epochs)
epochs = 100
model.fit(X_train_normalized, y_train, epochs=epochs, verbose=1)

# MODELİ DEĞERLENDİRME
y_pred_normalized = model.predict(X_test_normalized)
mse_normalized = mean_squared_error(y_test, y_pred_normalized)
print("Ortalama Karesel Hata (100 Epoch ile Normalize Edilmiş Veri):", mse_normalized)




# hidden layer sayısını arttıralım. Bu durum sinir ağımıza derinlik katacak.

'''
Hidden layer sayısını arttırırken istenilen özellikler:
D. Increase the number of hidden layers

Repeat Part C but use a neural network with 3 hidden layers, each of 10 nodes, and ReLU activation function.

❓How does the mean of the mean squared errors compare to that from Step C?

Cevap:Step C'de, normalize edilmiş veriyle 100 epoch ile eğitilen modelin ortalama karesel hata değerine bakmıştık.
Şimdi ise, D kısmında 3 gizli katmanlı sinir ağı kullanarak normalize edilmiş veriyle 100 epoch ile eğittiğimiz modelin
ortalama karesel hata değerini değerlendireceğiz.
D kısmında, daha derin bir sinir ağı yapısı kullandığımız için model daha fazla özellikleri ve karmaşıklıkları öğrenme
potansiyeline sahip olur. Bu nedenle, ortalama karesel hata değerinin D kısmında C kısmına göre düşük olması beklenir.

'''

# VERİYİ NORMALİZE ETME (Z-Score Normalizasyon)
X_mean = X_train.mean()
X_std = X_train.std()
X_train_normalized = (X_train - X_mean) / X_std
X_test_normalized = (X_test - X_mean) / X_std

# MODEL OLUŞTURMA (3 gizli katmanlı, her biri 10 düğüm)
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train_normalized.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# MODELİ DERLEME
model.compile(optimizer='adam', loss='mean_squared_error')

# MODELİ EĞİTME (100 Epochs)
epochs = 100
model.fit(X_train_normalized, y_train, epochs=epochs, verbose=1)

# MODELİ DEĞERLENDİRME
y_pred_normalized = model.predict(X_test_normalized)
mse_normalized = mean_squared_error(y_test, y_pred_normalized)
print("Ortalama Karesel Hata (3 Gizli Katman, 100 Epoch ile Normalize Edilmiş Veri):", mse_normalized)
