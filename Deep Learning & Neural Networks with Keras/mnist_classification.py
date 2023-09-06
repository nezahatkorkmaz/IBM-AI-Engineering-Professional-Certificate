# Gerekli kütüphaneleri ve veri kümesini yükleyelim
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist

# MNIST veri kümesini yükleyelim
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Eğitim verilerini inceleyelim
print(X_train.shape)

# Resimleri tek boyutlu vektörlere dönüştürelim
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Giriş verilerini 0-1 aralığına normalize edelim
X_train = X_train / 255
X_test = X_test / 255

# Sınıf etiketlerini kategorik hale getirelim
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Sınıflandırma modelini oluşturalım
def classification_model():
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Modeli oluşturalım
model = classification_model()

# Modeli eğitelim
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# Modelin performansını değerlendirelim
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))        

# Modeli kaydedelim ve yükleyelim
model.save('classification_model.h5')
from keras.models import load_model
pretrained_model = load_model('classification_model.h5')

