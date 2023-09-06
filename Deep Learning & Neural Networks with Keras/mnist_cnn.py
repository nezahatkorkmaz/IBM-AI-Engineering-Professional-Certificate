import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D  # Convolutional layers için
from keras.layers.convolutional import MaxPooling2D  # Pooling layers için
from keras.layers import Flatten  # Fully connected layers için

# Verileri yükle
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Veri şeklini düzenle: [örnek sayısı][pikseller][genişlik][yükseklik]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Verileri normalize et: 0-255 aralığını 0-1 aralığına dönüştür
X_train = X_train / 255
X_test = X_test / 255

# Etiketleri kategorik hale getir (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]  # Sınıf sayısı

def convolutional_model():
    # Modeli oluştur
    model = Sequential()

    # Convolutional katman ekle: 16 filtre, her biri 5x5 boyutunda
    # 'relu' aktivasyon fonksiyonu kullan
    # Giriş şekli: (28, 28, 1)
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))

    # Pooling katmanı ekle: 2x2 boyutunda
    # 2x2 boyutunda hareket eden bir filtre kullanarak maksimum değeri al
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Veriyi düzleştir: Convolutional katmanlardan gelen 3D veriyi 1D hale getir
    model.add(Flatten())

    # Tam bağlantılı (fully connected) katman ekle: 100 nöron, 'relu' aktivasyon fonksiyonu kullan
    model.add(Dense(100, activation='relu'))

    # Çıkış katmanını ekle: Sınıf sayısı kadar nöron, 'softmax' aktivasyon fonksiyonu kullan
    model.add(Dense(num_classes, activation='softmax'))

    # Modeli derle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Modeli oluştur
model = convolutional_model()

# Modeli eğit
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Modeli değerlendir
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))
