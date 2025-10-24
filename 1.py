from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000,28*28)).astype('float32')/255
x_test = x_test.reshape((10000,28*28)).astype('float32')/255

print(y_train[0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[0])

model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)), #hidden layer
    layers.Dense(10, activation='softmax')#output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
# history.history['loss','val_loss','accuracy','val_accuracy']

plt.plot(history.history['loss'], label = "train_loss")
plt.plot(history.history['val_loss'], label = "val_loss")
'''plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_loss'], label='val_loss')'''
plt.legend()
plt.show()

