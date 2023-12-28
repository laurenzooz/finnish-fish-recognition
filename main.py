import cv2
import imghdr
import numpy as np  
import matplotlib.pyplot as plt
import os
import tensorflow
from tensorflow import keras
from keras import layers, models, optimizers, losses, metrics, datasets


data = keras.utils.image_dataset_from_directory("data", image_size=(128, 128))


    
data = data.map(lambda x,y: (x/255, y))

#data.as_numpy_iterator().next()

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2)) 
model.add(layers.Conv2D(16, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train, epochs=20, validation_data=val)

model.save("test.model")

# model = models.load_model("test.model")

img = cv2.imread("hauki.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()

resize  = tensorflow.image.resize(img, (128, 128))
plt.imshow(resize.numpy().astype(int))
plt.show()

pred = model.predict(np.expand_dims(resize/255, 0))

print(pred)

