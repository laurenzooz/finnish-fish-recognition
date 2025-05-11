import cv2
import imghdr
import numpy as np  
import matplotlib.pyplot as plt
import os
import tensorflow
from tensorflow import keras
from keras import layers, models, optimizers, losses, metrics, datasets
'''
image_exts = ['jpeg','jpg', 'bmp', 'png']

data_dir = "data"


data = keras.utils.image_dataset_from_directory("data", image_size=(64, 64))


    
data = data.map(lambda x,y: (x/255, y))



#data.as_numpy_iterator().next()

train_size = int(len(data)*.9)
val_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(2, 2)) 
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train, epochs=10, validation_data=val)

model.save(os.path.join('models','imageclassifier.h5'))
'''
model = models.load_model("models/imageclassifier.h5")

img = cv2.imread("kissa.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()

resize  = tensorflow.image.resize(img, (64, 64))
plt.imshow(resize.numpy().astype(int))
plt.show()

pred = model.predict(np.array([resize]) / 255)
index = np.argmax(pred)

print(pred)
print(index)

if index == 1:
	print("Koira")
else:
	print("Kissa")



##koira palautti äskön 1  [0.31463856 0.6853615 ]
