import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import random
from walk_model import WalkModel

# Start
print("Running on TensorFlow", tf.__version__, "Keras", keras.__version__)

ACTION_SPACE = 8
OBSERVATION_SPACE = 9

# Compile
model = WalkModel()
model.compile(loss={'output': 'mse'}, 
              optimizer='sgd',
              metrics=['accuracy'])
print(model.summary())

# Callbacks
checkpoint = ModelCheckpoint(filepath='./model_checkpoint.h5', verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer]

print("Training...")

X_train = []
y_train = []
for j in range(10000):
    
    leg = random.random()

    y = np.zeros([ACTION_SPACE])
    y[0] = leg
    y[1] = leg
    y[2] = leg
    y[3] = leg
    y[4] = 0.0
    y[5] = 0.0
    y[6] = 0.0
    y[7] = 0.0

    x = np.zeros([OBSERVATION_SPACE])
    x[0] = random.random()
    x[1] = random.random()
    x[2] = random.random()
    x[3] = random.random()
    x[4] = random.random()
    x[5] = random.random()
    x[6] = random.random()
    x[7] = random.random()
    x[8] = leg
    X_train.append(x.reshape((OBSERVATION_SPACE,)))
    y_train.append(y.reshape((ACTION_SPACE,)))
X_train = np.array(X_train)
y_train = np.array(y_train)
model.fit(X_train, y_train, batch_size=10, epochs=5)

print("Trained. Saving...")

# Save
model.save('walk_model.h5')

print("Saved.")
