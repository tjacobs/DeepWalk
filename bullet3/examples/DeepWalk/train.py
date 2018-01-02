import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import random, math, time
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
    
    phase = random.random()

    # Input
    x = np.zeros([OBSERVATION_SPACE, 4])
    x[0] = random.random()
    x[1] = random.random()
    x[2] = random.random()
    x[3] = random.random()
    x[4] = random.random()
    x[5] = random.random()
    x[6] = random.random()
    x[7] = random.random()
    x[8] = phase

    # Output
    y = np.zeros([ACTION_SPACE])
    y[0] = (math.sin(phase * 3.1415) + 1) / 2
    y[1] = (math.sin(phase * 3.1415 + 3.1415/2)  + 1) / 2
    y[2] = (math.sin(phase * 3.1415)  + 1) / 2
    y[3] = (math.sin(phase * 3.1415 + 3.1415/2)  + 1) / 2
    y[4] = (math.cos(phase * 3.1415) + 1 ) / 2
    y[5] = (math.cos(phase * 3.1415 + 3.1415/2) + 1) / 2
    y[6] = (math.cos(phase * 3.1415) + 1) / 2
    y[7] = (math.cos(phase * 3.1415 + 3.1415/2) + 1) / 2

    X_train.append(x.reshape((OBSERVATION_SPACE, 4)))
    y_train.append(y.reshape((ACTION_SPACE,)))
X_train = np.array(X_train)
y_train = np.array(y_train)
model.fit(X_train, y_train, batch_size=10, epochs=8)

print("Trained. Saving...")

time.sleep(0.1)

# Save
model.save('walk_model.h5')

time.sleep(0.1)

print("Saved.")
