import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from walk_model import WalkModel

# Start
print("Running on TensorFlow", tf.__version__, "Keras", keras.__version__)

# Set configuration
train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 10
batch_size = 30
OBSERVATION_SPACE = 9

# Compile
model = WalkModel()
model.compile(loss={'output': 'categorical_crossentropy'}, 
              optimizer='sgd',
              metrics=['accuracy'])
print(model.summary())

# Callbacks
checkpoint = ModelCheckpoint(filepath='./model_checkpoint.h5', verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer]

#def zero_critic(epochs=100):
#    for i in range(epochs):
#        for j in range(OBSERVATION_SPACE):
#            X_train = []
#            y_train = []
#            
#            y = np.empty([1])
#            y[0]=0.0
#            x = to_onehot(OBSERVATION_SPACE,j)
#            X_train.append(x.reshape((OBSERVATION_SPACE,)))
#            y_train.append(y.reshape((1,)))
#            X_train = np.array(X_train)
#            y_train = np.array(y_train)
#            critic_model.fit(X_train, y_train, batch_size=1, nb_epoch=1, verbose=0)

#print("Zeroing out critic network...")
#sys.stdout.flush()
#zero_critic()
#print("Done!")
#plot_value(STATEGRID)

# Train
#model.fit_generator(
#    batch_generator(train_generator),
#    steps_per_epoch = nb_train_samples // batch_size,
#    validation_data = batch_generator(validation_generator),
#    validation_steps = nb_validation_samples // batch_size,
#    epochs=epochs,
#    callbacks=callbacks)

print("Saving...")

# Save
model.save('walk_model.h5')

print("Saved.")
