import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.layers import GlobalAveragePooling2D, Flatten
from keras.layers import Lambda
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Embedding, LSTM

# Define our model
def WalkModel():

    # Define model input
    main_input = Input(shape=(9, 4), name='main_input')

    x = Flatten()(main_input)
#    x1 = Input(shape=(9,), name='main_input')(main_input)
#    x2 = Input(shape=(9,), name='main_input')(main_input)
#    x3 = Input(shape=(9,), name='main_input')(main_input)
#    x4 = Input(shape=(9,), name='main_input')(main_input)
 #   x = keras.layers.concatenate([x1, x2, x3, x4], axis=-1)

    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(8, activation='linear', name='output')(x)

    # Create
    model = Model(inputs=main_input, outputs=[output])
    return model
    
