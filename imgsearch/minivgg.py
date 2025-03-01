from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
#from keras import Backend as K 
from keras import backend as K

class MiniVGG:
    @staticmethod
    #depth AKA channels
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        #channels first
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        #CONVOLUTION 1
        model.add(Conv2D(32, (3,3), padding="same",
            input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.25))

        # 2 CONVOLUTION
        #(conv then relu) * 2 then pooling
        model.add(Conv2D(64, (3,3), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3,3), padding="same",
            input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0,25))

        # 3 CONVOLUTION
        #(conv then relu) * 2 then pooling
        model.add(Conv2D(128, (3,3), padding="same", 
            input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3,3), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        #fully connected and RElu
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        #softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model






