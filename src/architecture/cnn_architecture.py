from keras.models import Model
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, Input
from keras.optimizers import Adam

from utils.utils import determine_num_classes

def create_emotion_model(dataset_name, learning_rate=0.001):
    input_shape = (48, 48, 1)
    inputs = Input(shape=input_shape)
    
    num_classes = determine_num_classes(dataset_name)

    # First conv block
    conv1 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second conv block
    conv2 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Third conv block
    conv3 = Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Fourth conv block
    conv4 = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Dense layers
    flatten = Flatten()(pool4)
    dense = Dense(128, activation='relu')(flatten)
    drop = Dropout(0.2)(dense)
    output = Dense(num_classes, activation='softmax')(drop)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    return model
