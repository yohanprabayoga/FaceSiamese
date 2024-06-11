import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def create_siamese_network(input_shape):
    input = Input(input_shape)
    
    x = Conv2D(64, (10,10), activation='relu')(input)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (7,7), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4,4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (4,4), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid')(x)
    
    return Model(input, x)

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = K.sum(K.square(featA - featB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def build_model(input_shape):
    imgA = Input(shape=input_shape)
    imgB = Input(shape=input_shape)
    
    base_network = create_siamese_network(input_shape)
    
    featA = base_network(imgA)
    featB = base_network(imgB)
    
    distance = Lambda(euclidean_distance)([featA, featB])
    
    model = Model(inputs=[imgA, imgB], outputs=distance)
    
    return model
