import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from keras import layers


def feature_extractor():
    inputs = layers.Input(shape=(105,105,3))

    c1 = layers.Conv2D(64, (10,10), activation='relu')(inputs)
    m1 = layers.MaxPooling2D(64, (2,2), padding='same')(c1)
    
    c2 = layers.Conv2D(128, (7,7), activation='relu')(m1)
    m2 = layers.MaxPooling2D(64, (2,2), padding='same')(c2)

    c3 = layers.Conv2D(128, (4,4), activation='relu')(m2)
    m3 = layers.MaxPooling2D(64, (2,2), padding='same')(c3)

    c4 = layers.Conv2D(256, (4,4), activation='relu')(m3)
    f1 = layers.Flatten()(c4)

    d1 = layers.Dense(4096, activation='sigmoid')(f1)

    return keras.Model(inputs=inputs, outputs=d1)



class Dist(layers.Layer):
    def __init__(self, **kwargs):
        super(Dist, self).__init__(**kwargs)

    def call(self, anchor_embedding, reference_embedding):
        return tf.math.abs(anchor_embedding - reference_embedding)



def build_model():

    image_embedding = feature_extractor()
    dist_layer = Dist()

    anchor_image_inputs = layers.Input(shape=(105,105,3))
    reference_image_input = layers.Input(shape=(105,105,3))

    anchor_embeddings = image_embedding(anchor_image_inputs)
    reference_embeddings = image_embedding(reference_image_input)

    distance = dist_layer(anchor_embeddings, reference_embeddings)

    final_output = layers.Dense(1, activation='sigmoid')(distance)

    return keras.Model(inputs=[anchor_image_inputs, reference_image_input], outputs=[final_output])
