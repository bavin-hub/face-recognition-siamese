import tensorflow as tf
from tensorflow import keras
import keras

def get_loss_and_optimizer():
    bce = keras.losses.BinaryCrossentropy()
    optim = keras.optimizers.Adam(learning_rate=1e-4)

    return bce, optim