import tensorflow as tf
from tensorflow import keras
import os

def get_processed_data():
    cwd = os.getcwd()
    anchor_path = os.path.join(cwd, "data\\anchor")
    positive_path = os.path.join(cwd, "data\\positive")
    negative_path = os.path.join(cwd, "data\\negative")

    anchor = tf.data.Dataset.list_files(anchor_path+'\*.jpg').take(300)
    positive = tf.data.Dataset.list_files(positive_path+'\*.jpg').take(300)
    negative = tf.data.Dataset.list_files(negative_path+'\*.jpg').take(300)

    def preprocess(file_path):
        bytes_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(bytes_img)
        img = tf.image.resize(img, (105,105))
        img = img/255.0
        return img

    positive_data = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(positive)))))
    negative_data = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(negative)))))

    data = positive_data.concatenate(negative_data)

    def preprocess_multiple(anchor, val, label):
        return (preprocess(anchor), preprocess(val), label)

    data = data.map(preprocess_multiple)\
            .shuffle(buffer_size=1024)\
            .cache()
    
    train_data = data.take(round(len(data)*0.7))
    train_data = train_data.batch(16)\
                            .prefetch(8)


    test_data = data.skip(round(len(data)*0.7))
    test_data = test_data.take(round(len(data)*0.3))
    test_data = test_data.batch(16)
    
    return train_data, test_data