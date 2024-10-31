import tensorflow as tf
from model import build_model
from loss_optimizer import get_loss_and_optimizer
from preprocess_data import get_processed_data
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=2, 
                    type = int,
                    help ='total epochs to be run')

model = build_model()
binary_crossentropy_loss, optimizer = get_loss_and_optimizer()

################ load and process the data ################

train_data, test_data = get_processed_data()

################ checkpoint ################

checkpoints_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoints_dir, "ckpt")
checkpoint = tf.train.Checkpoint(opt=optimizer, siamese_model=model)

################ training ################

@tf.function
def train_step(batch):

    with tf.GradientTape() as tape:
        x = batch[:2]
        y = batch[2]
        
        pred = model(x, training=True)
        loss = binary_crossentropy_loss(y, pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def train(data, epochs):
    for epoch in range(epochs):
        loss = None
        print('\n Epoch {}/{}'.format(epoch+1, epochs, loss))
        prog_bar = tf.keras.utils.Progbar(len(data))

        for idx, batch in enumerate(data):
            loss = train_step(batch)
            prog_bar.update(idx+1)
        
        if epoch % 2 ==0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print(f"Loss -> {loss}")

if __name__=="__main__":
    args = parser.parse_args()
    epochs = args.epochs
    train(train_data, epochs)

