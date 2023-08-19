import numpy as np


def load_dataset():
    with np.load('mnist.npz') as f:
        # convert from rgb to unit rgb (0-1)
        images = f['x_train'].astype('float32') / 255

        # reshape from (60000,28,28) into (60000,784)
        images = np.reshape(images, (60000, 784))

        # labels
        labels = f['y_train']

        # convert to output layer format
        labels = np.eye(10)[labels]

        return images, labels


def save_model(weights_input_to_hidden, weights_hidden_to_output):
    np.savez('model.npz', weights_hidden_to_output=weights_hidden_to_output,
             weights_input_to_hidden=weights_input_to_hidden)


def load_model():
    with np.load('model.npz') as f:
        return f['weights_input_to_hidden'], f['weights_hidden_to_output']
