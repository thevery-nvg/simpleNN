import numpy as np


def load_dataset():
    with np.load('mnist.npz') as f:
        # convert from rgb to unit rgb (0-1)
        x_train = f['x_train'].astype('float32') / 255

        # reshape from (60000,28,28) into (60000,784)
        x_train = np.reshape(x_train, (60000, 784))

        # labels
        y_train = f['y_train']

        # convert to output layer format
        y_train = np.eye(10)[y_train]

        return x_train, y_train


def save_model(weights_input_to_hidden, weights_hidden_to_output):
    np.savez('model.npz', weights_hidden_to_output=weights_hidden_to_output, weights_input_to_hidden=weights_input_to_hidden)


def load_model():
    with np.load('model.npz') as f:
        weights_input_to_hidden = f['weights_input_to_hidden']
        weights_hidden_to_output = f['weights_hidden_to_output']
        return weights_input_to_hidden, weights_hidden_to_output
