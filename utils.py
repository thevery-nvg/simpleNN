import numpy as np


def load_dataset():
    with np.load('new_data.npz') as f:
        # convert from rgb to unit rgb (0-1)
        images = f['x_train'].astype('float32') / 255

        # reshape from (60000,28,28) into (60000,784)
        images = np.reshape(images, (images.shape[0], 784))
        
        labels = f['y_train']
        labels = np.eye(10)[labels]
        return images, labels


def save_model(weights_input_to_hidden, weights_hidden_1_to_hidden_2, weights_hidden_2_to_output):
    np.savez('model.npz', weights_input_to_hidden=weights_input_to_hidden,
             weights_hidden_1_to_hidden_2=weights_hidden_1_to_hidden_2,
             weights_hidden_2_to_output=weights_hidden_2_to_output)


def load_model():
    with np.load('model.npz') as f:
        return f['weights_input_to_hidden'], f['weights_hidden_1_to_hidden_2'], f['weights_hidden_2_to_output']


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def tanh(x):
    return np.tanh(x)
