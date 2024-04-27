import numpy as np
import utils

images, labels = utils.load_dataset()

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (50, 784))
weights_hidden_1_to_hidden_2 = np.random.uniform(-0.5, 0.5, (50, 50))
weights_hidden_2_to_output = np.random.uniform(-0.5, 0.5, (10, 50))

bias_input_to_hidden = np.zeros((50, 1))
bias_hidden_1_to_hidden_2 = np.zeros((50, 1))
bias_hidden_to_output = np.zeros((10, 1))

epochs = 4
e_loss = 0
e_correct = 0
learning_rate = 0.05

for epoch in range(epochs):
    print(f'Epoch #{epoch}')
    for image, label in zip(images, labels):
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        # forward propagation first hidden layer
        hidden_1_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden_1 = utils.sigmoid(hidden_1_raw)

        # forward propagation second hidden layer
        hidden_2_raw = bias_hidden_1_to_hidden_2 + weights_hidden_1_to_hidden_2 @ hidden_1
        hidden_2 = utils.sigmoid(hidden_2_raw)

        # forward propagation (output layer)
        output_raw = bias_hidden_to_output + weights_hidden_2_to_output @ hidden_2
        output = utils.sigmoid(output_raw)

        # loss/error calculation
        e_loss += 1 / len(output_raw) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        # backpropagation (output layer)
        delta_output = output - label
        weights_hidden_2_to_output -= learning_rate * delta_output @ np.transpose(hidden_2)
        bias_hidden_to_output -= learning_rate * delta_output

        # backpropagation second hidden layer
        delta_hidden_2 = np.transpose(weights_hidden_2_to_output) @ delta_output * (hidden_2 * (1 - hidden_2))
        weights_hidden_1_to_hidden_2 -= learning_rate * delta_hidden_2 @ np.transpose(hidden_1)
        bias_hidden_1_to_hidden_2 -= learning_rate * delta_hidden_2

        # backpropagation first hidden layer
        delta_hidden_1 = np.transpose(weights_hidden_1_to_hidden_2) @ delta_hidden_2 * (hidden_1 * (1 - hidden_1))
        weights_input_to_hidden -= learning_rate * delta_hidden_1 @ np.transpose(image)
        bias_input_to_hidden -= learning_rate * delta_hidden_1

    print(f'Loss: {round((e_loss[0] / images.shape[0]) * 100, 3)}%')
    print(f'Accuracy: {round((e_correct / images.shape[0]) * 100, 3)}%')
    e_loss = 0
    e_correct = 0
    utils.save_model(weights_input_to_hidden, weights_hidden_1_to_hidden_2, weights_hidden_2_to_output)
