import random, utils
import numpy as np
import matplotlib.pyplot as plt

# CHECK
images, labels = utils.load_dataset()
weights_input_to_hidden, weights_hidden_to_output = utils.load_model()

bias_input_to_hidden = np.zeros((20, 1))
bias_hidden_to_output = np.zeros((10, 1))

test_image = random.choice(images)
# Predict
image = np.reshape(test_image, (-1, 1))
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw))
output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap='Greys')
plt.title(f'NN suggests the number is: {output.argmax()}')
plt.show()
