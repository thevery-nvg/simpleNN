import random, utils
import numpy as np
import matplotlib.pyplot as plt


images, labels = utils.load_dataset()
weights_input_to_hidden, weights_hidden_1_to_hidden_2, weights_hidden_2_to_output = utils.load_model()

bias_input_to_hidden = np.zeros((50, 1))  
bias_hidden_1_to_hidden_2 = np.zeros((50, 1))  
bias_hidden_to_output = np.zeros((10, 1))  


test_image = random.choice(images)


image = np.reshape(test_image, (-1, 1))

hidden_1_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden_1 = 1 / (1 + np.exp(-hidden_1_raw))

hidden_2_raw = bias_hidden_1_to_hidden_2 + weights_hidden_1_to_hidden_2 @ hidden_1
hidden_2 = 1 / (1 + np.exp(-hidden_2_raw))

output_raw = bias_hidden_to_output + weights_hidden_2_to_output @ hidden_2
output = 1 / (1 + np.exp(-output_raw))

plt.imshow(test_image.reshape(28, 28), cmap='Greys')
plt.title(f'NN suggests the number is: {output.argmax()}')
plt.show()
