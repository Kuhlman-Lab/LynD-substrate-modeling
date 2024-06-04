import numpy as np

import matplotlib.pyplot as plt

coef = np.load('coef.npy')
print(coef.shape)

coef = np.reshape(coef, (6, 20)).T
print(coef.shape)

plt.imshow(coef, cmap='magma')
plt.title('ML Model Feature Importance Map')
plt.xticks(np.arange(coef.shape[1]), np.arange(coef.shape[1]) + 1)
plt.yticks(np.arange(coef.shape[0]), 'ACDEFGHIKLMNPQRSTVWY')
plt.show()
